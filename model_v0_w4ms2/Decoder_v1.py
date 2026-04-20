
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ContextNet(nn.Module):
    def __init__(self, in_ch, hidden_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, hidden_ch, k=3, s=1, p=1, d=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=2, d=2),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=4, d=4),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=8, d=8),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=16, d=16),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1, d=1),
            nn.Conv2d(hidden_ch, 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def warp_image(img, flow):
    """
    img:  [B, C, H, W]
    flow: [B, 2, H, W] in pixel units
    """
    B, C, H, W = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing='ij'
    )
    base_grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).expand(B, -1, -1, -1)
    sample_grid = base_grid + flow

    sample_grid_x = 2.0 * sample_grid[:, 0] / max(W - 1, 1) - 1.0
    sample_grid_y = 2.0 * sample_grid[:, 1] / max(H - 1, 1) - 1.0
    sample_grid = torch.stack((sample_grid_x, sample_grid_y), dim=-1)

    return F.grid_sample(
        img,
        sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )


def sobel_grad_map(x):
    """
    x: [B, C, H, W]
    return: [B, 1, H, W]
    """
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[1,  2,  1],
         [0,  0,  0],
         [-1, -2, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


class RefinementHead(nn.Module):
    """
    Residual refinement at full resolution.
    Inputs:
        coarse_flow      : [B, 2, H, W]
        feat_up          : [B, C, H, W]
        img_src / img_tgt: [B, 3, H, W]
    Produces:
        delta_flow       : [B, 2, H, W]
    """
    def __init__(self, feat_ch, hidden_ch=64):
        super().__init__()
        # feat_up + flow + src + tgt + warped + residual_mag + edge
        in_ch = feat_ch + 2 + 3 + 3 + 3 + 1 + 1
        self.net = nn.Sequential(
            ConvBlock(in_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch // 2, k=3, s=1, p=1),
            nn.Conv2d(hidden_ch // 2, 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, coarse_flow, feat_up, img_src, img_tgt):
        warped_src = warp_image(img_src, coarse_flow)
        residual_mag = torch.abs(warped_src - img_tgt).mean(dim=1, keepdim=True)
        edge_map = sobel_grad_map(img_src)
        x = torch.cat(
            [feat_up, coarse_flow, img_src, img_tgt, warped_src, residual_mag, edge_map],
            dim=1,
        )
        return self.net(x)


class FlowDecoder_single_v1(nn.Module):
    """
    Compared with the original decoder:
    1. keep coarse low-resolution prediction
    2. keep context refinement at low resolution
    3. upsample features and flow
    4. run a full-resolution residual refinement head using image evidence
    """

    def __init__(self, in_ch=128, hidden_ch=128, upsample=8, use_prev_flow=True):
        super().__init__()
        self.upsample = upsample
        self.use_prev_flow = use_prev_flow

        effective_in = in_ch + (2 if self.use_prev_flow else 0)

        self.conv0 = nn.Sequential(
            ConvBlock(effective_in, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
        )

        self.flow_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)

        self.low_res_refine = nn.Sequential(
            ConvBlock(hidden_ch + 2, hidden_ch),
            ConvBlock(hidden_ch, hidden_ch),
        )
        self.delta_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        self.context_net = ContextNet(hidden_ch, hidden_ch=hidden_ch)

        self.full_res_refine = RefinementHead(feat_ch=hidden_ch, hidden_ch=hidden_ch)

    def forward(self, z, img_src=None, img_tgt=None, flow_prev=None):
        B, C, H, W = z.shape

        if self.use_prev_flow:
            if flow_prev is None:
                flow_prev_down = torch.zeros(B, 2, H, W, device=z.device, dtype=z.dtype)
            else:
                flow_prev_down = F.interpolate(
                    flow_prev,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ) / float(self.upsample)
            z = torch.cat((z, flow_prev_down), dim=1)

        feat = self.conv0(z)
        flow = self.flow_head(feat)

        x = torch.cat((feat, flow), dim=1)
        x = self.low_res_refine(x)
        delta = self.delta_head(x)
        flow_refined = flow + delta

        context_delta = self.context_net(x)
        if context_delta.shape[-2:] != flow_refined.shape[-2:]:
            context_delta = F.interpolate(
                context_delta,
                size=flow_refined.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        flow_low = flow_refined + context_delta

        flow_up = F.interpolate(
            flow_low,
            scale_factor=self.upsample,
            mode='bilinear',
            align_corners=False,
        ) * float(self.upsample)

        if img_src is None or img_tgt is None:
            return flow_up

        feat_up = F.interpolate(
            feat,
            scale_factor=self.upsample,
            mode='bilinear',
            align_corners=False,
        )
        delta_full = self.full_res_refine(flow_up, feat_up, img_src, img_tgt)
        flow_final = flow_up + delta_full
        return flow_final


class FlowDecoder(nn.Module):
    """
    Input:
        fused_seq: [B, Tm, C, H, W]
        imgs    : [B, Tm+1, 3, H_out, W_out]  (optional, used for refinement)
    Output:
        flows   : [B, Tm, 2, H_out, W_out]
    """

    def __init__(self, in_ch=128, hidden_ch=128, upsample=8, use_prev_flow=True):
        super().__init__()
        self.decoder = FlowDecoder_single_v1(in_ch, hidden_ch, upsample, use_prev_flow)
        self.use_prev_flow = use_prev_flow

    def forward(self, fused_seq, imgs=None):
        B, Tm, C, H, W = fused_seq.shape

        flows = []
        flow_prev = None
        for t in range(Tm):
            z = fused_seq[:, t]
            img_src = imgs[:, t] if imgs is not None else None
            img_tgt = imgs[:, t + 1] if imgs is not None else None

            if self.use_prev_flow:
                flow_t = self.decoder(z, img_src=img_src, img_tgt=img_tgt, flow_prev=flow_prev)
            else:
                flow_t = self.decoder(z, img_src=img_src, img_tgt=img_tgt)

            flows.append(flow_t)
            flow_prev = flow_t.detach()

        return torch.stack(flows, dim=1)
