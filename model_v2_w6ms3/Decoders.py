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
                bias=not norm
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# --------------------------------
# v0_Copied from Tests --- 04/06
# --------------------------------
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


class FlowDecoder_single(nn.Module):
    """
    Input:
        embeddings z: [B, C, H, W]

    Output:
        flow: [B, 2, H_out, W_out]
    """

    def __init__(self, in_ch=128, hidden_ch=128, upsample=4, use_prev_flow=True):
        super().__init__()
        self.upsample = upsample
        self.use_prev_flow = use_prev_flow

        effective_in = in_ch + (2 if self.use_prev_flow else 0)

        self.conv0 = nn.Sequential(
            ConvBlock(effective_in, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
        )

        self.flow_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        self.refine = nn.Sequential(
            ConvBlock(hidden_ch + 2, hidden_ch),
            ConvBlock(hidden_ch, hidden_ch),
        )
        self.delta_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        self.context_net = ContextNet(hidden_ch)

    def forward(self, z, flow_prev=None):
        B, C, H, W = z.shape

        if self.use_prev_flow:
            if flow_prev is None:
                flow_prev_downSample = torch.zeros(
                    B, 2, H, W, device=z.device, dtype=z.dtype
                )
            else:
                flow_prev_downSample = F.interpolate(
                    flow_prev,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )

            z = torch.cat((z, flow_prev_downSample), dim=1)  # always 128 + 2 = 130

        feat = self.conv0(z)
        flow = self.flow_head(feat)

        # high capacity.
        """
        flow_up = F.interpolate(flow, scale_factor=self.upsample, mode='bilinear', align_corners=False)
        feat_up = F.interpolate(feat, scale_factor=self.upsample, mode='bilinear', align_corners=False)

        x = torch.cat((feat_up, flow_up), dim=1)
        x = self.refine(x)

        delta = self.delta_head(x)
        flow_refined = flow_up + delta        
        """

        # a lower capacity version
        x = torch.cat((feat, flow), dim=1)
        x = self.refine(x)
        delta = self.delta_head(x)
        flow_refined = flow + delta

        context_delta = self.context_net(x)
        if context_delta.shape[-2:] != flow_refined.shape[-2:]:
            context_delta = F.interpolate(
                context_delta,
                size=flow_refined.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        flow_final = flow_refined + context_delta

        flow_final_low_capacity = F.interpolate(flow_final, scale_factor=8, mode='bilinear', align_corners=False)

        flow = flow_final_low_capacity
        return flow


class FlowDecoder(nn.Module):
    """
    Input:
        fused_seq: [B, Tm, C, H, W]
    Output:
        flows: [B, Tm, 2, H_out, W_out]
    """

    def __init__(self, in_ch=128, hidden_ch=128, upsample=4, use_prev_flow=True):
        super().__init__()
        self.decoder = FlowDecoder_single(in_ch, hidden_ch, upsample, use_prev_flow)
        self.use_prev_flow = use_prev_flow

    def _upsample_flow_to(self, flow, size_hw):
        """
        flow: [B, 2, h, w] in pixel units
        size_hw: (H, W)

        Returns:
            [B, 2, H, W] with flow magnitude scaled correctly
        """
        B, C, h, w = flow.shape
        H, W = size_hw

        if (h, w) == (H, W):
            return flow

        scale_y = H / h
        scale_x = W / w

        flow_up = F.interpolate(
            flow,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # flow is in pixel units, so values must also be scaled
        flow_up[:, 0] *= scale_x
        flow_up[:, 1] *= scale_y

        return flow_up

    def forward(self, fused_seq, flow_inits=None):
        B, Tm, C, h, w = fused_seq.shape

        flows = []
        flow_residuals = []
        flow_prev = None

        for t in range(Tm):
            z = fused_seq[:, t]

            if self.use_prev_flow:
                flow_res_t = self.decoder(z, flow_prev)
            else:
                flow_res_t = self.decoder(z)

            _, _, H, W = flow_res_t.shape

            if flow_inits is not None:
                init_t = self._upsample_flow_to(flow_inits[:, t], (H, W))
                flow_t = init_t + flow_res_t
            else:
                flow_t = flow_res_t

            flows.append(flow_t)
            flow_residuals.append(flow_res_t)
            flow_prev = flow_t.detach()

        flows = torch.stack(flows, dim=1)
        flow_residuals = torch.stack(flow_residuals, dim=1)
        return flows, flow_residuals

