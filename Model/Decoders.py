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
    """PWC-style dilated context refinement."""
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


class ConvexUpsampler(nn.Module):
    """RAFT-style convex upsampling for low-resolution flow.

    The mask predicts, for every low-resolution cell, an upsample x upsample grid of
    3x3 convex weights. The final high-resolution flow is a softmax-weighted local
    combination of the 3x3 neighboring low-resolution flow vectors.
    """
    def __init__(self, hidden_ch=64, upsample=8):
        super().__init__()
        self.upsample = upsample
        self.mask_head = nn.Sequential(
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            nn.Conv2d(hidden_ch, 9 * upsample * upsample, kernel_size=1, padding=0),
        )

    def forward(self, feat, flow_low):
        b, _, h, w = flow_low.shape
        s = self.upsample

        # RAFT convention: multiply low-res flow by the scale before convex mixing,
        # so output vectors are in full-resolution pixel units.
        flow = flow_low * float(s)

        mask = self.mask_head(feat)
        mask = mask.view(b, 1, 9, s, s, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, kernel_size=3, padding=1)
        up_flow = up_flow.view(b, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        return up_flow.view(b, 2, h * s, w * s)


class FlowDecoderSingle(nn.Module):
    """
    v26 decoder single step.

    Compared with Decoders_v20_fixed2:
      - keeps the same low-resolution flow head/refinement/context logic
      - replaces bilinear+residual high-res refinement with RAFT-style convex upsampling
    """
    def __init__(self, in_ch=128, hidden_ch=64, upsample=8, use_prev_flow=True):
        super().__init__()
        self.upsample = upsample
        self.use_prev_flow = use_prev_flow
        effective_in = in_ch + (2 if use_prev_flow else 0)

        self.conv0 = nn.Sequential(
            ConvBlock(effective_in, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
        )
        self.flow_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)

        self.low_refine = nn.Sequential(
            ConvBlock(hidden_ch + 2, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
        )
        self.low_delta_head = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        self.context_net = ContextNet(hidden_ch, hidden_ch=hidden_ch)
        self.convex_upsampler = ConvexUpsampler(hidden_ch=hidden_ch, upsample=upsample)

    def forward(self, z, flow_prev=None):
        b, _, h, w = z.shape
        if self.use_prev_flow:
            if flow_prev is None:
                flow_prev_low = torch.zeros(b, 2, h, w, device=z.device, dtype=z.dtype)
            else:
                _, _, Hprev, Wprev = flow_prev.shape
                flow_prev_low = F.interpolate(flow_prev, size=(h, w), mode="bilinear", align_corners=False)
                # Convert full-resolution pixel units to low-resolution pixel units.
                flow_prev_low[:, 0] *= w / Wprev
                flow_prev_low[:, 1] *= h / Hprev
            z = torch.cat([z, flow_prev_low], dim=1)

        feat = self.conv0(z)
        flow_low = self.flow_head(feat)

        low_x = self.low_refine(torch.cat([feat, flow_low], dim=1))
        flow_low = flow_low + self.low_delta_head(low_x) + self.context_net(low_x)

        flow_high = self.convex_upsampler(feat, flow_low)
        return flow_high


class FlowDecoder(nn.Module):
    """
    Input:
        fused_seq: [B,Tm,C,h,w]
        flow_inits: optional [B,Tm,2,h,w] or another low-resolution size
    Output:
        flows: [B,Tm,2,H,W]
        flow_residuals: [B,Tm,2,H,W]
    """
    def __init__(self, in_ch=128, hidden_ch=64, upsample=8, use_prev_flow=True):
        super().__init__()
        self.decoder = FlowDecoderSingle(in_ch, hidden_ch, upsample, use_prev_flow)
        self.use_prev_flow = use_prev_flow

    def _upsample_flow_to(self, flow, size_hw):
        b, c, h, w = flow.shape
        H, W = size_hw
        if (h, w) == (H, W):
            return flow
        flow_up = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        flow_up[:, 0] *= W / w
        flow_up[:, 1] *= H / h
        return flow_up

    def forward(self, fused_seq, flow_inits=None):
        b, tm, c, h, w = fused_seq.shape
        flows = []
        residuals = []
        flow_prev = None
        for t in range(tm):
            z = fused_seq[:, t]
            flow_res = self.decoder(z, flow_prev if self.use_prev_flow else None)
            _, _, H, W = flow_res.shape
            if flow_inits is not None:
                init_t = self._upsample_flow_to(flow_inits[:, t], (H, W))
                flow_t = init_t + flow_res
            else:
                flow_t = flow_res
            flows.append(flow_t)
            residuals.append(flow_res)
            flow_prev = flow_t.detach()
        return torch.stack(flows, dim=1), torch.stack(residuals, dim=1)
