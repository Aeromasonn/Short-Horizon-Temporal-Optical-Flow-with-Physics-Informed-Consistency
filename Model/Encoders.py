
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Common convolution -> BN -> LeakyReLU block."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SobelEdgeExtractor(nn.Module):
    """
    Compute Sobel edge magnitude from RGB images.
    Input:  [B, 3, H, W]
    Output: [B, 1, H, W]
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


class FeaturePyramidCNN(nn.Module):
    """
    Shared CNN backbone for Sobel-augmented RGB input.
    Input:  [B, 4, H, W]
    Output: [B, C, H/8, W/8]
    """
    def __init__(self, in_ch=4, base_ch=16, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, base_ch, k=3, s=2, p=1),
            ConvBlock(base_ch, base_ch, k=3, s=1, p=1),
            ConvBlock(base_ch, base_ch * 2, k=3, s=2, p=1),
            ConvBlock(base_ch * 2, base_ch * 2, k=3, s=1, p=1),
            ConvBlock(base_ch * 2, out_ch, k=3, s=2, p=1),
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, x):
        return self.net(x)


class LocalCorrelation(nn.Module):
    """PWC-style local correlation volume."""
    def __init__(self, radius=4):
        super().__init__()
        self.radius = radius

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.shape
        r = self.radius

        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)

        feat2_pad = F.pad(feat2, (r, r, r, r))
        corr_list = []

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                shifted = feat2_pad[:, :, dy + r:dy + r + h, dx + r:dx + r + w]
                corr = torch.sum(feat1 * shifted, dim=1, keepdim=True)
                corr_list.append(corr)

        return torch.cat(corr_list, dim=1)


class PairwiseFlowEncoder(nn.Module):
    """
    Sobel-enhanced pairwise encoder:
      image pair -> Sobel augmentation -> shared CNN -> local correlation -> fused pair embedding.
    """
    def __init__(self, feat_ch=64, corr_radius=4, embed_ch=128, predict_flow=True):
        super().__init__()
        self.sobel = SobelEdgeExtractor()
        self.feature_net = FeaturePyramidCNN(in_ch=4, base_ch=16, out_ch=feat_ch)
        self.corr = LocalCorrelation(radius=corr_radius)

        corr_ch = (2 * corr_radius + 1) ** 2
        fuse_in = feat_ch * 2 + corr_ch

        self.fuse = nn.Sequential(
            ConvBlock(fuse_in, 128, k=3, s=1, p=1),
            ConvBlock(128, embed_ch, k=3, s=1, p=1),
        )

        self.predict_flow = predict_flow
        if predict_flow:
            self.flow_head = nn.Sequential(
                ConvBlock(embed_ch, 64, k=3, s=1, p=1),
                nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            )

    def _augment_with_sobel(self, img):
        edge = self.sobel(img)
        return torch.cat([img, edge], dim=1)

    def forward(self, img1, img2):
        img1_aug = self._augment_with_sobel(img1)
        img2_aug = self._augment_with_sobel(img2)

        feat1 = self.feature_net(img1_aug)
        feat2 = self.feature_net(img2_aug)
        corr = self.corr(feat1, feat2)

        fused = torch.cat([feat1, corr, feat2], dim=1)
        pair_feat = self.fuse(fused)

        flow_init = self.flow_head(pair_feat) if self.predict_flow else None

        return {
            "feat1": feat1,
            "feat2": feat2,
            "corr": corr,
            "pair_feat": pair_feat,
            "flow_init": flow_init,
        }


class SequencePairEncoder(nn.Module):
    """Apply the Sobel-enhanced pairwise encoder to a frame sequence."""
    def __init__(self, feat_ch=64, corr_radius=4, embed_ch=128, predict_flow=True):
        super().__init__()
        self.pair_encoder = PairwiseFlowEncoder(
            feat_ch=feat_ch,
            corr_radius=corr_radius,
            embed_ch=embed_ch,
            predict_flow=predict_flow,
        )

    def forward(self, imgs):
        b, t, c, h, w = imgs.shape
        assert c == 3, f"Expected RGB input, got C={c}"

        pair_feats = []
        flow_inits = []
        corrs = []

        for idx in range(t - 1):
            out = self.pair_encoder(imgs[:, idx], imgs[:, idx + 1])
            pair_feats.append(out["pair_feat"])
            corrs.append(out["corr"])
            if out["flow_init"] is not None:
                flow_inits.append(out["flow_init"])

        pair_feats = torch.stack(pair_feats, dim=1)
        corrs = torch.stack(corrs, dim=1)
        flow_inits = torch.stack(flow_inits, dim=1) if flow_inits else None

        return {
            "pair_feats": pair_feats,
            "flow_inits": flow_inits,
            "corrs": corrs,
        }


class VisualBranchCNN(nn.Module):
    """Visual branch operating on raw RGB frames."""
    def __init__(self, in_ch=3, base_ch=32, out_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, base_ch, k=3, s=2, p=1),
            ConvBlock(base_ch, base_ch, k=3, s=1, p=1),
            ConvBlock(base_ch, base_ch * 2, k=3, s=2, p=1),
            ConvBlock(base_ch * 2, base_ch * 2, k=3, s=1, p=1),
            ConvBlock(base_ch * 2, base_ch * 4, k=3, s=2, p=1),
            ConvBlock(base_ch * 4, base_ch * 2, k=3, s=1, p=1),
            ConvBlock(base_ch * 2, out_ch, k=3, s=1, p=1),
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, imgs):
        b, t, c, h, w = imgs.shape
        x = imgs.reshape(b * t, c, h, w)
        x = self.encoder(x)
        _, c2, h2, w2 = x.shape
        return x.reshape(b, t, c2, h2, w2)


class MotionBranchCNN(nn.Module):
    """Motion branch operating on pairwise embeddings."""
    def __init__(self, in_ch=128, hidden_ch=128, out_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, out_ch, k=3, s=1, p=1),
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, motion_feats):
        b, tm, c, h, w = motion_feats.shape
        x = motion_feats.reshape(b * tm, c, h, w)
        x = self.encoder(x)
        _, c2, h2, w2 = x.shape
        return x.reshape(b, tm, c2, h2, w2)


class SpatialTemporalFusion_timeAware(nn.Module):
    """
    Lightweight pre-UNO fusion.
    Keeps the output contract [B, Tm, out_ch, H, W] while adding:
      - visual forward difference
      - motion causal difference
      - causal running mean context
    before the UNO latent integration stage.
    """
    def __init__(self, visual_ch=64, motion_ch=64, hidden_ch=128, out_ch=128):
        super().__init__()
        self.visual_ch = visual_ch
        self.motion_ch = motion_ch
        self.local_in = visual_ch + motion_ch
        self.temporal_hint_in = visual_ch + motion_ch
        self.pre_uno_in = self.local_in + self.temporal_hint_in + self.local_in

        self.local_proj = nn.Sequential(
            ConvBlock(self.pre_uno_in, hidden_ch, k=1, s=1, p=0),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, out_ch, k=3, s=1, p=1),
        )
        self.residual_proj = nn.Conv2d(self.local_in, out_ch, kernel_size=1)

    def forward(self, visual_feats, motion_feats):
        b, tv, cv, h, w = visual_feats.shape
        bm, tm, cm, hm, wm = motion_feats.shape

        assert b == bm
        assert (h, w) == (hm, wm)
        assert cv == self.visual_ch
        assert cm == self.motion_ch
        assert tv >= tm

        visual_aligned = visual_feats[:, :tm]

        if tv >= tm + 1:
            visual_next = visual_feats[:, 1:tm + 1]
            visual_delta = visual_next - visual_aligned
        else:
            visual_delta = torch.zeros_like(visual_aligned)

        motion_prev = torch.zeros_like(motion_feats)
        if tm > 1:
            motion_prev[:, 1:] = motion_feats[:, :-1]
        motion_delta = motion_feats - motion_prev

        pair_local = torch.cat([visual_aligned, motion_feats], dim=2)
        temporal_hint = torch.cat([visual_delta, motion_delta], dim=2)

        running_context = []
        running_sum = torch.zeros_like(pair_local[:, 0])
        for idx in range(tm):
            running_sum = running_sum + pair_local[:, idx]
            running_context.append(running_sum / float(idx + 1))
        running_context = torch.stack(running_context, dim=1)

        pre_uno = torch.cat([pair_local, temporal_hint, running_context], dim=2)
        pre_uno = pre_uno.reshape(b * tm, self.pre_uno_in, h, w)
        pair_local_2d = pair_local.reshape(b * tm, self.local_in, h, w)

        fused_seq = self.local_proj(pre_uno) + self.residual_proj(pair_local_2d)
        return fused_seq.reshape(b, tm, -1, h, w)


def flow_spatial_grads(flow):
    grad_x = torch.zeros_like(flow)
    grad_y = torch.zeros_like(flow)
    grad_x[..., :, 1:] = flow[..., :, 1:] - flow[..., :, :-1]
    grad_y[..., 1:, :] = flow[..., 1:, :] - flow[..., :-1, :]
    return grad_x, grad_y


def downsample_valid_mask(valid, size_hw):
    """
    valid: [B,H,W] or [B,1,H,W]
    returns: [B,1,h,w]
    """
    if valid.dim() == 3:
        valid = valid.unsqueeze(1).float()
    elif valid.dim() == 4:
        valid = valid.float()
    else:
        raise ValueError(f"Unexpected valid mask shape: {tuple(valid.shape)}")
    return F.interpolate(valid, size=size_hw, mode='nearest')


def build_uno_input_2d(fused_seq, flow_inits, valid_mask=None):
    """
    Build a 2D UNO input by stacking time into channels.
    Specifically:
        torch.cat([fused_seq, flow_inits, grad_x, grad_y], dim=2)
    This allows UNO to better take in time and spatial gradients.

    Inputs:
        fused_seq:   [B, Tm, Cf, H, W]
        flow_inits:  [B, Tm, 2, H, W]
        valid_mask:  [B, 1, H, W] or None
        x, y grads:  [B, Tm, 2, H, W]
    Output:
        uno_in:      [B, Tm*(Cf+6) + valid_extra, H, W]
    """
    b, tm, cf, h, w = fused_seq.shape
    if flow_inits is None:
        raise ValueError("flow_inits is required for UNO integration.")
    assert flow_inits.shape[:2] == (b, tm)
    assert flow_inits.shape[2] == 2
    assert flow_inits.shape[3:] == (h, w)

    grad_x, grad_y = flow_spatial_grads(flow_inits)

    step = torch.cat([fused_seq, flow_inits, grad_x, grad_y], dim=2)
    step = step.permute(0, 2, 1, 3, 4).contiguous()
    uno_in = step.view(b, tm * (cf + 6), h, w)

    if valid_mask is not None:
        assert valid_mask.shape == (b, 1, h, w)
        uno_in = torch.cat([uno_in, valid_mask], dim=1)

    return uno_in


class UNOLatentResidualHead(nn.Module):
    """
    Project UNO output back to per-pair latent residuals that refine the fused sequence.
    """
    def __init__(self, uno_out_ch, latent_ch, num_pairs):
        super().__init__()
        self.latent_ch = latent_ch
        self.num_pairs = num_pairs
        self.out_proj = nn.Sequential(
            ConvBlock(uno_out_ch, uno_out_ch, k=3, s=1, p=1),
            nn.Conv2d(uno_out_ch, latent_ch * num_pairs, kernel_size=1),
        )

    def forward(self, uno_feat, batch_size, num_pairs, height, width):
        delta = self.out_proj(uno_feat)
        return delta.view(batch_size, num_pairs, self.latent_ch, height, width)
