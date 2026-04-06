import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------
# v0_Copied from Tests --- 04/06
# --------------------------------

# ----------
# Encoder 1
# ----------
class ConvBlock(nn.Module):
    """
    Common Convolution Block
    """
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
    Compute Sobel edge magnitude from RGB image.
    Input:  [B, 3, H, W]
    Output: [B, 1, H, W]
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return mag


class FeaturePyramidCNN(nn.Module):
    """
    Shared CNN backbone for all input images.
    Input:  [B, 4, H, W] after RGB + Sobel concatenation
    Output: [B, C, H/8, W/8]
    """
    def __init__(self, in_ch=4, base_ch=16, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, base_ch, k=3, s=2, p=1),          # H/2
            ConvBlock(base_ch, base_ch, k=3, s=1, p=1),

            ConvBlock(base_ch, base_ch * 2, k=3, s=2, p=1),    # H/4
            ConvBlock(base_ch * 2, base_ch * 2, k=3, s=1, p=1),

            ConvBlock(base_ch * 2, out_ch, k=3, s=2, p=1),     # H/8
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, x):
        return self.net(x)


class LocalCorrelation(nn.Module):
    """
    PWC-style local correlation.
    For each pixel in feat1, correlate with a local window in feat2.

    Input:
        feat1, feat2: [B, C, H, W]
    Output:
        corr volume:  [B, (2r+1)^2, H, W]
    """
    def __init__(self, radius=4):
        super().__init__()
        self.radius = radius

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        r = self.radius

        # Normalize across channel for more stable dot product
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)

        feat2_pad = F.pad(feat2, (r, r, r, r))
        corr_list = []

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                shifted = feat2_pad[:, :, dy + r:dy + r + H, dx + r:dx + r + W]
                corr = torch.sum(feat1 * shifted, dim=1, keepdim=True)  # [B,1,H,W]
                corr_list.append(corr)

        corr_volume = torch.cat(corr_list, dim=1)  # [B, (2r+1)^2, H, W]
        return corr_volume


class PairwiseFlowEncoder(nn.Module):
    """
    First-stage encoder:
      image pair -> Sobel augmentation -> shared CNN -> correlation -> fused motion embedding

    Returns:
      pair_feat: [B, embed_ch, H/8, W/8]
      flow_init: [B, 2, H/8, W/8]   (optional coarse flow)
      corr:      [B, (2r+1)^2, H/8, W/8]
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
                nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
            )

    def _augment_with_sobel(self, img):
        edge = self.sobel(img)
        img_aug = torch.cat([img, edge], dim=1)  # [B, 4, H, W]
        return img_aug

    def forward(self, img1, img2):
        """
        img1, img2: [B, 3, H, W]
        """
        img1_aug = self._augment_with_sobel(img1)
        img2_aug = self._augment_with_sobel(img2)

        feat1 = self.feature_net(img1_aug)   # [B, C, H/8, W/8]
        feat2 = self.feature_net(img2_aug)   # [B, C, H/8, W/8]

        corr = self.corr(feat1, feat2)   # [B, corr_ch, H/8, W/8]

        fused = torch.cat([feat1, corr, feat2], dim=1)
        pair_feat = self.fuse(fused)

        if self.predict_flow:
            flow_init = self.flow_head(pair_feat)
        else:
            flow_init = None

        return {
            "feat1": feat1,
            "feat2": feat2,
            "corr": corr,
            "pair_feat": pair_feat,
            "flow_init": flow_init,
        }


class SequencePairEncoder(nn.Module):
    """
    Apply pairwise encoder over a sequence of frames.

    Input:
        imgs: [B, T, 3, H, W]

    Output:
        pair_feats: [B, T-1, C, H/8, W/8]
        flow_inits: [B, T-1, 2, H/8, W/8]
        corrs:      [B, T-1, K, H/8, W/8]
    """
    def __init__(self, feat_ch=64, corr_radius=4, embed_ch=128, predict_flow=True):
        super().__init__()
        self.pair_encoder = PairwiseFlowEncoder(
            feat_ch=feat_ch,
            corr_radius=corr_radius,
            embed_ch=embed_ch,
            predict_flow=predict_flow
        )

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        assert C == 3, f"Expected RGB input, got C={C}"

        pair_feats = []
        flow_inits = []
        corrs = []

        for t in range(T - 1):
            out = self.pair_encoder(imgs[:, t], imgs[:, t + 1])
            pair_feats.append(out["pair_feat"])
            corrs.append(out["corr"])
            if out["flow_init"] is not None:
                flow_inits.append(out["flow_init"])

        pair_feats = torch.stack(pair_feats, dim=1)   # [B, T-1, embed_ch, H/8, W/8]
        corrs = torch.stack(corrs, dim=1)             # [B, T-1, corr_ch, H/8, W/8]

        if len(flow_inits) > 0:
            flow_inits = torch.stack(flow_inits, dim=1)
        else:
            flow_inits = None

        return {
            "pair_feats": pair_feats,
            "flow_inits": flow_inits,
            "corrs": corrs,
        }

# ----------
# Encoder 2
# ----------

# ---------------------------------------------------
# Class 1: visual branch for raw images
# Input: raw RGB frames
# ---------------------------------------------------
class VisualBranchCNN(nn.Module):
    """
    Input:
        imgs: [B, T, 3, H, W]

    Output:
        visual_feats: [B, T, out_ch, H', W']
    """
    def __init__(self, in_ch=3, base_ch=32, out_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, base_ch, k=3, s=2, p=1),          # H/2
            ConvBlock(base_ch, base_ch, k=3, s=1, p=1),

            ConvBlock(base_ch, base_ch * 2, k=3, s=2, p=1),    # H/4
            ConvBlock(base_ch * 2, base_ch * 2, k=3, s=1, p=1),

            ConvBlock(base_ch * 2, base_ch * 4, k=3, s=2, p=1),  # H/8 -- Change specifically made here, to match with
                                                             # motion CNN behavior.
            ConvBlock(base_ch * 4, base_ch * 2, k=3, s=1, p=1),

            ConvBlock(base_ch * 2, out_ch, k=3, s=1, p=1),
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        x = imgs.reshape(B * T, C, H, W)
        x = self.encoder(x)
        _, C2, H2, W2 = x.shape
        x = x.reshape(B, T, C2, H2, W2)
        return x


# ---------------------------------------------------
# Class 2: motion / embedding branch
# Input: embeddings from encoder 1
# ---------------------------------------------------
class MotionBranchCNN(nn.Module):
    """
    Input:
        motion_feats: [B, Tm, Cm, H, W]

    Output:
        motion_out:   [B, Tm, out_ch, H, W]
    """
    def __init__(self, in_ch=128, hidden_ch=128, out_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, out_ch, k=3, s=1, p=1),
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, motion_feats):
        B, Tm, C, H, W = motion_feats.shape
        x = motion_feats.reshape(B * Tm, C, H, W)
        x = self.encoder(x)
        _, C2, H2, W2 = x.shape
        x = x.reshape(B, Tm, C2, H2, W2)
        return x


# ---------------------------------------------------
# Class 3: spatial-temporal fusion
# Frame-stacked 2D conv
# ---------------------------------------------------
class SpatialTemporalFusion(nn.Module):
    """
    Fuses:
      - visual features from raw frames
      - motion features from encoder 1

    Strategy:
      1. align temporal length
      2. concatenate along channel
      3. stack frames along channel dimension
      4. apply 2D convolutions for temporal fusion

    Example:
      visual_feats: [B, 4, Cv, H, W]
      motion_feats: [B, 3, Cm, H, W]

    We align visual_feats to pairwise time steps:
      use visual_feats[:, :-1]  -> [B, 3, Cv, H, W]

    Output:
      fused: [B, out_ch, H, W]
      OR sequence fused_seq: [B, Tm, hidden_ch, H, W] if needed
    """
    def __init__(self, visual_ch=64, motion_ch=64, num_pairs=3, hidden_ch=128, out_ch=128):
        super().__init__()
        self.num_pairs = num_pairs
        self.in_ch = (visual_ch + motion_ch) * num_pairs

        self.fusion = nn.Sequential(
            ConvBlock(self.in_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, visual_feats, motion_feats):
        """
        visual_feats: [B, T, Cv, H, W]
        motion_feats: [B, T-1, Cm, H, W]

        returns:
            fused: [B, out_ch, H, W]
        """
        B, Tv, Cv, H, W = visual_feats.shape
        Bm, Tm, Cm, Hm, Wm = motion_feats.shape

        assert B == Bm, f"Batch size mismatch, with vision B_size {B}, motion B_size {Bm}"
        assert H == Hm and W == Wm, f"Spatial size mismatch, with vision H {H} - {Hm}, W {W} - {Wm}"

        # align raw-frame visual features to pairwise motion time steps
        # example: from 4 frames keep first 3 features to align with 3 flows
        visual_aligned = visual_feats[:, :Tm]   # [B, Tm, Cv, H, W]

        # concatenate branch outputs at each time step
        fused_seq = torch.cat([visual_aligned, motion_feats], dim=2)   # [B, Tm, Cv+Cm, H, W]

        # frame-stacked 2D conv:
        # stack temporal dimension into channel dimension
        fused_seq = fused_seq.reshape(B, Tm * (Cv + Cm), H, W)

        fused = self.fusion(fused_seq)   # [B, out_ch, H, W] !!! Tm channel is collapsed
        return fused


class SpatialTemporalFusion_timeAware(nn.Module):
    """
    The same as class SpatialTemporalFusion. But returns:

    Output:
      fused: [B, Tm, out_ch, H, W]
    """

    def __init__(self, visual_ch=64, motion_ch=64, num_pairs=3, hidden_ch=128, out_ch=128):
        super().__init__()
        self.num_pairs = num_pairs
        self.visual_ch = visual_ch
        self.motion_ch = motion_ch
        self.fuse_in = visual_ch + motion_ch
        self.out_ch = out_ch

        self.fusion = nn.Sequential(
            ConvBlock(self.fuse_in, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, out_ch, k=3, s=1, p=1),
        )

    def forward(self, visual_feats, motion_feats):
        """
        visual_feats: [B, T, Cv, H, W]
        motion_feats: [B, T-1, Cm, H, W]

        returns:
            fused: [B, out_ch, H, W]
        """
        B, Tv, Cv, H, W = visual_feats.shape
        Bm, Tm, Cm, Hm, Wm = motion_feats.shape

        assert B == Bm, f"Batch size mismatch: {B} vs {Bm}"
        assert H == Hm and W == Wm, f"Spatial size mismatch: {(H, W)} vs {(Hm, Wm)}"
        assert Cv == self.visual_ch, f"Expected visual_ch={self.visual_ch}, got {Cv}"
        assert Cm == self.motion_ch, f"Expected motion_ch={self.motion_ch}, got {Cm}"
        assert Tv >= Tm, f"Need at least Tm visual steps, got Tv={Tv}, Tm={Tm}"

        # align raw-frame visual features to pairwise motion time steps
        # example: from 4 frames keep first 3 features to align with 3 flows
        visual_aligned = visual_feats[:, :Tm]  # [B, Tm, Cv, H, W]

        # concatenate branch outputs at each time step
        fused_seq = torch.cat([visual_aligned, motion_feats], dim=2)  # [B, Tm, Cv+Cm, H, W]

        # frame-stacked 2D conv:
        # stack temporal dimension into channel dimension
        fused_seq = fused_seq.reshape(B * Tm, Cv + Cm, H, W)
        fused_seq = self.fusion(fused_seq)  # [B*Tm, out_ch, H, W]
        fused_seq = fused_seq.reshape(B, Tm, self.out_ch, H, W)

        return fused_seq
