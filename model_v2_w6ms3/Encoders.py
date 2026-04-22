import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgen.executorch.api.et_cpp import return_type



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


class FeaturePyramidCNN(nn.Module):
    """
    Shared CNN backbone for all input images.
    Input:  [B, 3, H, W]
    Output: [B, C, H/8, W/8]
    """
    def __init__(self, in_ch=3, base_ch=16, out_ch=64):
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
      image pair -> shared CNN -> correlation -> fused motion embedding

    Returns:
      pair_feat: [B, embed_ch, H/8, W/8]
      flow_init: [B, 2, H/8, W/8]   (optional coarse flow)
      corr:      [B, (2r+1)^2, H/8, W/8]
    """
    def __init__(self, feat_ch=64, corr_radius=4, embed_ch=128, predict_flow=True):
        super().__init__()
        self.feature_net = FeaturePyramidCNN(in_ch=3, base_ch=16, out_ch=feat_ch)
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

    def forward(self, img1, img2):
        """
        img1, img2: [B, 3, H, W]
        """
        feat1 = self.feature_net(img1)   # [B, C, H/8, W/8]
        feat2 = self.feature_net(img2)   # [B, C, H/8, W/8]

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
class SpatialTemporalFusion_timeAware(nn.Module):
    """
    Lightweight pre-UNO fusion for a B*Tm workflow.

    This module intentionally removes the older heavy temporal mixing via
    frame-stacked 2D convolutions across all time steps. Instead it keeps the
    output contract expected by the downstream UNO + decoder stack:

      input  : visual_feats [B, T,  Cv, H, W]
               motion_feats [B, Tm, Cm, H, W]
      output : fused_seq    [B, Tm, out_ch, H, W]

    Temporal awareness is kept lightweight and local before UNO by injecting:
      1) visual forward difference  phi_v(t+1) - phi_v(t)
      2) motion causal difference   phi_m(t)   - phi_m(t-1)
      3) causal running mean of pairwise fused features

    The heavier multi-scale spatial reasoning is then left to UNO.
    """

    def __init__(self, visual_ch=64, motion_ch=64, num_pairs=3, hidden_ch=128, out_ch=128):
        super().__init__()
        self.num_pairs = num_pairs
        self.visual_ch = visual_ch
        self.motion_ch = motion_ch
        self.out_ch = out_ch

        # Current pairwise content.
        self.local_in = visual_ch + motion_ch

        # Lightweight temporal hints fed into UNO instead of performing a heavy
        # temporal fusion block here.
        self.temporal_hint_in = visual_ch + motion_ch

        # local pair content + temporal difference hints + causal context
        self.pre_uno_in = self.local_in + self.temporal_hint_in + self.local_in

        self.local_proj = nn.Sequential(
            ConvBlock(self.pre_uno_in, hidden_ch, k=1, s=1, p=0),
            ConvBlock(hidden_ch, hidden_ch, k=3, s=1, p=1),
            ConvBlock(hidden_ch, out_ch, k=3, s=1, p=1),
        )

        self.residual_proj = nn.Conv2d(self.local_in, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, visual_feats, motion_feats):
        """
        visual_feats: [B, T,  Cv, H, W]
        motion_feats: [B, Tm, Cm, H, W]

        returns:
            fused_seq: [B, Tm, out_ch, H, W]
        """
        B, Tv, Cv, H, W = visual_feats.shape
        Bm, Tm, Cm, Hm, Wm = motion_feats.shape

        assert B == Bm, f"Batch size mismatch: {B} vs {Bm}"
        assert H == Hm and W == Wm, f"Spatial size mismatch: {(H, W)} vs {(Hm, Wm)}"
        assert Cv == self.visual_ch, f"Expected visual_ch={self.visual_ch}, got {Cv}"
        assert Cm == self.motion_ch, f"Expected motion_ch={self.motion_ch}, got {Cm}"
        assert Tv >= Tm, f"Need at least Tm visual steps, got Tv={Tv}, Tm={Tm}"

        # Align raw-frame visual features to pairwise motion time steps.
        visual_aligned = visual_feats[:, :Tm]  # [B, Tm, Cv, H, W]

        # Optional next-step visual cue for local temporal awareness.
        if Tv >= Tm + 1:
            visual_next = visual_feats[:, 1:Tm + 1]
            visual_delta = visual_next - visual_aligned
        else:
            visual_delta = torch.zeros_like(visual_aligned)

        # Motion causal delta: each step knows how motion embedding changes from
        # the previous pair, while staying compatible with B*Tm processing.
        motion_prev = torch.zeros_like(motion_feats)
        if Tm > 1:
            motion_prev[:, 1:] = motion_feats[:, :-1]
        motion_delta = motion_feats - motion_prev

        # Current per-step pair content.
        pair_local = torch.cat([visual_aligned, motion_feats], dim=2)  # [B, Tm, Cv+Cm, H, W]

        # Temporal hints that will be handed to UNO as extra channels.
        temporal_hint = torch.cat([visual_delta, motion_delta], dim=2)  # [B, Tm, Cv+Cm, H, W]

        # Lightweight causal context summary over previous pairwise features.
        running_context = []
        running_sum = torch.zeros_like(pair_local[:, 0])
        for t in range(Tm):
            running_sum = running_sum + pair_local[:, t]
            running_context.append(running_sum / float(t + 1))
        running_context = torch.stack(running_context, dim=1)  # [B, Tm, Cv+Cm, H, W]

        # Keep the UNO workflow on B*Tm samples while preserving output shape.
        pre_uno = torch.cat([pair_local, temporal_hint, running_context], dim=2)
        pre_uno = pre_uno.reshape(B * Tm, self.pre_uno_in, H, W)
        pair_local_2d = pair_local.reshape(B * Tm, self.local_in, H, W)

        fused_seq = self.local_proj(pre_uno) + self.residual_proj(pair_local_2d)
        fused_seq = fused_seq.reshape(B, Tm, self.out_ch, H, W)

        return fused_seq


# ---------------------------------------------------
# Compatible with FullPipeline_v1.6
# Better time-aware UNO input
# Build 2D UNO input by stacking time into channels
# ---------------------------------------------------
def flow_spatial_grads(flow):
    grad_x = torch.zeros_like(flow)
    grad_y = torch.zeros_like(flow)

    grad_x[..., :, 1:] = flow[..., :, 1:] - flow[..., :, :-1]
    grad_y[..., 1:, :] = flow[..., 1:, :] - flow[..., :-1, :]

    return grad_x, grad_y

def downsample_valid_mask(valid, size_hw):
    if valid.dim()==3:
        valid = valid.unsqueeze(1).float()
    elif valid.dim()==4:
        valid = valid.float()
    else:
        raise ValueError(f'Unexpected shape {valid.shape}')

    valid_ds = F.interpolate(valid, size=size_hw, mode='nearest')
    return valid_ds

def build_uno_input_2d(fused_seq, flow_inits, valid_mask=None):
    """
    Inputs:
        fused_seq: [B, Tm, Cf, H, W]
        flow_inits: [B, Tm, 2, H, W]
        valid_mask: [B, 1, H, W] or None
    Output:
        uno_in: [B, Tm*(Cf + 2 + 4) + valid_extra, H, W]

    Include:
        fused latents; flow init; grad_x (flow init); grad_y (flow init)
    per timestep

    valid mask is global
    """
    B, Tm, Cf, H, W = fused_seq.shape
    assert flow_inits is not None, 'flow_inits required.'
    assert flow_inits.shape[:2] == (B, Tm)
    assert flow_inits.shape[2] == 2
    assert flow_inits.shape[3:] == (H, W)

    grad_x, grad_y = flow_spatial_grads(flow_inits)

    step = torch.cat(
        [fused_seq, flow_inits, grad_x, grad_y,], # Cf, 2, 2, 2
        dim=2
    )
    # [B, Tm, Cf+6, H, W]
    step = step.permute(0, 2, 1, 3, 4).contiguous()
    uno_in = step.view(B, Tm * (Cf + 6), H, W)

    if valid_mask is not None:
        assert valid_mask.shape == (B, 1, H, W)
        uno_in = torch.cat([uno_in, valid_mask], dim=1)

    return uno_in

class UNOLatentResidualHead(nn.Module):
    def __init__(self, out_ch, latent_ch, num_pairs):
        super().__init__()
        self.out_ch = out_ch
        self.latent_ch = latent_ch
        self.num_pairs = num_pairs

        self.out_proj = nn.Sequential(
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
            nn.Conv2d(out_ch, latent_ch*num_pairs, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, uno_feat, B, Tm, H, W):
        delta = self.out_proj(uno_feat)
        delta = delta.view(B, Tm, self.latent_ch, H, W)
        return delta

# ---------------------------------------------------
# Compatible with FullPipeline_v1.7
# Better time-aware UNO input
# Build 2D UNO input for integrating with pairwise encoder modules
# ---------------------------------------------------

def build_pair_uno_input(pair_feats, flow_inits=None, valid_mask=None, use_flow=False, use_grads=False):
    """
    Build UNO input for pairwise motion refinement.

    Input:
        pair_feats : [B, Tm, Cp, H, W]
        flow_inits : [B, Tm, 2,  H, W] or None
        valid_mask : [B, 1, H, W] or None
        use_flow   : whether to append flow_inits per timestep
        use_grads  : whether to append spatial grads of flow_inits per timestep

    Output:
        uno_in : [B, C_stack, H, W]

    Experiment Plan:
        v1: use_flow=False, use_grads=False   -> pair_feats only
        v2: use_flow=True,  use_grads=False   -> pair_feats + flow_inits
        v3: use_flow=True,  use_grads=True    -> pair_feats + flow_inits + grads
    """
    B, Tm, Cp, H, W = pair_feats.shape

    step_feats = [pair_feats]  # [B, Tm, Cp, H, W]

    if use_flow:
        if flow_inits is None:
            raise ValueError("flow_inits is required when use_flow=True")
        assert flow_inits.shape[:2] == (B, Tm), f"flow_inits shape mismatch: {flow_inits.shape}"
        assert flow_inits.shape[2] == 2, f"Expected flow_inits channel=2, got {flow_inits.shape[2]}"
        assert flow_inits.shape[3:] == (H, W), f"Spatial mismatch: {flow_inits.shape[3:]} vs {(H, W)}"
        step_feats.append(flow_inits)

        if use_grads:
            grad_x, grad_y = flow_spatial_grads(flow_inits)  # each [B, Tm, 2, H, W]
            step_feats.append(grad_x)
            step_feats.append(grad_y)

    step = torch.cat(step_feats, dim=2)  # [B, Tm, Cstep, H, W]
    step = step.permute(0, 2, 1, 3, 4).contiguous()  # [B, Cstep, Tm, H, W]
    uno_in = step.view(B, -1, H, W)  # stack time into channels

    if valid_mask is not None:
        assert valid_mask.shape == (B, 1, H, W), f"Expected valid_mask [B,1,H,W], got {valid_mask.shape}"
        uno_in = torch.cat([uno_in, valid_mask], dim=1)

    return uno_in

class PairwiseUNOResidualHead(nn.Module):
    """
    Map UNO output back to pairwise motion residuals.

    Input:
        uno_feat : [B, out_ch, H, W]

    Output:
        delta_pair_feats : [B, Tm, pair_ch, H, W]
    """
    def __init__(self, out_ch, pair_ch, num_pairs):
        super().__init__()
        self.out_ch = out_ch
        self.pair_ch = pair_ch
        self.num_pairs = num_pairs

        self.out_proj = nn.Sequential(
            ConvBlock(out_ch, out_ch, k=3, s=1, p=1),
            nn.Conv2d(out_ch, pair_ch * num_pairs, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, uno_feat, B, Tm, H, W):
        delta = self.out_proj(uno_feat)              # [B, Tm*pair_ch, H, W]
        delta = delta.view(B, Tm, self.pair_ch, H, W)
        return delta

class PairwiseMotionRefiner(nn.Module):
    """
    Residual operator refinement on pairwise motion embeddings.

    Pipeline:
        pair_feats -> build_pair_uno_input -> UNO -> residual head -> refined_pair_feats

    Intended use:
        refined_pair_feats = PairwiseMotionRefiner(...)(pair_feats, flow_inits, valid_lowres)
        motion_feats = motion_branch(refined_pair_feats)
    """
    def __init__(
        self,
        uno_module,
        out_ch=128,
        pair_ch=128,
        num_pairs=3,
        use_flow=False,
        use_grads=False,
        use_valid=True,
    ):
        super().__init__()
        self.uno = uno_module
        self.use_flow = use_flow
        self.use_grads = use_grads
        self.use_valid = use_valid

        self.residual_head = PairwiseUNOResidualHead(
            out_ch=out_ch,
            pair_ch=pair_ch,
            num_pairs=num_pairs,
        )

    def forward(self, pair_feats, flow_inits=None, valid_mask=None):
        """
        Input:
            pair_feats : [B, Tm, Cp, H, W]
            flow_inits : [B, Tm, 2, H, W] or None
            valid_mask : [B, 1, H, W] or None

        Output:
            refined_pair_feats : [B, Tm, Cp, H, W]
            pair_uno_input     : [B, C_stack, H, W]
            pair_uno_feat      : [B, out_ch, H, W]
            delta_pair_feats   : [B, Tm, Cp, H, W]
        """
        B, Tm, Cp, H, W = pair_feats.shape

        pair_uno_input = build_pair_uno_input(
            pair_feats=pair_feats,
            flow_inits=flow_inits,
            valid_mask=valid_mask if self.use_valid else None,
            use_flow=self.use_flow,
            use_grads=self.use_grads,
        )

        pair_uno_feat = self.uno(pair_uno_input)
        delta_pair_feats = self.residual_head(
            pair_uno_feat,
            B=B,
            Tm=Tm,
            H=H,
            W=W,
        )

        refined_pair_feats = pair_feats + delta_pair_feats

        return {
            "refined_pair_feats": refined_pair_feats,
            "pair_uno_input": pair_uno_input,
            "pair_uno_feat": pair_uno_feat,
            "delta_pair_feats": delta_pair_feats,
        }

