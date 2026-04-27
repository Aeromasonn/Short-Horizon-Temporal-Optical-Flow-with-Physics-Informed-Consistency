import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent  # model_v3_w6ms3

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from DataLoader import TempFlowDataset_disp
from Encoder_sober import (
    SequencePairEncoder,
    VisualBranchCNN,
    MotionBranchCNN,
    SpatialTemporalFusion_timeAware,
    build_uno_input_2d,
    downsample_valid_mask,
    UNOLatentResidualHead,
)
from Decoders_v20_fixed2 import FlowDecoder
from neuralop_seg.uno import UNO


def parse_args():
    parser = argparse.ArgumentParser(
        description="v24: v22 + multi-frame edge-aware smoothness, without v23 temporal composition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "v24_multiframe_smooth_config.json"),
        help="Path to the experiment config JSON.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint specified by experiment.checkpoint_name.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. If omitted with --resume, uses save_dir/checkpoint_name.",
    )
    parser.add_argument(
        "--additional_epochs",
        type=int,
        default=None,
        help="When resuming, train this many more epochs after the loaded checkpoint epoch.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sobel_grad_map(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def normalize_map(m: torch.Tensor) -> torch.Tensor:
    b = m.shape[0]
    m_flat = m.view(b, -1)
    m_min = m_flat.min(dim=1)[0].view(b, 1, 1, 1)
    m_max = m_flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (m - m_min) / (m_max - m_min + 1e-6)


def weighted_epe_loss(pred, gt, image, valid=None, edge_weight_scale=1.0):
    epe = torch.norm(pred - gt, dim=1, keepdim=True)
    img_edge = normalize_map(sobel_grad_map(image)).detach()
    weight = 1.0 + edge_weight_scale * img_edge

    if valid is not None:
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        valid = valid.float()
        weighted_valid = weight * valid
        return (epe * weighted_valid).sum() / (weighted_valid.sum() + 1e-6)

    return (epe * weight).sum() / (weight.sum() + 1e-6)


def make_sampling_grid(flow: torch.Tensor):
    """Return normalized grid and in-bound mask for forward flow used as backward sampling offsets.

    flow: [B,2,H,W] in full-resolution pixel units. grid samples image2 at x + flow(x),
    which reconstructs image1 when flow is image1 -> image2.
    """
    b, _, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).expand(b, -1, -1, -1)
    sample_grid = base_grid + flow

    mask = (
        (sample_grid[:, 0] >= 0.0) & (sample_grid[:, 0] <= max(w - 1, 1)) &
        (sample_grid[:, 1] >= 0.0) & (sample_grid[:, 1] <= max(h - 1, 1))
    ).float()

    sample_grid_x = 2.0 * sample_grid[:, 0] / max(w - 1, 1) - 1.0
    sample_grid_y = 2.0 * sample_grid[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((sample_grid_x, sample_grid_y), dim=-1)
    return grid, mask


def warp_image(img, flow):
    grid, _ = make_sampling_grid(flow)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def warp_valid_mask(flow):
    _, mask = make_sampling_grid(flow)
    return mask


def warp_flow(flow_to_sample, flow_grid):
    """Sample flow_to_sample at x + flow_grid(x)."""
    grid, _ = make_sampling_grid(flow_grid)
    return F.grid_sample(flow_to_sample, grid, mode="bilinear", padding_mode="border", align_corners=True)


def forward_backward_confidence(flow_fw, flow_bw, alpha=0.01, beta=0.5, gamma=2.0, floor=0.05):
    """Soft reliability confidence, not a hard sky mask.

    Forward-backward rule:
        flow_fw(x) + flow_bw(x + flow_fw(x)) should be close to 0.

    The output is in [floor, 1], except truly out-of-image samples are zero because
    photometric reconstruction is undefined outside the image. Low-texture sky can
    become low confidence, but it still participates through the floor.
    """
    bw_warped = warp_flow(flow_bw, flow_fw)
    fb_error = torch.norm(flow_fw + bw_warped, dim=1)
    mag = torch.norm(flow_fw, dim=1) + torch.norm(bw_warped, dim=1)
    scale = alpha * mag + beta
    conf = torch.exp(-gamma * fb_error / (scale + 1e-6))
    conf = torch.clamp(conf, min=floor, max=1.0)
    conf = conf * warp_valid_mask(flow_fw)
    return conf.detach(), fb_error.detach()


def align_reversed_backward_flows(flows_reversed):
    """[I3->I2,I2->I1,I1->I0] -> [I1->I0,I2->I1,I3->I2]."""
    return torch.flip(flows_reversed, dims=[1])


def multiframe_fb_confidences(flows_fw, flows_bw, alpha=0.01, beta=0.5, gamma=2.0, floor=0.05):
    confs, errors = [], []
    for t in range(flows_fw.shape[1]):
        c, e = forward_backward_confidence(
            flows_fw[:, t], flows_bw[:, t], alpha=alpha, beta=beta, gamma=gamma, floor=floor
        )
        confs.append(c)
        errors.append(e)
    return torch.stack(confs, dim=1), torch.stack(errors, dim=1)


def forward_backward_hard_mask(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5, detach=True):
    """UPFlow/UnFlow-style hard reliability mask for photometric loss.

    Valid if forward flow and warped backward flow are consistent:
        ||F_fw + warp(F_bw, F_fw)||^2 < alpha1 * (||F_fw||^2 + ||warp(F_bw)||^2) + alpha2

    alpha1=0.01 and alpha2=0.5 follow the common UnFlow / UFlow convention.
    The mask is used only to gate photometric loss; smoothness still applies everywhere.
    """
    bw_warped = warp_flow(flow_bw, flow_fw)
    fb_vec = flow_fw + bw_warped
    fb_err_sq = torch.sum(fb_vec * fb_vec, dim=1)

    fw_sq = torch.sum(flow_fw * flow_fw, dim=1)
    bw_sq = torch.sum(bw_warped * bw_warped, dim=1)
    threshold = alpha1 * (fw_sq + bw_sq) + alpha2

    mask = (fb_err_sq < threshold).float()
    mask = mask * warp_valid_mask(flow_fw)

    if detach:
        mask = mask.detach()
    return mask


def multiframe_fb_hard_masks(flows_fw, flows_bw, alpha1=0.01, alpha2=0.5):
    masks = []
    for t in range(flows_fw.shape[1]):
        masks.append(forward_backward_hard_mask(
            flows_fw[:, t], flows_bw[:, t], alpha1=alpha1, alpha2=alpha2
        ))
    return torch.stack(masks, dim=1)


def forward_backward_consistency_loss(flows_fw, flows_bw, valid_floor=0.05, robust_eps=0.01):
    """Explicit differentiable FB consistency penalty.

    Different from fb_error_mean logging, this keeps gradients and directly penalizes:
        F_fw(x) + F_bw(x + F_fw(x)) -> 0

    It uses a Charbonnier-style robust penalty and only removes truly out-of-image samples.
    valid_floor is NOT applied as a hard sky mask; it only keeps numerically stable weighting.
    """
    vals = []
    for t in range(flows_fw.shape[1]):
        fw = flows_fw[:, t]
        bw = flows_bw[:, t]
        bw_warped = warp_flow(bw, fw)
        fb_vec = fw + bw_warped
        fb_mag = torch.sqrt(torch.sum(fb_vec * fb_vec, dim=1) + robust_eps * robust_eps)
        valid = warp_valid_mask(fw)
        # Only invalid outside-image samples are removed. Sky is not masked.
        vals.append((fb_mag * valid).sum() / (valid.sum() + 1e-6))
    return torch.stack(vals).mean()


def ssim_map_loss(img1, img2, window_size=3, c1=0.01 ** 2, c2=0.03 ** 2):
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu12

    ssim_n = (2 * mu12 + c1) * (2 * sigma12 + c2)
    ssim_d = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = ssim_n / (ssim_d + 1e-6)
    return torch.clamp((1.0 - ssim_map) / 2.0, 0.0, 1.0).mean(dim=1)


def soft_census_transform(img, patch_size=7, eps=0.81):
    """Differentiable census-like transform.

    The previous hard comparison `(patch > center).float()` gives almost no useful gradient
    to the flow. This soft version follows the same idea but keeps gradients.
    """
    if img.shape[1] > 1:
        img = img.mean(dim=1, keepdim=True)
    b, c, h, w = img.shape
    p = patch_size // 2
    patches = F.unfold(F.pad(img, [p, p, p, p], mode="reflect"), kernel_size=patch_size)
    patches = patches.view(b, patch_size * patch_size, h, w)
    center = img
    diff = patches - center
    return diff / torch.sqrt(eps + diff * diff)


def pair_photometric_map(
    img_src,
    img_tgt,
    flow_src_to_tgt,
    lambda_l1=0.15,
    lambda_ssim=0.85,
    lambda_census=0.30,
    census_patch=7,
):
    """Photometric error for a source->target flow.

    Correct direction: sample target at x + flow_src_to_tgt(x), then compare to source.
    The old version sampled source and compared to target, which only matches a target->source flow.
    """
    warped_tgt = warp_image(img_tgt, flow_src_to_tgt)

    l1_map = (warped_tgt - img_src).abs().mean(dim=1)
    ssim_map = ssim_map_loss(warped_tgt, img_src)
    c_warped = soft_census_transform(warped_tgt, patch_size=census_patch)
    c_src = soft_census_transform(img_src, patch_size=census_patch)
    census_map = (c_warped - c_src).abs().mean(dim=1)

    photo = lambda_l1 * l1_map + lambda_ssim * ssim_map + lambda_census * census_map
    return photo, warp_valid_mask(flow_src_to_tgt)


def texture_confidence(img, floor=0.05):
    """Low-texture pixels get lower photometric confidence; floor prevents full removal."""
    tex = normalize_map(sobel_grad_map(img)).squeeze(1).detach()
    return torch.clamp(tex, min=floor, max=1.0)


def robust_weight_from_error(err, detach=True, beta=10.0):
    """Downweight very large photometric errors caused by occlusion/borders/illumination."""
    if detach:
        err = err.detach()
    med = err.flatten(1).median(dim=1)[0].view(-1, 1, 1)
    return torch.exp(-beta * torch.clamp(err - med, min=0.0))


def multiframe_photometric_loss(
    imgs,
    flows,
    lambda_l1=0.15,
    lambda_ssim=0.85,
    lambda_census=0.30,
    census_patch=7,
    reduction="mean",
    use_confidence=True,
    texture_floor=0.05,
    robust_beta=10.0,
    fb_conf=None,
):
    """
    imgs:  [B, T, C, H, W]
    flows: [B, T-1, 2, H, W], each flow is imgs[:, t] -> imgs[:, t+1]
    fb_conf: optional [B, T-1, H, W] soft forward-backward reliability confidence

    reduction:
      - "mean": weighted mean over adjacent pair losses
      - "min":  per-pixel min across adjacent pair losses, then mean
    """
    pair_losses = []
    pair_weights = []
    num_pairs = flows.shape[1]
    if imgs.shape[1] != num_pairs + 1:
        raise ValueError(f"Expected imgs length = flows length + 1, got {imgs.shape[1]} and {num_pairs}.")

    for t in range(num_pairs):
        pair_map, valid_warp = pair_photometric_map(
            imgs[:, t],
            imgs[:, t + 1],
            flows[:, t],
            lambda_l1=lambda_l1,
            lambda_ssim=lambda_ssim,
            lambda_census=lambda_census,
            census_patch=census_patch,
        )
        weight = valid_warp
        if use_confidence:
            weight = weight * texture_confidence(imgs[:, t], floor=texture_floor)
            weight = weight * robust_weight_from_error(pair_map, beta=robust_beta)
        if fb_conf is not None:
            weight = weight * fb_conf[:, t]
        pair_losses.append(pair_map)
        pair_weights.append(weight)

    losses = torch.stack(pair_losses, dim=1)   # [B,T-1,H,W]
    weights = torch.stack(pair_weights, dim=1) # [B,T-1,H,W]

    if reduction == "min":
        idx = losses.detach().argmin(dim=1, keepdim=True)
        chosen_loss = torch.gather(losses, 1, idx).squeeze(1)
        chosen_weight = torch.gather(weights, 1, idx).squeeze(1)
        return (chosen_loss * chosen_weight).sum() / (chosen_weight.sum() + 1e-6)
    if reduction == "mean":
        return (losses * weights).sum() / (weights.sum() + 1e-6)
    raise ValueError(f"Unsupported multiframe reduction: {reduction}")



def compose_adjacent_flows(flow_t_t1, flow_t1_t2):
    """Compose adjacent flows into a two-step flow.

    F_{t->t+2}(x) = F_{t->t+1}(x) + F_{t+1->t+2}(x + F_{t->t+1}(x))
    """
    return flow_t_t1 + warp_flow(flow_t1_t2, flow_t_t1)


def accumulated_multiframe_photometric_loss(
    imgs,
    flows,
    fb_conf=None,
    lambda_l1=0.15,
    lambda_ssim=0.85,
    lambda_census=0.30,
    census_patch=7,
    use_confidence=True,
    texture_floor=0.05,
    robust_beta=10.0,
    max_skip=2,
):
    """Longer-horizon photometric loss using composed adjacent flows.

    For 4 frames this supervises I0->I2 and I1->I3. This exposes accumulated
    errors without requiring an extra direct F_{t->t+2} head.
    """
    b, t_total, _, _, _ = imgs.shape
    if flows.shape[1] < 2 or max_skip < 2:
        return flows.new_tensor(0.0)

    total = flows.new_tensor(0.0)
    denom = flows.new_tensor(0.0)

    # v20 fixed2 uses skip=2 for stability. Larger skips can be added later by
    # repeated composition, but skip=2 already checks accumulation for 4 frames.
    skip = 2
    for t in range(t_total - skip):
        flow_acc = compose_adjacent_flows(flows[:, t], flows[:, t + 1])
        pair_map, valid_warp = pair_photometric_map(
            imgs[:, t],
            imgs[:, t + skip],
            flow_acc,
            lambda_l1=lambda_l1,
            lambda_ssim=lambda_ssim,
            lambda_census=lambda_census,
            census_patch=census_patch,
        )

        weight = valid_warp
        if use_confidence:
            weight = weight * texture_confidence(imgs[:, t], floor=texture_floor)
            weight = weight * robust_weight_from_error(pair_map, beta=robust_beta)
        if fb_conf is not None:
            # confidence product is still non-zero for low-confidence sky due to floor
            weight = weight * fb_conf[:, t] * fb_conf[:, t + 1]

        total = total + (pair_map * weight).sum()
        denom = denom + weight.sum()

    return total / (denom + 1e-6)


def accumulated_flow_smoothness_loss(flows, fb_conf=None):
    """Gently regularize composed two-step flow so accumulated errors do not explode."""
    if flows.shape[1] < 2:
        return flows.new_tensor(0.0)
    vals = []
    for t in range(flows.shape[1] - 1):
        flow_acc = compose_adjacent_flows(flows[:, t], flows[:, t + 1])
        dx = (flow_acc[:, :, :, 1:] - flow_acc[:, :, :, :-1]).abs().mean(dim=1)
        dy = (flow_acc[:, :, 1:, :] - flow_acc[:, :, :-1, :]).abs().mean(dim=1)
        if fb_conf is not None:
            c = fb_conf[:, t] * fb_conf[:, t + 1]
            dx_w = c[:, :, 1:]
            dy_w = c[:, 1:, :]
            vals.append((dx * dx_w).sum() / (dx_w.sum() + 1e-6))
            vals.append((dy * dy_w).sum() / (dy_w.sum() + 1e-6))
        else:
            vals.append(dx.mean())
            vals.append(dy.mean())
    return torch.stack(vals).mean()

def temporal_loss(flows):
    if flows.shape[1] < 2:
        return flows.new_tensor(0.0)
    return (flows[:, 1:] - flows[:, :-1]).abs().mean()



def edge_aware_smoothness_loss(flow, image):
    flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

    img_dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    img_dy = image[:, :, 1:, :] - image[:, :, :-1, :]

    weight_x = torch.exp(-torch.mean(torch.abs(img_dx), dim=1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(img_dy), dim=1, keepdim=True))

    loss_x = weight_x * torch.abs(flow_dx)
    loss_y = weight_y * torch.abs(flow_dy)
    return loss_x.mean() + loss_y.mean()


def multiframe_edge_aware_smoothness_loss(flows, imgs):
    """Average edge-aware smoothness over every adjacent flow.

    flows[:, t] is I_t -> I_{t+1}, so the edge image should be imgs[:, t].
    This keeps smoothness as a full-sequence regularizer instead of only applying
    it to the single GT-selected flow.
    """
    vals = []
    num_pairs = flows.shape[1]
    for t in range(num_pairs):
        vals.append(edge_aware_smoothness_loss(flows[:, t], imgs[:, t]))
    return torch.stack(vals).mean()


def select_gt_flow(pred_flows, src_idx):
    b = pred_flows.shape[0]
    out = []
    for idx in range(b):
        t = int(src_idx[idx].item())
        out.append(pred_flows[idx, t])
    return torch.stack(out, dim=0)



def save_checkpoint(save_path, epoch, modules, optimizer, stats=None, config=None):
    checkpoint = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "stats": stats,
        "config": config,
    }
    for name, module in modules.items():
        checkpoint[f"{name}_state_dict"] = module.state_dict()
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to: checkpoints/{save_path.name}")


def load_checkpoint(checkpoint_path, modules, optimizer, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    for name, module in modules.items():
        state_key = f"{name}_state_dict"
        if state_key not in checkpoint:
            raise KeyError(f"Missing {state_key} in checkpoint: {checkpoint_path}")
        module.load_state_dict(checkpoint[state_key])

    if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0))
    stats = checkpoint.get("stats", None)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Resuming after epoch: {start_epoch}")
    return start_epoch, stats


class V24Trainer:
    def __init__(self, cfg, modules, optimizer, writer, device):
        self.cfg = cfg
        self.modules = modules
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.global_step = 0

    def train_mode(self):
        for module in self.modules.values():
            module.train()

    def forward_pipeline(self, imgs, valid):
        pair_out = self.modules["pair_encoder"](imgs)
        pair_feats = pair_out["pair_feats"]
        flow_inits = pair_out["flow_inits"]
        corrs = pair_out["corrs"]

        if flow_inits is None:
            raise RuntimeError("v19 UNO integration requires predict_flow_init=True in the pair encoder.")

        visual_feats = self.modules["visual_branch"](imgs)
        motion_feats = self.modules["motion_branch"](pair_feats)
        fused_seq = self.modules["fusion"](visual_feats, motion_feats)

        valid_ds = None
        if self.cfg["model"]["uno_use_valid_mask"]:
            valid_ds = downsample_valid_mask(valid, fused_seq.shape[-2:])

        uno_in = build_uno_input_2d(fused_seq, flow_inits, valid_mask=valid_ds)
        uno_feat = self.modules["uno"](uno_in)

        b, tm, _, h, w = fused_seq.shape
        latent_delta = self.modules["latent_head"](uno_feat, b, tm, h, w)
        refined_seq = fused_seq + latent_delta

        flows, flow_residuals = self.modules["decoder"](refined_seq, flow_inits=flow_inits)

        return {
            "flows": flows,
            "flow_inits": flow_inits,
            "pair_feats": pair_feats,
            "corrs": corrs,
            "fused_seq": fused_seq,
            "latent_delta": latent_delta,
            "refined_seq": refined_seq,
            "flow_residuals": flow_residuals,
        }

    def train_step(self, batch):
        self.train_mode()

        imgs = batch["imgs"].to(self.device)
        gt_flow = batch["flow"].to(self.device)
        valid = batch["valid"].to(self.device)
        src_idx = batch["src_idx_in_seq"].to(self.device)
        img_src = batch["img_src"].to(self.device)

        out = self.forward_pipeline(imgs, valid)
        pred_flows = out["flows"]
        pred = select_gt_flow(pred_flows, src_idx)

        fb_conf = None
        fb_mask = None
        fb_error_mean = pred_flows.new_tensor(0.0)
        fb_conf_mean = pred_flows.new_tensor(1.0)
        fb_mask_ratio = pred_flows.new_tensor(1.0)
        loss_self_bw = pred_flows.new_tensor(0.0)
        loss_fb = pred_flows.new_tensor(0.0)
        fb_photo_weight = None
        if self.cfg["loss"].get("photo_use_fb_consistency", True):
            # Run the same model on reversed frames to obtain backward flows:
            # [I3->I2, I2->I1, I1->I0], then align to [I1->I0, I2->I1, I3->I2].
            imgs_rev = torch.flip(imgs, dims=[1])
            out_rev = self.forward_pipeline(imgs_rev, valid)
            bw_flows = align_reversed_backward_flows(out_rev["flows"])
            fb_conf, fb_errors = multiframe_fb_confidences(
                pred_flows,
                bw_flows,
                alpha=self.cfg["loss"].get("fb_alpha", 0.01),
                beta=self.cfg["loss"].get("fb_beta", 0.5),
                gamma=self.cfg["loss"].get("fb_gamma", 2.0),
                floor=self.cfg["loss"].get("fb_conf_floor", 0.05),
            )
            fb_error_mean = fb_errors.mean()
            fb_conf_mean = fb_conf.mean()

            if self.cfg["loss"].get("photo_use_hard_fb_mask", True):
                fb_mask = multiframe_fb_hard_masks(
                    pred_flows,
                    bw_flows,
                    alpha1=self.cfg["loss"].get("fb_mask_alpha1", 0.01),
                    alpha2=self.cfg["loss"].get("fb_mask_alpha2", 0.5),
                )
                fb_mask_ratio = fb_mask.mean()
                fb_photo_weight = fb_mask
            else:
                fb_photo_weight = fb_conf
                fb_mask_ratio = (fb_conf > self.cfg["loss"].get("fb_conf_floor", 0.05) + 1e-6).float().mean()

            loss_fb = forward_backward_consistency_loss(
                pred_flows,
                bw_flows,
                robust_eps=self.cfg["loss"].get("fb_loss_robust_eps", 0.01),
            )

            if self.cfg["loss"].get("photo_bidirectional", True):
                loss_self_bw = multiframe_photometric_loss(
                    imgs_rev,
                    out_rev["flows"],
                    lambda_l1=self.cfg["loss"].get("photo_lambda_l1", 0.15),
                    lambda_ssim=self.cfg["loss"].get("photo_lambda_ssim", 0.85),
                    lambda_census=self.cfg["loss"].get("photo_lambda_census", 0.30),
                    census_patch=self.cfg["loss"].get("photo_census_patch", 7),
                    reduction=self.cfg["loss"].get("photo_multiframe_reduction", "mean"),
                    use_confidence=self.cfg["loss"].get("photo_use_confidence", True),
                    texture_floor=self.cfg["loss"].get("photo_texture_floor", 0.05),
                    robust_beta=self.cfg["loss"].get("photo_robust_beta", 10.0),
                    fb_conf=torch.flip(fb_photo_weight, dims=[1]),
                )

        loss_flow = weighted_epe_loss(
            pred, gt_flow, img_src, valid,
            edge_weight_scale=self.cfg["loss"]["lambda_edge_weight"],
        )
        loss_self = multiframe_photometric_loss(
            imgs,
            pred_flows,
            lambda_l1=self.cfg["loss"].get("photo_lambda_l1", 0.15),
            lambda_ssim=self.cfg["loss"].get("photo_lambda_ssim", 0.85),
            lambda_census=self.cfg["loss"].get("photo_lambda_census", 0.30),
            census_patch=self.cfg["loss"].get("photo_census_patch", 7),
            reduction=self.cfg["loss"].get("photo_multiframe_reduction", "mean"),
            use_confidence=self.cfg["loss"].get("photo_use_confidence", True),
            texture_floor=self.cfg["loss"].get("photo_texture_floor", 0.05),
            robust_beta=self.cfg["loss"].get("photo_robust_beta", 10.0),
            fb_conf=fb_photo_weight if self.cfg["loss"].get("photo_use_fb_consistency", True) else None,
        )
        if self.cfg["loss"].get("photo_bidirectional", True):
            loss_self = 0.5 * (loss_self + loss_self_bw)

        loss_acc_photo = accumulated_multiframe_photometric_loss(
            imgs,
            pred_flows,
            fb_conf=fb_photo_weight if self.cfg["loss"].get("photo_use_fb_consistency", True) else None,
            lambda_l1=self.cfg["loss"].get("photo_lambda_l1", 0.15),
            lambda_ssim=self.cfg["loss"].get("photo_lambda_ssim", 0.85),
            lambda_census=self.cfg["loss"].get("photo_lambda_census", 0.30),
            census_patch=self.cfg["loss"].get("photo_census_patch", 7),
            use_confidence=self.cfg["loss"].get("photo_use_confidence", True),
            texture_floor=self.cfg["loss"].get("photo_texture_floor", 0.05),
            robust_beta=self.cfg["loss"].get("photo_robust_beta", 10.0),
            max_skip=self.cfg["loss"].get("accum_max_skip", 2),
        )
        loss_acc_smooth = accumulated_flow_smoothness_loss(pred_flows, fb_conf=fb_conf)

        loss_temp = temporal_loss(pred_flows)
        loss_smooth = multiframe_edge_aware_smoothness_loss(pred_flows, imgs)

        loss = (
            loss_flow
            + self.cfg["loss"]["lambda_self"] * loss_self
            + self.cfg["loss"].get("lambda_acc_photo", 0.05) * loss_acc_photo
            + self.cfg["loss"].get("lambda_acc_smooth", 0.0) * loss_acc_smooth
            + self.cfg["loss"].get("lambda_fb", 0.01) * loss_fb
            + self.cfg["loss"]["lambda_temp"] * loss_temp
            + self.cfg["loss"]["lambda_smooth"] * loss_smooth
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "loss_flow": float(loss_flow.item()),
            "loss_self": float(loss_self.item()),
            "loss_temp": float(loss_temp.item()),
            "loss_acc_photo": float(loss_acc_photo.item()),
            "loss_acc_smooth": float(loss_acc_smooth.item()),
            "loss_self_bw": float(loss_self_bw.item()),
            "loss_fb": float(loss_fb.item()),
            "loss_smooth": float(loss_smooth.item()),
            "latent_delta_mean_abs": float(out["latent_delta"].abs().mean().item()),
            "flow_residual_mean_abs": float(out["flow_residuals"].abs().mean().item()),
            "fb_error_mean": float(fb_error_mean.item()),
            "fb_conf_mean": float(fb_conf_mean.item()),
            "fb_mask_ratio": float(fb_mask_ratio.item()),
        }

        if self.global_step % self.cfg["train"]["log_every_n_steps"] == 0:
            for key, value in stats.items():
                self.writer.add_scalar(f"train_step/{key}", value, self.global_step)
        self.global_step += 1
        return stats



def build_modules(cfg, device):
    pair_encoder = SequencePairEncoder(
        feat_ch=cfg["model"]["pair_feat_ch"],
        corr_radius=cfg["model"]["corr_radius"],
        embed_ch=cfg["model"]["pair_embed_ch"],
        predict_flow=cfg["model"]["predict_flow_init"],
    ).to(device)

    visual_branch = VisualBranchCNN(
        in_ch=cfg["model"]["visual_in_ch"],
        base_ch=cfg["model"]["visual_base_ch"],
        out_ch=cfg["model"]["visual_out_ch"],
    ).to(device)

    motion_branch = MotionBranchCNN(
        in_ch=cfg["model"]["motion_in_ch"],
        hidden_ch=cfg["model"]["motion_hidden_ch"],
        out_ch=cfg["model"]["motion_out_ch"],
    ).to(device)

    fusion = SpatialTemporalFusion_timeAware(
        visual_ch=cfg["model"]["fusion_visual_ch"],
        motion_ch=cfg["model"]["fusion_motion_ch"],
        hidden_ch=cfg["model"]["fusion_hidden_ch"],
        out_ch=cfg["model"]["fusion_out_ch"],
    ).to(device)

    num_pairs = cfg["data"]["seq_len"] - 1
    uno_in_channels = num_pairs * (cfg["model"]["fusion_out_ch"] + 6)
    if cfg["model"]["uno_use_valid_mask"]:
        uno_in_channels += 1

    uno = UNO(
        in_channels=uno_in_channels,
        out_channels=cfg["model"]["uno_out_channels"][-1],
        hidden_channels=cfg["model"]["uno_hidden_channels"],
        lifting_channels=cfg["model"]["uno_lifting_channels"],
        projection_channels=cfg["model"]["uno_projection_channels"],
        n_layers=cfg["model"]["uno_n_layers"],
        uno_out_channels=cfg["model"]["uno_out_channels"],
        uno_n_modes=cfg["model"]["uno_n_modes"],
        uno_scalings=cfg["model"]["uno_scalings"],
        positional_embedding=None,
        fno_skip="linear",
        horizontal_skip="linear",
        channel_mlp_skip="linear",
    ).to(device)

    latent_head = UNOLatentResidualHead(
        uno_out_ch=cfg["model"]["uno_out_channels"][-1],
        latent_ch=cfg["model"]["fusion_out_ch"],
        num_pairs=num_pairs,
    ).to(device)

    decoder = FlowDecoder(
        in_ch=cfg["model"]["decoder_in_ch"],
        hidden_ch=cfg["model"]["decoder_hidden_ch"],
        upsample=cfg["model"]["decoder_upsample"],
        use_prev_flow=cfg["model"]["decoder_use_prev_flow"],
    ).to(device)

    return {
        "pair_encoder": pair_encoder,
        "visual_branch": visual_branch,
        "motion_branch": motion_branch,
        "fusion": fusion,
        "uno": uno,
        "latent_head": latent_head,
        "decoder": decoder,
    }



def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    set_seed(cfg["train"]["seed"])

    save_dir = (THIS_DIR / cfg["experiment"]["save_dir"]).resolve()
    tb_dir = (THIS_DIR / cfg["experiment"]["tensorboard_dir"] / cfg["experiment"]["experiment_name"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    config_dump_path = save_dir / cfg["experiment"]["config_dump_name"]
    with open(config_dump_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    writer = SummaryWriter(log_dir=str(tb_dir))

    dataset = TempFlowDataset_disp(
        root=cfg["data"]["data_root"],
        split=cfg["data"]["split"],
        image_folder=cfg["data"]["image_folder"],
        flow_type=cfg["data"]["flow_type"],
        disp_type=cfg["data"]["disp_type"],
        seq_len=cfg["data"]["seq_len"],
        center_frame_idx=cfg["data"]["center_frame_idx"],
        crop_size=tuple(cfg["data"]["crop_size"]),
        normalize=cfg["data"]["normalize"],
        stats_in=cfg["data"]["stats_file"],
        return_pair_only=cfg["data"]["return_pair_only"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"],
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    modules = build_modules(cfg, device)
    optimizer = torch.optim.AdamW(
        [p for module in modules.values() for p in module.parameters()],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    trainer = V24Trainer(cfg, modules, optimizer, writer, device)

    start_epoch = 0
    resume_stats = None
    if args.resume or args.resume_path is not None:
        resume_path = Path(args.resume_path) if args.resume_path is not None else save_dir / cfg["experiment"]["checkpoint_name"]
        start_epoch, resume_stats = load_checkpoint(resume_path, modules, optimizer, device)
        trainer.global_step = start_epoch * len(train_loader)
        if args.additional_epochs is not None:
            cfg["train"]["num_epochs"] = start_epoch + args.additional_epochs
        print(f"Training will run from epoch {start_epoch + 1} to {cfg['train']['num_epochs']}.")

    sanity_batch = next(iter(train_loader))
    with torch.no_grad():
        sanity_out = trainer.forward_pipeline(
            sanity_batch["imgs"].to(device),
            sanity_batch["valid"].to(device),
        )
    print("imgs:", sanity_batch["imgs"].shape)
    print("pred flows:", sanity_out["flows"].shape)
    print("flow_inits:", None if sanity_out["flow_inits"] is None else sanity_out["flow_inits"].shape)
    print("pair_feats:", sanity_out["pair_feats"].shape)
    print("fused_seq:", sanity_out["fused_seq"].shape)
    print("refined_seq:", sanity_out["refined_seq"].shape)
    print("flow_residuals:", sanity_out["flow_residuals"].shape)

    best_loss = float("inf")
    milestone_epochs = set(cfg["train"]["save_epoch_checkpoints"])

    for epoch in range(start_epoch, cfg["train"]["num_epochs"]):
        running = {
            "loss": 0.0,
            "loss_flow": 0.0,
            "loss_self": 0.0,
            "loss_temp": 0.0,
            "loss_acc_photo": 0.0,
            "loss_acc_smooth": 0.0,
            "loss_self_bw": 0.0,
            "loss_fb": 0.0,
            "loss_smooth": 0.0,
            "latent_delta_mean_abs": 0.0,
            "flow_residual_mean_abs": 0.0,
            "fb_error_mean": 0.0,
            "fb_conf_mean": 0.0,
            "fb_mask_ratio": 0.0,
        }
        n_batches = 0

        for batch in train_loader:
            stats = trainer.train_step(batch)
            for key, value in stats.items():
                if key not in running:
                    running[key] = 0.0
                running[key] += value
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        for key, value in avg.items():
            writer.add_scalar(f"train_epoch/{key}", value, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{cfg['train']['num_epochs']} | "
            f"loss={avg['loss']:.4f} | "
            f"flow={avg['loss_flow']:.4f} | "
            f"self={avg['loss_self']:.4f} | "
            f"temp={avg['loss_temp']:.4f} | "
            f"acc_photo={avg['loss_acc_photo']:.4f} | "
            f"acc_smooth={avg['loss_acc_smooth']:.4f} | "
            f"fb_loss={avg['loss_fb']:.4f} | "
            f"smooth={avg['loss_smooth']:.4f} | "
            f"latent={avg['latent_delta_mean_abs']:.4f} | "
            f"flow_res={avg['flow_residual_mean_abs']:.4f} | "
            f"fb_err={avg['fb_error_mean']:.4f} | "
            f"fb_conf={avg['fb_conf_mean']:.4f} | "
            f"fb_mask={avg['fb_mask_ratio']:.4f}"
        )

        latest_path = save_dir / cfg["experiment"]["checkpoint_name"]
        save_checkpoint(latest_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if (epoch + 1) == 200:
            save_path_200 = save_dir / "fullpipeline_v24_epoch_200.pth"
            save_checkpoint(save_path_200, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            best_path = save_dir / cfg["experiment"]["best_checkpoint_name"]
            save_checkpoint(best_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if (epoch + 1) in milestone_epochs:
            milestone_path = save_dir / f"{cfg['experiment']['experiment_name']}_epoch_{epoch + 1}.pth"
            save_checkpoint(milestone_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
