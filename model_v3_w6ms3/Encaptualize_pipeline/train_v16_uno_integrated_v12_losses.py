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
MODULE_DIR = THIS_DIR.parent

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
from Decoders import FlowDecoder
from neuralop_seg.uno import UNO


def parse_args():
    parser = argparse.ArgumentParser(
        description="v16: v15 UNO integrated backbone + v12 loss package (with lambda_temp kept at 0.1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "v16_uno_config.json"),
        help="Path to experiment config JSON.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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


def warp_image(img, flow):
    b, c, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device),
        torch.arange(w, device=img.device),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).expand(b, -1, -1, -1)
    sample_grid = base_grid + flow

    sample_grid_x = 2.0 * sample_grid[:, 0] / max(w - 1, 1) - 1.0
    sample_grid_y = 2.0 * sample_grid[:, 1] / max(h - 1, 1) - 1.0
    sample_grid = torch.stack((sample_grid_x, sample_grid_y), dim=-1)

    return F.grid_sample(img, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def census_transform(img, patch_size=7):
    if img.shape[1] > 1:
        img = img.mean(dim=1, keepdim=True)

    b, _, h, w = img.shape
    p = patch_size // 2
    padded = F.pad(img, [p, p, p, p], mode="reflect")

    patches = []
    for dy in range(patch_size):
        for dx in range(patch_size):
            patch = padded[:, :, dy:dy + h, dx:dx + w]
            patches.append((patch > img).float())

    return torch.cat(patches, dim=1)


def photometric_loss_v16(
    img_src,
    img_tgt,
    flow,
    valid=None,
    auto_mask=True,
    eps=1e-3,
    grad_gate=True,
    grad_thresh=0.03,
    use_census=True,
    census_patch=7,
):
    warped_src = warp_image(img_src, flow)

    if use_census:
        c_warped = census_transform(warped_src, census_patch)
        c_tgt = census_transform(img_tgt, census_patch)
        photo_warped = (c_warped - c_tgt).abs().mean(dim=1)
        photo_identity = (img_src - img_tgt).abs().mean(dim=1)
    else:
        photo_warped = (warped_src - img_tgt).abs().mean(dim=1)
        photo_identity = (img_src - img_tgt).abs().mean(dim=1)

    if auto_mask:
        mask = (photo_warped + eps < photo_identity).float()
    else:
        mask = torch.ones_like(photo_warped)

    if grad_gate:
        edge_map = normalize_map(sobel_grad_map(img_src)).detach()[:, 0]
        mask = mask * (edge_map > grad_thresh).float()

    if valid is not None:
        if valid.ndim == 4:
            valid = valid[:, 0]
        valid = valid.float()
        mask = mask * valid

    return (photo_warped * mask).sum() / (mask.sum() + 1e-6)


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


def binary_erosion(mask, kernel_size=11):
    inv = 1.0 - mask
    eroded_inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return 1.0 - eroded_inv


def build_high_confidence_sky_mask(image, grad_thresh=0.04, top_ratio=0.38, erosion_kernel=11):
    edge_map = normalize_map(sobel_grad_map(image)).detach()
    low_grad = (edge_map < grad_thresh).float()

    b, _, h, w = image.shape
    y = torch.linspace(0, 1, h, device=image.device).view(1, 1, h, 1)
    top_mask = (y < top_ratio).float().expand(b, 1, h, w)

    sky_mask = low_grad * top_mask
    sky_mask = binary_erosion(sky_mask, kernel_size=erosion_kernel)
    return sky_mask


def flow_grad_map(flow):
    fx = flow[:, 0:1]
    fy = flow[:, 1:2]
    grad = sobel_grad_map(fx) + sobel_grad_map(fy)
    return normalize_map(grad)


def boundary_weighted_epe_loss(pred, gt, image, valid=None, image_scale=1.0, flow_scale=2.0):
    epe = torch.norm(pred - gt, dim=1, keepdim=True)
    img_edge = normalize_map(sobel_grad_map(image)).detach()
    gt_edge = flow_grad_map(gt).detach()

    weight = 1.0 + image_scale * img_edge + flow_scale * gt_edge

    if valid is not None:
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        valid = valid.float()
        weight = weight * valid

    return (epe * weight).sum() / (weight.sum() + 1e-6)


def dominant_background_flow_loss(flow, image, grad_thresh=0.04, top_ratio=0.38, erosion_kernel=11):
    sky_mask = build_high_confidence_sky_mask(
        image,
        grad_thresh=grad_thresh,
        top_ratio=top_ratio,
        erosion_kernel=erosion_kernel,
    ).detach()

    mask_sum = sky_mask.sum(dim=(2, 3), keepdim=True)
    mean_flow = (flow * sky_mask).sum(dim=(2, 3), keepdim=True) / (mask_sum + 1e-6)
    residual = torch.abs(flow - mean_flow)
    return (residual * sky_mask).sum() / (sky_mask.sum() * flow.shape[1] + 1e-6)


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


class V16Trainer:
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

    def current_sky_lambda(self, epoch_idx: int) -> float:
        warm = max(int(self.cfg["loss"]["sky_warmup_epochs"]), 1)
        scale = min(1.0, float(epoch_idx + 1) / float(warm))
        return float(self.cfg["loss"]["lambda_sky"]) * scale

    def forward_pipeline(self, imgs, valid):
        pair_out = self.modules["pair_encoder"](imgs)
        pair_feats = pair_out["pair_feats"]
        flow_inits = pair_out["flow_inits"]
        corrs = pair_out["corrs"]

        if flow_inits is None:
            raise RuntimeError("v16 UNO integration requires predict_flow_init=True in the pair encoder.")

        visual_feats = self.modules["visual_branch"](imgs)
        motion_feats = self.modules["motion_branch"](pair_feats)
        fused_seq = self.modules["fusion"](visual_feats, motion_feats)

        valid_ds = None
        if self.cfg["model"]["uno_use_valid_mask"]:
            valid_ds = downsample_valid_mask(valid, fused_seq.shape[-2:])

        uno_in = build_uno_input_2d(fused_seq, flow_inits, valid_mask=valid_ds)
        uno_feat = self.modules["uno"](uno_in)

        b, tm, latent_ch, h, w = fused_seq.shape
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

    def train_step(self, batch, epoch_idx=0):
        self.train_mode()

        imgs = batch["imgs"].to(self.device)
        gt_flow = batch["flow"].to(self.device)
        valid = batch["valid"].to(self.device)
        src_idx = batch["src_idx_in_seq"].to(self.device)
        img_src = batch["img_src"].to(self.device)
        img_tgt = batch["img_tgt"].to(self.device)

        out = self.forward_pipeline(imgs, valid)
        pred_flows = out["flows"]
        pred = select_gt_flow(pred_flows, src_idx)

        loss_flow = weighted_epe_loss(
            pred,
            gt_flow,
            img_src,
            valid,
            edge_weight_scale=self.cfg["loss"]["lambda_edge_weight"],
        )
        loss_self = photometric_loss_v16(
            img_src,
            img_tgt,
            pred,
            valid=valid,
            auto_mask=self.cfg["loss"]["photo_auto_mask"],
            eps=self.cfg["loss"]["auto_mask_eps"],
            grad_gate=self.cfg["loss"]["photo_grad_gate"],
            grad_thresh=self.cfg["loss"]["photo_grad_thresh"],
            use_census=self.cfg["loss"]["photo_use_census"],
            census_patch=self.cfg["loss"]["census_patch"],
        )
        loss_boundary = boundary_weighted_epe_loss(
            pred,
            gt_flow,
            img_src,
            valid=valid,
            image_scale=self.cfg["loss"]["lambda_edge_weight"],
            flow_scale=self.cfg["loss"]["boundary_flow_scale"],
        )
        loss_temp = temporal_loss(pred_flows)
        loss_smooth = edge_aware_smoothness_loss(pred, img_src)
        loss_sky = dominant_background_flow_loss(
            pred,
            img_src,
            grad_thresh=self.cfg["loss"]["sky_grad_thresh"],
            top_ratio=self.cfg["loss"]["sky_top_ratio"],
            erosion_kernel=self.cfg["loss"]["sky_erosion_kernel"],
        )
        current_lambda_sky = self.current_sky_lambda(epoch_idx)

        loss = (
            loss_flow
            + self.cfg["loss"]["lambda_boundary"] * loss_boundary
            + self.cfg["loss"]["lambda_self"] * loss_self
            + self.cfg["loss"]["lambda_temp"] * loss_temp
            + self.cfg["loss"]["lambda_smooth"] * loss_smooth
            + current_lambda_sky * loss_sky
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "loss_flow": float(loss_flow.item()),
            "loss_self": float(loss_self.item()),
            "loss_boundary": float(loss_boundary.item()),
            "loss_temp": float(loss_temp.item()),
            "loss_smooth": float(loss_smooth.item()),
            "loss_sky": float(loss_sky.item()),
            "lambda_sky_used": float(current_lambda_sky),
            "latent_delta_mean_abs": float(out["latent_delta"].abs().mean().item()),
            "flow_residual_mean_abs": float(out["flow_residuals"].abs().mean().item()),
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

    trainer = V16Trainer(cfg, modules, optimizer, writer, device)

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

    for epoch in range(cfg["train"]["num_epochs"]):
        running = {
            "loss": 0.0,
            "loss_flow": 0.0,
            "loss_self": 0.0,
            "loss_boundary": 0.0,
            "loss_temp": 0.0,
            "loss_smooth": 0.0,
            "loss_sky": 0.0,
            "lambda_sky_used": 0.0,
            "latent_delta_mean_abs": 0.0,
            "flow_residual_mean_abs": 0.0,
        }
        n_batches = 0

        for batch in train_loader:
            stats = trainer.train_step(batch, epoch_idx=epoch)
            for key in running:
                running[key] += stats[key]
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        for key, value in avg.items():
            writer.add_scalar(f"train_epoch/{key}", value, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{cfg['train']['num_epochs']} | "
            f"loss={avg['loss']:.4f} | "
            f"flow={avg['loss_flow']:.4f} | "
            f"self={avg['loss_self']:.4f} | "
            f"boundary={avg['loss_boundary']:.4f} | "
            f"temp={avg['loss_temp']:.4f} | "
            f"smooth={avg['loss_smooth']:.4f} | "
            f"sky={avg['loss_sky']:.4f} | "
            f"sky_lambda={avg['lambda_sky_used']:.4f} | "
            f"latent={avg['latent_delta_mean_abs']:.4f} | "
            f"flow_res={avg['flow_residual_mean_abs']:.4f}"
        )

        latest_path = save_dir / cfg["experiment"]["checkpoint_name"]
        save_checkpoint(latest_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if (epoch + 1) == 200:
            save_path_200 = save_dir / "fullpipeline_v16_epoch_200.pth"
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
