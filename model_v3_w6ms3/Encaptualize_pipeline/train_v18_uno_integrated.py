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
from Decoders import FlowDecoder
from neuralop_seg.uno import UNO


def parse_args():
    parser = argparse.ArgumentParser(
        description="v18: v15 backbone + paper-style photometric loss + spatial augmentation consistency"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "v15_uno_config.json"),
        help="Path to the experiment config JSON.",
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



def ssim_loss(img1, img2, window_size=3, c1=0.01 ** 2, c2=0.03 ** 2):
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


def paper_photometric_loss(
    img_src,
    img_tgt,
    flow,
    valid=None,
    lambda_l1=0.15,
    lambda_ssim=0.85,
    lambda_census=0.30,
    census_patch=7,
):
    warped_src = warp_image(img_src, flow)

    l1_map = (warped_src - img_tgt).abs().mean(dim=1)
    ssim_map = ssim_loss(warped_src, img_tgt)
    c_warped = census_transform(warped_src, patch_size=census_patch)
    c_tgt = census_transform(img_tgt, patch_size=census_patch)
    census_map = (c_warped - c_tgt).abs().mean(dim=1)

    photo = lambda_l1 * l1_map + lambda_ssim * ssim_map + lambda_census * census_map

    if valid is not None:
        if valid.ndim == 4:
            valid = valid[:, 0]
        valid = valid.float()
        return (photo * valid).sum() / (valid.sum() + 1e-6)

    return photo.mean()


def make_translation_theta(batch_size, h, w, max_shift_px, device):
    dx = torch.empty(batch_size, device=device).uniform_(-max_shift_px, max_shift_px)
    dy = torch.empty(batch_size, device=device).uniform_(-max_shift_px, max_shift_px)

    tx = -2.0 * dx / max(w - 1, 1)
    ty = -2.0 * dy / max(h - 1, 1)

    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    return theta


def apply_spatial_transform(x, theta, mode="bilinear"):
    if x.ndim == 5:
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        theta_rep = theta.repeat_interleave(t, dim=0)
        grid = F.affine_grid(theta_rep, x_flat.size(), align_corners=True)
        out = F.grid_sample(
            x_flat,
            grid,
            mode=mode,
            padding_mode="border" if mode != "nearest" else "zeros",
            align_corners=True,
        )
        return out.reshape(b, t, c, h, w)

    b, c, h, w = x.shape
    grid = F.affine_grid(theta, x.size(), align_corners=True)
    return F.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode="border" if mode != "nearest" else "zeros",
        align_corners=True,
    )


def augmentation_consistency_loss(
    trainer,
    imgs,
    valid,
    src_idx,
    pred,
    max_shift_px=12.0,
):
    b, _, _, h, w = imgs.shape
    theta = make_translation_theta(b, h, w, max_shift_px=max_shift_px, device=imgs.device)

    valid_in = valid.float()
    if valid_in.ndim == 3:
        valid_in = valid_in.unsqueeze(1)

    imgs_aug = apply_spatial_transform(imgs, theta, mode="bilinear")
    valid_aug_4d = apply_spatial_transform(valid_in, theta, mode="nearest")
    valid_aug = valid_aug_4d[:, 0] if valid.ndim == 3 else valid_aug_4d

    out_aug = trainer.forward_pipeline(imgs_aug, valid_aug)
    pred_aug = select_gt_flow(out_aug["flows"], src_idx)

    pred_target = apply_spatial_transform(pred.detach(), theta, mode="bilinear")
    mask_aug = (valid_aug_4d > 0.5).float()

    diff = torch.abs(pred_aug - pred_target).sum(dim=1, keepdim=True)
    loss_aug = (diff * mask_aug).sum() / (mask_aug.sum() + 1e-6)

    return loss_aug, {
        "imgs_aug": imgs_aug,
        "valid_aug": valid_aug,
        "pred_aug": pred_aug,
        "pred_target": pred_target,
    }


def photometric_loss(img_src, img_tgt, flow, valid=None):
    return paper_photometric_loss(img_src, img_tgt, flow, valid=valid)


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


class V18Trainer:
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
            raise RuntimeError("v15 UNO integration requires predict_flow_init=True in the pair encoder.")

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

    def train_step(self, batch):
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
            pred, gt_flow, img_src, valid,
            edge_weight_scale=self.cfg["loss"]["lambda_edge_weight"],
        )
        loss_self = paper_photometric_loss(
            img_src,
            img_tgt,
            pred,
            valid=valid,
            lambda_l1=self.cfg["loss"].get("photo_lambda_l1", 0.15),
            lambda_ssim=self.cfg["loss"].get("photo_lambda_ssim", 0.85),
            lambda_census=self.cfg["loss"].get("photo_lambda_census", 0.30),
            census_patch=self.cfg["loss"].get("photo_census_patch", 7),
        )
        loss_aug, aug_out = augmentation_consistency_loss(
            self,
            imgs,
            valid,
            src_idx,
            pred,
            max_shift_px=float(self.cfg["loss"].get("aug_max_shift_px", 12.0)),
        )
        loss_temp = temporal_loss(pred_flows)
        loss_smooth = edge_aware_smoothness_loss(pred, img_src)

        loss = (
            loss_flow
            + self.cfg["loss"]["lambda_self"] * loss_self
            + self.cfg["loss"].get("lambda_aug", 0.05) * loss_aug
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
            "loss_aug": float(loss_aug.item()),
            "loss_temp": float(loss_temp.item()),
            "loss_smooth": float(loss_smooth.item()),
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

    trainer = V18Trainer(cfg, modules, optimizer, writer, device)

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
            "loss_aug": 0.0,
            "loss_temp": 0.0,
            "loss_smooth": 0.0,
            "latent_delta_mean_abs": 0.0,
            "flow_residual_mean_abs": 0.0,
        }
        n_batches = 0

        for batch in train_loader:
            stats = trainer.train_step(batch)
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
            f"aug={avg['loss_aug']:.4f} | "
            f"temp={avg['loss_temp']:.4f} | "
            f"smooth={avg['loss_smooth']:.4f} | "
            f"latent={avg['latent_delta_mean_abs']:.4f} | "
            f"flow_res={avg['flow_residual_mean_abs']:.4f}"
        )

        latest_path = save_dir / cfg["experiment"]["checkpoint_name"]
        save_checkpoint(latest_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)
        
        if (epoch + 1) == 200:
            save_path_200 = save_dir / "fullpipeline_v18_epoch_200.pth"
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
