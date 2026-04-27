
import argparse
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent  # model_v3_w6ms3 parent if placed inside experiment dir
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
    parser = argparse.ArgumentParser(description="Visualize predictions for v15 UNO integrated pipeline")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "v15_uno_config.json"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_checkpoint(checkpoint_path, modules, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "stats" in checkpoint and checkpoint["stats"] is not None:
        print(f"Checkpoint stats: {checkpoint['stats']}")

    missing_summary = {}
    unexpected_summary = {}
    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key not in checkpoint:
            raise KeyError(f"Missing key in checkpoint: {key}")
        missing, unexpected = module.load_state_dict(checkpoint[key], strict=False)
        missing_summary[name] = len(missing)
        unexpected_summary[name] = len(unexpected)
        module.eval()

    print("State load summary:")
    for name in modules:
        print(f"  {name}: missing={missing_summary[name]}, unexpected={unexpected_summary[name]}")

    return checkpoint


def forward_pipeline(modules, cfg, imgs, valid):
    pair_out = modules["pair_encoder"](imgs)
    pair_feats = pair_out["pair_feats"]
    flow_inits = pair_out["flow_inits"]
    corrs = pair_out["corrs"]

    if flow_inits is None:
        raise RuntimeError("v15 visualization expects predict_flow_init=True and non-None flow_inits.")

    visual_feats = modules["visual_branch"](imgs)
    motion_feats = modules["motion_branch"](pair_feats)
    fused_seq = modules["fusion"](visual_feats, motion_feats)

    valid_ds = None
    if cfg["model"]["uno_use_valid_mask"]:
        valid_ds = downsample_valid_mask(valid, fused_seq.shape[-2:])

    uno_in = build_uno_input_2d(fused_seq, flow_inits, valid_mask=valid_ds)
    uno_feat = modules["uno"](uno_in)

    b, tm, latent_ch, h, w = fused_seq.shape
    latent_delta = modules["latent_head"](uno_feat, b, tm, h, w)
    refined_seq = fused_seq + latent_delta

    flows, flow_residuals = modules["decoder"](refined_seq, flow_inits=flow_inits)

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


def tensor_img_to_np(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    img = img.permute(1, 2, 0).numpy()
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)
    return img


def flow_to_rgb(flow):
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)

    hue = (ang + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = mag / (np.max(mag) + 1e-6)
    val = np.clip(val, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def compute_epe_map(pred, gt, valid=None):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu()
    epe = torch.norm(pred - gt, dim=0)

    if valid is not None:
        if isinstance(valid, np.ndarray):
            valid = torch.from_numpy(valid)
        valid = valid.detach().cpu().float()
        if valid.ndim == 3:
            valid = valid[0]
        epe = epe * valid

    return epe.numpy()


def select_gt_flow_single(pred_flows, src_idx, b=0):
    t = int(src_idx[b].item())
    return pred_flows[b, t]


def visualize_batch_result_centered(batch, pred_flows, sample_idx=0, save_path=None):
    imgs = batch["imgs"][sample_idx]
    gt_flow = batch["flow"][sample_idx]
    valid = batch["valid"][sample_idx]
    src_idx = batch["src_idx_in_seq"]
    t = int(src_idx[sample_idx].item())

    num_imgs = imgs.shape[0]
    num_preds = pred_flows.shape[1]

    img_nps = [tensor_img_to_np(imgs[i]) for i in range(num_imgs)]
    pred_rgbs = [flow_to_rgb(pred_flows[sample_idx, i]) for i in range(num_preds)]
    gt_rgb = flow_to_rgb(gt_flow)
    matched_pred = select_gt_flow_single(pred_flows, src_idx, b=sample_idx)
    epe_map = compute_epe_map(matched_pred, gt_flow, None)

    cols = max(num_imgs, num_preds + 2)
    fig, axes = plt.subplots(2, cols, figsize=(5.5 * cols, 10))
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(cols):
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    human_start = 2 - t

    for i in range(num_imgs):
        axes[0, i].imshow(img_nps[i])
        axes[0, i].set_title(f"Frame {human_start + i}")

    for i in range(num_preds):
        axes[1, i].imshow(pred_rgbs[i])
        axes[1, i].set_title(f"Pred Flow {human_start + i}→{human_start + i + 1}")

    axes[1, num_preds].imshow(gt_rgb)
    axes[1, num_preds].set_title(f"GT Flow ({human_start + t}→{human_start + t + 1})")

    im = axes[1, num_preds + 1].imshow(epe_map, cmap="inferno")
    axes[1, num_preds + 1].set_title("EPE Map")
    plt.colorbar(im, ax=axes[1, num_preds + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {save_path}")
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    batch_size = args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"]
    split = args.split if args.split is not None else cfg["data"]["split"]

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (THIS_DIR / cfg["experiment"]["save_dir"] / cfg["experiment"]["best_checkpoint_name"]).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else (THIS_DIR / f"visualization_outputs_{checkpoint_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)
    print("config:", args.config)
    print("checkpoint:", checkpoint_path)
    print("output_dir:", output_dir)

    dataset = TempFlowDataset_disp(
        root=cfg["data"]["data_root"],
        split=split,
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

    vis_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    modules = build_modules(cfg, device)
    load_checkpoint(checkpoint_path, modules, device)

    saved = 0
    seen = 0

    for batch_idx, batch in enumerate(vis_loader):
        imgs = batch["imgs"].to(device)
        valid = batch["valid"].to(device)

        with torch.no_grad():
            out = forward_pipeline(modules, cfg, imgs, valid)

        pred_flows = out["flows"].detach().cpu()
        print(f"batch {batch_idx}: pred_flows shape = {tuple(pred_flows.shape)}")

        batch_n = pred_flows.shape[0]
        for sample_idx in range(batch_n):
            if seen < args.start_index:
                seen += 1
                continue

            if saved >= args.max_samples:
                break

            save_path = output_dir / f"vis_batch{batch_idx:03d}_sample{sample_idx:02d}.png"
            visualize_batch_result_centered(batch, pred_flows, sample_idx=sample_idx, save_path=save_path)
            saved += 1
            seen += 1

        if saved >= args.max_samples:
            break

    print(f"Visualization finished. Saved {saved} sample(s) to {output_dir}.")


if __name__ == "__main__":
    main()
