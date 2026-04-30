import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from Decoders_v26_convex import FlowDecoder
from neuralop_seg.uno import UNO


# -------------------------
# RAFT / Middlebury flow viz
# -------------------------
def make_colorwheel() -> np.ndarray:
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0

    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY

    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col += YG

    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC

    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col += CB

    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM

    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False) -> np.ndarray:
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        col0 = colorwheel[k0, i] / 255.0
        col1 = colorwheel[k1, i] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75

        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_image(flow_uv: np.ndarray, clip_flow: float | None = None, convert_to_bgr: bool = False) -> np.ndarray:
    if flow_uv.ndim != 3 or flow_uv.shape[2] != 2:
        raise ValueError(f"flow_to_image expects [H, W, 2], got {flow_uv.shape}")

    flow_uv = flow_uv.copy()
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


# -------------------------
# Model / data helpers
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize v26 UNO + convex-upsample decoder predictions with RAFT-style color mapping.")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--output_dir", type=str, default="visualization_outputs_raft")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default=None, help="Override dataset split from config.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_modules(cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.nn.Module]:
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
    if cfg["model"].get("uno_use_valid_mask", False):
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

    # v26 change:
    # This FlowDecoder comes from Decoders_v26_convex.py.
    # It keeps the same constructor/config keys as before, but internally uses
    # RAFT-style convex upsampling instead of the older bilinear/refinement decoder.
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


def load_checkpoint(modules: Dict[str, torch.nn.Module], checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key not in ckpt:
            raise KeyError(f"Missing {key} in checkpoint.")
        module.load_state_dict(ckpt[key], strict=True)
        module.eval()
    return ckpt


def forward_pipeline(modules: Dict[str, torch.nn.Module], cfg: Dict[str, Any], imgs: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
    pair_out = modules["pair_encoder"](imgs)
    pair_feats = pair_out["pair_feats"]
    flow_inits = pair_out["flow_inits"]

    if flow_inits is None:
        raise RuntimeError("v26 separate-operator requires predict_flow_init=True in the pair encoder.")

    visual_feats = modules["visual_branch"](imgs)
    motion_feats = modules["motion_branch"](pair_feats)
    fused_seq = modules["fusion"](visual_feats, motion_feats)

    valid_ds = None
    if cfg["model"].get("uno_use_valid_mask", False):
        valid_ds = downsample_valid_mask(valid, fused_seq.shape[-2:])

    # FULLY SEPARATE / STANDALONE OPERATOR:
    # UNO output is used as the decoder latent directly; no residual add-back.
    uno_in = build_uno_input_2d(fused_seq, flow_inits, valid_mask=valid_ds)
    uno_feat = modules["uno"](uno_in)

    b, tm, _, h, w = fused_seq.shape
    latent_delta = modules["latent_head"](uno_feat, b, tm, h, w)
    refined_seq = latent_delta

    dec_out = modules["decoder"](refined_seq, flow_inits=flow_inits)
    if isinstance(dec_out, tuple):
        flows, flow_residuals = dec_out
    else:
        flows, flow_residuals = dec_out, None

    return {
        "flows": flows,
        "flow_residuals": flow_residuals,
        "flow_inits": flow_inits,
    }


def select_gt_flow(pred_flows: torch.Tensor, src_idx: torch.Tensor) -> torch.Tensor:
    out = []
    for b in range(pred_flows.shape[0]):
        t = int(src_idx[b].item())
        out.append(pred_flows[b, t])
    return torch.stack(out, dim=0)


# -------------------------
# Visualization helpers
# -------------------------
def tensor_image_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))

    if np.isfinite(img).all():
        if img.min() >= 0.0 and img.max() <= 1.0:
            pass
        elif img.min() >= -1.0 and img.max() <= 1.0:
            img = (img + 1.0) / 2.0
        else:
            mn, mx = float(img.min()), float(img.max())
            img = (img - mn) / (mx - mn + 1e-6)
    else:
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn + 1e-6)

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def epe_map(pred: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    epe = torch.norm(pred - gt, dim=0).detach().cpu().float().numpy()
    vmax = np.percentile(epe, 99)
    vmax = max(vmax, 1e-6)
    norm = np.clip(epe / vmax, 0.0, 1.0)
    heat = np.stack([
        norm,
        np.zeros_like(norm),
        1.0 - norm,
    ], axis=-1)
    return (heat * 255.0).astype(np.uint8)


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 22], fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.array(pil)


def pad_to_same_height(images: list[np.ndarray]) -> list[np.ndarray]:
    max_h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] == max_h:
            padded.append(im)
            continue
        pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
        padded.append(np.concatenate([im, pad], axis=0))
    return padded


def make_panel(img_src: torch.Tensor, img_tgt: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, sample_name: str) -> Image.Image:
    src_np = add_label(tensor_image_to_uint8(img_src), "img_src")
    tgt_np = add_label(tensor_image_to_uint8(img_tgt), "img_tgt")

    pred_np = pred.detach().cpu().permute(1, 2, 0).float().numpy()
    gt_np = gt.detach().cpu().permute(1, 2, 0).float().numpy()

    pred_flow = add_label(flow_to_image(pred_np), "pred_flow (RAFT map)")
    gt_flow = add_label(flow_to_image(gt_np), "gt_flow (RAFT map)")
    epe_np = add_label(epe_map(pred, gt), "epe_map")

    row1 = pad_to_same_height([src_np, tgt_np, pred_flow])
    row2 = pad_to_same_height([gt_flow, epe_np])

    row1_img = np.concatenate(row1, axis=1)
    if len(row2) == 2:
        blank = np.zeros_like(row2[0])
        row2.append(blank)
    row2_img = np.concatenate(row2, axis=1)

    panel = np.concatenate([row1_img, row2_img], axis=0)
    panel = add_label(panel, sample_name)
    return Image.fromarray(panel)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    if args.split is not None:
        cfg["data"]["split"] = args.split

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modules = build_modules(cfg, device)
    ckpt = load_checkpoint(modules, args.checkpoint, device)
    print(f"Loaded checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            imgs = batch["imgs"].to(device)
            valid = batch["valid"].to(device)
            gt_flow = batch["flow"].to(device)
            src_idx = batch["src_idx_in_seq"].to(device)
            img_src = batch["img_src"].to(device)
            img_tgt = batch["img_tgt"].to(device)

            out = forward_pipeline(modules, cfg, imgs, valid)
            pred = select_gt_flow(out["flows"], src_idx)

            for b in range(pred.shape[0]):
                sample_name = f"batch{batch_idx:03d}_sample{b:02d}"
                panel = make_panel(img_src[b], img_tgt[b], pred[b], gt_flow[b], sample_name)
                save_path = output_dir / f"{sample_name}.png"
                panel.save(save_path)
                print(f"Saved {save_path}")
                saved += 1
                if saved >= args.max_samples:
                    print(f"Done. Saved {saved} visualization(s) to {output_dir}")
                    return

    print(f"Done. Saved {saved} visualization(s) to {output_dir}")


if __name__ == "__main__":
    main()
