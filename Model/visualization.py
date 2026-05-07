import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from DataLoader import TempFlowDataset_disp
from trainer import Trainer, build_modules, select_gt_flow, set_seed


# -------------------------
# RAFT / Middlebury flow visualization
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
        raise ValueError(f"flow_to_image expects [H,W,2], got {flow_uv.shape}")

    flow_uv = flow_uv.copy()
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    u = u / (rad_max + 1e-5)
    v = v / (rad_max + 1e-5)
    return flow_uv_to_colors(u, v, convert_to_bgr)


# -------------------------
# Args / config / checkpoint
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified visualization for v26 later / early / standalone models."
    )
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "config.json"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--architecture_type",
        type=str,
        choices=["later", "early", "standalone"],
        default=None,
        help="Override cfg['model']['architecture_type']; must match the checkpoint architecture.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multiple"],
        default="single",
        help="single = src/tgt/pred/GT/EPE panel; multiple = 4 frames + 3 predicted flows + GT flow.",
    )
    parser.add_argument("--output_dir", type=str, default="visualization_outputs")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.architecture_type is not None:
        cfg["model"]["architecture_type"] = args.architecture_type
    else:
        cfg["model"]["architecture_type"] = cfg["model"].get("architecture_type", "later")

    if args.split is not None:
        cfg["data"]["split"] = args.split
    return cfg


def load_checkpoint(modules: Dict[str, torch.nn.Module], checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key not in ckpt:
            raise KeyError(f"Missing {key} in checkpoint: {checkpoint_path}")
        module.load_state_dict(ckpt[key], strict=True)
        module.eval()
    return ckpt


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
    vmax = max(np.percentile(epe, 99), 1e-6)
    norm = np.clip(epe / vmax, 0.0, 1.0)
    heat = np.stack([norm, np.zeros_like(norm), 1.0 - norm], axis=-1)
    return (heat * 255.0).astype(np.uint8)


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 22], fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.array(pil)


def pad_to_same_height(images: list[np.ndarray]) -> list[np.ndarray]:
    max_h = max(im.shape[0] for im in images)
    out = []
    for im in images:
        if im.shape[0] == max_h:
            out.append(im)
        else:
            pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            out.append(np.concatenate([im, pad], axis=0))
    return out


def resize_to(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))


def make_single_panel(
    img_src: torch.Tensor,
    img_tgt: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    sample_name: str,
) -> Image.Image:
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
        row2.append(np.zeros_like(row2[0]))
    row2_img = np.concatenate(row2, axis=1)

    panel = np.concatenate([row1_img, row2_img], axis=0)
    panel = add_label(panel, sample_name)
    return Image.fromarray(panel)


def make_multiple_panel(
    imgs_seq: torch.Tensor,
    pred_flows: torch.Tensor,
    gt_flow: torch.Tensor,
    src_idx: torch.Tensor,
    sample_name: str,
) -> Image.Image:
    if imgs_seq.shape[0] < 4 or pred_flows.shape[0] < 3:
        raise ValueError(f"Expected 4 frames and 3 flows, got {imgs_seq.shape[0]} frames and {pred_flows.shape[0]} flows.")

    frame_imgs = []
    for t in range(4):
        frame_imgs.append(add_label(tensor_image_to_uint8(imgs_seq[t]), f"Frame {t + 1}"))

    flow_imgs = []
    for t in range(3):
        flow_np = pred_flows[t].detach().cpu().permute(1, 2, 0).float().numpy()
        flow_imgs.append(add_label(flow_to_image(flow_np), f"Pred Flow {t + 1}->{t + 2}"))

    gt_np = gt_flow.detach().cpu().permute(1, 2, 0).float().numpy()
    gt_t = int(src_idx.item())
    flow_imgs.append(add_label(flow_to_image(gt_np), f"GT Flow ({gt_t + 1}->{gt_t + 2})"))

    tile_h, tile_w = frame_imgs[0].shape[:2]
    target_size = (tile_w, tile_h)
    frame_imgs = [resize_to(im, target_size) for im in frame_imgs]
    flow_imgs = [resize_to(im, target_size) for im in flow_imgs]

    row1 = np.concatenate(frame_imgs, axis=1)
    row2 = np.concatenate(flow_imgs, axis=1)
    gap = np.ones((30, row1.shape[1], 3), dtype=np.uint8) * 255
    panel = np.concatenate([row1, gap, row2], axis=0)
    panel = add_label(panel, sample_name)
    return Image.fromarray(panel)


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = apply_overrides(load_config(args.config), args)
    arch = cfg["model"].get("architecture_type", "later")

    output_dir = Path(args.output_dir) / f"{arch}_{args.mode}"
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
    print("device:", device)
    print("architecture_type:", arch)
    print("mode:", args.mode)

    modules = build_modules(cfg, device)
    ckpt = load_checkpoint(modules, args.checkpoint, device)
    print(f"Loaded checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

    # Reuse the same forward_pipeline as training, so visualization follows model settings.
    visualizer = Trainer(cfg, modules, optimizer=None, writer=None, device=device)

    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            imgs = batch["imgs"].to(device)
            valid = batch["valid"].to(device)
            gt_flow = batch["flow"].to(device)
            src_idx = batch["src_idx_in_seq"].to(device)
            img_src = batch["img_src"].to(device)
            img_tgt = batch["img_tgt"].to(device)

            out = visualizer.forward_pipeline(imgs, valid)
            pred_flows = out["flows"]
            pred_gt = select_gt_flow(pred_flows, src_idx)

            for b in range(pred_flows.shape[0]):
                sample_name = f"batch{batch_idx:03d}_sample{b:02d}_{arch}_{args.mode}"
                if args.mode == "single":
                    panel = make_single_panel(img_src[b], img_tgt[b], pred_gt[b], gt_flow[b], sample_name)
                else:
                    panel = make_multiple_panel(imgs[b], pred_flows[b], gt_flow[b], src_idx[b], sample_name)

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
