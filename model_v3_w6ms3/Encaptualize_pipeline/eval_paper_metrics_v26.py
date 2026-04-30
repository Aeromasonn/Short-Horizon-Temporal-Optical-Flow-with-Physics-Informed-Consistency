#!/usr/bin/env python3
"""
Paper-style optical flow evaluation for our v26 variants.

Metrics:
  1. EPE_all_valid:
     Average endpoint error over all valid pixels in the evaluated split.

  2. Fl-epe / per_image_epe:
     KITTI/RAFT-style per-image EPE:
     compute mean EPE per image over valid pixels, then average over images.

  3. Fl-all / F1-all:
     KITTI optical flow outlier rate:
       outlier if EPE > 3 px AND EPE / ||GT flow|| > 0.05
     Reported as percentage of valid pixels.
     Lower is better.

Note:
  In many optical-flow papers and codebases, KITTI "F1-all" or "Fl-all" is NOT
  the precision/recall F1 score. It is the outlier percentage.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from DataLoader import TempFlowDataset_disp


def parse_args():
    parser = argparse.ArgumentParser(description="RAFT/KITTI-style EPE and Fl-all evaluation.")
    parser.add_argument("--train_script", type=str, required=True,
                        help="Variant training script, e.g. train_v26_convex.py")
    parser.add_argument("--config", type=str, required=True,
                        help="Variant config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Best checkpoint path.")
    parser.add_argument("--split", type=str, required=True,
                        help="Dataset split to evaluate, e.g. training or testing.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional quick debug limit. Omit for full split.")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_train_module(train_script: str):
    path = Path(train_script).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find train_script: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def find_trainer_class(mod):
    if hasattr(mod, "V26Trainer"):
        return mod.V26Trainer
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "forward_pipeline"):
            return obj
    raise AttributeError("Cannot find trainer class with forward_pipeline(...) in train_script.")


def load_checkpoint_to_modules(modules, checkpoint_path: str, device: torch.device):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    missing = []
    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key not in ckpt:
            missing.append(key)
        else:
            module.load_state_dict(ckpt[key], strict=True)
            module.eval()

    if missing:
        raise KeyError(
            "Checkpoint does not match this model. Missing keys:\n"
            + "\n".join(missing)
        )
    return ckpt


def select_gt_flow(pred_flows: torch.Tensor, src_idx: torch.Tensor) -> torch.Tensor:
    """
    pred_flows: [B,Tm,2,H,W]
    src_idx:    [B], identifies which adjacent pair has GT.
    returns:    [B,2,H,W]
    """
    selected = []
    for b in range(pred_flows.shape[0]):
        t = int(src_idx[b].item())
        selected.append(pred_flows[b, t])
    return torch.stack(selected, dim=0)


def compute_batch_kitti_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
    """
    pred, gt: [B,2,H,W]
    valid:    [B,H,W] or [B,1,H,W]

    Returns:
      batch pixel-level sums and per-image rows.
    """
    if valid.ndim == 4:
        valid = valid[:, 0]
    valid = valid.float() > 0.5

    epe = torch.norm(pred - gt, dim=1)                 # [B,H,W]
    gt_mag = torch.norm(gt, dim=1)                     # [B,H,W]

    # KITTI/RAFT Fl-all/F1-all outlier definition:
    # bad if endpoint error > 3 px AND relative endpoint error > 5%.
    # Add epsilon to avoid division by zero; zero-motion pixels require absolute criterion too.
    rel_err = epe / (gt_mag + 1e-9)
    bad = (epe > 3.0) & (rel_err > 0.05) & valid

    rows = []
    total_epe_sum = 0.0
    total_valid_pixels = 0
    total_bad_pixels = 0

    for b in range(pred.shape[0]):
        vb = valid[b]
        valid_count = int(vb.sum().item())

        if valid_count == 0:
            # Keep row but do not contribute to means.
            row = {
                "sample_in_batch": b,
                "valid_pixels": 0,
                "epe_valid": None,
                "fl_all_percent": None,
                "bad_pixels": 0,
            }
        else:
            epe_sum_b = float(epe[b][vb].sum().item())
            bad_count_b = int(bad[b].sum().item())

            row = {
                "sample_in_batch": b,
                "valid_pixels": valid_count,
                "epe_valid": epe_sum_b / valid_count,
                "fl_all_percent": 100.0 * bad_count_b / valid_count,
                "bad_pixels": bad_count_b,
            }

            total_epe_sum += epe_sum_b
            total_valid_pixels += valid_count
            total_bad_pixels += bad_count_b

        rows.append(row)

    return {
        "epe_sum": total_epe_sum,
        "valid_pixels": total_valid_pixels,
        "bad_pixels": total_bad_pixels,
        "rows": rows,
    }


def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg["data"]["split"] = args.split

    # Keep loader safe for large KITTI crops.
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["num_workers"] = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("variant train_script:", args.train_script)
    print("config:", args.config)
    print("checkpoint:", args.checkpoint)
    print("split:", args.split)

    train_mod = import_train_module(args.train_script)
    if not hasattr(train_mod, "build_modules"):
        raise AttributeError("train_script must define build_modules(cfg, device).")

    modules = train_mod.build_modules(cfg, device)
    ckpt = load_checkpoint_to_modules(modules, args.checkpoint, device)
    epoch = ckpt.get("epoch", "unknown")
    print("loaded checkpoint epoch:", epoch)

    TrainerClass = find_trainer_class(train_mod)
    trainer = TrainerClass(cfg, modules, optimizer=None, writer=None, device=device)

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
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    for module in modules.values():
        module.eval()

    total_epe_sum = 0.0
    total_valid_pixels = 0
    total_bad_pixels = 0
    per_image_epe_values: List[float] = []
    per_image_fl_values: List[float] = []
    csv_rows = []
    sample_global_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            try:
                imgs = batch["imgs"].to(device)
                valid = batch["valid"].to(device)
                gt_flow = batch["flow"].to(device)
                src_idx = batch["src_idx_in_seq"].to(device)
            except KeyError as e:
                raise KeyError(
                    f"Missing key {e} in dataset batch. "
                    "This usually means the selected split has no GT flow/valid mask. "
                    "Official KITTI test data usually has no public GT, so use a validation split "
                    "or evaluate test predictions through the KITTI server."
                )

            out = trainer.forward_pipeline(imgs, valid)
            pred = select_gt_flow(out["flows"], src_idx)

            metrics = compute_batch_kitti_metrics(pred, gt_flow, valid)
            total_epe_sum += metrics["epe_sum"]
            total_valid_pixels += metrics["valid_pixels"]
            total_bad_pixels += metrics["bad_pixels"]

            for row in metrics["rows"]:
                row_out = {
                    "global_sample_idx": sample_global_idx,
                    "batch_idx": batch_idx,
                    "sample_in_batch": row["sample_in_batch"],
                    "src_idx_in_seq": int(src_idx[row["sample_in_batch"]].item()),
                    "valid_pixels": row["valid_pixels"],
                    "epe_valid": row["epe_valid"],
                    "fl_all_percent": row["fl_all_percent"],
                    "bad_pixels": row["bad_pixels"],
                }
                csv_rows.append(row_out)

                if row["epe_valid"] is not None:
                    per_image_epe_values.append(float(row["epe_valid"]))
                    per_image_fl_values.append(float(row["fl_all_percent"]))

                sample_global_idx += 1

            running_epe = total_epe_sum / max(total_valid_pixels, 1)
            running_fl = 100.0 * total_bad_pixels / max(total_valid_pixels, 1)
            print(
                f"[{batch_idx + 1}/{len(loader)}] "
                f"samples={sample_global_idx} | "
                f"EPE_all_valid={running_epe:.6f} | "
                f"Fl-all/F1-all={running_fl:.4f}%"
            )

    if total_valid_pixels == 0:
        raise RuntimeError(
            "No valid pixels found. Check split/data paths/valid masks. "
            "Official KITTI test split may not include ground truth."
        )

    # Two EPE summaries:
    # - EPE_all_valid: one global average over every valid pixel.
    # - Fl-epe/per_image_epe: per-image mean EPE averaged over samples, as used by RAFT/torchvision for KITTI.
    epe_all_valid = total_epe_sum / total_valid_pixels
    fl_all_percent = 100.0 * total_bad_pixels / total_valid_pixels
    per_image_epe = sum(per_image_epe_values) / max(len(per_image_epe_values), 1)
    per_image_fl_all = sum(per_image_fl_values) / max(len(per_image_fl_values), 1)

    summary = {
        "train_script": args.train_script,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "split": args.split,
        "num_samples": sample_global_idx,
        "valid_pixels": int(total_valid_pixels),
        "bad_pixels": int(total_bad_pixels),

        "EPE_all_valid": epe_all_valid,
        "Fl_epe_per_image": per_image_epe,

        "Fl_all_percent": fl_all_percent,
        "F1_all_percent": fl_all_percent,

        "per_image_Fl_all_percent": per_image_fl_all,

        "metric_note": (
            "KITTI/RAFT-style: Fl-all/F1-all is outlier percentage, not precision/recall F1. "
            "A valid pixel is bad if EPE > 3 px AND EPE / ||GT flow|| > 0.05."
        ),
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "global_sample_idx",
            "batch_idx",
            "sample_in_batch",
            "src_idx_in_seq",
            "valid_pixels",
            "epe_valid",
            "fl_all_percent",
            "bad_pixels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n========== PAPER-STYLE SUMMARY ==========")
    print(f"split: {args.split}")
    print(f"samples: {sample_global_idx}")
    print(f"EPE_all_valid: {epe_all_valid:.6f}")
    print(f"Fl-epe / per_image_epe: {per_image_epe:.6f}")
    print(f"Fl-all / F1-all: {fl_all_percent:.4f}%")
    print(f"saved JSON: {out_json}")
    print(f"saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
