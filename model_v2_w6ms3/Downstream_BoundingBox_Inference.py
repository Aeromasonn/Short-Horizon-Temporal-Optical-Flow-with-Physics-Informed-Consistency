from My.Encoder_sober import *
from torch.utils.data import Dataset, DataLoader
from Detector import *
from DataLoader import *
from My.Decoders_v20_fixed2 import *
from neuralop_seg.uno import UNO

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from My.Encoder_sober import *


THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent


# ============================================================
# Args / config
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-only flow-object detector. Loads flow ckpt + detector ckpt and saves prediction PNGs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "v22_hard_fbmask_config.json"),
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--flow_ckpt",
        type=str,
        default="My/fullpipeline_v22_best.pth",
        help="Checkpoint for the optical-flow pipeline.",
    )
    parser.add_argument(
        "--detector_ckpt",
        type=str,
        default="ckpts/detector_epoch_050.pt",
        help="Checkpoint for the downstream Faster R-CNN detector.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="preds",
        help="Directory to save prediction visualizations.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--flow_index", type=int, default=1)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--use_mag", action="store_true", default=True)
    parser.add_argument("--no_mag", dest="use_mag", action="store_false")
    parser.add_argument("--use_valid", action="store_true", default=False)
    parser.add_argument(
        "--save_gt",
        action="store_true",
        default=False,
        help="If set, save a two-row GT + prediction visualization. Otherwise save prediction-only PNGs.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Dataset utilities
# ============================================================

def read_rgb_img(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def read_kitti_flow(path):
    flow_png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if flow_png is None:
        raise FileNotFoundError(f"Flow not found: {path}")

    flow_png = flow_png.astype(np.float32)

    valid = flow_png[:, :, 0] > 0
    u = (flow_png[:, :, 2] - 32768.0) / 64.0
    v = (flow_png[:, :, 1] - 32768.0) / 64.0

    flow = np.stack([u, v], axis=-1).astype(np.float32)
    valid = valid.astype(np.float32)

    return flow, valid


def read_kitti_disp(path):
    disp_png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if disp_png is None:
        raise FileNotFoundError(f"Disparity not found: {path}")

    disp_png = disp_png.astype(np.float32)
    disp = disp_png / 256.0
    valid = (disp_png > 0).astype(np.float32)

    return disp, valid


def read_obj_map(path):
    obj = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if obj is None:
        raise FileNotFoundError(f"Object map not found: {path}")
    return obj.astype(np.int32)


def center_crop(arr, crop_size):
    crop_h, crop_w = crop_size
    h, w = arr.shape[:2]

    if crop_h > h or crop_w > w:
        raise RuntimeError(f"Crop size {crop_size} larger than image size {h}x{w}")

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    if arr.ndim == 3:
        return arr[top:top + crop_h, left:left + crop_w, :]
    if arr.ndim == 2:
        return arr[top:top + crop_h, left:left + crop_w]
    raise RuntimeError(f"Unexpected arr.ndim={arr.ndim}")


def obj_map_to_boxes(obj_map, min_area=20, return_masks=False):
    boxes = []
    labels = []
    obj_ids = []
    masks = []

    unique_ids = np.unique(obj_map)

    for obj_id in unique_ids:
        if obj_id == 0:
            continue

        mask = obj_map == obj_id
        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        x1 = xs.min()
        x2 = xs.max()
        y1 = ys.min()
        y2 = ys.max()

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        area = w * h

        if area < min_area:
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(1)
        obj_ids.append(int(obj_id))

        if return_masks:
            masks.append(mask.astype(np.uint8))

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "obj_ids": torch.tensor(obj_ids, dtype=torch.long),
    }

    if return_masks:
        if len(masks) > 0:
            target["masks"] = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
        else:
            H, W = obj_map.shape
            target["masks"] = torch.zeros((0, H, W), dtype=torch.uint8)

    return target


def detection_collate_fn(batch):
    out = {}

    for k in batch[0].keys():
        if k == "label":
            out[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]

    return out


class TempFlowDataset_ObjMap(Dataset):
    """
    KITTI Flow / Scene Flow dataset with aligned obj_map labels.

    Also returns:
        sample_idx: original dataset index
        image_ids: e.g. ["000000_09", "000000_10", ...]
        src_img_id / tgt_img_id when available
    """

    def __init__(
        self,
        root,
        split="training",
        image_folder="image_2",
        flow_type="flow_occ",
        disp_type="disp_occ",
        seq_len=5,
        crop_size=(352, 1216),
        normalize=True,
        stats_in=None,
        return_pair_only=False,
        min_obj_area=20,
        return_masks=False,
        require_obj_map=True,
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.image_folder = image_folder
        self.flow_type = flow_type
        self.disp_type = disp_type
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.normalize = normalize
        self.stats_in = stats_in
        self.return_pair_only = return_pair_only
        self.min_obj_area = min_obj_area
        self.return_masks = return_masks
        self.require_obj_map = require_obj_map

        self.additional_frames_dir = self.root / "Additional_frames" / split / image_folder
        self.flow_dir = self.root / "Flow" / split / flow_type
        self.obj_map_dir = self.root / "Flow" / split / "obj_map"
        self.disp0_dir = self.root / "Flow" / split / f"{disp_type}_0"
        self.disp1_dir = self.root / "Flow" / split / f"{disp_type}_1"

        if not self.additional_frames_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {self.additional_frames_dir}")
        if not self.flow_dir.exists():
            raise FileNotFoundError(f"Missing flow folder: {self.flow_dir}")
        if self.require_obj_map and not self.obj_map_dir.exists():
            raise FileNotFoundError(f"Missing obj_map folder: {self.obj_map_dir}")
        if not self.disp0_dir.exists():
            raise FileNotFoundError(f"Missing disparity folder: {self.disp0_dir}")
        if not self.disp1_dir.exists():
            raise FileNotFoundError(f"Missing disparity folder: {self.disp1_dir}")

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found.")

        self.stats = self._load_stats()

    def _build_samples(self):
        flow_files = sorted(self.flow_dir.glob("*.png"))
        samples = []
        half = self.seq_len // 2

        for flow_path in flow_files:
            stem = flow_path.stem
            seq_id, frame_str = stem.split("_")
            flow_frame = int(frame_str)

            if self.return_pair_only:
                frame_indices = [flow_frame, flow_frame + 1]
            else:
                start = flow_frame - half + 1
                frame_indices = list(range(start, start + self.seq_len))

            img_paths = []
            valid_sample = True

            for t in frame_indices:
                img_path = self.additional_frames_dir / f"{seq_id}_{t:02d}.png"
                if not img_path.exists():
                    valid_sample = False
                    break
                img_paths.append(str(img_path))

            if not valid_sample:
                continue

            obj_map_path = self.obj_map_dir / f"{seq_id}_{flow_frame:02d}.png"
            if self.require_obj_map and not obj_map_path.exists():
                continue

            disp_name = f"{seq_id}_{flow_frame:02d}.png"
            disp0_path = self.disp0_dir / disp_name
            disp1_path = self.disp1_dir / disp_name

            if not disp0_path.exists() or not disp1_path.exists():
                continue

            samples.append({
                "seq_id": seq_id,
                "flow_frame": flow_frame,
                "frame_indices": frame_indices,
                "img_paths": img_paths,
                "image_ids": [f"{seq_id}_{t:02d}" for t in frame_indices],
                "flow_path": str(flow_path),
                "obj_map_path": str(obj_map_path),
                "disp0_path": str(disp0_path),
                "disp1_path": str(disp1_path),
            })

        return samples

    def _load_stats(self):
        if self.stats_in is not None and os.path.exists(self.stats_in):
            with open(self.stats_in, "r") as f:
                return json.load(f)

        stats = self.compute_stats()

        if self.stats_in is not None:
            dirname = os.path.dirname(self.stats_in)
            if dirname != "":
                os.makedirs(dirname, exist_ok=True)

            with open(self.stats_in, "w") as f:
                json.dump(stats, f, indent=2)

        return stats

    def compute_stats(self):
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        seen = set()

        for sample in self.samples:
            for img_path in sample["img_paths"]:
                if img_path in seen:
                    continue

                seen.add(img_path)
                img = read_rgb_img(img_path)

                if self.crop_size is not None:
                    img = center_crop(img, self.crop_size)

                h, w, _ = img.shape
                flat = img.reshape(-1, 3)
                channel_sum += flat.sum(axis=0)
                channel_sq_sum += (flat ** 2).sum(axis=0)
                pixel_count += h * w

        mean = channel_sum / pixel_count
        var = channel_sq_sum / pixel_count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "num_unique_frames": len(seen),
            "num_samples": len(self.samples),
        }

    def _normalize_img(self, img):
        mean = np.array(self.stats["mean"], dtype=np.float32)
        std = np.array(self.stats["std"], dtype=np.float32)
        return (img - mean) / std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = []

        for img_path in sample["img_paths"]:
            img = read_rgb_img(img_path)

            if self.crop_size is not None:
                img = center_crop(img, self.crop_size)

            if self.normalize:
                img = self._normalize_img(img)

            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)

        flow, valid = read_kitti_flow(sample["flow_path"])

        if self.crop_size is not None:
            flow = center_crop(flow, self.crop_size)
            valid = center_crop(valid, self.crop_size)

        flow = torch.from_numpy(flow).permute(2, 0, 1).contiguous()
        valid = torch.from_numpy(valid).contiguous()

        obj_map = read_obj_map(sample["obj_map_path"])

        if self.crop_size is not None:
            obj_map = center_crop(obj_map, self.crop_size)

        target = obj_map_to_boxes(
            obj_map,
            min_area=self.min_obj_area,
            return_masks=self.return_masks,
        )

        disp0, disp_valid0 = read_kitti_disp(sample["disp0_path"])
        disp1, disp_valid1 = read_kitti_disp(sample["disp1_path"])

        if self.crop_size is not None:
            disp0 = center_crop(disp0, self.crop_size)
            disp1 = center_crop(disp1, self.crop_size)
            disp_valid0 = center_crop(disp_valid0, self.crop_size)
            disp_valid1 = center_crop(disp_valid1, self.crop_size)

        disp0 = torch.from_numpy(disp0).unsqueeze(0).contiguous()
        disp1 = torch.from_numpy(disp1).unsqueeze(0).contiguous()
        disp_valid0 = torch.from_numpy(disp_valid0).unsqueeze(0).contiguous()
        disp_valid1 = torch.from_numpy(disp_valid1).unsqueeze(0).contiguous()

        disp = torch.stack([disp0, disp1], dim=0)
        disp_valid = torch.stack([disp_valid0, disp_valid1], dim=0)

        output = {
            "imgs": imgs,
            "flow": flow,
            "valid": valid,
            "label": target,
            "obj_map": torch.from_numpy(obj_map).long(),
            "disp": disp,
            "disp_valid": disp_valid,
            "seq_id": sample["seq_id"],
            "flow_frame": torch.tensor(sample["flow_frame"], dtype=torch.long),
            "frame_indices": torch.tensor(sample["frame_indices"], dtype=torch.long),
            "image_ids": sample["image_ids"],
            "sample_idx": torch.tensor(idx, dtype=torch.long),
        }

        if imgs.shape[0] >= 2:
            gt_src = sample["flow_frame"]
            indices = sample["frame_indices"]

            if gt_src in indices and (gt_src + 1) in indices:
                src_pos = indices.index(gt_src)
                tgt_pos = indices.index(gt_src + 1)

                output["img_src"] = imgs[src_pos]
                output["img_tgt"] = imgs[tgt_pos]
                output["src_idx_in_seq"] = torch.tensor(src_pos, dtype=torch.long)
                output["tgt_idx_in_seq"] = torch.tensor(tgt_pos, dtype=torch.long)
                output["src_img_id"] = f"{sample['seq_id']}_{gt_src:02d}"
                output["tgt_img_id"] = f"{sample['seq_id']}_{gt_src + 1:02d}"

        return output


# ============================================================
# Model construction / checkpoints
# ============================================================

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


class FlowObjectDetector(nn.Module):
    def __init__(
        self,
        num_classes,
        in_ch=3,
        backbone_name="mobilenet",
        pretrained_backbone=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_ch = in_ch
        self.backbone_name = backbone_name

        if backbone_name == "mobilenet":
            self.detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone="DEFAULT" if pretrained_backbone else None,
            )
        elif backbone_name == "resnet50":
            self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone="DEFAULT" if pretrained_backbone else None,
            )
        else:
            raise ValueError(f"Unknown backbone_name={backbone_name}. Use 'mobilenet' or 'resnet50'.")

        if in_ch != 3:
            self._replace_first_conv(in_ch)

        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def _replace_first_conv(self, in_ch):
        first_conv_parent = None
        first_conv_name = None
        old_conv = None

        for name, module in self.detector.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                old_conv = module
                parts = name.split(".")
                parent = self.detector.backbone

                for p in parts[:-1]:
                    parent = getattr(parent, p)

                first_conv_parent = parent
                first_conv_name = parts[-1]
                break

        if old_conv is None:
            raise RuntimeError("Could not find first Conv2d in detector backbone.")

        new_conv = nn.Conv2d(
            in_ch,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
        )

        with torch.no_grad():
            if in_ch == 3:
                new_conv.weight.copy_(old_conv.weight)
            elif in_ch < 3:
                new_conv.weight.copy_(old_conv.weight[:, :in_ch])
            else:
                new_conv.weight[:, :old_conv.in_channels].copy_(old_conv.weight)
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                for c in range(old_conv.in_channels, in_ch):
                    new_conv.weight[:, c:c + 1].copy_(mean_weight)

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        setattr(first_conv_parent, first_conv_name, new_conv)

    def forward(self, images, targets=None):
        return self.detector(images, targets)


def load_checkpoint(ckpt_path, modules, device, optimizer=None, strict=True):
    ckpt = torch.load(ckpt_path, map_location=device)

    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key in ckpt:
            module.load_state_dict(ckpt[key], strict=strict)
        else:
            print(f"[Warning] Missing key: {key}")

        module.to(device)
        module.eval()

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", None)
    stats = ckpt.get("stats", None)
    config = ckpt.get("config", None)

    print(f"[Flow CKPT] loaded from: {ckpt_path} (epoch={epoch})")
    return {"epoch": epoch, "stats": stats, "config": config}


def load_detector_ckpt(ckpt_path, device="cuda", num_classes=2, in_ch=3, backbone_name="mobilenet"):
    detector = FlowObjectDetector(
        num_classes=num_classes,
        in_ch=in_ch,
        backbone_name=backbone_name,
        pretrained_backbone=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    detector.load_state_dict(ckpt["detector_state_dict"], strict=True)
    detector.eval()

    print(f"[Detector CKPT] loaded from: {ckpt_path}")
    print(f"[Detector CKPT] epoch={ckpt.get('epoch', 'unknown')}")
    return detector, ckpt


# ============================================================
# Inference
# ============================================================

def build_flow_detector_input(pred_flows, valid=None, flow_index=0, use_mag=True, use_valid=False):
    flow = pred_flows[:, flow_index]
    inputs = [flow]

    if use_mag:
        mag = torch.norm(flow, dim=1, keepdim=True)
        inputs.append(mag)

    if use_valid:
        if valid is None:
            raise ValueError("valid is required when use_valid=True")

        if valid.ndim == 3:
            valid = valid.unsqueeze(1)

        inputs.append(valid.float())

    return torch.cat(inputs, dim=1)


def forward_pipeline(modules, imgs, valid, uno_use_valid_mask=True, device="cuda"):
    pair_encoder = modules["pair_encoder"].to(device)
    pair_out = pair_encoder(imgs)

    pair_feats = pair_out["pair_feats"].to(device)
    flow_inits = pair_out["flow_inits"].to(device)
    corrs = pair_out["corrs"].to(device)

    if flow_inits is None:
        raise RuntimeError("UNO integration requires predict_flow_init=True in the pair encoder.")

    visual_feats = modules["visual_branch"](imgs)
    motion_feats = modules["motion_branch"](pair_feats)
    fused_seq = modules["fusion"](visual_feats, motion_feats)

    valid_ds = None
    if uno_use_valid_mask:
        valid_ds = downsample_valid_mask(valid, fused_seq.shape[-2:])

    uno_in = build_uno_input_2d(fused_seq, flow_inits, valid_mask=valid_ds)
    uno_feat = modules["uno"](uno_in)

    b, tm, _, h, w = fused_seq.shape
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


@torch.no_grad()
def inference_one_batch(
    batch,
    flow_modules,
    detector,
    device,
    flow_index=0,
    use_mag=True,
    use_valid=False,
    score_thresh=0.3,
):
    detector.eval()
    for m in flow_modules.values():
        m.eval()

    imgs = batch["imgs"].to(device)

    valid = batch.get("valid", None)
    if valid is not None:
        valid = valid.to(device)

    flow_out = forward_pipeline(
        modules=flow_modules,
        imgs=imgs,
        valid=valid,
        device=device,
    )

    pred_flows = flow_out["flows"]

    flow_x = build_flow_detector_input(
        pred_flows=pred_flows,
        valid=valid,
        flow_index=flow_index,
        use_mag=use_mag,
        use_valid=use_valid,
    )

    det_images = [x for x in flow_x]
    det_out = detector(det_images)

    detections = []
    for det in det_out:
        keep = det["scores"] >= score_thresh
        detections.append({
            "boxes": det["boxes"][keep].detach().cpu(),
            "labels": det["labels"][keep].detach().cpu(),
            "scores": det["scores"][keep].detach().cpu(),
        })

    return {
        "pred_flows": pred_flows.detach().cpu(),
        "detector_input": flow_x.detach().cpu(),
        "detections": detections,
    }


# ============================================================
# Visualization / saving
# ============================================================

ID_TO_CLASS_NAME = {1: "Object"}


def flow_to_rgb(flow):
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255, 0, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def draw_boxes(ax, boxes, labels=None, scores=None, color="red", linewidth=2, prefix=""):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        text = prefix
        if labels is not None:
            cls_id = int(labels[i])
            cls_name = ID_TO_CLASS_NAME.get(cls_id, str(cls_id))
            text += cls_name

        if scores is not None:
            text += f" {float(scores[i]):.2f}"

        if text.strip():
            ax.text(
                x1,
                max(y1 - 3, 0),
                text,
                color=color,
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
            )


def get_batch_string(batch, key, sample_idx, default="unknown"):
    if key not in batch:
        return default

    value = batch[key]

    if isinstance(value, list):
        return str(value[sample_idx])

    if isinstance(value, torch.Tensor):
        v = value[sample_idx]
        if v.ndim == 0:
            return str(int(v.item()))
        return "_".join(str(int(x)) for x in v.detach().cpu().flatten().tolist())

    return str(value)


def make_save_name(batch, sample_idx, flow_index):
    seq_id = get_batch_string(batch, "seq_id", sample_idx, "seq")
    flow_frame = get_batch_string(batch, "flow_frame", sample_idx, "frame")
    sample_idx_str = get_batch_string(batch, "sample_idx", sample_idx, str(sample_idx))

    # Prefer source->target image ids returned by the dataloader.
    if "src_img_id" in batch and "tgt_img_id" in batch:
        src_id = get_batch_string(batch, "src_img_id", sample_idx, "")
        tgt_id = get_batch_string(batch, "tgt_img_id", sample_idx, "")
        if src_id and tgt_id:
            return f"{sample_idx_str}_{src_id}_to_{tgt_id}_flow{flow_index}.png"

    # Fallback: derive from seq_id + frame_indices.
    if "frame_indices" in batch:
        frame_indices = batch["frame_indices"][sample_idx].detach().cpu().tolist()
        if 0 <= flow_index < len(frame_indices) - 1:
            src_id = f"{seq_id}_{int(frame_indices[flow_index]):02d}"
            tgt_id = f"{seq_id}_{int(frame_indices[flow_index + 1]):02d}"
            return f"{sample_idx_str}_{src_id}_to_{tgt_id}_flow{flow_index}.png"

    return f"{sample_idx_str}_{seq_id}_{flow_frame}_flow{flow_index}.png"


def save_prediction_png(
    result,
    batch,
    sample_idx,
    flow_index,
    out_path,
    save_gt=False,
    score_thresh=0.3,
):
    pred_flow = result["pred_flows"][sample_idx, flow_index]
    flow_rgb = flow_to_rgb(pred_flow)

    pred = result["detections"][sample_idx]
    gt = batch["label"][sample_idx]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if save_gt:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # =======================
        # TOP: GT
        # =======================
        ax_gt = axes[0]
        ax_gt.imshow(flow_rgb)
        ax_gt.axis("off")

        draw_boxes(
            ax_gt,
            boxes=gt["boxes"],
            labels=gt["labels"],
            scores=None,
            color="lime",
            linewidth=2,
            prefix="GT ",
        )
        ax_gt.set_title("Ground Truth (Green)")

        # =======================
        # BOTTOM: Predictions
        # =======================
        ax_pred = axes[1]
        ax_pred.imshow(flow_rgb)
        ax_pred.axis("off")

        if pred["boxes"].numel() > 0:
            scores = pred["scores"]

            keep = scores >= score_thresh
            if keep.any():
                draw_boxes(
                    ax_pred,
                    boxes=pred["boxes"][keep],
                    labels=pred["labels"][keep],
                    scores=scores[keep],
                    color="red",
                    linewidth=2,
                    prefix="Pred ",
                )

        ax_pred.set_title(f"Predictions (Red) | score_thresh={score_thresh}")

    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.imshow(flow_rgb)
        ax.axis("off")

        if pred["boxes"].numel() > 0:
            scores = pred["scores"]
            keep = scores >= score_thresh

            if keep.any():
                draw_boxes(
                    ax,
                    boxes=pred["boxes"][keep],
                    labels=pred["labels"][keep],
                    scores=scores[keep],
                    color="red",
                    linewidth=2,
                    prefix="Pred ",
                )

        ax.set_title(out_path.stem)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)

    if "train" in cfg and "seed" in cfg["train"]:
        set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TempFlowDataset_ObjMap(
        root=cfg["data"]["data_root"],
        split=cfg["data"]["split"],
        image_folder=cfg["data"]["image_folder"],
        flow_type=cfg["data"]["flow_type"],
        disp_type=cfg["data"]["disp_type"],
        seq_len=cfg["data"]["seq_len"],
        crop_size=tuple(cfg["data"]["crop_size"]),
        normalize=cfg["data"]["normalize"],
        stats_in=cfg["data"]["stats_file"],
        return_pair_only=cfg["data"]["return_pair_only"],
    )

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = cfg.get("train", {}).get("num_workers", 0)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.get("train", {}).get("pin_memory", True),
        collate_fn=detection_collate_fn,
    )

    print(f"[Data] samples={len(dataset)} batch_size={args.batch_size}")

    flow_modules = build_modules(cfg, device)
    load_checkpoint(
        ckpt_path=args.flow_ckpt,
        modules=flow_modules,
        device=device,
        optimizer=None,
        strict=True,
    )

    detector_in_ch = 2 + int(args.use_mag) + int(args.use_valid)
    detector, _ = load_detector_ckpt(
        ckpt_path=args.detector_ckpt,
        device=device,
        num_classes=2,
        in_ch=detector_in_ch,
        backbone_name="mobilenet",
    )

    saved = 0

    for batch_i, batch in enumerate(tqdm(loader, desc="Inference")):
        if args.max_batches is not None and batch_i >= args.max_batches:
            break

        result = inference_one_batch(
            batch=batch,
            flow_modules=flow_modules,
            detector=detector,
            device=device,
            flow_index=args.flow_index,
            use_mag=args.use_mag,
            use_valid=args.use_valid,
            score_thresh=args.score_thresh,
        )

        B = result["pred_flows"].shape[0]

        for sample_i in range(B):
            if args.max_samples is not None and saved >= args.max_samples:
                break

            save_name = make_save_name(batch, sample_i, args.flow_index)
            save_path = out_dir / save_name

            save_prediction_png(
                result=result,
                batch=batch,
                sample_idx=sample_i,
                flow_index=args.flow_index,
                out_path=save_path,
                save_gt=args.save_gt,
                score_thresh=args.score_thresh,
            )

            saved += 1

        if args.max_samples is not None and saved >= args.max_samples:
            break

    print(f"[Done] saved {saved} PNG files to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
