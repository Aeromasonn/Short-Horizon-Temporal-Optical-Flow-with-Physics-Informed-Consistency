from My.Encoder_sober import *
from torch.utils.data import Dataset, DataLoader
from Detector import *
from DataLoader import *
from My.Decoders_v20_fixed2 import *
from neuralop_seg.uno import UNO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import random
import argparse
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict
from typing import List, Dict, Optional, Tuple
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from My.Encoder_sober import *


THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="v22: v21 + UPFlow/UnFlow-style hard forward-backward photometric mask"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "v22_hard_fbmask_config.json"),
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

def save_detector_ckpt(save_path, epoch, detector, optimizer=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "detector_state_dict": detector.state_dict(),
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(ckpt, save_path)
    print(f"[CKPT] saved detector checkpoint to: {save_path}")

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

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
    elif arr.ndim == 2:
        return arr[top:top + crop_h, left:left + crop_w]
    else:
        raise RuntimeError(f"Unexpected arr.ndim={arr.ndim}")


def obj_map_to_boxes(obj_map, min_area=20, return_masks=False):
    """
    obj_map:
        [H,W], integer instance id map

    Returns detection target:
        boxes:  [N,4], xyxy
        labels: [N], all ones
        obj_ids:[N]
        optional masks: [N,H,W]
    """

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

    Uses:
        Additional_frames/training/image_2/000000_08.png ...
        Flow/training/flow_occ/000000_10.png
        Flow/training/obj_map/000000_10.png

    Output:
        imgs:          [T,3,H,W]
        flow:          [2,H,W]
        valid:         [H,W]
        label:         dict with boxes/labels/obj_ids
        seq_id:        str, e.g. "000000"
        flow_frame:    long, usually 10
        frame_indices: [T]
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

        self.additional_frames_dir = (
            self.root / "Additional_frames" / split / image_folder
        )

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
            stem = flow_path.stem          # e.g. 000030_10
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

        return output

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
    """
    Faster R-CNN detector for flow-like inputs.

    Default:
        MobileNetV3-FPN Faster R-CNN, expecting 3-channel input:
        [u, v, magnitude]

    backbone_name:
        "mobilenet" -> fasterrcnn_mobilenet_v3_large_fpn
        "resnet50"  -> fasterrcnn_resnet50_fpn
    """

    def __init__(
        self,
        num_classes,
        in_ch=3,
        backbone_name="mobilenet",
        pretrained_backbone=True,
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
            raise ValueError(
                f"Unknown backbone_name={backbone_name}. "
                "Use 'mobilenet' or 'resnet50'."
            )

        # If in_ch != 3, modify first conv.
        # If in_ch == 3, we keep the pretrained backbone untouched.
        if in_ch != 3:
            self._replace_first_conv(in_ch)

        # Replace classification head
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes,
        )

    def _replace_first_conv(self, in_ch):
        """
        Replace the first Conv2d layer in either MobileNetV3-FPN or ResNet50-FPN.
        """

        first_conv_parent = None
        first_conv_name = None
        old_conv = None

        # Search for first Conv2d
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
        """
        images:
            list[Tensor[C,H,W]]

        targets:
            list[dict] during training
            None during inference
        """
        return self.detector(images, targets)


def build_flow_detector_input(pred_flows, valid=None, flow_index=0, use_mag=True, use_valid=False):
    """
    pred_flows:
        [B, Tm, 2, H, W]

    valid:
        [B, H, W] or [B, 1, H, W]

    Returns:
        x: [B, C, H, W]
    """

    flow = pred_flows[:, flow_index]  # [B, 2, H, W]

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

    x = torch.cat(inputs, dim=1)
    return x


def prepare_detection_targets(labels, device):
    """
    labels:
        batch["label"], a list of dictionaries.

    Each dict should contain:
        boxes:  [N,4]
        labels: [N]

    Returns:
        targets: list[dict]
    """

    targets = []

    for lab in labels:
        boxes = lab["boxes"].to(device).float()
        cls = lab["labels"].to(device).long()

        target = {
            "boxes": boxes,
            "labels": cls,
        }

        targets.append(target)

    return targets

def filter_boxes_by_flow_activity(
    boxes,
    labels,
    flow,          # [2,H,W]
    valid=None,    # [H,W] or None
    min_inside_mag=0.5,
    min_contrast=0.25,
    min_valid_frac=0.3,
):
    """
    Keep boxes only if the flow inside the box is sufficiently active
    or sufficiently different from nearby background.
    """

    if boxes.numel() == 0:
        return boxes, labels

    device = flow.device
    boxes = boxes.to(device)
    labels = labels.to(device)

    mag = torch.norm(flow, dim=0)  # [H,W]
    H, W = mag.shape

    keep = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.round().long()

        x1 = x1.clamp(0, W - 1)
        x2 = x2.clamp(0, W)
        y1 = y1.clamp(0, H - 1)
        y2 = y2.clamp(0, H)

        if x2 <= x1 or y2 <= y1:
            keep.append(False)
            continue

        inside = mag[y1:y2, x1:x2]

        if valid is not None:
            v = valid[y1:y2, x1:x2].float()
            valid_frac = v.mean().item()

            if valid_frac < min_valid_frac:
                keep.append(False)
                continue

            inside_mag = (inside * v).sum() / (v.sum() + 1e-6)
        else:
            inside_mag = inside.mean()

        # expanded surrounding region
        pad_x = max(4, int(0.25 * (x2 - x1)))
        pad_y = max(4, int(0.25 * (y2 - y1)))

        bx1 = max(0, x1 - pad_x)
        bx2 = min(W, x2 + pad_x)
        by1 = max(0, y1 - pad_y)
        by2 = min(H, y2 + pad_y)

        patch = mag[by1:by2, bx1:bx2].clone()

        # remove inside box from background patch
        local_x1 = x1 - bx1
        local_x2 = x2 - bx1
        local_y1 = y1 - by1
        local_y2 = y2 - by1

        bg_mask = torch.ones_like(patch, dtype=torch.bool)
        bg_mask[local_y1:local_y2, local_x1:local_x2] = False

        if bg_mask.sum() > 0:
            bg_mag = patch[bg_mask].mean()
            contrast = torch.abs(inside_mag - bg_mag)
        else:
            contrast = torch.tensor(0.0, device=device)

        inside_score = torch.quantile(inside.flatten(), 0.9)
        is_active = (inside_score >= min_inside_mag) or (contrast >= min_contrast)
        keep.append(bool(is_active.item()))

    keep = torch.tensor(keep, dtype=torch.bool, device=device)

    return boxes[keep].cpu(), labels[keep].cpu()

def forward_pipeline(modules, imgs, valid, uno_use_valid_mask=True, device="cuda"):
        pair_encoder = modules['pair_encoder'].to(device)
        pair_out = pair_encoder(imgs)
        # pair_out = modules["pair_encoder"](imgs).to(device)
        pair_feats = pair_out["pair_feats"].to(device)
        flow_inits = pair_out["flow_inits"].to(device)
        corrs = pair_out["corrs"].to(device)

        if flow_inits is None:
            raise RuntimeError("v19 UNO integration requires predict_flow_init=True in the pair encoder.")

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

def flow_detection_forward_pipeline(
    batch,
    flow_modules,
    detector,
    device,
    flow_index=0,
    use_mag=True,
    use_valid=False,
    train_detector=True,
):
    """
    Full forward pipeline:

    RGB sequence
        -> optical flow pipeline
        -> pred_flows
        -> flow detector input
        -> object detector
    """

    imgs = batch["imgs"].to(device)

    valid = batch.get("valid", None)
    if valid is not None:
        valid = valid.to(device)

    # 1. Optical flow forward pass
    out_flow = forward_pipeline(
        flow_modules, imgs, valid
    )

    pred_flows = out_flow["flows"]  # [B, Tm, 2, H, W]

    # 2. Build detector input
    flow_x = build_flow_detector_input(
        pred_flows=pred_flows,
        valid=valid,
        flow_index=flow_index,
        use_mag=use_mag,
        use_valid=use_valid,
    )

    # Faster R-CNN expects list[Tensor[C,H,W]]
    det_images = [x for x in flow_x]

    # 3. Build detection targets
    targets = None
    if train_detector:
        targets = prepare_detection_targets(batch["label"], device)

    # 4. Detector forward
    det_out = detector(det_images, targets)

    return {
        "flow_out": out_flow,
        "pred_flows": pred_flows,
        "detector_input": flow_x,
        "det_out": det_out,
    }

class FlowDetectionTrainer:
    def __init__(
        self,
        flow_modules,
        detector,
        train_loader,
        val_loader,
        optimizer,
        device,
        flow_index=0,
        use_mag=True,
        use_valid=False,
        freeze_flow=True,
        use_gtFlow_for_training=False,
    ):
        self.flow_modules = flow_modules
        self.detector = detector
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        self.flow_index = flow_index
        self.use_mag = use_mag
        self.use_valid = use_valid
        self.freeze_flow = freeze_flow
        self.use_gt = use_gtFlow_for_training

        self.detector.to(device)

        for module in self.flow_modules.values():
            module.to(device)

        if freeze_flow:
            for module in self.flow_modules.values():
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def train_one_epoch(self, epoch):
        self.detector.train()

        if self.freeze_flow:
            for module in self.flow_modules.values():
                module.eval()
        else:
            for module in self.flow_modules.values():
                module.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}")

        for batch in pbar:
            self.optimizer.zero_grad()

            imgs = batch["imgs"].to(self.device)

            valid = batch.get("valid", None)
            if valid is not None:
                valid = valid.to(self.device)

            # --------------------------------------------------
            # 1. Flow forward pass
            #    If freeze_flow=True, only this part has no_grad.
            # --------------------------------------------------
            if self.freeze_flow:
                with torch.no_grad():
                    flow_out = forward_pipeline(
                        imgs=imgs,
                        valid=valid,
                        modules=self.flow_modules
                    )
            else:
                flow_out = forward_pipeline(
                    imgs=imgs,
                    valid=valid,
                    modules=self.flow_modules
                )

            pred_flows = flow_out["flows"]  # [B, Tm, 2, H, W]
            gt_flows = batch["flow"]

            # --------------------------------------------------
            # 2. Build detector input
            #    - With pred flow or GT flow
            # --------------------------------------------------
            if self.use_gt:
                gt_flows = gt_flows.to(self.device)
                flow_x = build_flow_detector_input(
                    pred_flows=gt_flows.unsqueeze(1),
                    valid=valid,
                    flow_index=self.flow_index,
                    use_mag=self.use_mag,
                    use_valid=self.use_valid,
                )
            else:
                flow_x = build_flow_detector_input(
                    pred_flows=pred_flows,
                    valid=valid,
                    flow_index=self.flow_index,
                    use_mag=self.use_mag,
                    use_valid=self.use_valid,
                )

            det_images = [x for x in flow_x]

            targets = prepare_detection_targets(
                batch["label"],
                self.device,
            )

            # Optional: skip batch if all images have zero boxes
            has_box = any(t["boxes"].numel() > 0 for t in targets)
            if not has_box:
                continue

            # --------------------------------------------------
            # 3. Detector forward pass
            #    This part needs gradients.
            # --------------------------------------------------
            loss_dict = self.detector(det_images, targets)

            loss = sum(v for v in loss_dict.values())

            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value

            pbar.set_postfix({
                "loss": loss_value,
                "cls": loss_dict.get(
                    "loss_classifier",
                    torch.tensor(0.0, device=self.device)
                ).item(),
                "box": loss_dict.get(
                    "loss_box_reg",
                    torch.tensor(0.0, device=self.device)
                ).item(),
                "obj": loss_dict.get(
                    "loss_objectness",
                    torch.tensor(0.0, device=self.device)
                ).item(),
                "rpn": loss_dict.get(
                    "loss_rpn_box_reg",
                    torch.tensor(0.0, device=self.device)
                ).item(),
            })

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def inference_batch(self, batch, score_thresh=0.3):
        self.detector.eval()

        for module in self.flow_modules.values():
            module.eval()

        result = flow_detection_forward_pipeline(
            batch=batch,
            flow_modules=self.flow_modules,
            detector=self.detector,
            device=self.device,
            flow_index=self.flow_index,
            use_mag=self.use_mag,
            use_valid=self.use_valid,
            train_detector=False,
        )

        detections = result["det_out"]

        filtered = []

        for det in detections:
            keep = det["scores"] >= score_thresh

            filtered.append({
                "boxes": det["boxes"][keep].detach().cpu(),
                "labels": det["labels"][keep].detach().cpu(),
                "scores": det["scores"][keep].detach().cpu(),
            })

        result["detections"] = filtered
        return result

    @torch.no_grad()
    def validate_visual_only(self, max_batches=5, score_thresh=0.3):
        self.detector.eval()

        all_results = []

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            result = self.inference_batch(batch, score_thresh=score_thresh)
            all_results.append(result["detections"])

        return all_results

    def fit(self, epochs):
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)

            print(f"[Epoch {epoch}] train_loss = {train_loss:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
            })

        return history


    @torch.no_grad()
    def inference_batch(self, batch, score_thresh=0.3):
        self.detector.eval()

        for module in self.flow_modules.values():
            module.eval()

        result = flow_detection_forward_pipeline(
            batch=batch,
            flow_modules=self.flow_modules,
            detector=self.detector,
            device=self.device,
            flow_index=self.flow_index,
            use_mag=self.use_mag,
            use_valid=self.use_valid,
            train_detector=False,
        )

        detections = result["det_out"]

        filtered = []

        for det in detections:
            keep = det["scores"] >= score_thresh

            filtered.append({
                "boxes": det["boxes"][keep].detach().cpu(),
                "labels": det["labels"][keep].detach().cpu(),
                "scores": det["scores"][keep].detach().cpu(),
            })

        result["detections"] = filtered
        return result

    @torch.no_grad()
    def validate_visual_only(self, max_batches=5, score_thresh=0.3):
        self.detector.eval()

        all_results = []

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            result = self.inference_batch(batch, score_thresh=score_thresh)
            all_results.append(result["detections"])

        return all_results

    def fit(self, epochs):
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)

            print(f"[Epoch {epoch}] train_loss = {train_loss:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
            })

        return history

def load_checkpoint(ckpt_path, modules, device, optimizer=None, strict=True):
    ckpt = torch.load(ckpt_path, map_location=device)

    # ---- load module weights ----
    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key in ckpt:
            module.load_state_dict(ckpt[key], strict=strict)
        else:
            print(f"[Warning] Missing key: {key}")

        module.to(device)

    # ---- load optimizer (optional) ----
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ---- metadata ----
    epoch = ckpt.get("epoch", None)
    stats = ckpt.get("stats", None)
    config = ckpt.get("config", None)

    print(f"Loaded checkpoint complete from: {ckpt_path} (epoch={epoch})")

    return {
        "epoch": epoch,
        "stats": stats,
        "config": config,
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
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TempFlowDataset_ObjMap(
        root=cfg["data"]["data_root"],
        split=cfg["data"]["split"],
        image_folder=cfg["data"]["image_folder"],
        flow_type=cfg["data"]["flow_type"],
        disp_type=cfg["data"]["disp_type"],
        seq_len=cfg["data"]["seq_len"],
        # center_frame_idx=cfg["data"]["center_frame_idx"],
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
        collate_fn=detection_collate_fn if cfg['Downstream']['use_collate_fn'] else None
    )
    print('Dataset loaded')

    modules = build_modules(cfg, device)
    print('Modules loaded')
    optimizer = torch.optim.AdamW(
        [p for module in modules.values() for p in module.parameters()],
        lr=1e-4,
        weight_decay=1e-4
    )

    meta = load_checkpoint(
        ckpt_path="My/fullpipeline_v22_best.pth",
        modules=modules,
        device=device,
        optimizer=optimizer,  # optional
    )
    for m in modules.values():
        m.to(device)
        print(m)

    ID_TO_CLASS_NAME = {
        1: "object",
    }

    detector = FlowObjectDetector(
        num_classes=2,
        in_ch=3,  # u, v, magnitude
        pretrained_backbone=True,
    )

    optimizer = torch.optim.AdamW(
        detector.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    trainer = FlowDetectionTrainer(
        flow_modules=modules,
        detector=detector,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        device=device,
        flow_index=cfg['Downstream']['flow_idx'],
        use_mag=True,
        use_valid=False,
        freeze_flow=True,
        use_gtFlow_for_training=cfg['Downstream']['use_gtFlow_for_training']
    )

    print('Training start')
    epoch = cfg['Downstream']['train_ep']
    trainer.fit(epochs=epoch)

    save_detector_ckpt(
        save_path=f"ckpts/detector_epoch_{epoch:03d}.pt",
        epoch=epoch,
        detector=detector,
        optimizer=optimizer,
    )

    print(f'Training concluded, ckpt saved to ckpts/detector_epoch_{epoch:03d}.pt')


if __name__ == "__main__":
    main()