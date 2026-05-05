from PIL import Image
import torch
from networkx.algorithms.distance_measures import center
from torchvision import transforms
import numpy as np

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from torch.utils.data import Dataset, DataLoader
# --------------------------------
# v0_Copied from Tests --- 04/06
# --------------------------------
def read_rgb_img(path):
    """
    reads RGB image as float in [0,1] (normalized) of shape (H,W,3)
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Image not found at {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalize
    return img


def read_kitti_flow(path):
    """
    read KITTI flow GT PNG for optical flow (2D)

    Return:
        flow: (H,W,2), float32
        valid: (H,W), float32, 1 for valid flow, 0 for invalid

    KITTI flow PNG stores 3 channels in 16-bit PNG.
    Common conventions:
        valid = channel 0>0
        u = (channel 2 - 2^15) / 64
        v = (channel 1 - 2^15) / 64
    """
    flow_png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if flow_png is None:
        raise FileNotFoundError(f'Flow not found at {path}')
    flow_png = flow_png.astype(np.float32)

    # KITTI convention processing
    valid = flow_png[:, :, 0] > 0
    u = (flow_png[:, :, 2] - 32768.0) / 64.0
    v = (flow_png[:, :, 1] - 32768.0) / 64.0

    flow = np.stack([u, v], axis=-1).astype(np.float32)
    valid = valid.astype(np.float32)
    return flow, valid


def read_kitti_disp(path):
    """
    Read KITTI disparity PNG for scene flow (3D)

    Returns:
        disp  : (H, W) float32
        valid : (H, W) float32
    """
    disp_png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp_png is None:
        raise FileNotFoundError(f'Could not read disparity file: {path}')

    disp_png = disp_png.astype(np.float32)

    # KITTI disparity is stored scaled by 256
    disp = disp_png / 256.0
    valid = (disp_png > 0).astype(np.float32)

    return disp, valid


def center_crop(arr, crop_size):
    """
    KITTI raw images comes in 1242*375 and 1241*376.
    Do center crop to ensure image shape consistency.
    """
    crop_h, crop_w = crop_size
    h, w = arr.shape[:2]

    if crop_h > h or crop_w > w:
        raise RuntimeError(f'Crop size {crop_size} is larger than image size {h}x{w}')

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    if arr.ndim == 3:
        return arr[top:top + crop_h, left:left + crop_w, :]
    elif arr.ndim == 2:
        return arr[top:top + crop_h, left:left + crop_w]
    else:
        raise RuntimeError(f'Unexpected arr.ndim {arr.ndim}')

class TempFlowDataset_disp(Dataset):
    """
    root                : dataset root directory
    split               : 'training' / 'testing'
    image_folder        : image subfolder (e.g., image_2)
    flow_type           : flow type (unused but reserved)
    seq_len             : number of frames per sample
    center_frame_idx    : center frame index (KITTI default ~10)
    crop_size           : center crop size (default (352, 1216)
    normalize           : whether to apply z-score normalization
    stats_in            : path to precomputed stats (json)
    return_pair_only    : if True, only return 2 frames (t, t+1)

    additional_frames_dir : path to RGB frames
    flow_dir              : path to optical flow GT
    samples               : list of valid samples
    stats                 : normalization stats (mean/std)

    Out:
    {
        'imgs'            : (T, 3, H, W)
        'flow'            : (2, H, W)
        'valid'           : (H, W)
        'seq_id'          : str
        'flow_frame'      : int
        'frame_indices'   : (T,)
        optional but helpful to include:
            'img_src', 'img_tgt'
            'src_idx_in_seq', 'tgt_idx_in_seq'
    }
    """

    def __init__(self,
                 root, split='training', image_folder='image_2', flow_type='flow_occ',
                 disp_type='disp_occ',
                 seq_len=5, center_frame_idx=10, crop_size=(352, 1216),
                 normalize=True, stats_in=None, return_pair_only=False):

        super().__init__()

        self.root = Path(root)
        self.split = split
        self.image_folder = image_folder
        self.flow_type = flow_type
        self.disp_type = disp_type
        self.seq_len = seq_len
        self.center_frame_idx = center_frame_idx
        self.crop_size = crop_size
        self.normalize = normalize
        self.stats_in = stats_in
        self.return_pair_only = return_pair_only

        self.additional_frames_dir = self.root / 'Additional_frames' / split / image_folder
        self.flow_dir = self.root / 'Flow' / split / flow_type
        self.disp0_dir = self.root / 'Flow' / split / f'{disp_type}_0'
        self.disp1_dir = self.root / 'Flow' / split / f'{disp_type}_1'

        if not self.additional_frames_dir.exists():
            raise FileNotFoundError(f'Image folder missing: {self.additional_frames_dir}')
        if not self.flow_dir.exists():
            raise FileNotFoundError(f'Flow folder missing: {self.flow_dir}')
        if not self.disp0_dir.exists():
            raise FileNotFoundError(f'Disparity folder missing: {self.disp0_dir}')
        if not self.disp1_dir.exists():
            raise FileNotFoundError(f'Disparity folder missing: {self.disp1_dir}')

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError('No valid samples found.')

        self.stats = self._load_stats()

    def _build_samples(self):
        flow_files = sorted(self.flow_dir.glob('*.png'))
        samples = []

        half = self.seq_len // 2

        for flow_path in flow_files:
            stem = flow_path.stem
            seq_id, frame_id = stem.split('_')
            frame_id = int(frame_id)

            if self.return_pair_only:
                frame_indices = [frame_id, frame_id + 1]
            else:
                start = frame_id - half + 1
                frame_indices = list(range(start, start + self.seq_len))

            img_paths = []
            valid_sample = True

            for t in frame_indices:
                img_name = f"{seq_id}_{t:02d}.png"
                img_path = self.additional_frames_dir / img_name
                if not img_path.exists():
                    valid_sample = False
                    break
                img_paths.append(str(img_path))

            if not valid_sample:
                continue

            disp_name = f'{seq_id}_{frame_id:02d}.png'
            disp0_path = self.disp0_dir / disp_name
            disp1_path = self.disp1_dir / disp_name

            if not disp0_path.exists() or not disp1_path.exists():
                continue

            samples.append({
                "seq_id": seq_id,
                "flow_frame": frame_id,
                "img_paths": img_paths,
                "flow_path": str(flow_path),
                "disp0_path": str(disp0_path),
                "disp1_path": str(disp1_path),
                "frame_indices": frame_indices,
            })

        return samples

    def _load_stats(self):
        """
        Load the stats if there is any
        """
        if self.stats_in is not None and os.path.exists(self.stats_in):
            with open(self.stats_in, 'r') as f:
                return json.load(f)

        stats = self.compute_stats()

        if self.stats_in is not None:
            dir_name = os.path.dirname(self.stats_in)
            if dir_name != '':
                # if given folder
                os.makedirs(os.path.dirname(self.stats_in), exist_ok=True)
            else:
                # else save to current directory
                pass
            with open(self.stats_in, "w") as f:
                json.dump(stats, f, indent=2)

        return stats

    def compute_stats(self):
        """
        Compute RGB mean/std over dataset
        """
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0

        seen = set()

        for sample in self.samples:
            for img_path in sample['img_paths']:
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
            'mean': mean.tolist(),
            'std': std.tolist(),
            'num_unique_frames': len(seen),
            'num_samples': len(self.samples),
        }

    def _normalize_img(self, img):
        mean = np.array(self.stats['mean'], dtype=np.float32)
        std = np.array(self.stats['std'], dtype=np.float32)
        return (img - mean) / std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = []

        for img_path in sample['img_paths']:
            img = read_rgb_img(img_path)
            if self.crop_size is not None:
                img = center_crop(img, self.crop_size)
            if self.normalize:
                img = self._normalize_img(img)
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)  # (T, 3, H, W)

        flow, valid = read_kitti_flow(sample['flow_path'])
        if self.crop_size is not None:
            flow = center_crop(flow, self.crop_size)
            valid = center_crop(valid, self.crop_size)
        flow = torch.from_numpy(flow).permute(2, 0, 1).contiguous()  # (2, H, W)
        valid = torch.from_numpy(valid).contiguous()  # (H, W)

        disp0, disp_valid0 = read_kitti_disp(sample['disp0_path'])
        disp1, disp_valid1 = read_kitti_disp(sample['disp1_path'])

        if self.crop_size is not None:
            disp0 = center_crop(disp0, self.crop_size)
            disp1 = center_crop(disp1, self.crop_size)
            disp_valid0 = center_crop(disp_valid0, self.crop_size)
            disp_valid1 = center_crop(disp_valid1, self.crop_size)

        disp0 = torch.from_numpy(disp0).unsqueeze(0).contiguous()  # (1, H, W)
        disp1 = torch.from_numpy(disp1).unsqueeze(0).contiguous()  # (1, H, W)
        disp_valid0 = torch.from_numpy(disp_valid0).unsqueeze(0).contiguous()  # (1, H, W)
        disp_valid1 = torch.from_numpy(disp_valid1).unsqueeze(0).contiguous()  # (1, H, W)

        disp = torch.stack([disp0, disp1], dim=0)  # (2, 1, H, W)
        disp_valid = torch.stack([disp_valid0, disp_valid1], dim=0)  # (2, 1, H, W)

        output = {
            'imgs': imgs,
            'flow': flow,
            'valid': valid,
            'disp': disp,
            'disp_valid': disp_valid,
            'seq_id': sample['seq_id'],
            'flow_frame': torch.tensor(sample['flow_frame'], dtype=torch.long),
            'frame_indices': torch.tensor(sample['frame_indices'], dtype=torch.long),
        }

        if imgs.shape[0] >= 2:
            gt_src = sample['flow_frame']
            indices = sample['frame_indices']
            if gt_src in indices and (gt_src + 1) in indices:
                src_pos = indices.index(gt_src)
                tgt_pos = indices.index(gt_src + 1)
                output['img_src'] = imgs[src_pos]
                output['img_tgt'] = imgs[tgt_pos]
                output['src_idx_in_seq'] = torch.tensor(src_pos, dtype=torch.long)
                output['tgt_idx_in_seq'] = torch.tensor(tgt_pos, dtype=torch.long)

        return output


def read_obj_map(path):
    obj = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if obj is None:
        raise FileNotFoundError(f"Object map not found: {path}")

    return obj.astype(np.int32)

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