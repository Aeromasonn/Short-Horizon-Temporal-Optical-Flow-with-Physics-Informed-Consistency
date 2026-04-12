import os
import sys
import importlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.utils.data import DataLoader


# =========================
# Path setup
# =========================
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =========================
# Imports from project
# =========================
from model_v0_w4ms2.DataLoader import TempFlowDataset_disp

import model_v0_w4ms2.Encoders_sobel_v1 as Encoders
importlib.reload(Encoders)
from model_v0_w4ms2.Encoders_sobel_v1 import (
    SequencePairEncoder,
    VisualBranchCNN,
    MotionBranchCNN,
    SpatialTemporalFusion_timeAware,
)

import model_v0_w4ms2.Decoders as Decoders
importlib.reload(Decoders)
from model_v0_w4ms2.Decoders import FlowDecoder


# =========================
# Config
# =========================
DATA_ROOT = '../../Data'
STATS_FILE = 'stats.json'
CROP_SIZE = (352, 1216)
SEQ_LEN = 4
CENTER_FRAME_IDX = 10
BATCH_SIZE = 2

VIS_SPLIT = 'training'
MAX_SAMPLES = 10
START_INDEX = 0

CHECKPOINT_PATH = Path('checkpoints_v2/fullpipeline_v2_best.pth')
OUTPUT_DIR = Path('visualization_outputs_v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Data prep
# =========================
dataset = TempFlowDataset_disp(
    root=DATA_ROOT,
    split=VIS_SPLIT,
    image_folder='image_2',
    flow_type='flow_occ',
    disp_type='disp_occ',
    seq_len=SEQ_LEN,
    center_frame_idx=CENTER_FRAME_IDX,
    crop_size=CROP_SIZE,
    normalize=True,
    stats_in=STATS_FILE,
    return_pair_only=False,
)

vis_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# Encoders / Decoder
# =========================
pair_encoder = SequencePairEncoder(
    feat_ch=64,
    corr_radius=4,
    embed_ch=128,
    predict_flow=True,
)

visual_branch = VisualBranchCNN(
    in_ch=3,
    base_ch=32,
    out_ch=64,
)

motion_branch = MotionBranchCNN(
    in_ch=128,
    hidden_ch=128,
    out_ch=64,
)

fusion = SpatialTemporalFusion_timeAware(
    visual_ch=64,
    motion_ch=64,
    hidden_ch=128,
    out_ch=128,
)

decoder = FlowDecoder(
    in_ch=128,
    hidden_ch=64,
    upsample=8,
    use_prev_flow=True,
)


# =========================
# Helpers
# =========================
def forward_pipeline(imgs, pair_encoder, visual_branch, motion_branch, fusion, decoder):
    pair_out = pair_encoder(imgs)
    pair_feats = pair_out["pair_feats"]
    flow_inits = pair_out["flow_inits"]
    corrs = pair_out["corrs"]

    visual_feats = visual_branch(imgs)
    motion_feats = motion_branch(pair_feats)

    fused_seq = fusion(visual_feats, motion_feats)
    flows = decoder(fused_seq)

    return {
        "flows": flows,
        "flow_inits": flow_inits,
        "pair_feats": pair_feats,
        "corrs": corrs,
        "fused_seq": fused_seq,
    }


import numpy as np
import matplotlib.pyplot as plt
import torch


def flow_to_rgb(flow):
    """
    flow: torch.Tensor or np.ndarray of shape [2, H, W]
    returns: np.ndarray [H, W, 3] in [0,1]
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]

    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)  # [-pi, pi]

    hue = (ang + np.pi) / (2 * np.pi)   # [0,1]
    sat = np.ones_like(hue)
    val = mag / (np.max(mag) + 1e-6)
    val = np.clip(val, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1)

    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb


def tensor_img_to_np(img):
    """
    img: [3, H, W], normalized or unnormalized
    returns: [H, W, 3]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()

    img = img.permute(1, 2, 0).numpy()

    # simple min-max for display
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return img


def compute_epe_map(pred, gt, valid=None):
    """
    pred, gt: [2, H, W]
    valid: [H, W] or None
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu()

    epe = torch.norm(pred - gt, dim=0)

    if valid is not None:
        if isinstance(valid, np.ndarray):
            valid = torch.from_numpy(valid)
        valid = valid.detach().cpu().float()
        epe = epe * valid

    return epe.numpy()

def select_gt_flow_single(pred_flows, src_idx, b=0):
    """
    pred_flows: [B, Tm, 2, H, W]
    src_idx: [B]
    b: sample index in batch
    """
    t = int(src_idx[b].item())
    return pred_flows[b, t]


def load_checkpoint(checkpoint_path, pair_encoder, visual_branch, motion_branch, fusion, decoder, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pair_encoder.load_state_dict(checkpoint['pair_encoder_state_dict'])
    visual_branch.load_state_dict(checkpoint['visual_branch_state_dict'])
    motion_branch.load_state_dict(checkpoint['motion_branch_state_dict'])
    fusion.load_state_dict(checkpoint['fusion_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print(f'Loaded checkpoint from: {checkpoint_path}')
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'stats' in checkpoint and checkpoint['stats'] is not None:
        print(f"Checkpoint stats: {checkpoint['stats']}")


def visualize_batch_result(batch, pred_flows, sample_idx=0, save_path=None):
    img_src = batch["img_src"][sample_idx]
    img_tgt = batch["img_tgt"][sample_idx]
    gt_flow = batch["flow"][sample_idx]
    valid = batch["valid"][sample_idx]
    src_idx = batch["src_idx_in_seq"]

    pred = select_gt_flow_single(pred_flows, src_idx, b=sample_idx)

    img_src_np = tensor_img_to_np(img_src)
    img_tgt_np = tensor_img_to_np(img_tgt)
    pred_rgb = flow_to_rgb(pred)
    gt_rgb = flow_to_rgb(gt_flow)
    epe_map = compute_epe_map(pred, gt_flow, valid)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(img_src_np)
    axes[0].set_title("Source Image")
    axes[0].axis("off")

    axes[1].imshow(img_tgt_np)
    axes[1].set_title("Target Image")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Predicted Flow")
    axes[2].axis("off")

    axes[3].imshow(gt_rgb)
    axes[3].set_title("GT Flow")
    axes[3].axis("off")

    im = axes[4].imshow(epe_map, cmap="inferno")
    axes[4].set_title("EPE Map")
    axes[4].axis("off")
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Saved visualization to: {save_path}')

    plt.close(fig)


# =========================
# Load model
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

pair_encoder = pair_encoder.to(device)
visual_branch = visual_branch.to(device)
motion_branch = motion_branch.to(device)
fusion = fusion.to(device)
decoder = decoder.to(device)

load_checkpoint(
    CHECKPOINT_PATH,
    pair_encoder,
    visual_branch,
    motion_branch,
    fusion,
    decoder,
    device,
)

pair_encoder.eval()
visual_branch.eval()
motion_branch.eval()
fusion.eval()
decoder.eval()


# =========================
# Visualization loop
# =========================
saved = 0
seen = 0

for batch_idx, batch in enumerate(vis_loader):
    imgs = batch["imgs"].to(device)

    with torch.no_grad():
        out = forward_pipeline(
            imgs,
            pair_encoder,
            visual_branch,
            motion_branch,
            fusion,
            decoder,
        )

    pred_flows = out["flows"].cpu()
    batch_size = pred_flows.shape[0]

    for sample_idx in range(batch_size):
        if seen < START_INDEX:
            seen += 1
            continue

        if saved >= MAX_SAMPLES:
            break

        save_path = OUTPUT_DIR / f'vis_batch{batch_idx:03d}_sample{sample_idx:02d}.png'
        visualize_batch_result(batch, pred_flows, sample_idx=sample_idx, save_path=save_path)

        saved += 1
        seen += 1

    if saved >= MAX_SAMPLES:
        break

print('Visualization finished.')
