
import os
import sys
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =========================
# Path setup
# =========================
project_root = os.path.abspath("../..")
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
DATA_ROOT = '../../../Data'
STATS_FILE = 'stats.json'
CROP_SIZE = (352, 1216)
SEQ_LEN = 4
CENTER_FRAME_IDX = 10
BATCH_SIZE = 2
NUM_EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-4

LAMBDA_TEMP = 0.1
LAMBDA_SMOOTH = 0.01
LAMBDA_SELF = 0.2
LAMBDA_EDGE_WEIGHT = 1.0

# v10 update:
# 1) reduce sky weight
# 2) use high-confidence eroded sky mask
# 3) use smooth-sky prior instead of zero-flow penalty
# 4) ramp sky loss gradually in early training
LAMBDA_SKY = 0.02
SKY_GRAD_THRESH = 0.04
SKY_TOP_RATIO = 0.38
SKY_EROSION_KERNEL = 11
SKY_WARMUP_EPOCHS = 10

PHOTO_GRAD_THRESH = 0.03
CENSUS_PATCH = 7

SAVE_DIR = Path('checkpoints')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_NAME = 'fullpipeline_v10_latest.pth'
BEST_CHECKPOINT_NAME = 'fullpipeline_v10_best.pth'
AUTO_MASK_EPS = 1e-3


# =========================
# Data prep
# =========================
dataset = TempFlowDataset_disp(
    root=DATA_ROOT,
    split='training',
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

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

batch = next(iter(train_loader))
print('batch imgs:', batch['imgs'].shape)


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
# Losses / helpers
# =========================
def weighted_epe_loss(pred, gt, image, valid=None, edge_weight_scale=1.0):
    """
    Edge-weighted supervised EPE.
    Edge pixels receive higher weight, but supervision remains grounded in GT flow.
    """
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
    B, C, H, W = img.shape

    yy, xx = torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing='ij'
    )
    base_grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).expand(B, -1, -1, -1)
    sample_grid = base_grid + flow

    sample_grid_x = 2.0 * sample_grid[:, 0] / max(W - 1, 1) - 1.0
    sample_grid_y = 2.0 * sample_grid[:, 1] / max(H - 1, 1) - 1.0
    sample_grid = torch.stack((sample_grid_x, sample_grid_y), dim=-1)

    return F.grid_sample(img, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)


def census_transform(img, patch_size=7):
    """
    Rank-based local descriptor used in classical self-supervised optical flow.
    It is more robust than raw L1 photometric loss in weak-texture regions.
    """
    if img.shape[1] > 1:
        img = img.mean(dim=1, keepdim=True)

    B, _, H, W = img.shape
    p = patch_size // 2
    padded = F.pad(img, [p, p, p, p], mode='reflect')

    patches = []
    for dy in range(patch_size):
        for dx in range(patch_size):
            patch = padded[:, :, dy:dy + H, dx:dx + W]
            patches.append((patch > img).float())

    return torch.cat(patches, dim=1)


def photometric_loss_v10(
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
    """
    v10 keeps the v9 robust self-supervised formulation:
    1. Census distance for more robust matching
    2. Auto-mask to ignore pixels where identity is already as good as warping
    3. Gradient gate to suppress unreliable supervision in textureless regions
    """
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


def sobel_grad_map(x):
    """
    x: [B, C, H, W]
    return: gradient magnitude map [B, 1, H, W]
    """
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[1,  2,  1],
         [0,  0,  0],
         [-1, -2, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def normalize_map(m):
    """Normalize each sample to [0, 1]."""
    b = m.shape[0]
    m_flat = m.view(b, -1)
    m_min = m_flat.min(dim=1)[0].view(b, 1, 1, 1)
    m_max = m_flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (m - m_min) / (m_max - m_min + 1e-6)


def binary_erosion(mask, kernel_size=11):
    """
    Morphological erosion for a binary-like mask in [0, 1].
    Keeps only confident interior pixels and removes noisy boundaries.
    """
    inv = 1.0 - mask
    eroded_inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return 1.0 - eroded_inv


def build_high_confidence_sky_mask(image, grad_thresh=0.04, top_ratio=0.38, erosion_kernel=11):
    """
    v10 sky mask:
    - only low-gradient pixels
    - only in the upper part of the image
    - eroded to remove boundary leakage into trees/buildings
    """
    edge_map = normalize_map(sobel_grad_map(image)).detach()
    low_grad = (edge_map < grad_thresh).float()

    B, _, H, W = image.shape
    y = torch.linspace(0, 1, H, device=image.device).view(1, 1, H, 1)
    top_mask = (y < top_ratio).float().expand(B, 1, H, W)

    sky_mask = low_grad * top_mask
    sky_mask = binary_erosion(sky_mask, kernel_size=erosion_kernel)
    return sky_mask


def masked_sky_smoothness_loss(flow, image, grad_thresh=0.04, top_ratio=0.38, erosion_kernel=11):
    """
    v10 sky prior:
    Instead of forcing sky flow magnitude to zero, only encourage smooth,
    spatially consistent motion inside a high-confidence sky region.

    This is safer under camera motion, because sky can still move globally.
    """
    sky_mask = build_high_confidence_sky_mask(
        image,
        grad_thresh=grad_thresh,
        top_ratio=top_ratio,
        erosion_kernel=erosion_kernel,
    ).detach()

    flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

    mask_x = sky_mask[:, :, :, 1:] * sky_mask[:, :, :, :-1]
    mask_y = sky_mask[:, :, 1:, :] * sky_mask[:, :, :-1, :]

    loss_x = (torch.abs(flow_dx) * mask_x).sum() / (mask_x.sum() * flow.shape[1] + 1e-6)
    loss_y = (torch.abs(flow_dy) * mask_y).sum() / (mask_y.sum() * flow.shape[1] + 1e-6)
    return loss_x + loss_y


def select_gt_flow(pred_flows, src_idx):
    B = pred_flows.shape[0]
    out = []
    for b in range(B):
        t = int(src_idx[b].item())
        out.append(pred_flows[b, t])
    return torch.stack(out, dim=0)


def forward_pipeline(imgs, pair_encoder, visual_branch, motion_branch, fusion, decoder):
    pair_out = pair_encoder(imgs)
    pair_feats = pair_out['pair_feats']
    flow_inits = pair_out['flow_inits']
    corrs = pair_out['corrs']

    visual_feats = visual_branch(imgs)
    motion_feats = motion_branch(pair_feats)

    fused_seq = fusion(visual_feats, motion_feats)
    flows = decoder(fused_seq)

    return {
        'flows': flows,
        'flow_inits': flow_inits,
        'pair_feats': pair_feats,
        'corrs': corrs,
        'fused_seq': fused_seq,
    }


class Trainer:
    def __init__(
        self,
        pair_encoder,
        visual_branch,
        motion_branch,
        fusion,
        decoder,
        optimizer,
        device,
        lambda_temp=0.1,
        lambda_smooth=0.01,
        lambda_self=0.2,
        lambda_sky=0.02,
        edge_weight_scale=1.0,
        sky_grad_thresh=0.04,
        sky_top_ratio=0.38,
        sky_erosion_kernel=11,
        sky_warmup_epochs=10,
        photo_grad_thresh=0.03,
    ):
        self.pair_encoder = pair_encoder
        self.visual_branch = visual_branch
        self.motion_branch = motion_branch
        self.fusion = fusion
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device

        self.lambda_temp = lambda_temp
        self.lambda_smooth = lambda_smooth
        self.lambda_self = lambda_self
        self.lambda_sky = lambda_sky
        self.edge_weight_scale = edge_weight_scale
        self.sky_grad_thresh = sky_grad_thresh
        self.sky_top_ratio = sky_top_ratio
        self.sky_erosion_kernel = sky_erosion_kernel
        self.sky_warmup_epochs = sky_warmup_epochs
        self.photo_grad_thresh = photo_grad_thresh

    def train_mode(self):
        self.pair_encoder.train()
        self.visual_branch.train()
        self.motion_branch.train()
        self.fusion.train()
        self.decoder.train()

    def eval_mode(self):
        self.pair_encoder.eval()
        self.visual_branch.eval()
        self.motion_branch.eval()
        self.fusion.eval()
        self.decoder.eval()

    def current_sky_lambda(self, epoch_idx):
        warm = max(self.sky_warmup_epochs, 1)
        scale = min(1.0, float(epoch_idx + 1) / float(warm))
        return self.lambda_sky * scale

    def train_step(self, batch, epoch_idx=0):
        self.train_mode()

        imgs = batch['imgs'].to(self.device)
        gt_flow = batch['flow'].to(self.device)
        valid = batch['valid'].to(self.device)
        src_idx = batch['src_idx_in_seq'].to(self.device)
        img_src = batch['img_src'].to(self.device)
        img_tgt = batch['img_tgt'].to(self.device)

        out = forward_pipeline(
            imgs,
            self.pair_encoder,
            self.visual_branch,
            self.motion_branch,
            self.fusion,
            self.decoder,
        )

        pred_flows = out['flows']
        pred = select_gt_flow(pred_flows, src_idx)

        loss_flow = weighted_epe_loss(
            pred,
            gt_flow,
            img_src,
            valid,
            edge_weight_scale=self.edge_weight_scale,
        )
        loss_self = photometric_loss_v10(
            img_src,
            img_tgt,
            pred,
            valid=valid,
            auto_mask=True,
            eps=AUTO_MASK_EPS,
            grad_gate=True,
            grad_thresh=self.photo_grad_thresh,
            use_census=True,
            census_patch=CENSUS_PATCH,
        )
        loss_temp = temporal_loss(pred_flows)
        loss_smooth = edge_aware_smoothness_loss(pred, img_src)
        loss_sky = masked_sky_smoothness_loss(
            pred,
            img_src,
            grad_thresh=self.sky_grad_thresh,
            top_ratio=self.sky_top_ratio,
            erosion_kernel=self.sky_erosion_kernel,
        )

        current_lambda_sky = self.current_sky_lambda(epoch_idx)

        loss = (
            loss_flow
            + self.lambda_self * loss_self
            + self.lambda_temp * loss_temp
            + self.lambda_smooth * loss_smooth
            + current_lambda_sky * loss_sky
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': float(loss.item()),
            'loss_flow': float(loss_flow.item()),
            'loss_self': float(loss_self.item()),
            'loss_temp': float(loss_temp.item()),
            'loss_smooth': float(loss_smooth.item()),
            'loss_sky': float(loss_sky.item()),
            'lambda_sky_used': float(current_lambda_sky),
        }


def save_checkpoint(save_path, epoch, trainer, optimizer, stats=None):
    checkpoint = {
        'epoch': epoch,
        'pair_encoder_state_dict': trainer.pair_encoder.state_dict(),
        'visual_branch_state_dict': trainer.visual_branch.state_dict(),
        'motion_branch_state_dict': trainer.motion_branch.state_dict(),
        'fusion_state_dict': trainer.fusion.state_dict(),
        'decoder_state_dict': trainer.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'config': {
            'crop_size': CROP_SIZE,
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'lambda_temp': LAMBDA_TEMP,
            'lambda_smooth': LAMBDA_SMOOTH,
            'lambda_self': LAMBDA_SELF,
            'lambda_sky': LAMBDA_SKY,
            'edge_weight_scale': LAMBDA_EDGE_WEIGHT,
            'sky_grad_thresh': SKY_GRAD_THRESH,
            'sky_top_ratio': SKY_TOP_RATIO,
            'sky_erosion_kernel': SKY_EROSION_KERNEL,
            'sky_warmup_epochs': SKY_WARMUP_EPOCHS,
            'photo_grad_thresh': PHOTO_GRAD_THRESH,
            'census_patch': CENSUS_PATCH,
            'auto_mask_eps': AUTO_MASK_EPS,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
        },
    }
    torch.save(checkpoint, save_path)
    print(f'Saved checkpoint to: {save_path}')


# =========================
# Train setup
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

pair_encoder = pair_encoder.to(device)
visual_branch = visual_branch.to(device)
motion_branch = motion_branch.to(device)
fusion = fusion.to(device)
decoder = decoder.to(device)

optimizer = torch.optim.AdamW(
    list(pair_encoder.parameters())
    + list(visual_branch.parameters())
    + list(motion_branch.parameters())
    + list(fusion.parameters())
    + list(decoder.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

trainer = Trainer(
    pair_encoder=pair_encoder,
    visual_branch=visual_branch,
    motion_branch=motion_branch,
    fusion=fusion,
    decoder=decoder,
    optimizer=optimizer,
    device=device,
    lambda_temp=LAMBDA_TEMP,
    lambda_smooth=LAMBDA_SMOOTH,
    lambda_self=LAMBDA_SELF,
    lambda_sky=LAMBDA_SKY,
    edge_weight_scale=LAMBDA_EDGE_WEIGHT,
    sky_grad_thresh=SKY_GRAD_THRESH,
    sky_top_ratio=SKY_TOP_RATIO,
    sky_erosion_kernel=SKY_EROSION_KERNEL,
    sky_warmup_epochs=SKY_WARMUP_EPOCHS,
    photo_grad_thresh=PHOTO_GRAD_THRESH,
)


# =========================
# Forward sanity check
# =========================
batch = next(iter(train_loader))
imgs = batch['imgs'].to(device)

pair_encoder.eval()
visual_branch.eval()
motion_branch.eval()
fusion.eval()
decoder.eval()

with torch.no_grad():
    out = forward_pipeline(
        imgs,
        pair_encoder,
        visual_branch,
        motion_branch,
        fusion,
        decoder,
    )

print('imgs:', imgs.shape)
print('pred flows:', out['flows'].shape)
print('pair_feats:', out['pair_feats'].shape)
print('fused_seq:', out['fused_seq'].shape)
if out['flow_inits'] is not None:
    print('flow_inits:', out['flow_inits'].shape)
print('gt flow:', batch['flow'].shape)
print('valid:', batch['valid'].shape)
print('src_idx_in_seq:', batch['src_idx_in_seq'])


# =========================
# Training loop
# =========================
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    running = {
        'loss': 0.0,
        'loss_flow': 0.0,
        'loss_self': 0.0,
        'loss_temp': 0.0,
        'loss_smooth': 0.0,
        'loss_sky': 0.0,
        'lambda_sky_used': 0.0,
    }
    n_batches = 0

    for batch in train_loader:
        stats = trainer.train_step(batch, epoch_idx=epoch)
        for k in running:
            running[k] += stats[k]
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in running.items()}

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"loss={avg['loss']:.4f} | "
        f"flow={avg['loss_flow']:.4f} | "
        f"self={avg['loss_self']:.4f} | "
        f"temp={avg['loss_temp']:.4f} | "
        f"smooth={avg['loss_smooth']:.4f} | "
        f"sky={avg['loss_sky']:.4f} | "
        f"sky_lambda={avg['lambda_sky_used']:.4f}"
    )

    latest_path = SAVE_DIR / CHECKPOINT_NAME
    save_checkpoint(latest_path, epoch + 1, trainer, optimizer, stats=avg)

    if avg['loss'] < best_loss:
        best_loss = avg['loss']
        best_path = SAVE_DIR / BEST_CHECKPOINT_NAME
        save_checkpoint(best_path, epoch + 1, trainer, optimizer, stats=avg)

print('Training finished.')
