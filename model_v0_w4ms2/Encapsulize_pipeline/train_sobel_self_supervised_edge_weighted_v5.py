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

SAVE_DIR = Path('checkpoints')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_NAME = 'fullpipeline_v5_latest.pth'
BEST_CHECKPOINT_NAME = 'fullpipeline_v5_best.pth'


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

    pred:  [B, 2, H, W]
    gt:    [B, 2, H, W]
    image: [B, 3, H, W]
    valid: [B, 1, H, W] or [B, H, W] or None
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



def photometric_loss(img_src, img_tgt, flow, valid=None):
    warped_src = warp_image(img_src, flow)
    photo = (warped_src - img_tgt).abs().mean(dim=1)

    if valid is not None:
        if valid.ndim == 4:
            valid = valid[:, 0]
        valid = valid.float()
        return (photo * valid).sum() / (valid.sum() + 1e-6)

    return photo.mean()



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
        edge_weight_scale=1.0,
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
        self.edge_weight_scale = edge_weight_scale

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

    def train_step(self, batch):
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

        loss_flow = weighted_epe_loss(pred, gt_flow, img_src, valid, edge_weight_scale=self.edge_weight_scale)
        loss_self = photometric_loss(img_src, img_tgt, pred, valid)
        loss_temp = temporal_loss(pred_flows)
        loss_smooth = edge_aware_smoothness_loss(pred, img_src)

        loss = (
            loss_flow
            + self.lambda_self * loss_self
            + self.lambda_temp * loss_temp
            + self.lambda_smooth * loss_smooth
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
            'edge_weight_scale': LAMBDA_EDGE_WEIGHT,
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
    edge_weight_scale=LAMBDA_EDGE_WEIGHT,
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
    }
    n_batches = 0

    for batch in train_loader:
        stats = trainer.train_step(batch)
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
        f"smooth={avg['loss_smooth']:.4f}"
    )

    latest_path = SAVE_DIR / CHECKPOINT_NAME
    save_checkpoint(latest_path, epoch + 1, trainer, optimizer, stats=avg)

    if avg['loss'] < best_loss:
        best_loss = avg['loss']
        best_path = SAVE_DIR / BEST_CHECKPOINT_NAME
        save_checkpoint(best_path, epoch + 1, trainer, optimizer, stats=avg)

print('Training finished.')
