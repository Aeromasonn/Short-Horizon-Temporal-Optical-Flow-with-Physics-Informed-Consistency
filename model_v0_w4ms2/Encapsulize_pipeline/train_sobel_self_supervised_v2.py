import os
import sys
import importlib
from pathlib import Path

import torch
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
DATA_ROOT = '../../Data'
STATS_FILE = 'stats.json'
CROP_SIZE = (352, 1216)
SEQ_LEN = 4
CENTER_FRAME_IDX = 10
BATCH_SIZE = 2

NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = Path('checkpoints_v2')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / 'fullpipeline_v2_latest.pth'
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / 'fullpipeline_v2_best.pth'


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
# Helpers / Training
# =========================
def epe_loss(pred, gt, valid=None):
    """
    pred:  [B, 2, H, W]
    gt:    [B, 2, H, W]
    valid: [B, H, W] or [B, 1, H, W] or None
    """
    epe = torch.norm(pred - gt, dim=1)  # [B, H, W]

    if valid is not None:
        if valid.ndim == 4:
            valid = valid[:, 0]
        valid = valid.float()
        return (epe * valid).sum() / (valid.sum() + 1e-6)

    return epe.mean()




def warp_image(img, flow):
    """
    img:  [B, 3, H, W]
    flow: [B, 2, H, W] in pixel units
    """
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
    """
    img_src: [B, 3, H, W]
    img_tgt: [B, 3, H, W]
    flow:    [B, 2, H, W]
    valid:   [B, H, W] or [B, 1, H, W] or None
    """
    warped_src = warp_image(img_src, flow)
    photo = (warped_src - img_tgt).abs().mean(dim=1)

    if valid is not None:
        if valid.ndim == 4:
            valid = valid[:, 0]
        valid = valid.float()
        return (photo * valid).sum() / (valid.sum() + 1e-6)

    return photo.mean()

def temporal_loss(flows):
    """
    flows: [B, Tm, 2, H, W]
    """
    if flows.shape[1] < 2:
        return flows.new_tensor(0.0)
    return (flows[:, 1:] - flows[:, :-1]).abs().mean()


def smoothness_loss(flow):
    """
    flow: [B, 2, H, W]
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def select_gt_flow(pred_flows, src_idx):
    """
    pred_flows: [B, Tm, 2, H, W]
    src_idx:    [B]
    Returns:
        [B, 2, H, W]
    """
    B = pred_flows.shape[0]
    out = []
    for b in range(B):
        t = int(src_idx[b].item())
        out.append(pred_flows[b, t])
    return torch.stack(out, dim=0)

def forward_pipeline(imgs, pair_encoder, visual_branch, motion_branch, fusion, decoder):
    """
    imgs: [B, T, 3, H, W]

    Returns dict with:
      flows:      [B, Tm, 2, H, W]
      flow_inits: [B, Tm, 2, h, w] or None
      pair_feats: [B, Tm, Cp, h, w]
      corrs:      [B, Tm, K, h, w]
      fused_seq:  [B, Tm, Cf, h, w]
    """
    pair_out = pair_encoder(imgs)
    pair_feats = pair_out["pair_feats"]      # [B, Tm, 128, h, w]
    flow_inits = pair_out["flow_inits"]      # [B, Tm, 2, h, w] or None
    corrs = pair_out["corrs"]

    visual_feats = visual_branch(imgs)       # [B, T, 64, h, w]
    motion_feats = motion_branch(pair_feats) # [B, Tm, 64, h, w]
    
    fused_seq = fusion(visual_feats, motion_feats)  # [B, Tm, 128, h, w]
    flows = decoder(fused_seq)                      # [B, Tm, 2, H, W]

    return {
        "flows": flows,
        "flow_inits": flow_inits,
        "pair_feats": pair_feats,
        "corrs": corrs,
        "fused_seq": fused_seq,
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

        imgs = batch["imgs"].to(self.device)                  # [B, T, 3, H, W]
        gt_flow = batch["flow"].to(self.device)               # [B, 2, H, W]
        valid = batch["valid"].to(self.device)                # [B, H, W] or [B,1,H,W]
        src_idx = batch["src_idx_in_seq"].to(self.device)     # [B]
        img_src = batch["img_src"].to(self.device)            # [B, 3, H, W]
        img_tgt = batch["img_tgt"].to(self.device)            # [B, 3, H, W]

        out = forward_pipeline(
            imgs,
            self.pair_encoder,
            self.visual_branch,
            self.motion_branch,
            self.fusion,
            self.decoder,
        )

        pred_flows = out["flows"]                             # [B, Tm, 2, H, W]
        pred = select_gt_flow(pred_flows, src_idx)           # [B, 2, H, W]

        loss_flow = epe_loss(pred, gt_flow, valid)
        loss_self = photometric_loss(img_src, img_tgt, pred, valid)
        loss_temp = temporal_loss(pred_flows)
        loss_smooth = smoothness_loss(pred)

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
            "loss": float(loss.item()),
            "loss_flow": float(loss_flow.item()),
            "loss_self": float(loss_self.item()),
            "loss_temp": float(loss_temp.item()),
            "loss_smooth": float(loss_smooth.item()),
        }

    @torch.no_grad()
    def val_step(self, batch):
        self.eval_mode()

        imgs = batch["imgs"].to(self.device)
        gt_flow = batch["flow"].to(self.device)
        valid = batch["valid"].to(self.device)
        src_idx = batch["src_idx_in_seq"].to(self.device)
        img_src = batch["img_src"].to(self.device)
        img_tgt = batch["img_tgt"].to(self.device)

        out = forward_pipeline(
            imgs,
            self.pair_encoder,
            self.visual_branch,
            self.motion_branch,
            self.fusion,
            self.decoder,
        )

        pred_flows = out["flows"]
        pred = select_gt_flow(pred_flows, src_idx)

        loss_flow = epe_loss(pred, gt_flow, valid)
        loss_self = photometric_loss(img_src, img_tgt, pred, valid)
        loss_temp = temporal_loss(pred_flows)
        loss_smooth = smoothness_loss(pred)

        loss = (
            loss_flow
            + self.lambda_self * loss_self
            + self.lambda_temp * loss_temp
            + self.lambda_smooth * loss_smooth
        )

        return {
            "loss": float(loss.item()),
            "loss_flow": float(loss_flow.item()),
            "loss_self": float(loss_self.item()),
            "loss_temp": float(loss_temp.item()),
            "loss_smooth": float(loss_smooth.item()),
        }


def save_checkpoint(
    checkpoint_path,
    epoch,
    stats,
    pair_encoder,
    visual_branch,
    motion_branch,
    fusion,
    decoder,
    optimizer,
):
    checkpoint = {
        'epoch': epoch,
        'stats': stats,
        'pair_encoder_state_dict': pair_encoder.state_dict(),
        'visual_branch_state_dict': visual_branch.state_dict(),
        'motion_branch_state_dict': motion_branch.state_dict(),
        'fusion_state_dict': fusion.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved checkpoint to: {checkpoint_path}')


# =========================
# Device / Optimizer / Trainer
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

pair_encoder = pair_encoder.to(device)
visual_branch = visual_branch.to(device)
motion_branch = motion_branch.to(device)
fusion = fusion.to(device)
decoder = decoder.to(device)

optimizer = torch.optim.AdamW(
    list(pair_encoder.parameters()) +
    list(visual_branch.parameters()) +
    list(motion_branch.parameters()) +
    list(fusion.parameters()) +
    list(decoder.parameters()),
    lr=LEARNING_RATE,
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
    lambda_self=0.2,
)


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

    save_checkpoint(
        LATEST_CHECKPOINT_PATH,
        epoch=epoch + 1,
        stats=avg,
        pair_encoder=pair_encoder,
        visual_branch=visual_branch,
        motion_branch=motion_branch,
        fusion=fusion,
        decoder=decoder,
        optimizer=optimizer,
    )

    if avg['loss'] < best_loss:
        best_loss = avg['loss']
        save_checkpoint(
            BEST_CHECKPOINT_PATH,
            epoch=epoch + 1,
            stats=avg,
            pair_encoder=pair_encoder,
            visual_branch=visual_branch,
            motion_branch=motion_branch,
            fusion=fusion,
            decoder=decoder,
            optimizer=optimizer,
        )

print('Training finished.')
