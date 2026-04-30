import torch
import torch.nn.functional as F

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

def _to_bhw_mask(mask):
    """
    mask: [B,H,W] or [B,1,H,W] or None
    returns: [B,H,W] float or None
    """
    if mask is None:
        return None
    if mask.ndim == 4:
        mask = mask[:, 0]
    return mask.float()


def masked_mean(x, mask=None, eps=1e-6):
    """
    x:    [B,H,W] or [B,C,H,W]
    mask: [B,H,W] or [B,1,H,W] or None
    """
    if mask is None:
        return x.mean()

    mask = _to_bhw_mask(mask)

    if x.ndim == 4:
        mask = mask.unsqueeze(1)  # [B,1,H,W]

    return (x * mask).sum() / (mask.sum() * (x.shape[1] if x.ndim == 4 else 1) + eps)

def upsample_flow_to(flow, size_hw):
    """
    flow: [B, 2, h, w] in pixel units
    size_hw: (H, W)
    """
    B, C, h, w = flow.shape
    H, W = size_hw

    if (h, w) == (H, W):
        return flow

    scale_y = H / h
    scale_x = W / w

    flow_up = F.interpolate(
        flow,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    flow_up[:, 0] *= scale_x
    flow_up[:, 1] *= scale_y
    return flow_up

def smoothness_loss_masked(flow, mask=None, edge_aware_img=None, eps=1e-6):
    """
    flow: [B,2,H,W]
    mask: [B,H,W] or [B,1,H,W] or None

    If edge_aware_img is provided, reduce smoothing across strong image edges.
    edge_aware_img: [B,3,H,W]
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]   # [B,2,H,W-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]   # [B,2,H-1,W]

    if edge_aware_img is not None:
        img_dx = edge_aware_img[:, :, :, 1:] - edge_aware_img[:, :, :, :-1]
        img_dy = edge_aware_img[:, :, 1:, :] - edge_aware_img[:, :, :-1, :]

        # standard edge-aware weights
        wx = torch.exp(-img_dx.abs().mean(dim=1, keepdim=True))
        wy = torch.exp(-img_dy.abs().mean(dim=1, keepdim=True))

        dx = dx.abs() * wx
        dy = dy.abs() * wy
    else:
        dx = dx.abs()
        dy = dy.abs()

    if mask is None:
        return dx.mean() + dy.mean()

    mask = _to_bhw_mask(mask)
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]     # [B,H,W-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]     # [B,H-1,W]

    loss_x = masked_mean(dx, mask_x)
    loss_y = masked_mean(dy, mask_y)
    return loss_x + loss_y

def flow_magnitude_loss(flow, mask=None):
    """
    Penalize |flow| on selected pixels.
    flow: [B,2,H,W]
    mask: [B,H,W] or [B,1,H,W] or None
    """
    mag = torch.norm(flow, dim=1)  # [B,H,W]
    return masked_mean(mag, mask)

def multi_frame_flow_loss(
    pred_flows,
    gt_flow,
    valid,
    src_idx,

    # optional inputs (for different versions)
    flow_inits=None,
    flow_res=None,
    latent_del=None,
    pair_delta=None,

    img_src=None,
    img_tgt=None,

    # weights
    lambda_epe=1.0,
    lambda_photo=0.05,
    lambda_temp=0.1,
    lambda_sm_valid=0.02,
    lambda_sm_invalid=0.10,
    lambda_mag_invalid=0.02,
    lambda_flow_res=0.01,

    # optional regularizations (default OFF)
    lambda_init_improve=0.0,
    lambda_latent_delta=0.0,
    lambda_pair_delta=0.0,

    use_edge_aware_smooth=True,
):
    """
    Supports:
      - baseline (no UNO)
      - post-fusion UNO (latent_del)
      - pairwise UNO (pair_delta)

    All optional terms are gated by:
      (tensor is not None) AND (lambda > 0)
    """

    valid = _to_bhw_mask(valid)
    invalid = 1.0 - valid

    pred_main = select_gt_flow(pred_flows, src_idx)

    # optional selects
    init_main = select_gt_flow(flow_inits, src_idx) if flow_inits is not None else None
    flow_res_main = select_gt_flow(flow_res, src_idx) if flow_res is not None else None

    # -------------------------
    # core losses
    # -------------------------
    loss_epe = epe_loss(pred_main, gt_flow, valid)

    if img_src is not None and img_tgt is not None:
        loss_photo = photometric_loss(img_src, img_tgt, pred_main, valid)
    else:
        loss_photo = pred_main.new_tensor(0.0)

    loss_temp = temporal_loss(pred_flows)

    edge_img = img_src if (use_edge_aware_smooth and img_src is not None) else None
    loss_sm_valid = smoothness_loss_masked(pred_main, valid, edge_aware_img=edge_img)
    loss_sm_invalid = smoothness_loss_masked(pred_main, invalid, edge_aware_img=edge_img)

    loss_mag_invalid = flow_magnitude_loss(pred_main, invalid)

    # -------------------------
    # optional: flow residual
    # -------------------------
    if flow_res_main is not None and lambda_flow_res > 0:
        loss_flow_res = flow_magnitude_loss(flow_res_main, torch.ones_like(valid))
    else:
        loss_flow_res = pred_main.new_tensor(0.0)

    # -------------------------
    # optional: post-fusion UNO
    # -------------------------
    if latent_del is not None and lambda_latent_delta > 0:
        loss_latent_delta = latent_del.abs().mean()
    else:
        loss_latent_delta = pred_main.new_tensor(0.0)

    # -------------------------
    # optional: pairwise UNO
    # -------------------------
    if pair_delta is not None and lambda_pair_delta > 0:
        loss_pair_delta = pair_delta.abs().mean()
    else:
        loss_pair_delta = pred_main.new_tensor(0.0)

    # -------------------------
    # optional: init-improve
    # -------------------------
    if (
        flow_inits is not None
        and lambda_init_improve > 0.0
    ):
        init_main_up = upsample_flow_to(init_main, gt_flow.shape[-2:])

        epe_pred = torch.norm(pred_main - gt_flow, dim=1)
        epe_init = torch.norm(init_main_up - gt_flow, dim=1)

        penalty = F.relu(epe_pred - epe_init) * valid
        loss_init_improve = penalty.sum() / (valid.sum() + 1e-6)
    else:
        loss_init_improve = pred_main.new_tensor(0.0)

    # -------------------------
    # total
    # -------------------------
    total = (
        lambda_epe * loss_epe +
        lambda_photo * loss_photo +
        lambda_temp * loss_temp +
        lambda_sm_valid * loss_sm_valid +
        lambda_sm_invalid * loss_sm_invalid +
        lambda_mag_invalid * loss_mag_invalid +
        lambda_flow_res * loss_flow_res +
        lambda_latent_delta * loss_latent_delta +
        lambda_pair_delta * loss_pair_delta +
        lambda_init_improve * loss_init_improve
    )

    loss_dict = {
        "Total": total.detach(),
        "flow": loss_epe.detach(),
        "photo": loss_photo.detach(),
        "temp": loss_temp.detach(),
        "sm_valid": loss_sm_valid.detach(),
        "sm_inv": loss_sm_invalid.detach(),
        "mag_inv": loss_mag_invalid.detach(),
        "flow_res": loss_flow_res.detach(),
        "latent_delta": loss_latent_delta.detach(),
        "pair_delta": loss_pair_delta.detach(),
        "init_improve": loss_init_improve.detach(),
    }

    return total, loss_dict