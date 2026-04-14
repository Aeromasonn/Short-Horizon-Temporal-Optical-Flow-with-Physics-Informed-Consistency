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
    img_src=None,
    img_tgt=None,
    lambda_epe=1.0,
    lambda_photo=0.05,
    lambda_temp=0.1,
    lambda_sm_valid=0.02,
    lambda_sm_invalid=0.10,
    lambda_mag_invalid=0.02,
    use_edge_aware_smooth=True,
):
    """
    pred_flows: [B,Tm,2,H,W]
    gt_flow:    [B,2,H,W]
    valid:      [B,H,W] or [B,1,H,W]
    src_idx:    [B]
    img_src:    [B,3,H,W] or None
    img_tgt:    [B,3,H,W] or None

    Returns:
        total_loss, loss_dict
    """
    valid = _to_bhw_mask(valid)
    invalid = 1.0 - valid

    pred_main = select_gt_flow(pred_flows, src_idx)  # [B,2,H,W]

    # 1) supervised EPE on valid GT pixels
    loss_epe = epe_loss(pred_main, gt_flow, valid)

    # 2) optional photometric loss on valid pixels
    if img_src is not None and img_tgt is not None:
        loss_photo = photometric_loss(img_src, img_tgt, pred_main, valid)
    else:
        loss_photo = pred_main.new_tensor(0.0)

    # 3) temporal consistency across predicted flow sequence
    loss_temp = temporal_loss(pred_flows)

    # 4) spatial smoothness on valid pixels
    edge_img = img_src if (use_edge_aware_smooth and img_src is not None) else None
    loss_sm_valid = smoothness_loss_masked(pred_main, valid, edge_aware_img=edge_img)

    # 5) stronger smoothness on invalid pixels
    loss_sm_invalid = smoothness_loss_masked(pred_main, invalid, edge_aware_img=edge_img)

    # 6) small-flow prior on invalid pixels
    loss_mag_invalid = flow_magnitude_loss(pred_main, invalid)

    total = (
        lambda_epe * loss_epe +
        lambda_photo * loss_photo +
        lambda_temp * loss_temp +
        lambda_sm_valid * loss_sm_valid +
        lambda_sm_invalid * loss_sm_invalid +
        lambda_mag_invalid * loss_mag_invalid
    )

    loss_dict = {
        "loss": total.detach(),
        "epe": loss_epe.detach(),
        "photo": loss_photo.detach(),
        "temp": loss_temp.detach(),
        "smooth_valid": loss_sm_valid.detach(),
        "smooth_invalid": loss_sm_invalid.detach(),
        "mag_invalid": loss_mag_invalid.detach(),
    }

    return total, loss_dict