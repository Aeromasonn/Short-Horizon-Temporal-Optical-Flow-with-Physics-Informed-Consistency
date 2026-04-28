import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from My.Encoder_sober import *


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

# train_flow_detector.py

import torch
from tqdm import tqdm


# train_flow_detector.py

import torch
from tqdm import tqdm


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

            # --------------------------------------------------
            # 2. Build detector input
            #    This must be outside no_grad.
            # --------------------------------------------------
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