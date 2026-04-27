# train_v22_continue_full_log.py
#
# Continue v22 training for +100 epochs while keeping full epoch-level logging,
# matching the original v22 print/log style more closely.

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_v22_hard_fbmask import *


def load_checkpoint(modules, optimizer, ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key in ckpt:
            module.load_state_dict(ckpt[key])
        else:
            print(f"Warning: missing {key} in checkpoint")

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        print("Warning: optimizer_state_dict not found; optimizer starts fresh")

    start_epoch = ckpt.get("epoch", 0)
    print(f"Resume from epoch {start_epoch}")
    return start_epoch


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    modules = build_modules(cfg, device)

    optimizer = torch.optim.AdamW(
        [p for module in modules.values() for p in module.parameters()],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # Change this if you want to resume from best checkpoint instead.
    ckpt_path = "checkpoints/fullpipeline_v22_latest.pth"
    start_epoch = load_checkpoint(modules, optimizer, ckpt_path, device)

    dataset = TempFlowDataset_disp(
        root=cfg["data"]["data_root"],
        split=cfg["data"]["split"],
        image_folder=cfg["data"]["image_folder"],
        flow_type=cfg["data"]["flow_type"],
        disp_type=cfg["data"]["disp_type"],
        seq_len=cfg["data"]["seq_len"],
        center_frame_idx=cfg["data"]["center_frame_idx"],
        crop_size=tuple(cfg["data"]["crop_size"]),
        normalize=cfg["data"]["normalize"],
        stats_in=cfg["data"]["stats_file"],
        return_pair_only=cfg["data"]["return_pair_only"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    writer = SummaryWriter(log_dir="runs/v22_continue_full_log")
    trainer = V22Trainer(cfg, modules, optimizer, writer, device)

    extra_epochs = 100
    end_epoch = start_epoch + extra_epochs
    print(f"Continue training: {start_epoch} -> {end_epoch}")

    for epoch in range(start_epoch, end_epoch):
        running = {}
        n_batches = 0

        for batch in train_loader:
            stats = trainer.train_step(batch)

            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + float(value)

            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        # Full TensorBoard epoch-level logging
        for key, value in avg.items():
            writer.add_scalar(f"train_epoch_continue/{key}", value, epoch + 1)

        print(
            f"[Continue] Epoch {epoch + 1}/{end_epoch} | "
            f"loss={avg.get('loss', 0.0):.4f} | "
            f"flow={avg.get('loss_flow', 0.0):.4f} | "
            f"self={avg.get('loss_self', 0.0):.4f} | "
            f"temp={avg.get('loss_temp', 0.0):.4f} | "
            f"acc_photo={avg.get('loss_acc_photo', 0.0):.4f} | "
            f"acc_smooth={avg.get('loss_acc_smooth', 0.0):.4f} | "
            f"fb_loss={avg.get('loss_fb', 0.0):.4f} | "
            f"smooth={avg.get('loss_smooth', 0.0):.4f} | "
            f"latent={avg.get('latent_delta_mean_abs', 0.0):.4f} | "
            f"flow_res={avg.get('flow_residual_mean_abs', 0.0):.4f} | "
            f"fb_err={avg.get('fb_error_mean', 0.0):.4f} | "
            f"fb_conf={avg.get('fb_conf_mean', 0.0):.4f} | "
            f"fb_mask={avg.get('fb_mask_ratio', 0.0):.4f}"
        )

        # Save normal latest continue checkpoint
        latest_path = save_dir / "fullpipeline_v22_continue_latest.pth"
        save_checkpoint(latest_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        # Save milestone every 10 epochs to avoid too many files
        if (epoch + 1) % 10 == 0 or (epoch + 1) == end_epoch:
            milestone_path = save_dir / f"fullpipeline_v22_continue_epoch_{epoch + 1}.pth"
            save_checkpoint(milestone_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

    writer.close()
    print("Continue training finished.")


if __name__ == "__main__":
    main()
