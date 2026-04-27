# train_v22_continue.py

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

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    print(f"Resume from epoch {start_epoch}")
    return start_epoch


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    modules = build_modules(cfg, device)

    optimizer = torch.optim.AdamW(
        [p for module in modules.values() for p in module.parameters()],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

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
        pin_memory=True,
    )

    writer = SummaryWriter(log_dir="runs/v22_continue")

    trainer = V22Trainer(cfg, modules, optimizer, writer, device)

    extra_epochs = 100
    end_epoch = start_epoch + extra_epochs

    print(f"Continue training: {start_epoch} → {end_epoch}")

    for epoch in range(start_epoch, end_epoch):
        running_loss = 0
        n_batches = 0

        for batch in train_loader:
            stats = trainer.train_step(batch)
            running_loss += stats["loss"]
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)

        print(f"[Continue] Epoch {epoch+1}/{end_epoch} | loss={avg_loss:.4f}")

        save_path = Path(f"checkpoints/v22_continue_epoch_{epoch+1}.pth")
        save_checkpoint(
            save_path,
            epoch + 1,
            modules,
            optimizer,
            stats={"loss": avg_loss},
            config=cfg,
        )

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
