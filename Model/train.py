import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .DataLoader import TempFlowDataset_disp
from .trainer import Trainer, build_modules, save_checkpoint, set_seed

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent  

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified training launcher: later / early / standalone operator architectures"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "config.json"),
        help="Path to the unified experiment config JSON.",
    )
    parser.add_argument(
        "--architecture_type",
        type=str,
        choices=["later", "early", "standalone"],
        default=None,
        help="Override cfg['model']['architecture_type'].",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.architecture_type is not None:
        cfg["model"]["architecture_type"] = args.architecture_type
        cfg["experiment"]["experiment_name"] = f"{args.architecture_type}"
        cfg["experiment"]["checkpoint_name"] = f"fullpipeline_{args.architecture_type}_latest.pth"
        cfg["experiment"]["best_checkpoint_name"] = f"fullpipeline_{args.architecture_type}_best.pth"
        cfg["experiment"]["config_dump_name"] = f"config_{args.architecture_type}.json"
    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    set_seed(cfg["train"]["seed"])

    save_dir = (THIS_DIR / cfg["experiment"]["save_dir"]).resolve()
    tb_dir = (THIS_DIR / cfg["experiment"]["tensorboard_dir"] / cfg["experiment"]["experiment_name"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    config_dump_path = save_dir / cfg["experiment"]["config_dump_name"]
    with open(config_dump_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    writer = SummaryWriter(log_dir=str(tb_dir))

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
        shuffle=cfg["train"]["shuffle"],
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("architecture_type:", cfg["model"].get("architecture_type", "later"))

    modules = build_modules(cfg, device)
    optimizer = torch.optim.AdamW(
        [p for module in modules.values() for p in module.parameters()],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    trainer = Trainer(cfg, modules, optimizer, writer, device)

    sanity_batch = next(iter(train_loader))
    with torch.no_grad():
        sanity_out = trainer.forward_pipeline(
            sanity_batch["imgs"].to(device),
            sanity_batch["valid"].to(device),
        )
    print("imgs:", sanity_batch["imgs"].shape)
    print("pred flows:", sanity_out["flows"].shape)
    print("flow_inits:", None if sanity_out["flow_inits"] is None else sanity_out["flow_inits"].shape)
    print("pair_feats:", sanity_out["pair_feats"].shape)
    print("fused_seq:", sanity_out["fused_seq"].shape)
    print("refined_seq:", sanity_out["refined_seq"].shape)
    print("flow_residuals:", sanity_out["flow_residuals"].shape)

    best_loss = float("inf")
    milestone_epochs = set(cfg["train"]["save_epoch_checkpoints"])

    for epoch in range(cfg["train"]["num_epochs"]):
        running = {
            "loss": 0.0,
            "loss_flow": 0.0,
            "loss_self": 0.0,
            "loss_temp": 0.0,
            "loss_acc_photo": 0.0,
            "loss_acc_smooth": 0.0,
            "loss_self_bw": 0.0,
            "loss_fb": 0.0,
            "loss_smooth": 0.0,
            "latent_delta_mean_abs": 0.0,
            "flow_residual_mean_abs": 0.0,
            "fb_error_mean": 0.0,
            "fb_conf_mean": 0.0,
            "fb_mask_ratio": 0.0,
        }
        n_batches = 0

        for batch in train_loader:
            stats = trainer.train_step(batch)
            for key, value in stats.items():
                if key not in running:
                    running[key] = 0.0
                running[key] += value
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        for key, value in avg.items():
            writer.add_scalar(f"train_epoch/{key}", value, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{cfg['train']['num_epochs']} | "
            f"loss={avg['loss']:.4f} | "
            f"flow={avg['loss_flow']:.4f} | "
            f"self={avg['loss_self']:.4f} | "
            f"temp={avg['loss_temp']:.4f} | "
            f"acc_photo={avg['loss_acc_photo']:.4f} | "
            f"acc_smooth={avg['loss_acc_smooth']:.4f} | "
            f"fb_loss={avg['loss_fb']:.4f} | "
            f"smooth={avg['loss_smooth']:.4f} | "
            f"latent={avg['latent_delta_mean_abs']:.4f} | "
            f"flow_res={avg['flow_residual_mean_abs']:.4f} | "
            f"fb_err={avg['fb_error_mean']:.4f} | "
            f"fb_conf={avg['fb_conf_mean']:.4f} | "
            f"fb_mask={avg['fb_mask_ratio']:.4f}"
        )

        latest_path = save_dir / cfg["experiment"]["checkpoint_name"]
        save_checkpoint(latest_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if (epoch + 1) == 200:
            save_path_200 = save_dir / f"fullpipeline_v26_{cfg['model'].get('architecture_type', 'later')}_epoch_200.pth"
            save_checkpoint(save_path_200, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            best_path = save_dir / cfg["experiment"]["best_checkpoint_name"]
            save_checkpoint(best_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if (epoch + 1) in milestone_epochs:
            milestone_path = save_dir / f"{cfg['experiment']['experiment_name']}_epoch_{epoch + 1}.pth"
            save_checkpoint(milestone_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
