import argparse
import torch
from pprint import pprint


def main():
    parser = argparse.ArgumentParser(description="Read checkpoint (.pth) info")
    parser.add_argument("pth_path", type=str, help="Path to .pth file")
    args = parser.parse_args()

    ckpt = torch.load(args.pth_path, map_location="cpu")

    print("\n========== BASIC INFO ==========")
    print("Epoch:", ckpt.get("epoch", "N/A"))

    stats = ckpt.get("stats", {})
    if stats:
        print("\n========== STATS ==========")
        for k, v in stats.items():
            print(f"{k}: {v}")
    else:
        print("No stats found")

    print("\n========== OPTIMIZER ==========")
    if "optimizer_state_dict" in ckpt:
        print("Optimizer state loaded")
    else:
        print("No optimizer state")

    print("\n========== MODEL MODULES ==========")
    for k in ckpt.keys():
        if k.endswith("_state_dict"):
            print(k)

    print("\n========== CONFIG ==========")
    config = ckpt.get("config", None)
    if config:
        pprint(config)
    else:
        print("No config saved")

    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()