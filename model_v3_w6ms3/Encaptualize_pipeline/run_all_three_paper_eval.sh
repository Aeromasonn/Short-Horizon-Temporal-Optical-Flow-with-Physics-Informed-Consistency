#!/bin/bash
set -e

# ============================================================
# Run paper-style optical-flow evaluation for all three models
# on both train and test splits.
#
# Metrics:
#   EPE_all_valid
#   Fl-epe / per_image_epe
#   Fl-all / F1-all percentage
#
# Note:
#   Official KITTI testing split usually has no public GT.
#   If your "testing" folder has no flow/valid labels, this script will fail there.
#   For a report, use your held-out validation split if available.
# ============================================================

TRAIN_SPLIT=${TRAIN_SPLIT:-training}
TEST_SPLIT=${TEST_SPLIT:-testing}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
OUT_DIR=${OUT_DIR:-paper_eval_results}

mkdir -p "$OUT_DIR"

run_one () {
  MODEL_NAME=$1
  SPLIT=$2
  TRAIN_SCRIPT=$3
  CONFIG=$4
  CKPT=$5

  echo ""
  echo "============================================================"
  echo "Model:      $MODEL_NAME"
  echo "Split:      $SPLIT"
  echo "Checkpoint: $CKPT"
  echo "============================================================"

  python3 eval_paper_metrics_v26.py \
    --train_script "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --output_json "$OUT_DIR/${MODEL_NAME}_${SPLIT}_summary.json" \
    --output_csv "$OUT_DIR/${MODEL_NAME}_${SPLIT}_per_sample.csv"
}

# ------------------------------
# Train split
# ------------------------------
run_one "convex" "$TRAIN_SPLIT" \
  "train_v26_convex.py" \
  "v26_convex_config.json" \
  "checkpoints/fullpipeline_v26_best.pth"

run_one "early" "$TRAIN_SPLIT" \
  "train_v26_early_operator.py" \
  "v26_early_operator_config.json" \
  "checkpoints/fullpipeline_v26_early_best.pth"

run_one "separate" "$TRAIN_SPLIT" \
  "train_v26_separate_operator.py" \
  "v26_separate_operator_config.json" \
  "checkpoints/fullpipeline_v26_separate_best.pth"

# ------------------------------
# Test split
# ------------------------------
run_one "convex" "$TEST_SPLIT" \
  "train_v26_convex.py" \
  "v26_convex_config.json" \
  "checkpoints/fullpipeline_v26_best.pth"

run_one "early" "$TEST_SPLIT" \
  "train_v26_early_operator.py" \
  "v26_early_operator_config.json" \
  "checkpoints/fullpipeline_v26_early_best.pth"

run_one "separate" "$TEST_SPLIT" \
  "train_v26_separate_operator.py" \
  "v26_separate_operator_config.json" \
  "checkpoints/fullpipeline_v26_separate_best.pth"

# ------------------------------
# Compact final table
# ------------------------------
python3 - <<'PY'
import json
from pathlib import Path

out_dir = Path("paper_eval_results")
if not out_dir.exists():
    out_dir = Path(__import__("os").environ.get("OUT_DIR", "paper_eval_results"))

rows = []
for path in sorted(out_dir.glob("*_summary.json")):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    model = path.name.split("_")[0]
    split = d["split"]
    rows.append((
        model,
        split,
        d["EPE_all_valid"],
        d["Fl_epe_per_image"],
        d["Fl_all_percent"],
        d["num_samples"],
        d["checkpoint_epoch"],
    ))

print("\n================ FINAL PAPER-STYLE TABLE ================")
print(f"{'Model':<10} {'Split':<10} {'EPE_all':>12} {'Fl-epe':>12} {'Fl-all/F1%':>12} {'N':>6} {'Epoch':>8}")
print("-" * 78)
for model, split, epe_all, fl_epe, fl_all, n, epoch in rows:
    print(f"{model:<10} {split:<10} {epe_all:>12.6f} {fl_epe:>12.6f} {fl_all:>12.4f} {n:>6} {str(epoch):>8}")
print("=========================================================")
print(f"Full JSON/CSV files saved in: {out_dir}")
PY
