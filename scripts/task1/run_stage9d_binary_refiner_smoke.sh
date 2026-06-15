#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p quality_reports/logs /tmp/cathaction-matplotlib

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

PYTHON=(/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion python -u)

STAGE7_CONFIG="configs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7.yaml"
STAGE7_CKPT="outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt"
REFINER_CONFIG="configs/task1/smp_fpn_convnext_small_512_binary_hardneg_stage9d.yaml"
REFINER_OUT="outputs/task1/smp_fpn_convnext_small_512_binary_hardneg_stage9d_smoke"
COARSE_TRAIN_DIR="outputs/task1/stage9d_train_coarse_stage7_hflip"
GATE_OUT="outputs/task1/stage9d_binary_gate_smoke32"
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"
STAGE9A_POST="outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv"

echo "==== CATHACTION Task 1 Stage 9D binary refiner smoke ===="
echo "start_time=$(date --iso-8601=seconds)"

echo "---- train_coarse_prediction_smoke_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/predict_tta_ensemble_original_space.py \
  --model convnext_small640_toolness "$STAGE7_CONFIG" "$STAGE7_CKPT" 1.0 \
  --tta hflip \
  --manifest configs/task1/splits/released_train.csv \
  --max-samples 128 \
  --batch-size 4 \
  --num-workers 8 \
  --device cuda \
  --output-dir "$COARSE_TRAIN_DIR" \
  > quality_reports/logs/task1_stage9d_train_coarse_stage7_smoke.log 2>&1
echo "---- train_coarse_prediction_smoke_done $(date --iso-8601=seconds) ----"

echo "---- binary_refiner_train_smoke_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/train_baseline.py \
  --config "$REFINER_CONFIG" \
  --output-dir "$REFINER_OUT" \
  --max-train-samples 128 \
  --max-eval-samples 64 \
  --epochs 2 \
  --device cuda \
  > quality_reports/logs/task1_stage9d_binary_refiner_train_smoke.log 2>&1
echo "---- binary_refiner_train_smoke_done $(date --iso-8601=seconds) ----"

echo "---- binary_gate_smoke_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/predict_binary_roi_gate_original_space.py \
  --refiner-model binary_stage9d "$REFINER_CONFIG" "$REFINER_OUT/best_checkpoint.pt" 1.0 \
  --tta hflip \
  --manifest "$EVAL_MANIFEST" \
  --coarse-predictions-csv "$STAGE9A_POST" \
  --output-dir "$GATE_OUT" \
  --max-samples 32 \
  --device cuda \
  --threshold 0.65 \
  --roi-size 512 \
  --roi-stride 256 \
  --max-rois 12 \
  --patch-batch-size 4 \
  > quality_reports/logs/task1_stage9d_binary_gate_smoke.log 2>&1
echo "---- binary_gate_smoke_done $(date --iso-8601=seconds) ----"

echo "---- binary_gate_eval_smoke_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/evaluate_baseline.py \
  --config "$STAGE7_CONFIG" \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$GATE_OUT/predictions.csv" \
  --output-json "$GATE_OUT/eval.json" \
  --device cuda \
  > quality_reports/logs/task1_stage9d_binary_gate_eval_smoke.log 2>&1
"${PYTHON[@]}" scripts/task1/evaluate_mslnet_style.py \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$GATE_OUT/predictions.csv" \
  --output-json "$GATE_OUT/eval_mslnet_style.json" \
  > quality_reports/logs/task1_stage9d_binary_gate_mslnet_eval_smoke.log 2>&1
echo "---- binary_gate_eval_smoke_done $(date --iso-8601=seconds) ----"

"${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path

out = Path("outputs/task1/stage9d_binary_gate_smoke32")
task1 = json.loads((out / "eval.json").read_text())["metrics"]
msl = json.loads((out / "eval_mslnet_style.json").read_text())["sample_mean"]
print("task1_dice=", task1["dice"])
print("task1_label_1=", task1["per_class_dice"]["label_1"])
print("task1_label_2=", task1["per_class_dice"]["label_2"])
print("mslnet_dice=", msl["dice"])
print("mslnet_f1_r3=", msl["f1_r3"])
PY

echo "end_time=$(date --iso-8601=seconds)"
