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
REFINER_CONFIG="configs/task1/smp_fpn_convnext_small_512_binary_hardneg_stage9d.yaml"
REFINER_CKPT="outputs/task1/smp_fpn_convnext_small_512_binary_hardneg_stage9d/best_checkpoint.pt"
GATE_OUT="${1:-outputs/task1/stage9d_binary_gate_full_t065}"
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"
STAGE9A_POST="outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv"

echo "==== CATHACTION Task 1 Stage 9D binary refiner gate/eval ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "refiner_ckpt=$REFINER_CKPT"
echo "gate_out=$GATE_OUT"

echo "---- binary_gate_full_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/predict_binary_roi_gate_original_space.py \
  --refiner-model binary_stage9d "$REFINER_CONFIG" "$REFINER_CKPT" 1.0 \
  --tta hflip \
  --manifest "$EVAL_MANIFEST" \
  --coarse-predictions-csv "$STAGE9A_POST" \
  --output-dir "$GATE_OUT" \
  --device cuda \
  --threshold 0.65 \
  --roi-size 512 \
  --roi-stride 256 \
  --max-rois 16 \
  --patch-batch-size 4 \
  > quality_reports/logs/task1_stage9d_binary_gate_full_t065.log 2>&1
echo "---- binary_gate_full_done $(date --iso-8601=seconds) ----"

echo "---- binary_gate_eval_full_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/evaluate_baseline.py \
  --config "$STAGE7_CONFIG" \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$GATE_OUT/predictions.csv" \
  --output-json "$GATE_OUT/eval.json" \
  --device cuda \
  > quality_reports/logs/task1_stage9d_binary_gate_eval_full_t065.log 2>&1
"${PYTHON[@]}" scripts/task1/evaluate_mslnet_style.py \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$GATE_OUT/predictions.csv" \
  --output-json "$GATE_OUT/eval_mslnet_style.json" \
  --per-sample-csv "$GATE_OUT/eval_mslnet_style_per_sample.csv" \
  > quality_reports/logs/task1_stage9d_binary_gate_mslnet_eval_full_t065.log 2>&1
echo "---- binary_gate_eval_full_done $(date --iso-8601=seconds) ----"

"${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path

out = Path("outputs/task1/stage9d_binary_gate_full_t065")
task1 = json.loads((out / "eval.json").read_text())["metrics"]
msl_payload = json.loads((out / "eval_mslnet_style.json").read_text())
msl = msl_payload["sample_mean"]
msl_domains = msl_payload["sample_mean_by_domain"]
print("task1_dice=", task1["dice"])
print("task1_label_1=", task1["per_class_dice"]["label_1"])
print("task1_label_2=", task1["per_class_dice"]["label_2"])
print("task1_animal=", task1["by_domain"].get("animal", {}).get("dice"))
print("task1_phantom=", task1["by_domain"].get("phantom", {}).get("dice"))
print("mslnet_dice=", msl["dice"])
print("mslnet_iou=", msl["iou"])
print("mslnet_ahd=", msl["ahd"])
print("mslnet_f1_r3=", msl["f1_r3"])
print("mslnet_precision_r3=", msl["precision_r3"])
print("mslnet_recall_r3=", msl["recall_r3"])
print("mslnet_animal_f1_r3=", msl_domains.get("animal", {}).get("f1_r3"))
print("mslnet_phantom_f1_r3=", msl_domains.get("phantom", {}).get("f1_r3"))
PY

echo "end_time=$(date --iso-8601=seconds)"
