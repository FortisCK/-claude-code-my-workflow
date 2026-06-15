#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.yaml"
TRAIN_LOG="quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.log"
MASTER_LOG="${1:-quality_reports/logs/task1_stage9b_convnext_small_640_toolness_cbdice_hd_master.log}"
EVAL_DIR="outputs/task1/stage9b_convnext_small_640_toolness_cbdice_hd_eval"
RAW_EVAL_JSON="$EVAL_DIR/raw_hflip_original_full.json"
RAW_PRED_DIR="$EVAL_DIR/raw_predictions"
RAW_PRED_EVAL_JSON="$RAW_PRED_DIR/eval.json"
POST_DIR="$EVAL_DIR/remove_small_min32"
POST_EVAL_JSON="$POST_DIR/eval.json"
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"
CHECKPOINT="outputs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b/best_checkpoint.pt"

mkdir -p "$(dirname "$MASTER_LOG")" "$EVAL_DIR" /tmp/cathaction-matplotlib

exec >> "$MASTER_LOG" 2>&1

echo "==== CATHACTION Task 1 Stage 9B ConvNeXt-Small 640 toolness cbDice HD ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "config=$CONFIG"
echo "train_log=$TRAIN_LOG"
echo "eval_manifest=$EVAL_MANIFEST"
echo "conda_env=cardiac-diffusion"

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

PYTHON=(/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion python -u)

"${PYTHON[@]}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; aborting Stage 9B training.")
print("cuda_device_count=", torch.cuda.device_count())
print("cuda_device_name=", torch.cuda.get_device_name(0))
PY

echo "---- train_start $(date --iso-8601=seconds) ----"
scripts/task1/run_stage2_shootout.sh "$CONFIG" "$TRAIN_LOG"
echo "---- train_done $(date --iso-8601=seconds) ----"

echo "---- full_eval_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/evaluate_tta_ensemble_original_space.py \
  --model convnext_small_640_toolness_cbdice_hd "$CONFIG" "$CHECKPOINT" 1.0 \
  --tta hflip \
  --eval-manifest "$EVAL_MANIFEST" \
  --batch-size 4 \
  --num-workers 8 \
  --device cuda \
  --output-json "$RAW_EVAL_JSON" \
  > "quality_reports/logs/task1_stage9b_raw_full_eval.log" 2>&1
echo "raw_eval_json=$RAW_EVAL_JSON"
echo "---- full_eval_done $(date --iso-8601=seconds) ----"

echo "---- raw_prediction_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/predict_tta_ensemble_original_space.py \
  --model convnext_small_640_toolness_cbdice_hd "$CONFIG" "$CHECKPOINT" 1.0 \
  --tta hflip \
  --manifest "$EVAL_MANIFEST" \
  --batch-size 4 \
  --num-workers 8 \
  --device cuda \
  --output-dir "$RAW_PRED_DIR" \
  > "quality_reports/logs/task1_stage9b_raw_predict.log" 2>&1
echo "---- raw_prediction_done $(date --iso-8601=seconds) ----"

echo "---- raw_prediction_eval_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/evaluate_baseline.py \
  --config "$CONFIG" \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$RAW_PRED_DIR/predictions.csv" \
  --output-json "$RAW_PRED_EVAL_JSON" \
  --device cuda \
  > "quality_reports/logs/task1_stage9b_raw_prediction_eval.log" 2>&1
echo "---- raw_prediction_eval_done $(date --iso-8601=seconds) ----"

echo "---- postprocess_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/apply_morphology_postprocess.py \
  --predictions-csv "$RAW_PRED_DIR/predictions.csv" \
  --output-dir "$POST_DIR" \
  > "quality_reports/logs/task1_stage9b_remove_small_min32.log" 2>&1
echo "---- postprocess_done $(date --iso-8601=seconds) ----"

echo "---- post_eval_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/evaluate_baseline.py \
  --config "$CONFIG" \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$POST_DIR/predictions.csv" \
  --output-json "$POST_EVAL_JSON" \
  --device cuda \
  > "quality_reports/logs/task1_stage9b_post_eval.log" 2>&1
echo "---- post_eval_done $(date --iso-8601=seconds) ----"

"${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path

paths = {
    "raw_hflip": Path("outputs/task1/stage9b_convnext_small_640_toolness_cbdice_hd_eval/raw_hflip_original_full.json"),
    "raw_png": Path("outputs/task1/stage9b_convnext_small_640_toolness_cbdice_hd_eval/raw_predictions/eval.json"),
    "post": Path("outputs/task1/stage9b_convnext_small_640_toolness_cbdice_hd_eval/remove_small_min32/eval.json"),
}
for name, path in paths.items():
    if not path.exists():
        continue
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", payload)
    domains = metrics.get("by_domain", {})
    print(
        name,
        "dice=", metrics["dice"],
        "label_1=", metrics["per_class_dice"]["label_1"],
        "label_2=", metrics["per_class_dice"]["label_2"],
        "animal=", domains.get("animal", {}).get("dice"),
        "phantom=", domains.get("phantom", {}).get("dice"),
    )
PY

echo "end_time=$(date --iso-8601=seconds)"
