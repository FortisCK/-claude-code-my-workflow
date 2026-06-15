#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${1:-outputs/task1/stage9a_seven_model_add_both_010_010}"
RAW_DIR="$OUT_ROOT/raw_predictions"
POST_DIR="$OUT_ROOT/remove_small_min32"
LOG_DIR="quality_reports/logs"
RAW_EVAL_JSON="$RAW_DIR/eval.json"
POST_EVAL_JSON="$POST_DIR/eval.json"
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"

mkdir -p "$RAW_DIR" "$POST_DIR" "$LOG_DIR" /tmp/cathaction-matplotlib

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export STAGE9A_OUT_ROOT="$OUT_ROOT"

PYTHON=(/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion python -u)

CONVNEXT640_CONFIG="configs/task1/smp_fpn_convnext_tiny_640_rescue_stable.yaml"
CONVNEXT640_CKPT="outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt"
EFFICIENTNET_B3_CONFIG="configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml"
EFFICIENTNET_B3_CKPT="outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt"
CONVNEXTV2_BASE_CONFIG="configs/task1/smp_fpn_convnextv2_base_512_stage4a.yaml"
CONVNEXTV2_BASE_CKPT="outputs/task1/smp_fpn_convnextv2_base_512_stage4a/best_checkpoint.pt"
CONVNEXT_SMALL512_CONFIG="configs/task1/smp_fpn_convnext_small_512_stage4a.yaml"
CONVNEXT_SMALL512_CKPT="outputs/task1/smp_fpn_convnext_small_512_stage4a/best_checkpoint.pt"
CONVNEXT_SMALL640_CONFIG="configs/task1/smp_fpn_convnext_small_640_stage5.yaml"
CONVNEXT_SMALL640_CKPT="outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt"
CONVNEXT_SMALL640_CLDICE_CONFIG="configs/task1/smp_fpn_convnext_small_640_cldice_stage6.yaml"
CONVNEXT_SMALL640_CLDICE_CKPT="outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/best_checkpoint.pt"
CONVNEXT_SMALL640_TOOLNESS_CONFIG="configs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7.yaml"
CONVNEXT_SMALL640_TOOLNESS_CKPT="outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt"

echo "==== CATHACTION Task 1 Stage 9A seven-model postprocess ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "out_root=$OUT_ROOT"
echo "raw_dir=$RAW_DIR"
echo "post_dir=$POST_DIR"
echo "eval_manifest=$EVAL_MANIFEST"

"${PYTHON[@]}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; aborting seven-model full prediction.")
print("cuda_device_count=", torch.cuda.device_count())
print("cuda_device_name=", torch.cuda.get_device_name(0))
PY

echo "---- raw_prediction_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/predict_tta_ensemble_original_space.py \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.2600 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.2600 \
  --model convnextv2_base "$CONVNEXTV2_BASE_CONFIG" "$CONVNEXTV2_BASE_CKPT" 0.0936 \
  --model convnext_small512 "$CONVNEXT_SMALL512_CONFIG" "$CONVNEXT_SMALL512_CKPT" 0.0936 \
  --model convnext_small640 "$CONVNEXT_SMALL640_CONFIG" "$CONVNEXT_SMALL640_CKPT" 0.0928 \
  --model convnext_small640_cldice "$CONVNEXT_SMALL640_CLDICE_CONFIG" "$CONVNEXT_SMALL640_CLDICE_CKPT" 0.1000 \
  --model convnext_small640_toolness "$CONVNEXT_SMALL640_TOOLNESS_CONFIG" "$CONVNEXT_SMALL640_TOOLNESS_CKPT" 0.1000 \
  --tta hflip \
  --manifest "$EVAL_MANIFEST" \
  --batch-size 4 \
  --num-workers 8 \
  --device cuda \
  --output-dir "$RAW_DIR" \
  > "$LOG_DIR/task1_stage9a_seven_model_predict.log" 2>&1
echo "---- raw_prediction_done $(date --iso-8601=seconds) ----"

echo "---- raw_eval_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/evaluate_baseline.py \
  --config "$CONVNEXT_SMALL640_TOOLNESS_CONFIG" \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$RAW_DIR/predictions.csv" \
  --output-json "$RAW_EVAL_JSON" \
  --device cuda \
  > "$LOG_DIR/task1_stage9a_seven_model_raw_eval.log" 2>&1
echo "---- raw_eval_done $(date --iso-8601=seconds) ----"

echo "---- postprocess_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/apply_morphology_postprocess.py \
  --predictions-csv "$RAW_DIR/predictions.csv" \
  --output-dir "$POST_DIR" \
  > "$LOG_DIR/task1_stage9a_seven_model_remove_small_min32.log" 2>&1
echo "---- postprocess_done $(date --iso-8601=seconds) ----"

echo "---- post_eval_start $(date --iso-8601=seconds) ----"
"${PYTHON[@]}" scripts/task1/evaluate_baseline.py \
  --config "$CONVNEXT_SMALL640_TOOLNESS_CONFIG" \
  --eval-manifest "$EVAL_MANIFEST" \
  --predictions-csv "$POST_DIR/predictions.csv" \
  --output-json "$POST_EVAL_JSON" \
  --device cuda \
  > "$LOG_DIR/task1_stage9a_seven_model_post_eval.log" 2>&1
echo "---- post_eval_done $(date --iso-8601=seconds) ----"

"${PYTHON[@]}" - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["STAGE9A_OUT_ROOT"])
paths = {
    "raw": out_root / "raw_predictions" / "eval.json",
    "post": out_root / "remove_small_min32" / "eval.json",
}
for name, path in paths.items():
    if not path.exists():
        continue
    metrics = json.loads(path.read_text())["metrics"]
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
