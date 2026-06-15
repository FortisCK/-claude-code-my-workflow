#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7.yaml"
TRAIN_LOG="quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_aux_stage7.log"
MASTER_LOG="${1:-quality_reports/logs/task1_stage7_convnext_small_640_toolness_aux_master.log}"
EVAL_DIR="outputs/task1/stage7_convnext_small_640_toolness_aux_eval"
EVAL_JSON="$EVAL_DIR/convnext_small_640_toolness_aux_hflip_original_full.json"
EVAL_LOG="quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_aux_stage7_full_eval.log"
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"
CHECKPOINT="outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt"

mkdir -p "$(dirname "$MASTER_LOG")" "$EVAL_DIR" /tmp/cathaction-matplotlib

exec >> "$MASTER_LOG" 2>&1

echo "==== CATHACTION Task 1 Stage 7 ConvNeXt-Small 640 toolness auxiliary ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "config=$CONFIG"
echo "train_log=$TRAIN_LOG"
echo "eval_json=$EVAL_JSON"
echo "eval_manifest=$EVAL_MANIFEST"
echo "conda_env=cardiac-diffusion"

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

echo "---- train_start $(date --iso-8601=seconds) ----"
scripts/task1/run_stage2_shootout.sh "$CONFIG" "$TRAIN_LOG"
echo "---- train_done $(date --iso-8601=seconds) ----"

echo "---- full_eval_start $(date --iso-8601=seconds) ----"
/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion \
  python -u scripts/task1/evaluate_tta_ensemble_original_space.py \
    --model convnext_small_640_toolness_aux "$CONFIG" "$CHECKPOINT" 1.0 \
    --tta hflip \
    --eval-manifest "$EVAL_MANIFEST" \
    --batch-size 4 \
    --num-workers 8 \
    --device cuda \
    --output-json "$EVAL_JSON" \
    > "$EVAL_LOG" 2>&1
echo "full_eval_json=$EVAL_JSON"
echo "full_eval_log=$EVAL_LOG"
echo "---- full_eval_done $(date --iso-8601=seconds) ----"

/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/task1/stage7_convnext_small_640_toolness_aux_eval/convnext_small_640_toolness_aux_hflip_original_full.json")
metrics = json.loads(path.read_text())["metrics"]
domains = metrics.get("by_domain", {})
print(
    path.name,
    "dice=", metrics["dice"],
    "label_1=", metrics["per_class_dice"]["label_1"],
    "label_2=", metrics["per_class_dice"]["label_2"],
    "animal=", domains.get("animal", {}).get("dice"),
    "phantom=", domains.get("phantom", {}).get("dice"),
)
PY

echo "end_time=$(date --iso-8601=seconds)"
