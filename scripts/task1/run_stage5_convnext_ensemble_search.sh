#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-outputs/task1/stage5_convnext_ensemble_search}"
mkdir -p "$OUT_DIR" quality_reports/logs /tmp/cathaction-matplotlib

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

PYTHON=(/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion python -u)
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"

CONVNEXT640_CONFIG="configs/task1/smp_fpn_convnext_tiny_640_rescue_stable.yaml"
CONVNEXT640_CKPT="outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt"
EFFICIENTNET_B3_CONFIG="configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml"
EFFICIENTNET_B3_CKPT="outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt"
CONVNEXT_SMALL_CONFIG="configs/task1/smp_fpn_convnext_small_512_stage4a.yaml"
CONVNEXT_SMALL_CKPT="outputs/task1/smp_fpn_convnext_small_512_stage4a/best_checkpoint.pt"
CONVNEXTV2_BASE_CONFIG="configs/task1/smp_fpn_convnextv2_base_512_stage4a.yaml"
CONVNEXTV2_BASE_CKPT="outputs/task1/smp_fpn_convnextv2_base_512_stage4a/best_checkpoint.pt"
EFFICIENTNET_B5_CONFIG="configs/task1/smp_unet_efficientnet_b5_512_stage4a.yaml"
EFFICIENTNET_B5_CKPT="outputs/task1/smp_unet_efficientnet_b5_512_stage4a/best_checkpoint.pt"

echo "==== CATHACTION Task 1 Stage 5 ConvNeXt ensemble search ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "out_dir=$OUT_DIR"
echo "eval_manifest=$EVAL_MANIFEST"
echo "current_champion_dice=0.6438677932546133"

run_eval() {
  local name="$1"
  shift
  local output_json="$OUT_DIR/${name}.json"
  local output_log="$OUT_DIR/${name}.log"

  echo "---- eval_start $(date --iso-8601=seconds) $name ----"
  "${PYTHON[@]}" scripts/task1/evaluate_tta_ensemble_original_space.py \
    "$@" \
    --tta hflip \
    --eval-manifest "$EVAL_MANIFEST" \
    --batch-size 4 \
    --num-workers 8 \
    --device cuda \
    --output-json "$output_json" \
    > "$output_log" 2>&1
  echo "json=$output_json"
  echo "log=$output_log"
  echo "---- eval_done $(date --iso-8601=seconds) $name ----"
}

run_eval "w045_045_010_convnextv2" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.45 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.45 \
  --model convnextv2_base "$CONVNEXTV2_BASE_CONFIG" "$CONVNEXTV2_BASE_CKPT" 0.10

run_eval "w040_040_020_convnextv2" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.40 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.40 \
  --model convnextv2_base "$CONVNEXTV2_BASE_CONFIG" "$CONVNEXTV2_BASE_CKPT" 0.20

run_eval "w050_040_010_convnextv2" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.50 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.40 \
  --model convnextv2_base "$CONVNEXTV2_BASE_CONFIG" "$CONVNEXTV2_BASE_CKPT" 0.10

run_eval "w040_050_010_convnextv2" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.40 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.50 \
  --model convnextv2_base "$CONVNEXTV2_BASE_CONFIG" "$CONVNEXTV2_BASE_CKPT" 0.10

run_eval "w045_045_010_convnext_small" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.45 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.45 \
  --model convnext_small "$CONVNEXT_SMALL_CONFIG" "$CONVNEXT_SMALL_CKPT" 0.10

run_eval "w040_040_020_convnext_small" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.40 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.40 \
  --model convnext_small "$CONVNEXT_SMALL_CONFIG" "$CONVNEXT_SMALL_CKPT" 0.20

run_eval "w040_040_010_010_convnextv2_small" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.40 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.40 \
  --model convnextv2_base "$CONVNEXTV2_BASE_CONFIG" "$CONVNEXTV2_BASE_CKPT" 0.10 \
  --model convnext_small "$CONVNEXT_SMALL_CONFIG" "$CONVNEXT_SMALL_CKPT" 0.10

run_eval "w045_045_010_efficientnet_b5" \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.45 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.45 \
  --model efficientnet_b5 "$EFFICIENTNET_B5_CONFIG" "$EFFICIENTNET_B5_CKPT" 0.10

echo "---- summary_start $(date --iso-8601=seconds) ----"
export STAGE5_OUT_DIR="$OUT_DIR"
"${PYTHON[@]}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["STAGE5_OUT_DIR"])
rows = []
for path in sorted(out.glob("*.json")):
    metrics = json.loads(path.read_text())["metrics"]
    domains = metrics.get("by_domain", {})
    rows.append(
        (
            metrics["dice"],
            path.name,
            metrics["per_class_dice"]["label_1"],
            metrics["per_class_dice"]["label_2"],
            domains.get("animal", {}).get("dice"),
            domains.get("phantom", {}).get("dice"),
        )
    )
for dice, name, label_1, label_2, animal, phantom in sorted(rows, reverse=True):
    print(
        name,
        "dice=", dice,
        "label_1=", label_1,
        "label_2=", label_2,
        "animal=", animal,
        "phantom=", phantom,
    )
PY
echo "end_time=$(date --iso-8601=seconds)"
