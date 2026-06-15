#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-outputs/task1/stage5_weight_refine}"
mkdir -p "$OUT_DIR" /tmp/cathaction-matplotlib

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

PYTHON=(/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion python -u)
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"
OUTPUT_JSON="$OUT_DIR/five_model_weight_refine_full.json"
OUTPUT_LOG="$OUT_DIR/five_model_weight_refine_full.log"

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

echo "==== CATHACTION Task 1 Stage 5 weight refine ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "out_dir=$OUT_DIR"
echo "output_json=$OUTPUT_JSON"
echo "incumbent_dice=0.6500661421073305"

"${PYTHON[@]}" scripts/task1/evaluate_weight_grid_original_space.py \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.35 \
  --model efficientnet_b3 "$EFFICIENTNET_B3_CONFIG" "$EFFICIENTNET_B3_CKPT" 0.35 \
  --model convnextv2_base "$CONVNEXTV2_BASE_CONFIG" "$CONVNEXTV2_BASE_CKPT" 0.10 \
  --model convnext_small512 "$CONVNEXT_SMALL512_CONFIG" "$CONVNEXT_SMALL512_CKPT" 0.10 \
  --model convnext_small640 "$CONVNEXT_SMALL640_CONFIG" "$CONVNEXT_SMALL640_CKPT" 0.10 \
  --candidate incumbent_035_035_010_010_010:0.35,0.35,0.10,0.10,0.10 \
  --candidate main_c640_plus_036_034_010_010_010:0.36,0.34,0.10,0.10,0.10 \
  --candidate main_effb3_plus_034_036_010_010_010:0.34,0.36,0.10,0.10,0.10 \
  --candidate main_c640_plus2_037_033_010_010_010:0.37,0.33,0.10,0.10,0.10 \
  --candidate main_effb3_plus2_033_037_010_010_010:0.33,0.37,0.10,0.10,0.10 \
  --candidate tail_v2_plus_035_035_012_009_009:0.35,0.35,0.12,0.09,0.09 \
  --candidate tail_s512_plus_035_035_009_012_009:0.35,0.35,0.09,0.12,0.09 \
  --candidate tail_s640_plus_035_035_009_009_012:0.35,0.35,0.09,0.09,0.12 \
  --candidate tail_v2_plus2_035_035_014_008_008:0.35,0.35,0.14,0.08,0.08 \
  --candidate tail_s512_plus2_035_035_008_014_008:0.35,0.35,0.08,0.14,0.08 \
  --candidate tail_s640_plus2_035_035_008_008_014:0.35,0.35,0.08,0.08,0.14 \
  --candidate less_tail_equal_0375_0375_0083_0083_0084:0.375,0.375,0.083,0.083,0.084 \
  --candidate more_tail_equal_0325_0325_0117_0117_0116:0.325,0.325,0.117,0.117,0.116 \
  --candidate smalls_plus_034_034_008_012_012:0.34,0.34,0.08,0.12,0.12 \
  --candidate s640_plus_main_high_036_036_008_008_012:0.36,0.36,0.08,0.08,0.12 \
  --candidate s512_plus_main_high_036_036_008_012_008:0.36,0.36,0.08,0.12,0.08 \
  --candidate v2_plus_main_high_036_036_012_008_008:0.36,0.36,0.12,0.08,0.08 \
  --tta hflip \
  --eval-manifest "$EVAL_MANIFEST" \
  --batch-size 4 \
  --num-workers 8 \
  --device cuda \
  --output-json "$OUTPUT_JSON" \
  > "$OUTPUT_LOG" 2>&1

echo "---- summary_start $(date --iso-8601=seconds) ----"
export STAGE5_WEIGHT_REFINE_JSON="$OUTPUT_JSON"
"${PYTHON[@]}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["STAGE5_WEIGHT_REFINE_JSON"])
payload = json.loads(path.read_text())
for row in payload["ranked_candidates"]:
    print(
        row["name"],
        "dice=", row["dice"],
        "label_1=", row["label_1"],
        "label_2=", row["label_2"],
        "animal=", row["animal"],
        "phantom=", row["phantom"],
    )
PY
echo "end_time=$(date --iso-8601=seconds)"
