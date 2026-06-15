#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-outputs/task1/stage3_convnext640_eval}"
mkdir -p "$OUT_DIR" /tmp/cathaction-matplotlib

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

PYTHON=(/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion env PYTHONPATH=src:. MPLCONFIGDIR=/tmp/cathaction-matplotlib python -u)

EVAL_MANIFEST="configs/task1/splits/released_eval.csv"
CONVNEXT512_CONFIG="configs/task1/smp_fpn_convnext_tiny_512_rescue_stable.yaml"
CONVNEXT512_CKPT="outputs/task1/smp_fpn_convnext_tiny_512_rescue_stable/best_checkpoint.pt"
EFFICIENTNET_CONFIG="configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml"
EFFICIENTNET_CKPT="outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt"
CONVNEXT640_CONFIG="configs/task1/smp_fpn_convnext_tiny_640_rescue_stable.yaml"
CONVNEXT640_CKPT="outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt"

echo "==== Stage 3 ConvNeXt 640 evaluation ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "out_dir=$OUT_DIR"

"${PYTHON[@]}" scripts/task1/evaluate_baseline.py \
  --config "$CONVNEXT640_CONFIG" \
  --checkpoint "$CONVNEXT640_CKPT" \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-json "$OUT_DIR/convnext640_best_resized_full.json" \
  --device cuda

"${PYTHON[@]}" scripts/task1/evaluate_tta_ensemble_original_space.py \
  --model convnext512 "$CONVNEXT512_CONFIG" "$CONVNEXT512_CKPT" 0.5 \
  --model efficientnet "$EFFICIENTNET_CONFIG" "$EFFICIENTNET_CKPT" 0.5 \
  --tta hflip \
  --eval-manifest "$EVAL_MANIFEST" \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --output-json "$OUT_DIR/incumbent_512_ensemble_w050_050_hflip_original_full.json"

"${PYTHON[@]}" scripts/task1/evaluate_tta_ensemble_original_space.py \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 1.0 \
  --tta hflip \
  --eval-manifest "$EVAL_MANIFEST" \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --output-json "$OUT_DIR/convnext640_best_hflip_original_full.json"

"${PYTHON[@]}" scripts/task1/evaluate_tta_ensemble_original_space.py \
  --model convnext640 "$CONVNEXT640_CONFIG" "$CONVNEXT640_CKPT" 0.5 \
  --model efficientnet "$EFFICIENTNET_CONFIG" "$EFFICIENTNET_CKPT" 0.5 \
  --tta hflip \
  --eval-manifest "$EVAL_MANIFEST" \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --output-json "$OUT_DIR/ensemble_convnext640_efficientnet512_w050_050_hflip_original_full.json"

"${PYTHON[@]}" -c "
import json
from pathlib import Path
out = Path('$OUT_DIR')
for path in sorted(out.glob('*.json')):
    payload = json.loads(path.read_text())
    metrics = payload['metrics']
    print(
        path.name,
        'dice=', metrics['dice'],
        'label_1=', metrics['per_class_dice']['label_1'],
        'label_2=', metrics['per_class_dice']['label_2'],
        'animal=', metrics['by_domain'].get('animal', {}).get('dice'),
        'phantom=', metrics['by_domain'].get('phantom', {}).get('dice'),
    )
"

echo "end_time=$(date --iso-8601=seconds)"
