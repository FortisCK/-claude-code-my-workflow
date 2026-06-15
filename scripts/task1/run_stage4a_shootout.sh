#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p quality_reports/logs outputs/task1/stage4a_eval /tmp/cathaction-matplotlib

CONDA=(/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion)
EVAL_MANIFEST="configs/task1/splits/released_eval.csv"

CONFIGS=(
  "configs/task1/smp_segformer_mit_b3_512_stage4a.yaml"
  "configs/task1/smp_segformer_mit_b4_512_stage4a.yaml"
  "configs/task1/smp_fpn_convnext_small_512_stage4a.yaml"
  "configs/task1/smp_fpn_convnextv2_base_512_stage4a.yaml"
  "configs/task1/smp_unet_efficientnet_b5_512_stage4a.yaml"
)

echo "==== CATHACTION Task 1 Stage 4A aggressive architecture shootout ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "conda_env=cardiac-diffusion"
echo "eval_manifest=$EVAL_MANIFEST"
echo "configs=${CONFIGS[*]}"

for config in "${CONFIGS[@]}"; do
  stem="$(basename "$config" .yaml)"
  train_log="quality_reports/logs/${stem}.log"
  eval_log="quality_reports/logs/${stem}_full_eval.log"
  eval_json="outputs/task1/stage4a_eval/${stem}_full_released.json"
  output_dir="outputs/task1/${stem#task1_}"

  echo "---- train_start $(date --iso-8601=seconds) $stem ----"
  scripts/task1/run_stage2_shootout.sh "$config" "$train_log"
  echo "---- train_done $(date --iso-8601=seconds) $stem ----"

  echo "---- full_eval_start $(date --iso-8601=seconds) $stem ----"
  "${CONDA[@]}" python -u scripts/task1/evaluate_baseline.py \
    --config "$config" \
    --checkpoint "$output_dir/best_checkpoint.pt" \
    --eval-manifest "$EVAL_MANIFEST" \
    --output-json "$eval_json" \
    --device cuda \
    > "$eval_log" 2>&1
  echo "full_eval_json=$eval_json"
  echo "---- full_eval_done $(date --iso-8601=seconds) $stem ----"
done

echo "---- summary_start $(date --iso-8601=seconds) ----"
"${CONDA[@]}" python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("outputs/task1/stage4a_eval").glob("*_full_released.json")):
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
