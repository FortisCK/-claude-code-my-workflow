#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_PATH="${1:-quality_reports/logs/task1_monai_unet_cldice_domain_balanced_640.log}"
if [[ $# -gt 0 ]]; then
  shift
fi
mkdir -p "$(dirname "$LOG_PATH")" /tmp/cathaction-matplotlib

exec >> "$LOG_PATH" 2>&1

echo "==== CATHACTION Task 1 MONAI clDice domain-balanced 640 run ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "config=configs/task1/monai_unet_cldice_domain_balanced_640.yaml"
echo "conda_env=cardiac-diffusion"

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion \
  python -u scripts/task1/train_baseline.py \
  --config configs/task1/monai_unet_cldice_domain_balanced_640.yaml \
  "$@"
status=$?
echo "end_time=$(date --iso-8601=seconds)"
echo "exit_status=$status"
exit "$status"
