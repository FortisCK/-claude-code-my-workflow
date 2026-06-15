#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_PATH="${1:-quality_reports/logs/task1_monai_unet_dicece_full_512_overnight.log}"
mkdir -p "$(dirname "$LOG_PATH")" /tmp/cathaction-matplotlib

exec >> "$LOG_PATH" 2>&1

echo "==== CATHACTION Task 1 MONAI overnight run ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "config=configs/task1/monai_unet_dicece_full_512_overnight.yaml"
echo "conda_env=cardiac-diffusion"

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1

exec /home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion \
  python -u -c "import sys; sys.path[:0]=['src','.']; from cathaction.training.task1_baseline import main_train; raise SystemExit(main_train(['--config','configs/task1/monai_unet_dicece_full_512_overnight.yaml']))"
