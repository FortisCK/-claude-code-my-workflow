#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_PATH [LOG_PATH] [train_baseline.py args...]" >&2
  exit 2
fi

CONFIG_PATH="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_STEM="$(basename "$CONFIG_PATH" .yaml)"
LOG_PATH="quality_reports/logs/${CONFIG_STEM}.log"
if [[ $# -gt 0 && "$1" == *.log ]]; then
  LOG_PATH="$1"
  shift
fi

mkdir -p "$(dirname "$LOG_PATH")" /tmp/cathaction-matplotlib

exec >> "$LOG_PATH" 2>&1

echo "==== CATHACTION Task 1 Stage 2 architecture shootout ===="
echo "start_time=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "config=$CONFIG_PATH"
echo "conda_env=cardiac-diffusion"

export MPLCONFIGDIR=/tmp/cathaction-matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

/home/mingzhang/miniconda3/bin/conda run --no-capture-output -n cardiac-diffusion \
  python -u scripts/task1/train_baseline.py \
  --config "$CONFIG_PATH" \
  "$@"
status=$?
echo "end_time=$(date --iso-8601=seconds)"
echo "exit_status=$status"
exit "$status"
