#!/usr/bin/env bash
# CATHACTION submission container entrypoint. Dispatches to the Task 1 or Task 2
# fully-automated inference wrapper based on the TASK env var. No user interaction.
set -euo pipefail

TASK="${TASK:-task1}"
INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
DEVICE="${DEVICE:-cuda}"

case "${TASK}" in
  task1)
    exec python scripts/task1/run_task1_submission_inference.py \
      --image-dir "${INPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --device "${DEVICE}" \
      --execute
    ;;
  task2)
    exec python scripts/task2/run_task2_submission_inference.py \
      --image-dir "${INPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --device "${DEVICE}" \
      --execute
    ;;
  *)
    echo "Unknown TASK='${TASK}' (expected 'task1' or 'task2')" >&2
    exit 2
    ;;
esac
