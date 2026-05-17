#!/usr/bin/env bash
# run_pipeline.sh — wait for GPU, then run preprocess_all + generate_motion_pairs.
#
# Usage:
#   nohup bash scripts/python/run_pipeline.sh > experiments/runs/pipeline.log 2>&1 &
#   tail -f experiments/runs/pipeline.log
#
# Environment overrides (all optional):
#   FREE_THRESHOLD_MB  — GPU free threshold (default 40000 = 40 GB)
#   CONSEC_HITS        — confirmed-idle polls (default 5)
#   POLL_INTERVAL      — seconds between polls (default 60)
#   PRE_LIMIT          — cap preprocess_all to N cases (debug; default unlimited)
#   PAIR_LIMIT         — cap generate_motion_pairs to N cases (debug; default unlimited)
#   PAIR_VARIANTS      — variants per case (default 4)
#   PAIR_N_VIEWS       — projection views (default 1000)
#   PAIR_N_PHASES      — cardiac phases (default 48)
#   CONDA_ENV          — conda env to activate (default cardiac-diffusion)
#
# After completion writes a sentinel: experiments/runs/pipeline_DONE
# Failure writes:                       experiments/runs/pipeline_FAILED

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="experiments/runs"
mkdir -p "$OUT_DIR"
SENTINEL_DONE="$OUT_DIR/pipeline_DONE"
SENTINEL_FAILED="$OUT_DIR/pipeline_FAILED"
rm -f "$SENTINEL_DONE" "$SENTINEL_FAILED"

CONDA_ENV="${CONDA_ENV:-cardiac-diffusion}"
PYTHON_BIN="/home/mingzhang/miniconda3/envs/${CONDA_ENV}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[run_pipeline] ERROR: python not found at $PYTHON_BIN" >&2
    touch "$SENTINEL_FAILED"
    exit 1
fi

# Phase 0 — wait for GPU
echo "[$(date)] Phase 0: waiting for GPU"
bash scripts/python/gpu_wait.sh
gpu_wait_rc=$?
if [[ $gpu_wait_rc -ne 0 ]]; then
    echo "[run_pipeline] gpu_wait.sh exited rc=$gpu_wait_rc — aborting" >&2
    touch "$SENTINEL_FAILED"
    exit "$gpu_wait_rc"
fi

# Phase 1 — TotalSegmentator + preprocess
echo "[$(date)] Phase 1: preprocess_all"
PRE_ARGS=()
if [[ -n "${PRE_LIMIT:-}" ]]; then PRE_ARGS+=(--limit "$PRE_LIMIT"); fi
"$PYTHON_BIN" -m scripts.python.preprocess_all --device gpu "${PRE_ARGS[@]}"
pre_rc=$?
if [[ $pre_rc -ne 0 && $pre_rc -ne 2 ]]; then
    # rc 2 = partial success (some cases failed); 0 = full success
    echo "[run_pipeline] preprocess_all FAILED rc=$pre_rc — aborting" >&2
    touch "$SENTINEL_FAILED"
    exit "$pre_rc"
fi
echo "[$(date)] Phase 1 rc=$pre_rc (0=full / 2=partial)"

# Brief pause so VRAM from TotalSegmentator clears before tomosipo
sleep 30

# Phase 2 — motion-pair generation
echo "[$(date)] Phase 2: generate_motion_pairs"
PAIR_ARGS=(--variants "${PAIR_VARIANTS:-4}" \
           --n-views "${PAIR_N_VIEWS:-1000}" \
           --n-phases "${PAIR_N_PHASES:-48}")
if [[ -n "${PAIR_LIMIT:-}" ]]; then PAIR_ARGS+=(--limit "$PAIR_LIMIT"); fi
"$PYTHON_BIN" -m scripts.python.generate_motion_pairs --device cuda "${PAIR_ARGS[@]}"
pair_rc=$?
echo "[$(date)] Phase 2 rc=$pair_rc"

# Final sentinel
if [[ $pre_rc -eq 0 && $pair_rc -eq 0 ]]; then
    touch "$SENTINEL_DONE"
    echo "[$(date)] Pipeline DONE → $SENTINEL_DONE"
    exit 0
else
    # Partial success counts as "completed with failures"
    touch "$SENTINEL_DONE"
    echo "[$(date)] Pipeline completed WITH FAILURES (preprocess_rc=$pre_rc, pair_rc=$pair_rc)" >&2
    echo "Check experiments/runs/preprocess_all/failed.txt and experiments/runs/generate_motion_pairs/failed.txt"
    exit 2
fi
