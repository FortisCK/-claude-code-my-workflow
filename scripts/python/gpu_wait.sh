#!/usr/bin/env bash
# gpu_wait.sh — block until the GPU is reliably free.
#
# Polls every POLL_INTERVAL seconds. Triggers (exits 0) once `memory.free`
# has been ≥ FREE_THRESHOLD_MB for CONSEC_HITS consecutive polls AND the
# foreign-process count is 0. The consec-hits guard prevents triggering on
# a transient memory dip during another job's evaluation pass.
#
# Usage:
#   bash scripts/python/gpu_wait.sh                  # defaults below
#   FREE_THRESHOLD_MB=40000 CONSEC_HITS=5 POLL_INTERVAL=60 bash scripts/python/gpu_wait.sh
#
# Notes:
#   - Single-GPU server (RTX 6000 Ada, 49 GB).
#   - Default 40 GB free × 5 polls × 60 s = 5 min of confirmed idleness.

set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"
FREE_THRESHOLD_MB="${FREE_THRESHOLD_MB:-40000}"
CONSEC_HITS="${CONSEC_HITS:-5}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

echo "[gpu_wait] GPU=$GPU_INDEX  threshold=${FREE_THRESHOLD_MB} MiB free  consec=$CONSEC_HITS  interval=${POLL_INTERVAL}s"

hits=0
poll_count=0
while true; do
    poll_count=$((poll_count + 1))
    # memory.free in MiB
    free_mb=$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits)
    # Number of LIVE compute processes on this GPU. nvidia-smi keeps reporting
    # ghost PIDs whose CUDA context wasn't released cleanly (process_name shows
    # as "[Not Found]") — those don't actually hold the GPU and must not block
    # us. We filter by checking /proc/<pid>/status existence.
    n_proc=0
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if [[ -e "/proc/$pid/status" ]]; then
            n_proc=$((n_proc + 1))
        fi
    done < <(nvidia-smi --id="$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

    ts=$(date +'%Y-%m-%d %H:%M:%S')
    if (( free_mb >= FREE_THRESHOLD_MB )) && (( n_proc == 0 )); then
        hits=$((hits + 1))
        echo "[$ts] [gpu_wait] poll=$poll_count  free=${free_mb} MiB  n_proc=$n_proc  hits=$hits/$CONSEC_HITS"
        if (( hits >= CONSEC_HITS )); then
            echo "[$ts] [gpu_wait] threshold sustained — releasing."
            exit 0
        fi
    else
        if (( hits > 0 )); then
            echo "[$ts] [gpu_wait] poll=$poll_count  free=${free_mb} MiB  n_proc=$n_proc  hits reset 0"
        elif (( poll_count % 10 == 0 )); then
            # Quiet log: only every 10th idle-poll while waiting
            echo "[$ts] [gpu_wait] poll=$poll_count  free=${free_mb} MiB  n_proc=$n_proc  (waiting)"
        fi
        hits=0
    fi
    sleep "$POLL_INTERVAL"
done
