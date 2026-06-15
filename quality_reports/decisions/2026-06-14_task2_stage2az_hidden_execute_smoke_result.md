# Task2 Stage2AZ Hidden Raw-Image Execute Smoke Result

Date: 2026-06-14  
Status: implemented and verified

## Purpose

Stage2AY proved that the raw-image hidden wrapper can generate a correct dry-run
manifest. Stage2AZ verifies that the same wrapper can execute a tiny no-GT
pipeline end to end and produce a final Stage2AQ prediction CSV.

## Implemented

- Added `--proposal-limit` to `scripts/task2/run_stage2aq_hidden_pipeline.py`.
- The limit is passed only to raw-image proposal exporters:
  - `scripts/task2/export_yolo_nogt_proposals.py`
  - `scripts/task2/export_sequence_tip_nogt_proposals.py`
- The wrapper now also passes `--batch-size` to YOLO proposal export and
  `--batch-size` / `--workers` to Stage2X proposal export.
- Added `--fallback-domain` to `scripts/task2/run_stage2_champion_pipeline.py`
  and wired it from the hidden wrapper into the final Stage2AE/Stage2AQ export.

## Why This Was Needed

Two hidden/Docker-style failure modes were exposed during smoke testing:

1. Stage2X proposal export used its default multiprocessing workers, which is
   brittle in restricted/sandboxed environments. The wrapper now propagates the
   caller's worker count.
2. Final champion export requires known domain metadata. Raw hidden filenames
   may not encode `animal`, `phantom`, or `human`, so a wrapper-level
   `--fallback-domain` must be passed through to the final exporter when domain
   metadata is unavailable.

## Execute Smoke

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/run_stage2aq_hidden_pipeline.py \
  --generate-proposals \
  --proposal-limit 2 \
  --raw-image-dir datasets/collision_detection/images \
  --split hidden_smoke \
  --work-dir outputs/task2/stage2az_hidden_execute_smoke/limit2_cpu_phantom \
  --manifest outputs/task2/stage2az_hidden_execute_smoke/limit2_cpu_phantom_manifest.json \
  --device cpu \
  --workers 0 \
  --batch-size 2 \
  --fallback-domain phantom \
  --execute
```

Result: completed successfully.

Key row counts including CSV headers:

| Artifact | Rows including header |
| --- | ---: |
| YOLO proposals | 17 |
| Stage2X proposals | 201 |
| yolo-only candidates | 17 |
| multisource candidates | 117 |
| yolo-only Stage2U rows | 17 |
| multisource Stage2U rows | 117 |
| final Stage2AQ predictions | 27 |

The final Stage2AQ prediction CSV is:

```text
outputs/task2/stage2az_hidden_execute_smoke/limit2_cpu_phantom/
  hidden_stage2ae_stage2aq_predictions/stage2aq/
  hidden_smoke_domain_policy_predictions.csv
```

It contains 26 prediction rows for the 2-image smoke input.

## GPU Note

The current Codex execution sandbox could not access the GPU driver:

- `nvidia-smi` failed to communicate with the NVIDIA driver.
- `torch.cuda.is_available()` returned `False` in both `cathaction-task1` and
  `cardiac-diffusion`.

Therefore this smoke was run on CPU. This does not change the project
assumption that the workstation GPU can be used from a correctly configured
interactive environment; it only records the limitation of this execution
context.

## Regression Verification

Focused checks:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile \
  scripts/task2/run_stage2aq_hidden_pipeline.py \
  scripts/task2/run_stage2_champion_pipeline.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_stage2aw_hidden_pipeline.py \
  tests/test_task2_champion_pipeline.py
```

Result: 9 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 106 passed.

## Remaining Gap

The pipeline now executes from raw images to internal Stage2AQ prediction CSV.
The remaining challenge-submission gap is still the official Task2 result-file
schema, which is not specified in the available 2026 PDF. Once the official
validation package or platform instructions are released, add a final formatter
from the internal CSV schema to the required official file.

