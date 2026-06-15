# Task2 Stage2BA Submission Inference Entrypoint Result

Date: 2026-06-14  
Status: implemented and verified

## Purpose

Stage2BA wraps the verified raw-image Stage2AQ hidden pipeline behind a
submission-facing command. The intent is to make the future Docker entrypoint
simple and stable:

```bash
python scripts/task2/run_task2_submission_inference.py \
  --image-dir <official_input_images> \
  --output-dir <official_output_dir> \
  --execute
```

## Implemented

Added:

```text
scripts/task2/run_task2_submission_inference.py
```

The script:

- accepts either `--image-dir` or `--image-list-csv`;
- writes a submission-facing manifest;
- builds the raw-image Stage2AQ hidden command with `--generate-proposals`;
- supports `--proposal-limit` for smoke tests;
- propagates `--fallback-domain`, `--device`, `--batch-size`, and `--workers`;
- after execution, copies the internal Stage2AQ prediction CSV to a stable
  output filename:

```text
<output-dir>/task2_predictions_internal.csv
```

## Dry-Run Verification

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/run_task2_submission_inference.py \
  --image-dir datasets/collision_detection/images \
  --output-dir outputs/task2/stage2ba_submission_entrypoint/dryrun \
  --split hidden \
  --device cpu \
  --batch-size 2 \
  --workers 0 \
  --proposal-limit 2
```

Result:

- Manifest: `outputs/task2/stage2ba_submission_entrypoint/dryrun/task2_submission_inference_manifest.json`
- Mode: `dry_run`
- Generated command includes:
  - `scripts/task2/run_stage2aq_hidden_pipeline.py`
  - `--generate-proposals`
  - `--raw-image-dir`
  - `--fallback-domain phantom`
  - `--proposal-limit 2`
  - stable internal result path

## Test Verification

Focused checks:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile \
  scripts/task2/run_task2_submission_inference.py \
  scripts/task2/run_stage2aq_hidden_pipeline.py \
  scripts/task2/run_stage2_champion_pipeline.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_submission_inference.py \
  tests/test_task2_stage2aw_hidden_pipeline.py \
  tests/test_task2_champion_pipeline.py
```

Result: 12 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 109 passed.

## Official Format Status

The script does not claim to generate the official challenge result file. The
available 2026 CATHACTION PDF says a predefined result file will be required,
but does not define the concrete Task2 schema. Until the official validation
package or platform instructions are released, this entrypoint exports the
current internal Stage2AQ prediction CSV and records that caveat in its
manifest.

## Remaining Work

Once the official schema is published, add:

```text
scripts/task2/format_task2_official_submission.py
```

and call it from `run_task2_submission_inference.py` after
`task2_predictions_internal.csv` is produced.

