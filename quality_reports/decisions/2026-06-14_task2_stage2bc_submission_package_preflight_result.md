# Task2 Stage2BC Submission Package Preflight Result

Date: 2026-06-14  
Status: implemented and verified

## Purpose

Stage2BC adds a read-only preflight for Task2 submission package readiness. It
checks that the submission-facing inference entrypoint, hidden Stage2AQ wrapper,
no-GT proposal/ranker/export scripts, and required trained weights are present
before running inference or building a Docker image.

## Implemented

Added:

```text
scripts/task2/check_task2_submission_package.py
```

The script checks:

- `scripts/task2/run_task2_submission_inference.py`
- `scripts/task2/run_stage2aq_hidden_pipeline.py`
- `scripts/task2/export_yolo_nogt_proposals.py`
- `scripts/task2/export_sequence_tip_nogt_proposals.py`
- `scripts/task2/build_nogt_candidate_pool.py`
- `scripts/task2/infer_stage2u_quality_ranker.py`
- `scripts/task2/run_stage2_champion_pipeline.py`
- `scripts/task2/export_stage2ab_domain_policy_predictions.py`
- YOLO Stage2L weights
- Stage2X class1 checkpoint
- Stage2U quality-ranker checkpoint
- optional input image directory or image-list CSV
- optional output directory creatability

It writes a JSON manifest and returns nonzero under `--strict` if required
checks fail.

## Verification Command

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/check_task2_submission_package.py \
  --image-dir datasets/collision_detection/images \
  --output-dir outputs/task2/stage2bc_submission_package_preflight/example_output \
  --manifest quality_reports/decisions/2026-06-14_task2_submission_package_preflight.json \
  --strict
```

Result:

```text
failure_count=0
```

Manifest:

```text
quality_reports/decisions/2026-06-14_task2_submission_package_preflight.json
```

## Required Weight Checks

| Artifact | Size |
| --- | ---: |
| YOLO Stage2L weights | 19,255,962 bytes |
| Stage2X class1 checkpoint | 358,019,609 bytes |
| Stage2U quality-ranker checkpoint | 334,124,917 bytes |

## Test Verification

Focused checks:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile \
  scripts/task2/check_task2_submission_package.py \
  scripts/task2/run_task2_submission_inference.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_submission_package_preflight.py \
  tests/test_task2_submission_inference.py
```

Result: 6 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 112 passed.

## Remaining Gap

The preflight intentionally does not check a final official result formatter
because the available 2026 CATHACTION PDF does not specify the concrete Task2
result-file schema. Once official instructions are published, the formatter and
its required output path should be added to this preflight.

