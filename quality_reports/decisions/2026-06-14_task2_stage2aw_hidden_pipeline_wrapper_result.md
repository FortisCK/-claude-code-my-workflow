# Task2 Stage2AW Hidden Stage2AQ Pipeline Wrapper Result

Date: 2026-06-14

## Decision

Add a dry-run-first hidden-test wrapper:

`scripts/task2/run_stage2aq_hidden_pipeline.py`

The wrapper connects the no-GT pieces into the Stage2AQ export path:

1. yolo-only proposals -> no-GT candidate pool;
2. multi-source proposals -> no-GT candidate pool;
3. yolo-only candidates -> Stage2U no-GT ranker inference;
4. multi-source candidates -> Stage2U no-GT ranker inference;
5. Stage2AE/Stage2AQ clean-prediction export.

It is dry-run by default and requires `--execute` to run commands.

## Why

Stage2AQ needs two branches at hidden-test time:

- a yolo-only branch used as the Stage2AE baseline;
- a multi-source branch used only for Stage2AQ phantom-class1 replacement.

Stage2AW makes this explicit in one manifest instead of relying on manually
remembered command sequences.

## Public-Validation Dry Run

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/run_stage2aq_hidden_pipeline.py \
  --split valid_combined \
  --split valid_phantom \
  --split valid_animal \
  --work-dir outputs/task2/stage2aw_hidden_pipeline/public_valid_dryrun \
  --manifest outputs/task2/stage2aw_hidden_pipeline/public_valid_dryrun_manifest.json \
  --device cpu \
  --workers 0 \
  --batch-size 2
```

Output:

`outputs/task2/stage2aw_hidden_pipeline/public_valid_dryrun_manifest.json`

Preflight result:

- missing required inputs: 0
- frozen Stage2U checkpoint found;
- default proposal source directories found;
- all public-validation `{split}_proposals.csv` files found;
- generated five commands in the expected order.

## Scope

This wrapper starts from precomputed proposal CSVs. It does not yet:

- run raw-image proposal generation from official input images;
- convert clean internal CSVs into the exact challenge result-file format;
- package the pipeline into Docker.

Those are the next submission-hardening tasks.

## Verification

Focused tests:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest \
  tests/test_task2_stage2aw_hidden_pipeline.py \
  tests/test_task2_stage2av_nogt_candidate_pool.py \
  tests/test_task2_stage2u_nogt_inference.py
```

Result: 7 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest tests/test_task2_*.py
```

Result: 96 passed.
