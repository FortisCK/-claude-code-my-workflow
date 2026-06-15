# Task2 Stage2AY Hidden Raw-Image Wrapper Result

Date: 2026-06-14  
Status: implemented and verified

## Purpose

Stage2AW could dry-run the hidden pipeline only when no-GT proposal CSVs already existed. Stage2AY extends it so a hidden/public-validation run can start from raw images and produce the current Stage2AQ prediction CSV without any ground-truth fields.

## Implemented

- Updated `scripts/task2/run_stage2aq_hidden_pipeline.py`.
- Added `--generate-proposals` mode.
- Added raw input options:
  - `--raw-image-dir`
  - `--raw-image-list-csv`
- Added default hidden-ready proposal generators:
  - YOLO Stage2L: `outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt`
  - Stage2X class1 tip localizer: `outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt`
- The generated command chain is:
  1. `export_yolo_stage2l_nogt_proposals_<split>`
  2. `export_stage2x_class1_nogt_proposals_<split>`
  3. `build_yolo_only_nogt_candidate_pool`
  4. `build_multisource_nogt_candidate_pool`
  5. `infer_yolo_only_stage2u`
  6. `infer_multisource_stage2u`
  7. `export_stage2ae_stage2aq_hidden_predictions`

## Hidden/Docker Readiness Fix

In raw-image mode, the candidate-pool commands now use the provided `--raw-image-dir` as their fallback `--image-dir`. The preflight no longer treats the local default `datasets/collision_detection/images` as a hard dependency when proposals are generated from raw hidden images. This matters because a Docker/hidden-test mount may use a different input path.

## Dry-Run Verification

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/run_stage2aq_hidden_pipeline.py \
  --generate-proposals \
  --raw-image-dir datasets/collision_detection/images \
  --split hidden \
  --work-dir outputs/task2/stage2ay_hidden_raw_image_pipeline/public_images_dryrun \
  --manifest outputs/task2/stage2ay_hidden_raw_image_pipeline/public_images_dryrun_manifest.json \
  --device cpu \
  --workers 0 \
  --batch-size 2
```

Result:

- Manifest: `outputs/task2/stage2ay_hidden_raw_image_pipeline/public_images_dryrun_manifest.json`
- `missing_required_inputs`: empty
- Preflight passed:
  - Stage2U checkpoint
  - YOLO weights
  - Stage2X checkpoint
  - raw image directory

## Regression Verification

Focused checks:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile \
  scripts/task2/run_stage2aq_hidden_pipeline.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_stage2aw_hidden_pipeline.py
```

Result: 3 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 106 passed.

## Official Submission-Format Finding

The 2026 CATHACTION challenge PDF states that participants will submit Docker containers and a result file in a predefined format, and that detailed submission instructions will include data input/output structure and evaluation software. It also states that validation data is scheduled for release on 2026-07-10 and the submission deadline is 2026-08-23.

However, the available PDF does not define the concrete Task2 result-file schema: no required column names, JSON/CSV layout, confidence-field name, coordinate convention, or directory name is specified. Therefore, Stage2AY intentionally stops at our internal clean no-GT prediction CSV. A final official formatter should be added once the official validation package or platform instructions are published.

## Current Practical Output

The current internal hidden-ready output target is:

```text
<work-dir>/<prediction-name>/stage2aq/<split>_domain_policy_predictions.csv
```

For the dry run above:

```text
outputs/task2/stage2ay_hidden_raw_image_pipeline/public_images_dryrun/
  hidden_stage2ae_stage2aq_predictions/stage2aq/hidden_domain_policy_predictions.csv
```

This file is not claimed to be the final official submission format yet.

