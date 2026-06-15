# Task 2 Stage2L Combined Validation Started

Date: 2026-06-10

## Decision

Stop using the animal-only validation YAML for Stage2L checkpoint selection.
Use a combined held-out validation list instead:

- animal: `video_2_animal`, 107 frames;
- phantom: `valid_phantom_balanced_small`, 824 frames.

The combined validation panel has 931 frames total, with original class counts:

- class 0: 427;
- class 1: 504.

## Rationale

An animal-only validation set can select a checkpoint that improves animal
normal/collision proposals while degrading phantom behavior. Task 2 hidden data
may include multiple domains, and the final method cannot assume domain labels
are known at inference time. The combined panel gives the YOLO proposal detector
a checkpoint-selection signal from both held-out animal and held-out phantom
data, while still preserving separate panels for post-hoc domain diagnosis.

## Artifacts

- Split script: `scripts/task2/prepare_stage2l_proposal_splits.py`
- Split test: `tests/test_task2_stage2l_proposal_splits.py`
- Summary: `configs/task2/splits_stage2l_proposal/summary.json`
- Training YAML:
  `configs/task2/collision_detection_stage2l_agnostic_train_v0_v1_val_v2_valid_combined_balanced.local.yaml`
- Training log:
  `quality_reports/logs/task2_yolo11s_1024_stage2l_agnostic_train_v0_v1_val_v2_combined_bal_e20.log`
- Output directory:
  `outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20`

## Verification

- `python -m py_compile scripts/task2/prepare_stage2l_proposal_splits.py tests/test_task2_stage2l_proposal_splits.py`
- `python -m pytest -q tests/test_task2_stage2l_proposal_splits.py`
- Training startup log confirms:
  - CUDA is available;
  - `nc=1`, class name `tool_roi`;
  - train labels are scanned from `datasets/collision_detection_agnostic/labels`;
  - validation labels are scanned from `datasets/collision_detection_agnostic/labels`;
  - validation count is 931 images.

## Running Process

- PID: `2879379`
- Epochs: 20
- Image size: 1024
- Batch size: 32
- Validation: combined animal-v2 + phantom-balanced-small

