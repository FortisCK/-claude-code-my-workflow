# Task 2 Stage2L Animal-Normal-Aware Proposal Started

Date: 2026-06-10

## Objective

Train a class-agnostic YOLO proposal detector that includes animal normal
examples in training.

## Generated Artifacts

- Plan:
  - `quality_reports/plans/2026-06-10_task2-stage2l-animal-normal-proposal.md`
- Split script:
  - `scripts/task2/prepare_stage2l_proposal_splits.py`
- Test:
  - `tests/test_task2_stage2l_proposal_splits.py`
- Split directory:
  - `configs/task2/splits_stage2l_proposal/`
- Main training YAML:
  - `configs/task2/collision_detection_stage2l_agnostic_train_v0_v1_val_v2_animal.local.yaml`

## Split

Train:

- original `train_clean` phantom data
- `video_0_animal` normal procedure
- `video_1_animal` mostly-collision procedure

Validation:

- `video_2_animal`

Training panel:

- total: 35,375 samples
- animal: 291
- phantom: 35,084
- original class counts: normal 17,742, collision 17,633
- agnostic class counts: class 0 `tool_roi` = 35,375

Validation panel:

- total: 107 samples
- original class counts: normal 15, collision 92
- agnostic class counts: class 0 `tool_roi` = 107

## Verification

Commands:

```bash
python3 -m py_compile scripts/task2/prepare_stage2l_proposal_splits.py
conda run -n cathaction-task1 pytest -q tests/test_task2_stage2l_proposal_splits.py
```

Result:

- `1 passed`

## Training

An initial run was stopped because generated image paths resolved through the
symlink to the original two-class dataset, causing Ultralytics to read original
labels. The split writer was fixed to preserve the agnostic image root.

Current run:

- PID: 2878468
- log: `quality_reports/logs/task2_yolo11s_1024_stage2l_agnostic_train_v0_v1_val_v2_e20_fix1.log`
- output: `outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_e20_fix1`
- model: `yolo11s.pt`
- epochs: 20
- image size: 1024
- batch: 32
- validation: `video_2_animal`

Startup check:

- training labels scanned from `datasets/collision_detection_agnostic/labels`
- validation labels scanned from `datasets/collision_detection_agnostic/labels`
- no label class-count errors after the path fix
