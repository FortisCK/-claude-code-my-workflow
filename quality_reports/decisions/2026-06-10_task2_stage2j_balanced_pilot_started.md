# Task2 Stage2J Balanced Pilot Started

Date: 2026-06-10

## What Changed

Added reproducible Stage2J split generation for domain-balanced Task 2 training:

- Script: `scripts/task2/prepare_stage2j_balanced_splits.py`
- Test: `tests/test_task2_stage2j_balanced_splits.py`
- Plan: `quality_reports/plans/2026-06-10_task2-stage2j-domain-balanced-training-panels.md`
- Generated split directory: `configs/task2/splits_stage2j_balanced/`
- Generated summary: `configs/task2/splits_stage2j_balanced/summary.json`

Validation command:

```bash
conda run -n cathaction-task1 pytest -q tests/test_task2_stage2j_balanced_splits.py
```

Result: `1 passed`.

## Generated Panels

The Stage2J generator does not move raw files. It writes deterministic list files and YOLO YAMLs.

Shared validation panels:

- `valid_phantom_full`: 11,024 phantom samples, class0=10,612, class1=412.
- `valid_phantom_balanced_small`: 824 phantom samples, class0=412, class1=412.

Pilot folds:

- `pilot_train_v1_val_v2`
  - train: 10,000 phantom samples plus `video_1_animal` repeated 50x.
  - train records: 19,150.
  - held-out animal val: `video_2_animal`, 107 samples.
- `pilot_train_v2_val_v1`
  - train: 10,000 phantom samples plus `video_2_animal` repeated 50x.
  - train records: 15,350.
  - held-out animal val: `video_1_animal`, 183 samples.

## Started Run

Started the first short pilot:

- Fold: `pilot_train_v1_val_v2`
- YAML: `configs/task2/collision_detection_stage2j_pilot_train_v1_val_v2_animal.local.yaml`
- Output: `outputs/task2/yolo_stage2j_balanced/yolo11s_1024_pilot_v1_val_v2_e15`
- Log: `quality_reports/logs/task2_yolo11s_1024_stage2j_pilot_v1_val_v2_e15.log`
- PID at launch: `3974991`

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_yolo.py \
  --model yolo11s.pt \
  --data configs/task2/collision_detection_stage2j_pilot_train_v1_val_v2_animal.local.yaml \
  --epochs 15 \
  --imgsz 1024 \
  --batch 32 \
  --workers 8 \
  --project outputs/task2/yolo_stage2j_balanced \
  --name yolo11s_1024_pilot_v1_val_v2_e15 \
  --patience 8 \
  --close-mosaic 3 \
  --device 0
```

Startup check passed:

- CUDA visible.
- GPU memory around 20.5 GB during training.
- Training list scanned: 19,150 images, 0 corrupt.
- Held-out animal validation scanned: 107 images, 0 corrupt.

## Next Checks

After the run finishes, validate the best checkpoint on:

1. held-out animal validation from the training log;
2. `configs/task2/collision_detection_stage2j_pilot_train_v1_val_v2_valid_phantom_balanced_small.local.yaml`;
3. `configs/task2/collision_detection_stage2j_pilot_train_v1_val_v2_valid_phantom_full.local.yaml`.

Decision criterion:

- Continue Stage2J if animal mAP improves while phantom does not collapse.
- Stop Stage2J if it only reproduces Stage2H animal specialization.
