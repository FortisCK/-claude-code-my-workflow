# Task 2 YOLO11s Baseline Started

Date: 2026-06-04

## Status

Task 2 baseline training has been started in a detached `screen` session.

This is the first serious detector baseline after Task 1 was frozen. It uses a
modern public YOLO checkpoint rather than a frame classifier, because the local
Task 2 labels are one-box-per-frame YOLO annotations and the CathAction paper
reports AP/mAP object-detection metrics.

## Active Run

Screen session:

```bash
screen -ls
```

Expected active session:

```text
2739466.task2_yolo11s_1024
```

Training log:

```bash
tail -f quality_reports/logs/task2_yolo11s_1024_clean_combined_e100.log
```

Output directory:

```text
outputs/task2/yolo/yolo11s_1024_clean_combined_e100/
```

Main checkpoints once training progresses:

```text
outputs/task2/yolo/yolo11s_1024_clean_combined_e100/weights/best.pt
outputs/task2/yolo/yolo11s_1024_clean_combined_e100/weights/last.pt
```

## Command

The run was started with:

```bash
screen -dmS task2_yolo11s_1024 bash -lc 'cd /home/mingzhang/cathaction && env PYTHONUNBUFFERED=1 /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_yolo.py --model yolo11s.pt --data configs/task2/collision_detection_clean_combined.local.yaml --epochs 100 --imgsz 1024 --batch 32 --workers 8 --device 0 --project outputs/task2/yolo --name yolo11s_1024_clean_combined_e100 --patience 30 --close-mosaic 10 > quality_reports/logs/task2_yolo11s_1024_clean_combined_e100.log 2>&1'
```

Configuration:

| Item | Value |
| --- | --- |
| Model | `yolo11s.pt` |
| Pretraining | public Ultralytics checkpoint |
| Data YAML | `configs/task2/collision_detection_clean_combined.local.yaml` |
| Train split | clean train = raw `train_phantom` minus validation overlaps |
| Validation split | combined `valid_phantom + valid_animal` |
| Image size | `1024` |
| Batch size | `32` |
| Epochs | `100` |
| Device | `cuda:0` |
| GPU | NVIDIA RTX 6000 Ada Generation, 49GB |

## Startup Verification

Completed before the full run:

- Task 2 parser unit tests: `4 passed`.
- Split generation: clean train, valid phantom, valid animal, combined val, and tiny smoke YAMLs generated.
- CPU tiny YOLO smoke:
  - `64` train / `64` val;
  - `0` corrupt labels/images;
  - completed successfully.
- GPU tiny YOLO smoke:
  - model `yolo11s.pt`;
  - `imgsz=256`;
  - `64` train / `64` val;
  - CUDA visible and training completed successfully.
- Full run startup:
  - train scan: `35084` images, `0` backgrounds, `0` corrupt;
  - val scan: `11422` images, `0` backgrounds, `0` corrupt;
  - GPU memory at startup: about `19GB / 49GB`.

## Environment Notes

The regular sandboxed shell cannot see CUDA/NVML:

```text
torch.cuda.is_available() == False
```

The host shell outside the sandbox can see the GPU:

```text
cuda_available True
device_count 1
NVIDIA RTX 6000 Ada Generation
```

Therefore, long GPU training must be launched outside the sandbox, currently
via `screen`.

`cardiac-diffusion` now has:

| Package | Version |
| --- | --- |
| `torch` | `2.11.0+cu128` |
| `ultralytics` | `8.4.60` |
| `opencv-python` | `4.10.0` |
| `numpy` | `1.26.4` |

`opencv-python-headless` was removed after it left `cv2` as a broken namespace.

## Follow-Up After Training

1. Inspect `results.csv` and identify best epoch by `metrics/mAP50-95(B)`.
2. Evaluate `best.pt` separately on:
   - `configs/task2/collision_detection_clean_valid_phantom.local.yaml`;
   - `configs/task2/collision_detection_clean_valid_animal.local.yaml`.
3. Generate validation overlays for true positives, false positives, and class confusions.
4. Decide next baseline:
   - `YOLO11m` or `YOLO11s` at `1280`;
   - temporal smoothing / adjacent-frame fusion;
   - Task1 segmentation-guided tip prior.

