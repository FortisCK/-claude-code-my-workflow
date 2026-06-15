# Task 2 Stage2I Reverse Fold Started and Phantom Sanity Check

Date: 2026-06-10

## Purpose

Stage2I checks two follow-up questions from Stage2H:

1. Does animal adaptation work in the reverse direction?
2. Does the Stage2H animal-adapted model preserve phantom-domain performance?

## Phantom Sanity Check

Model:

```text
outputs/task2/yolo_adapt_animal/yolo11s_1024_train_v1_val_v2_e40/weights/best.pt
```

Validation YAML:

```text
configs/task2/collision_detection_clean_valid_phantom.local.yaml
```

Output directory:

```text
runs/detect/outputs/task2/yolo_adapt_animal_eval/stage2h_best_on_valid_phantom
```

Result on `valid_phantom`:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 11024 | 11024 | 0.152 | 0.114 | 0.0722 | 0.0259 |
| normal | 10612 | 10612 | 0.304 | 0.229 | 0.143 | 0.0514 |
| collision | 412 | 412 | 0.000 | 0.000 | 0.00149 | 0.000395 |

Interpretation:

- Stage2H adaptation strongly improves held-out animal collision.
- It does not preserve phantom performance.
- This supports domain-specific behavior and rules out a naive “train on a little animal and use directly everywhere” strategy.
- A final route needs domain-balanced training/calibration rather than pure animal specialization.

## Reverse Fold Training

Started PID:

```text
3769168
```

Command:

```bash
setsid env YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/matplotlib-codex WANDB_DISABLED=true \
  /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_yolo.py \
  --model yolo11s.pt \
  --data configs/task2/collision_detection_adapt_animal_train_v2_val_v1.local.yaml \
  --epochs 40 \
  --imgsz 1024 \
  --batch 32 \
  --workers 8 \
  --project outputs/task2/yolo_adapt_animal \
  --name yolo11s_1024_train_v2_val_v1_e40 \
  --patience 10 \
  --close-mosaic 5
```

Log:

```text
quality_reports/logs/task2_yolo11s_1024_adapt_animal_train_v2_val_v1_e40.log
```

Output directory:

```text
outputs/task2/yolo_adapt_animal/yolo11s_1024_train_v2_val_v1_e40
```

Split:

- Train: original `train_clean` plus `video_2_animal`
- Validation: held-out `video_1_animal`
- Train sample count: 35,191
- Validation sample count: 183

Startup status:

- CUDA available: yes
- Model: YOLO11s pretrained checkpoint, 2 output classes
- Training and validation label scans completed without corrupt samples

