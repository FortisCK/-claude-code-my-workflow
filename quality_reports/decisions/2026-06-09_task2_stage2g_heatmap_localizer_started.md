# Task 2 Stage2G Heatmap Localizer Started

Date: 2026-06-09

## Why

Stage2F showed that existing YOLO candidate pools are recoverable for class0
and phantom class1, but fail completely on the stratified animal class1 sample:
top50 IoU>=0.25 recall was 0.0. A reranker cannot recover missing candidates,
so the first-stage localizer must change.

## Actions

Stopped the unproductive collision-only YOLO run:

```text
outputs/task2/yolo_collision/yolo11s_1024_collision_only_clean_combined_e80/
```

It reached epoch 27 with mAP50 about 0.00116 and mAP50-95 about 0.00038.

Added:

```text
src/cathaction/data/task2_heatmap.py
tests/test_task2_heatmap.py
scripts/task2/train_heatmap_localizer.py
```

Verification:

```text
python3 -m py_compile scripts/task2/train_heatmap_localizer.py src/cathaction/data/task2_heatmap.py tests/test_task2_heatmap.py
conda run -n cathaction-task1 pytest -q tests/test_task2_heatmap.py tests/test_task2_dataset.py
```

Smoke run passed:

```text
outputs/task2/heatmap_localizer/smoke_cpu_i64_e1/
```

## Current Run

Background PID:

```text
3316060
```

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/train_heatmap_localizer.py \
  --device auto \
  --epochs 12 \
  --input-size 256 \
  --channels 16,32,64,128 \
  --strides 2,2,2 \
  --batch-size 64 \
  --workers 8 \
  --valid-samples-per-class 256 \
  --primary-metric valid_animal/class1_recall_iou_0.25 \
  --name monai_unet_heatmap256_stage2g_e12
```

Log:

```text
quality_reports/logs/task2_heatmap_monai_unet256_stage2g_e12.log
```

Output:

```text
outputs/task2/heatmap_localizer/monai_unet_heatmap256_stage2g_e12/
```

## Readout

Primary diagnostic metric:

```text
valid_animal/class1_recall_iou_0.25
```

The route is promising only if animal class1 recall/mean IoU becomes non-zero.
