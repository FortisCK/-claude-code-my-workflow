# Task 2 Stage2E Collision-Specialist Detector Started

Date: 2026-06-09

## Goal

Start a collision-specialist detector to directly address the current Task 2
failure mode: class-1/collision proposal recall is very low, especially on the
animal validation subset.

## Implementation

Added:

- `scripts/task2/prepare_collision_specialist_yolo.py`
- `tests/test_task2_collision_specialist_yolo.py`

Generated:

- Dataset mirror: `datasets/collision_detection_collision_only/`
- Splits: `configs/task2/splits_collision_only/`
- YAMLs:
  - `configs/task2/collision_detection_collision_only_clean_combined.local.yaml`
  - `configs/task2/collision_detection_collision_only_clean_valid_phantom.local.yaml`
  - `configs/task2/collision_detection_collision_only_clean_valid_animal.local.yaml`
  - `configs/task2/collision_detection_collision_only_clean_smoke.local.yaml`

Label policy:

- Source class `1` is rewritten to single YOLO class `0 collision_roi`.
- Source class `0` is rewritten to an empty YOLO label file and used as a
  negative/background image.

Generated split summary:

| Split | Base samples | Collision positives | Empty negatives |
| --- | ---: | ---: | ---: |
| train_clean | 35,084 | 17,453 | 17,631 |
| valid_combined | 11,422 | 684 | 10,738 |
| valid_phantom | 11,024 | 412 | 10,612 |
| valid_animal | 398 | 272 | 126 |

## Verification

Passed:

```bash
python3 -m py_compile scripts/task2/prepare_collision_specialist_yolo.py scripts/task2/train_yolo.py
conda run -n cathaction-task1 pytest -q tests/test_task2_collision_specialist_yolo.py tests/test_task2_class_agnostic_yolo.py tests/test_task2_dataset.py
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_yolo.py \
  --model yolo11n.yaml \
  --data configs/task2/collision_detection_collision_only_clean_smoke.local.yaml \
  --epochs 1 \
  --imgsz 320 \
  --batch 16 \
  --workers 2 \
  --project outputs/task2/yolo_collision \
  --name yolo11n_320_collision_only_smoke \
  --patience 1 \
  --close-mosaic 0 \
  --no-plots
```

The smoke run confirmed that Ultralytics accepts empty-label background images:
`64 images, 32 backgrounds, 0 corrupt`.

## Training Started

Run:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_yolo.py \
  --model yolo11s.pt \
  --data configs/task2/collision_detection_collision_only_clean_combined.local.yaml \
  --epochs 80 \
  --imgsz 1024 \
  --batch 32 \
  --workers 8 \
  --project outputs/task2/yolo_collision \
  --name yolo11s_1024_collision_only_clean_combined_e80 \
  --patience 20 \
  --close-mosaic 10
```

Log:

- `quality_reports/logs/task2_yolo11s_1024_collision_only_clean_combined_e80.log`

Output:

- `outputs/task2/yolo_collision/yolo11s_1024_collision_only_clean_combined_e80/`

The log confirmed full training started:

- train scan: `35084 images, 17631 backgrounds, 0 corrupt`
- val scan: `11422 images, 10738 backgrounds, 0 corrupt`
- GPU memory: about `19.4G`
- epoch 1 entered batch training

## Next Evaluation

After the first useful checkpoint is available, evaluate proposal recall with:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_proposals.py \
  --data-root datasets/collision_detection \
  --weights outputs/task2/yolo_collision/yolo11s_1024_collision_only_clean_combined_e80/weights/best.pt \
  --name yolo11s_1024_collision_only_clean_combined_e80 \
  --imgsz 1024 \
  --top-k 1,3,5,10 \
  --batch-size 32 \
  --source-chunk-size 64 \
  --progress-every 1000
```

Primary diagnostics:

- combined class-1 top5 IoU>=0.5, previous class-agnostic value: `0.0365`
- animal class-1 top5 IoU>=0.5, previous class-agnostic value: `0.0`
- overall collision AP/mAP after merging with the existing normal detector, if
  proposal recall improves
