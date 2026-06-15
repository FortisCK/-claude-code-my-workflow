# Task 2 Stage2C Class-Agnostic Proposal Detector Started

Date: 2026-06-08

## Purpose

Stage2B showed that the Task 2 two-stage path is blocked mainly by coarse
localization. The GT-ROI ConvNeXt classifier is strong, but YOLO-derived ROIs
are often mislocalized. Stage2C starts a class-agnostic proposal detector so
YOLO learns only to localize the tool/contact ROI; normal/collision
classification remains a second-stage ROI task.

## New Artifacts

Plan:

- `quality_reports/plans/2026-06-08_task2-stage2c-class-agnostic-proposal-detector.md`

Data preparation:

- `scripts/task2/prepare_class_agnostic_yolo.py`
- `tests/test_task2_class_agnostic_yolo.py`
- `datasets/collision_detection_agnostic/`
  - `images` is a symlink to `datasets/collision_detection/images`
  - `labels` contains rewritten one-class labels
- `configs/task2/splits_agnostic/`
- `configs/task2/collision_detection_agnostic_clean_combined.local.yaml`
- `configs/task2/collision_detection_agnostic_clean_valid_phantom.local.yaml`
- `configs/task2/collision_detection_agnostic_clean_valid_animal.local.yaml`
- `configs/task2/collision_detection_agnostic_clean_smoke.local.yaml`

Proposal evaluation:

- `scripts/task2/evaluate_yolo_proposals.py`
- `tests/test_task2_yolo_proposals.py`

## Data Generation Result

Command:

```bash
python3 scripts/task2/prepare_class_agnostic_yolo.py
```

Summary:

- labels written: 46,506
- all rewritten labels have class id `0`
- original class counts:
  - class 0: 28,369
  - class 1: 18,137
- train clean samples: 35,084
- valid combined samples: 11,422
- valid phantom samples: 11,024
- valid animal samples: 398

Sanity checks:

- `configs/task2/collision_detection_agnostic_clean_combined.local.yaml` has
  only one class: `0: tool_roi`
- image lists point to `datasets/collision_detection_agnostic/images/...`
- label lists point to `datasets/collision_detection_agnostic/labels/...`
- `datasets/collision_detection_agnostic/images` is a symlink to the original
  image directory, so images are not duplicated

## Verification

Commands:

```bash
python3 -m py_compile scripts/task2/prepare_class_agnostic_yolo.py scripts/task2/train_yolo.py
conda run -n cathaction-task1 pytest -q tests/test_task2_class_agnostic_yolo.py tests/test_task2_dataset.py
python3 -m py_compile scripts/task2/evaluate_yolo_proposals.py
conda run -n cathaction-task1 pytest -q tests/test_task2_yolo_proposals.py tests/test_task2_class_agnostic_yolo.py
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_yolo.py --model yolo11n.yaml --data configs/task2/collision_detection_agnostic_clean_smoke.local.yaml --epochs 1 --imgsz 320 --batch 8 --workers 2 --project outputs/task2/yolo_proposal --name yolo11n_320_agnostic_smoke --patience 1 --close-mosaic 0 --no-plots
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_proposals.py --weights outputs/task2/yolo_proposal/yolo11n_320_agnostic_smoke/weights/best.pt --name yolo11n_320_agnostic_smoke_eval64 --imgsz 320 --limit 64 --batch-size 16 --source-chunk-size 64 --top-k 1,5,10 --progress-every 64
```

Results:

- Python compilation passed.
- Unit tests passed.
- YOLO smoke training read agnostic labels successfully.
- Proposal evaluator smoke completed and wrote:
  - `outputs/task2/yolo_proposal_eval/yolo11n_320_agnostic_smoke_eval64/`

## Full Training Started

Run:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_yolo.py \
  --model yolo11s.pt \
  --data configs/task2/collision_detection_agnostic_clean_combined.local.yaml \
  --epochs 100 \
  --imgsz 1024 \
  --batch 32 \
  --workers 8 \
  --project outputs/task2/yolo_proposal \
  --name yolo11s_1024_agnostic_clean_combined_e100 \
  --patience 30 \
  --close-mosaic 10
```

Launched in detached `screen` with stdout/stderr redirected to:

```text
quality_reports/logs/task2_yolo11s_1024_agnostic_clean_combined_e100.log
```

Output directory:

```text
outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/
```

Status at launch check:

- CUDA visible in `cardiac-diffusion`
- GPU memory: about 20.4 GB
- GPU utilization: about 95%
- process name: `.../cardiac-diffusion/bin/python`
- PID reported by `nvidia-smi`: `814525`
- training entered epoch 1

Note: `screen -ls` reported the socket as `Dead ???`, but the Python process
continued running and the log kept updating. Monitor through the log and
`nvidia-smi` rather than relying on reattaching to the screen.

## Monitoring

Use:

```bash
tail -f quality_reports/logs/task2_yolo11s_1024_agnostic_clean_combined_e100.log
nvidia-smi
```

When training finishes, evaluate proposal recall with:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_proposals.py \
  --weights outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt \
  --name yolo11s_1024_agnostic_clean_combined_e100 \
  --imgsz 1024 \
  --top-k 1,3,5,10 \
  --batch-size 32 \
  --source-chunk-size 64 \
  --progress-every 1000
```

Primary comparison targets from Stage2B:

- top1 `valid_combined` IoU>=0.5 recall: 0.3810
- top5 `valid_combined` IoU>=0.5 recall: 0.4931

If class-agnostic proposal recall improves materially, the next stage is
Stage2D: class-agnostic YOLO topK proposals + ConvNeXt ROI classifier using
`proposal_conf * ROI_prob`.
