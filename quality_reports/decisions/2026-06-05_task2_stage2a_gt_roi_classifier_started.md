# Task 2 Stage2A GT-ROI Classifier Started

Date: 2026-06-05

## Status

Stage2A GT-ROI collision classifier is running in a detached `screen` session.

This is a diagnostic experiment for the two-stage Task 2 direction:

1. use known GT bbox to crop a tip-centered ROI;
2. train a binary classifier for `normal` vs `collision`;
3. measure whether collision state is learnable once localization is removed
   from the problem.

This does not replace the final detector yet. It measures the classification
upper-bound under GT localization.

## Active Run

Screen session:

```bash
screen -ls
```

Expected active session:

```text
task2_roi_convnext_tiny
```

Training log:

```bash
tail -f quality_reports/logs/task2_roi_convnext_tiny_gtroi224_scale8_e30.log
```

Output directory:

```text
outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/
```

Checkpoints:

```text
outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/best.pt
outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/last.pt
```

## Command

```bash
screen -dmS task2_roi_convnext_tiny bash -lc 'cd /home/mingzhang/cathaction && env PYTHONUNBUFFERED=1 /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/train_roi_classifier.py --name convnext_tiny_gtroi224_scale8_e30 --model convnext_tiny --pretrained --epochs 30 --batch-size 128 --workers 8 --device auto --input-size 224 --crop-scale 8 --min-crop-size 224 --max-crop-size 512 --lr 0.0003 --weight-decay 0.0001 --balanced-sampler --class-weight > quality_reports/logs/task2_roi_convnext_tiny_gtroi224_scale8_e30.log 2>&1'
```

## Configuration

| Item | Value |
| --- | --- |
| Model | `convnext_tiny` |
| Backend | `timm` via `auto` |
| Pretraining | public ImageNet ConvNeXt-Tiny weights |
| Input size | `224` |
| Crop | GT bbox center, square ROI |
| Crop scale | `8x` max bbox side |
| Crop size clamp | min `224`, max `512` pixels before resize |
| Train split | `configs/task2/splits/train_clean_labels.txt` |
| Validation splits | combined, phantom, animal |
| Epochs | `30` |
| Batch size | `128` |
| Optimizer | AdamW, `lr=3e-4`, `weight_decay=1e-4` |
| Class handling | class-weighted CE + balanced sampler |
| Device | `cuda:0` |

## Split Counts

| Split | Class 0 | Class 1 | Total |
| --- | ---: | ---: | ---: |
| train_clean | `17631` | `17453` | `35084` |
| valid_combined | `10738` | `684` | `11422` |
| valid_phantom | `10612` | `412` | `11024` |
| valid_animal | `126` | `272` | `398` |

Working class-name assumption remains:

- `0` = normal
- `1` = collision

## Verification

Completed before the full run:

- `python3 -m py_compile src/cathaction/data/task2_roi.py scripts/task2/train_roi_classifier.py`
- `pytest tests/test_task2_dataset.py tests/test_task2_roi.py`
  - result: `6 passed`
- smoke run:
  - `32` train samples;
  - `32` validation samples;
  - one epoch completed;
  - checkpoints and metrics were written.

## First Epoch Signal

After epoch 1 of the full run:

| Metric | valid_combined | valid_phantom | valid_animal |
| --- | ---: | ---: | ---: |
| accuracy | `0.6419` | `0.6546` | `0.2915` |
| balanced accuracy | `0.5296` | `0.6608` | `0.4603` |
| collision recall | `0.4020` | `0.6675` | `0.0000` |
| collision F1 | `0.1185` | `0.1262` | `0.0000` |
| average precision | `0.3306` | `0.5136` | `0.5455` |
| AUROC | `0.4761` | `0.7729` | `0.0236` |

Interpretation caveat:

- ROI classification AP/AUROC are not detector mAP and cannot be directly
  compared to YOLO mAP.
- The first epoch already shows useful signal on phantom but poor animal
  separation; later epochs will decide whether this is stable or a transient
  threshold/ranking issue.
- Animal AP should be interpreted against its high positive prevalence
  (`272 / 398` collision); the low AUROC at epoch 1 is a warning sign.

## Next Checks

1. Let the 30-epoch run finish unless validation clearly collapses.
2. Inspect `metrics.csv` for best combined AP and animal AUROC.
3. If GT-ROI classification is useful:
   - tune crop scale/context;
   - test fixed larger context;
   - connect YOLO-predicted boxes to the classifier.
4. If animal remains poor:
   - train/evaluate domain-balanced ROI classifier;
   - separate phantom and animal thresholds;
   - test temporal ROI windows.

