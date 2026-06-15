# Task 2 Stage2C Class-Agnostic Proposal Detector Result

Date: 2026-06-09

## Goal

Train YOLO as a class-agnostic `tool_roi` proposal detector, then measure whether
it improves candidate-box coverage for the two-stage Task 2 pipeline.

The target metric for this stage is proposal recall, not final two-class mAP.

## Inputs

- Detector: `outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt`
- Training log: `quality_reports/logs/task2_yolo11s_1024_agnostic_clean_combined_e100.log`
- Proposal eval output: `outputs/task2/yolo_proposal_eval/yolo11s_1024_agnostic_clean_combined_e100/summary_metrics.json`
- Splits:
  - `configs/task2/splits/valid_combined_labels.txt`
  - `configs/task2/splits/valid_phantom_labels.txt`
  - `configs/task2/splits/valid_animal_labels.txt`

## Training Result

The class-agnostic detector trained for 39 epochs before stopping. Its best YOLO
validation metrics were:

| Metric | Best epoch | Value |
| --- | ---: | ---: |
| mAP50 | 13 | 0.20386 |
| mAP50-95 | 9 | 0.06935 |
| Precision | 20 | 0.37901 |
| Recall | 5 | 0.33917 |

For context, the earlier two-class YOLO11s baseline had best validation
`mAP50=0.11561` and `mAP50-95=0.04206`.

## Proposal Coverage

IoU>=0.5 recall by split:

| Split | top1 | top5 | top10 |
| --- | ---: | ---: | ---: |
| valid_combined | 0.3247 | 0.5190 | 0.5981 |
| valid_phantom | 0.3276 | 0.5289 | 0.6108 |
| valid_animal | 0.2437 | 0.2437 | 0.2437 |

Compared with the Stage2B two-class detector:

| Split | Old top1 | New top1 | Old top5 | New top5 |
| --- | ---: | ---: | ---: | ---: |
| valid_combined | 0.3810 | 0.3247 | 0.4931 | 0.5190 |
| valid_phantom | 0.3850 | 0.3276 | 0.5011 | 0.5289 |
| valid_animal | 0.2714 | 0.2437 | 0.2714 | 0.2437 |

Class-conditioned IoU>=0.5 recall shows the main failure mode:

| Split / GT class | top1 | top5 | top10 |
| --- | ---: | ---: | ---: |
| valid_combined / class 0 | 0.3453 | 0.5497 | 0.6288 |
| valid_combined / class 1 | 0.0015 | 0.0365 | 0.1155 |
| valid_animal / class 0 | 0.7698 | 0.7698 | 0.7698 |
| valid_animal / class 1 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

Class-agnostic training improves multi-candidate coverage for the dominant
normal-like class, but it does not solve collision-region localization.

The result is useful but not sufficient:

- Positive: combined top5 coverage improves from 0.4931 to 0.5190, and top10
  reaches 0.5981.
- Negative: top1 coverage drops from 0.3810 to 0.3247.
- Critical failure: collision-class proposal recall is almost absent, especially
  on animal where class-1 proposal recall is 0.0 at top1/top5/top10.

This means the class-agnostic detector is not a strong final localization stage
by itself. It may still help a multi-candidate two-stage system, but the next
meaningful improvement should directly address collision ROI localization and
domain imbalance rather than only reranking more generic proposals.

## Next Recommendation

Run one class-agnostic top5/top10 ROI-fusion check with `score_mode=det_roi` to
measure whether the modest proposal-coverage gain translates into final mAP.
If it does not clearly beat the two-class YOLO baseline, switch to a
collision-aware localization strategy:

1. Train a higher-weight collision detector or oversampled collision proposal
   detector.
2. Evaluate with class-conditioned proposal recall as a first-class metric.
3. Add animal-aware validation tracking, because animal collision is currently
   the weakest mode.
