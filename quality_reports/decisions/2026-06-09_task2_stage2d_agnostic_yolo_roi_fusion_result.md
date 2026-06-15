# Task 2 Stage2D Class-Agnostic YOLO + ROI Fusion Result

Date: 2026-06-09

## Goal

Test whether the Stage2C class-agnostic YOLO proposal detector improves final
Task 2 detector-style mAP when combined with the existing Stage2A ConvNeXt ROI
classifier.

## Inputs

- Detector:
  `outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt`
- ROI classifier:
  `outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/best.pt`
- Fusion output:
  `outputs/task2/yolo_roi_fusion/yolo11s1024_agnostic_convnext_tiny_gtroi224_stage2d_top5_detroi/summary_metrics.json`

## Command

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_roi_fusion.py \
  --yolo-weights outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt \
  --name yolo11s1024_agnostic_convnext_tiny_gtroi224_stage2d_top5_detroi \
  --candidate-mode topk \
  --top-k 5 \
  --score-mode det_roi \
  --det-batch-size 32 \
  --source-chunk-size 64 \
  --roi-batch-size 256 \
  --progress-every 1000
```

## Result

Comparison against the prior two-class YOLO top5 `det_roi` fusion:

| Run | Split | top1 IoU>=0.5 | top5 best IoU>=0.5 | ROI cls AP | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| two-class YOLO top5 det_roi | valid_combined | 0.3810 | 0.4931 | 0.2373 | 0.1111 | 0.0406 |
| class-agnostic YOLO top5 det_roi | valid_combined | 0.3247 | 0.5190 | 0.5196 | 0.0948 | 0.0356 |
| two-class YOLO top5 det_roi | valid_phantom | 0.3850 | 0.5011 | 0.2602 | 0.1069 | 0.0409 |
| class-agnostic YOLO top5 det_roi | valid_phantom | 0.3276 | 0.5289 | 0.1788 | 0.0918 | 0.0355 |
| two-class YOLO top5 det_roi | valid_animal | 0.2714 | 0.2714 | 0.9021 | 0.4167 | 0.0860 |
| class-agnostic YOLO top5 det_roi | valid_animal | 0.2437 | 0.2437 | 0.9825 | 0.3812 | 0.0658 |

## Interpretation

The class-agnostic detector improves top5 proposal coverage on combined and
phantom validation, but the improvement does not translate into final mAP.

Main observations:

- Combined top5 proposal recall improves from 0.4931 to 0.5190.
- Combined final mAP50-95 drops from 0.0406 to 0.0356.
- Phantom final mAP50-95 drops from 0.0409 to 0.0355.
- Animal final mAP50-95 drops from 0.0860 to 0.0658.
- The Stage2C diagnostic already showed class-1/collision proposal recall is
  extremely low, so top10 is unlikely to be a high-value next run without a
  different scoring/localization strategy.

This confirms that the current two-stage pipeline is localization-limited and
collision-proposal-limited. Class-agnostic detection is not enough; it finds
more generic candidate boxes, but not the boxes that matter for collision AP.

## Recommendation

Do not promote the class-agnostic YOLO fusion to the current Task 2 best.

The next stage should be collision-aware localization:

1. Train/evaluate a collision-oversampled or collision-weighted detector.
2. Track proposal recall separately for class 0 and class 1, not only combined.
3. Treat animal collision recall as a hard diagnostic because it is currently
   the clearest failure mode.
