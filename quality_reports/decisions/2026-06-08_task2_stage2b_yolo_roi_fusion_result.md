# Task 2 Stage2B YOLO + ROI Fusion Result

Date: 2026-06-08

## Goal

Test whether the strong Stage2A GT-ROI classifier can improve Task 2 detector-style
mAP when the ROI is supplied by the existing YOLO11s detector instead of the
ground-truth bbox.

## Inputs

- Detector: `outputs/task2/yolo/yolo11s_1024_clean_combined_e100/weights/best.pt`
- ROI classifier: `outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/best.pt`
- Splits:
  - `configs/task2/splits/valid_combined_labels.txt`
  - `configs/task2/splits/valid_phantom_labels.txt`
  - `configs/task2/splits/valid_animal_labels.txt`

## Implementation

Added:

- `src/cathaction/metrics/detection.py`
  - Detection ground-truth/prediction records.
  - `xyxy` IoU.
  - 101-point interpolated AP.
  - mAP50 and mAP50-95 over class ids 0 and 1.
- `tests/test_task2_detection_metrics.py`
- `scripts/task2/evaluate_yolo_roi_fusion.py`
  - Runs YOLO on validation images.
  - Supports top-1, top-K, or all YOLO candidate boxes per image.
  - Crops the same ROI shape used in Stage2A.
  - Runs the ConvNeXt-Tiny ROI classifier.
  - Emits two detector scores per candidate box:
    - class 0 score = `1 - p_collision`
    - class 1 score = `p_collision`
  - Also supports `score_mode=det_roi`, where class scores are multiplied by
    the YOLO candidate confidence.
  - Also emits an argmax-class detector diagnostic.

The first full run attempted to pass all 11,422 combined images into one
Ultralytics `predict(..., stream=True)` call and failed with CUDA OOM during
preprocessing. The script was fixed to pass images in chunks via
`--source-chunk-size`, default 64.

## Verification

Commands:

```bash
python3 -m py_compile src/cathaction/metrics/detection.py scripts/task2/evaluate_yolo_roi_fusion.py
conda run -n cathaction-task1 pytest -q tests/test_task2_detection_metrics.py tests/test_task2_roi.py tests/test_task2_dataset.py
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_roi_fusion.py --name yolo11s1024_convnext_tiny_gtroi224_stage2b_smoke128_chunked --limit 128 --det-batch-size 32 --source-chunk-size 64 --roi-batch-size 64 --progress-every 64
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_roi_fusion.py --name yolo11s1024_convnext_tiny_gtroi224_stage2b --det-batch-size 32 --source-chunk-size 64 --roi-batch-size 256 --progress-every 1000
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_roi_fusion.py --name yolo11s1024_convnext_tiny_gtroi224_stage2b_top5 --candidate-mode topk --top-k 5 --det-batch-size 32 --source-chunk-size 64 --roi-batch-size 256 --progress-every 1000
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_yolo_roi_fusion.py --name yolo11s1024_convnext_tiny_gtroi224_stage2b_top5_detroi --candidate-mode topk --top-k 5 --score-mode det_roi --det-batch-size 32 --source-chunk-size 64 --roi-batch-size 256 --progress-every 1000
```

Results:

- Unit/smoke tests passed.
- Chunked 128-sample smoke completed without OOM.
- Full validation completed and wrote:
  - `outputs/task2/yolo_roi_fusion/yolo11s1024_convnext_tiny_gtroi224_stage2b/summary_metrics.json`
  - `outputs/task2/yolo_roi_fusion/yolo11s1024_convnext_tiny_gtroi224_stage2b_top5/summary_metrics.json`
  - `outputs/task2/yolo_roi_fusion/yolo11s1024_convnext_tiny_gtroi224_stage2b_top5_detroi/summary_metrics.json`
  - per-split metrics JSON files
  - per-split prediction CSV files

## Full Metrics

| Split | Samples | Top-box IoU>=0.5 | ROI cls AP | ROI cls AUROC | Fusion mAP50 | Fusion mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid_combined | 11,422 | 0.3810 | 0.3119 | 0.7869 | 0.0890 | 0.0291 |
| valid_phantom | 11,024 | 0.3850 | 0.2410 | 0.6818 | 0.0902 | 0.0297 |
| valid_animal | 398 | 0.2714 | 0.8856 | 0.8793 | 0.3748 | 0.0784 |

Class AP details for two-class fusion:

| Split | class 0 AP50 | class 0 AP50-95 | class 1 AP50 | class 1 AP50-95 |
| --- | ---: | ---: | ---: | ---: |
| valid_combined | 0.1643 | 0.0546 | 0.0137 | 0.0035 |
| valid_phantom | 0.1586 | 0.0540 | 0.0217 | 0.0054 |
| valid_animal | 0.7497 | 0.1569 | 0.0000 | 0.0000 |

Multi-candidate fusion comparison:

| Run | Split | Best-candidate IoU>=0.5 | ROI cls AP | Fusion mAP50 | Fusion mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| top1, ROI score | valid_combined | 0.3810 | 0.3119 | 0.0890 | 0.0291 |
| top5, ROI score | valid_combined | 0.4931 | 0.2373 | 0.0448 | 0.0159 |
| top5, YOLO-conf * ROI score | valid_combined | 0.4931 | 0.2373 | 0.1111 | 0.0406 |
| top1, ROI score | valid_phantom | 0.3850 | 0.2410 | 0.0902 | 0.0297 |
| top5, ROI score | valid_phantom | 0.5011 | 0.2602 | 0.0518 | 0.0181 |
| top5, YOLO-conf * ROI score | valid_phantom | 0.5011 | 0.2602 | 0.1069 | 0.0409 |
| top1, ROI score | valid_animal | 0.2714 | 0.8856 | 0.3748 | 0.0784 |
| top5, ROI score | valid_animal | 0.2714 | 0.9021 | 0.1031 | 0.0220 |
| top5, YOLO-conf * ROI score | valid_animal | 0.2714 | 0.9021 | 0.4167 | 0.0860 |

For context, the YOLO11s baseline training log reports best epoch 9:

- `metrics/mAP50(B) = 0.11561`
- `metrics/mAP50-95(B) = 0.04206`

## Interpretation

Stage2A showed that collision classification is learnable when the ROI is
centered on the ground-truth bbox. Stage2B shows that directly replacing the
ground-truth bbox with YOLO candidates is not enough unless candidate scoring
and localization are also improved.

The bottleneck is candidate localization plus candidate-score calibration, not
the ROI classifier capacity:

- Combined top-box localization recall at IoU 0.5 is only 38.1%.
- Combined top5 best-candidate localization recall at IoU 0.5 improves to
  49.3%, but this still leaves about half the frames without a usable IoU-0.5
  proposal.
- ROI-only scoring fails with multiple candidates because false candidate boxes
  can receive high ROI collision scores and pollute AP ranking.
- Multiplying ROI probabilities by YOLO confidence recovers performance:
  combined mAP50-95 rises from 0.0159 to 0.0406 for top5 candidates.
- Animal localization is worse, with IoU 0.5 recall 27.1%.
- Collision class AP remains extremely low in detector-style evaluation.

The best fusion variant tested here, top5 with `score_mode=det_roi`, is close
to the YOLO11s best baseline but does not clearly exceed it on combined
validation: 0.0406 vs YOLO training-log best 0.0421 mAP50-95. Its value is
diagnostic: Task 2 needs a better proposal/localization stage before the ROI
classifier can become a meaningful improvement.

## Next Direction

Highest-priority options:

1. Train a class-agnostic tool-tip/contact-region detector: collapse class ids
   during detection and let the ROI classifier handle normal vs collision.
2. Improve candidate scoring/calibration if using multi-candidate fusion:
   `YOLO_conf * ROI_prob` is much better than ROI-only scores.
3. Use temporal/case-level consistency after per-frame candidates are available.
4. Consider a heatmap/keypoint-style localizer if YOLO boxes remain unstable.

Do not spend more time tuning the Stage2A ROI classifier until candidate
localization recall is improved.
