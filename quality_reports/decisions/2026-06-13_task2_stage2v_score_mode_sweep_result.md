# Task2 Stage2V Class-Specific Score-Mode Sweep Result

Date: 2026-06-13

## Context

Stage2V moved from a single global ranking mode to class-aware prediction fusion. The prior Stage2V best used:

- class 0 / normal: yolo-only run, `prob_iou75_source_rank_decay_roi`
- class 1 / collision: mixed-source run, `blend_rank_decay_roi`

That gave `valid_combined` mAP50 `0.1978528427` and mAP50-95 `0.0482545457`.

## Top-k Export Sweep

Using the prior class-aware fusion, I swept `--topk-per-sample-class` over `{1, 3, 5, 10, 20, 50, 100}`.

| top-k per sample/class | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| 1 | 0.1867 | 0.0419 |
| 3 | 0.1900 | 0.0451 |
| 5 | 0.1936 | 0.0466 |
| 10 | 0.1963 | 0.0477 |
| 20 | 0.1973 | 0.0481 |
| 50 | 0.1979 | 0.0483 |
| 100 | 0.1979 | 0.0483 |

Conclusion: top-k filtering is useful for output size control, but it does not improve the validation metric once k is reasonably high. Keep the full or top-50 export for analysis; do not treat top-k as a performance lever.

## Class-Specific Score Search

I audited saved prediction rows from these Stage2U/Stage2V eval runs:

- mixed-source: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval`
- yolo-only: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`
- augmented: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_stage2p_augmented_eval`
- refined fine-tune: `outputs/task2/stage2u_quality_ranker/stage2u_refined_source_finetune_train100k_fullvalid_augmented_eval`

Best `valid_combined` class-wise AP50-95 choices:

| class | run | score mode | AP50 | AP50-95 |
| --- | --- | --- | ---: | ---: |
| class 0 / normal | yolo-only | `prob_iou75_source_rank_decay_roi` | 0.1581350491 | 0.0675076495 |
| class 1 / collision | yolo-only | `roi` | 0.2438010127 | 0.0309732432 |

This implies a better combined class-aware export than the prior class0-yolo/class1-mixed fusion.

## Adopted Export

Command pattern:

```bash
python3 scripts/task2/export_stage2v_class_aware_predictions.py \
  --class0-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --class0-score-mode prob_iou75_source_rank_decay_roi \
  --class1-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --class1-score-mode roi \
  --output-dir outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export
```

Output:

- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/eval_metrics.json`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/valid_combined_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/valid_phantom_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/valid_animal_predictions.csv`

Clean-export verification: the prediction CSVs contain only inference fields:

`sample_id, video_id, frame_index, domain, class_id, score, x1, y1, x2, y2`

No validation-only `gt_*` columns are present.

## Metrics

| split | mAP50 | mAP50-95 | prediction rows |
| --- | ---: | ---: | ---: |
| valid_combined | 0.2009680309 | 0.0492404464 | 40970 |
| valid_phantom | 0.0960399493 | 0.0408605706 | 39338 |
| valid_animal | 0.4998403066 | 0.0500025663 | 1632 |

Compared with the previous Stage2V class-aware export:

| model/export | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| class0 yolo + class1 mixed | 0.1978528427 | 0.0482545457 |
| class0 yolo + class1 yolo-roi | 0.2009680309 | 0.0492404464 |

The new setting improves the aggregate primary metric by about `+0.0010` mAP50-95 and aggregate mAP50 by about `+0.0031`.

## Decision

Adopt `stage2v_class0_yolo_iou75_class1_yolo_roi_export` as the current Task2 validation best for the public `valid_combined` aggregate.

Keep the prior class0-yolo/class1-mixed export as a comparison point because it is slightly better on `valid_phantom` mAP50-95 (`0.04138` vs `0.04086`). For the current objective, the combined validation metric is the better selection target, so the yolo/yolo-roi class-aware export is the active best.

## Remaining Risk

This is score-mode selection on public validation data, so it should be treated as model selection rather than proof of hidden-test generalization. The class 0 signal on animal remains weak, and the method still depends on the quality of the YOLO candidate pool.
