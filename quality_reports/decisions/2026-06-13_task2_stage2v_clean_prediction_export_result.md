# Task2 Stage2V Result: Clean Prediction Export

Date: 2026-06-13
Status: completed

## Question

Can the current best class-aware fused predictions be exported without validation-only fields, while preserving the exact validation metrics?

## Implementation

Added:

- `scripts/task2/export_clean_predictions.py`
- `tests/test_task2_export_clean_predictions.py`

The exporter reads internal fused prediction rows and writes clean internal prediction CSVs containing only:

- `sample_id`
- `video_id`
- `frame_index`
- `domain`
- `class_id`
- `score`
- `x1`, `y1`, `x2`, `y2`
- `source`
- `source_rank`
- `score_mode`
- `policy_name`

Validation-only fields such as `gt_class` and `gt_iou` are removed.

The script also supports optional per-sample/per-class top-k pruning:

- `--topk-per-sample-class K`

## Verification

Static and unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/task2/export_clean_predictions.py
PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_export_clean_predictions.py \
  tests/test_task2_stage2u_quality_ranker.py
```

Result:

- 7 tests passed.

## Exported Files

Source:

- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/*_fused_predictions.csv`

Clean outputs:

- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/clean_predictions/valid_combined_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/clean_predictions/valid_phantom_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/clean_predictions/valid_animal_predictions.csv`

Row counts:

| Split | Rows |
| --- | ---: |
| valid_combined | `87470` |
| valid_phantom | `80538` |
| valid_animal | `6932` |

## Metric Check

The clean prediction CSVs were read back and re-evaluated with the project mAP implementation.

| Split | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| valid_combined | `0.197853` | `0.048255` |
| valid_phantom | `0.099890` | `0.041376` |
| valid_animal | `0.499327` | `0.049951` |

These match the class-aware fusion `eval_metrics.json` exactly to numerical precision.

## Decision

Use the clean prediction CSVs as the current internal Task2 prediction artifact for the best validation policy.

This is still not the final CATHACTION hidden-test submission format, but it is now free of validation labels and can be converted once the official result schema is confirmed.

## Next Direction

Build a single inference/export driver for hidden test that performs:

1. candidate generation;
2. Stage2U inference for yolo-only class 0 candidates;
3. Stage2U inference for mixed-source class 1 candidates;
4. class-aware fusion;
5. clean prediction export;
6. final official-format conversion.
