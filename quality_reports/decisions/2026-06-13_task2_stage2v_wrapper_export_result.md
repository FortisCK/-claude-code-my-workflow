# Task2 Stage2V Result: Single-Entry Class-Aware Export Wrapper

Date: 2026-06-13
Status: completed

## Question

Can the current best Stage2V class-aware fusion policy be exported through one reproducible command, rather than manually running fusion and clean-export scripts in sequence?

## Implementation

Added:

- `scripts/task2/export_stage2v_class_aware_predictions.py`

The wrapper:

1. reads class 0 prediction rows from one Stage2U eval run;
2. reads class 1 prediction rows from another Stage2U eval run;
3. applies class-specific score modes;
4. writes clean internal prediction CSVs without validation-only fields;
5. computes mAP when GT columns are available in the source eval runs;
6. supports optional `--topk-per-sample-class`.

Current policy:

- class 0: `stage2u_warm_stage2o_fullvalid_yolo_only_eval`, `prob_iou75_source_rank_decay_roi`;
- class 1: `stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval`, `blend_rank_decay_roi`.

## Verification

Static check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/evaluate_stage2u_class_fusion.py \
  scripts/task2/export_stage2v_class_aware_predictions.py
```

Wrapper command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/task2/export_stage2v_class_aware_predictions.py \
  --class0-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --class0-score-mode prob_iou75_source_rank_decay_roi \
  --class1-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval \
  --class1-score-mode blend_rank_decay_roi \
  --output-dir outputs/task2/stage2u_quality_ranker/stage2v_class_aware_clean_export_wrapper
```

Output files:

- `outputs/task2/stage2u_quality_ranker/stage2v_class_aware_clean_export_wrapper/valid_combined_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class_aware_clean_export_wrapper/valid_phantom_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class_aware_clean_export_wrapper/valid_animal_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class_aware_clean_export_wrapper/eval_metrics.json`

Validation:

| Split | Rows | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: |
| valid_combined | `87470` | `0.197853` | `0.048255` |
| valid_phantom | `80538` | `0.099890` | `0.041376` |
| valid_animal | `6932` | `0.499327` | `0.049951` |

All clean prediction CSVs have only:

- `sample_id`, `video_id`, `frame_index`, `domain`;
- `class_id`, `score`, `x1`, `y1`, `x2`, `y2`;
- `source`, `source_rank`, `score_mode`, `policy_name`.

No `gt_*` fields are exported.

## Decision

Use `export_stage2v_class_aware_predictions.py` as the current internal export entry point for Task2 Stage2V predictions.

This closes the reproducibility gap between the best validation result and a prediction artifact that can be converted to the official challenge format.

## Remaining Gap

This is not yet a complete hidden-test inference pipeline. It assumes the two source Stage2U eval runs already exist.

The hidden-test pipeline still needs:

1. candidate generation on hidden data;
2. Stage2U inference on yolo-only class 0 candidates;
3. Stage2U inference on mixed-source class 1 candidates;
4. this wrapper for class-aware clean export;
5. final official result-file conversion once the official schema is available.
