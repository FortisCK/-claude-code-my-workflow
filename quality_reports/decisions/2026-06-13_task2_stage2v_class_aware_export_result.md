# Task2 Stage2V Result: Class-Aware Fusion Prediction Export

Date: 2026-06-13
Status: completed

## Question

Can the current best class-aware Stage2U fusion policy be exported as concrete prediction CSVs, rather than existing only as an in-memory evaluation?

## Implementation

Extended:

- `scripts/task2/evaluate_stage2u_class_fusion.py`

New option:

- `--prediction-dir`

When provided, the script writes:

- `{split}_fused_predictions.csv`

Each row contains:

- `sample_id`, `video_id`, `frame_index`, `domain`;
- `class_id`, `score`, `x1`, `y1`, `x2`, `y2`;
- source metadata;
- score mode and policy name;
- validation-only diagnostic fields `gt_class` and `gt_iou`.

These CSVs are not yet the official hidden-test challenge format. They are the internal prediction artifact needed to build the final submission formatter once official result-file details are fixed.

## Export Run

Command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 scripts/task2/evaluate_stage2u_class_fusion.py \
  --class0-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --class0-score-mode prob_iou75_source_rank_decay_roi \
  --class1-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval \
  --class1-score-mode blend_rank_decay_roi \
  --output-json outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/eval_metrics.json \
  --prediction-dir outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval
```

Outputs:

- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/valid_combined_fused_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/valid_phantom_fused_predictions.csv`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/valid_animal_fused_predictions.csv`

Row counts:

| Split | Rows |
| --- | ---: |
| valid_combined | `87470` |
| valid_phantom | `80538` |
| valid_animal | `6932` |

## Verification

Static check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/task2/evaluate_stage2u_class_fusion.py
```

Metric recomputation from exported fused CSVs:

| Split | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| valid_combined | `0.197853` | `0.048255` |
| valid_phantom | `0.099890` | `0.041376` |
| valid_animal | `0.499327` | `0.049951` |

The recomputed values match `eval_metrics.json` exactly to numerical precision.

## Decision

Use the class-aware fused prediction CSVs as the current internal Task2 prediction artifact.

Current best policy:

- class 0: `yolo_stage2l` only, `prob_iou75_source_rank_decay_roi`;
- class 1: original mixed YOLO+geometry, `blend_rank_decay_roi`.

Current best validation score:

- valid_combined mAP50: `0.1979`;
- valid_combined mAP50-95: `0.0483`.

## Next Direction

The next step is not another score-mode sweep. Build the final hidden-test inference/export path around the same policy:

1. generate candidate pools for hidden test;
2. run Stage2U on the required source-filtered and mixed candidate sets;
3. fuse class-specific predictions using this policy;
4. convert internal prediction CSVs to the official CATHACTION result file format once confirmed.
