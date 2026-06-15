# Task2 Stage2V Result: Class-Aware Stage2U Source/Score Fusion

Date: 2026-06-13
Status: completed

## Question

The YOLO-only Stage2U evaluation improved mAP50-95 but hurt class 1 and mAP50. Can a class-aware policy combine the strengths of different source/score choices?

Evidence before fusion:

- Mixed-source Stage2U improves class 1/collision.
- YOLO-only Stage2U improves class 0/normal and phantom high-IoU AP.

## Audit Finding

Best mixed-source Stage2U:

| Split | Class | AP50 | AP50-95 |
| --- | --- | ---: | ---: |
| valid_combined | class 0 | `0.1375` | `0.0537` |
| valid_combined | class 1 | `0.2376` | `0.0290` |
| valid_phantom | class 0 | `0.1420` | `0.0554` |
| valid_phantom | class 1 | `0.0362` | `0.0129` |

Best YOLO-only Stage2U:

| Split | Class | AP50 | AP50-95 |
| --- | --- | ---: | ---: |
| valid_combined | class 0 | `0.1581` | `0.0675` |
| valid_combined | class 1 | `0.1701` | `0.0212` |
| valid_phantom | class 0 | `0.1635` | `0.0698` |
| valid_phantom | class 1 | `0.0212` | `0.0082` |

Interpretation:

- YOLO-only is better for class 0 at nearly every IoU threshold.
- Mixed-source is better for class 1.
- A global source filter is therefore suboptimal.

## Implementation

Added:

- `scripts/task2/evaluate_stage2u_class_fusion.py`

The script combines saved Stage2U eval runs by class:

- class 0 predictions from one run/score mode;
- class 1 predictions from another run/score mode;
- then recomputes the standard detection mAP.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/task2/evaluate_stage2u_class_fusion.py
```

Fusion run:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 scripts/task2/evaluate_stage2u_class_fusion.py \
  --class0-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --class0-score-mode prob_iou75_source_rank_decay_roi \
  --class1-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval \
  --class1-score-mode blend_rank_decay_roi \
  --output-json outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_class1_mixed_fusion_eval/eval_metrics.json
```

## Results

Policy:

- class 0 / normal:
  - source: `yolo_stage2l` only
  - score: `prob_iou75_source_rank_decay_roi`
- class 1 / collision:
  - source: original mixed YOLO+geometry
  - score: `blend_rank_decay_roi`

| Split | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| valid_combined | `0.1979` | `0.0483` |
| valid_phantom | `0.0999` | `0.0414` |
| valid_animal | `0.4993` | `0.0500` |

Comparison:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 | valid_phantom mAP50-95 |
| --- | ---: | ---: | ---: |
| Mixed-source Stage2U | `0.1875` | `0.0414` | `0.0342` |
| YOLO-only Stage2U | `0.1641` | `0.0444` | `0.0390` |
| Class-aware fusion | `0.1979` | `0.0483` | `0.0414` |

## Decision

Adopt class-aware Stage2U source/score fusion as the current best Task2 validation policy.

This is the first recent change that improves both:

- valid_combined mAP50: `0.1875 -> 0.1979`;
- valid_combined mAP50-95: `0.0414 -> 0.0483`.

The gain comes from using different source/score policies for different classes instead of treating the two labels symmetrically.

## Caveats

- This is currently a fusion of saved eval runs, not yet integrated into a single submission/export pipeline.
- It uses validation labels only for metric computation, not for prediction selection, but it does choose the class-specific policy based on validation analysis. This is acceptable for model selection on the public validation split, but should be documented.
- The animal class 0 AP remains essentially zero.
- High-IoU AP is still low in absolute terms; candidate generation remains the main bottleneck.

## Next Direction

1. Integrate class-aware fusion into the Task2 prediction/export path.
2. Make score/source policy explicit in config:
   - class 0: `yolo_stage2l`, `prob_iou75_source_rank_decay_roi`;
   - class 1: original mixed source, `blend_rank_decay_roi`.
3. Continue candidate generation redesign, now with a sharper target:
   - class 0 benefits from tighter YOLO high-IoU boxes;
   - class 1 needs broader mixed-source recall and better high-IoU localization.
