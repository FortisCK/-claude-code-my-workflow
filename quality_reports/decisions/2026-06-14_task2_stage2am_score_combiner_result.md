# Task2 Stage2AM Score Combiner Result

Date: 2026-06-14

## Question

After Stage2AL failed, can a conservative score-level combiner improve the
Stage2AE ranking without changing the CNN backbone or candidate pool?

## Implementation

Scripts added:

- `scripts/task2/sweep_stage2am_score_combiner.py`
- `scripts/task2/sweep_stage2am_fast_score_combiner.py`

The first implementation was too slow for broad sweeps because it repeatedly
constructed detection objects and recomputed AP. The fast implementation caches
candidate rows, per-class IoU, and score arrays, then computes one-to-one AP
directly. A baseline check confirms it reproduces Stage2AE exactly.

## Commands

Baseline check:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/sweep_stage2am_fast_score_combiner.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --box-transform-json outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_1.25_valid_combined.json \
  --score-mode prob_iou75_source_rank_decay_roi \
  --score-mode rank_decay_roi \
  --score-mode prob_iou75_roi \
  --alpha 0.0 \
  --blend-type linear \
  --output-json outputs/task2/stage2am_score_combiner/stage2ae_box_score_combiner_fast_baseline_check.json
```

Narrow score-combiner sweep:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/sweep_stage2am_fast_score_combiner.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --box-transform-json outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_1.25_valid_combined.json \
  --score-mode roi \
  --score-mode rank_decay_roi \
  --score-mode sqrt_source_roi \
  --score-mode prob_iou75_roi \
  --score-mode prob_iou75_source_rank_decay_roi \
  --score-mode pred_iou_source_rank_decay_roi \
  --score-mode prob_iou50_sqrt_source_roi \
  --alpha 0.0 \
  --alpha 0.1 \
  --alpha 0.25 \
  --alpha 0.5 \
  --alpha 1.0 \
  --output-json outputs/task2/stage2am_score_combiner/stage2ae_box_score_combiner_fast_narrow.json \
  --prediction-dir outputs/task2/stage2am_score_combiner/stage2am_fast_narrow_predictions
```

## Baseline Verification

The alpha-0 baseline exactly reproduced Stage2AE:

| Split | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| `valid_combined` | 0.21128164745000094 | 0.06327005557430049 |
| `valid_phantom` | 0.10462265654206386 | 0.043011390577405134 |
| `valid_animal` | 0.4997355309073923 | 0.14317027939394036 |

This validates the fast evaluator against the existing Stage2AE metrics.

## Narrow Sweep Result

Selected policy:

- class0/phantom: geometric blend of `prob_iou75_source_rank_decay_roi` and `prob_iou75_roi`, alpha 0.25;
- class0/animal: unchanged;
- class1/phantom: linear blend of `rank_decay_roi` and `roi`, alpha 0.5;
- class1/animal: `pred_iou_source_rank_decay_roi`.

Metrics:

| Method | Split | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: |
| Stage2AE baseline | `valid_combined` | 0.21128164745000094 | 0.06327005557430049 |
| Stage2AM narrow | `valid_combined` | 0.10276277699944569 | 0.04288029774366862 |
| Stage2AE baseline | `valid_phantom` | 0.10462265654206386 | 0.043011390577405134 |
| Stage2AM narrow | `valid_phantom` | 0.1065624943208877 | 0.04408741257034943 |
| Stage2AE baseline | `valid_animal` | 0.4997355309073923 | 0.14317027939394036 |
| Stage2AM narrow | `valid_animal` | 0.5 | 0.24489571162085144 |

Per-class notes:

- `valid_phantom` class1 AP50 improved from 0.045713631775941704 to 0.04906319642353235.
- `valid_phantom` class0 AP50 improved from 0.16353168130818602 to 0.16406179221824305.
- `valid_combined` class1 AP50 collapsed from 0.2644282457502355 to 0.04671318922609221.

## Interpretation

Stage2AM shows that simple score blending can improve phantom locally, but it
breaks cross-domain score calibration. The animal class1 scores and phantom
false positives are no longer ordered in a way that preserves combined AP.

This is exactly the risk Stage2AK exposed: per-domain optimization can improve
within-domain ranking while destroying the global precision-recall curve used
by combined mAP.

## Decision

Do not promote Stage2AM.

Stage2AE remains the default/balanced candidate. Stage2AI remains the
high-`mAP50-95` alternative.

## Next Direction

If continuing score-level work, it must optimize the global combined PR curve
directly, not independent domain AP. Viable next directions:

1. Learn a global cross-domain calibration factor with Stage2AE as fallback.
2. Optimize class1 score scale explicitly so phantom gains do not bury animal
   true positives.
3. Use official validation, once released, to choose between Stage2AE and
   Stage2AI rather than overfitting current public validation.

