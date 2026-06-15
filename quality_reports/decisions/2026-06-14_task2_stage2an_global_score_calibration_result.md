# Task2 Stage2AN Global Score Calibration Result

Date: 2026-06-14

## Question

Can a small global class/domain score-scale layer improve Stage2AE without the
per-domain AP collapse observed in Stage2AM?

## Implementation

Scripts added:

- `scripts/task2/sweep_stage2an_global_score_scale.py`
- `scripts/task2/sweep_stage2an_fast_global_score_scale.py`

The object-based implementation was too slow. The fast implementation caches
clean prediction rows, ground-truth boxes, class/domain metadata, score arrays,
and IoU-to-GT arrays, then performs AP sweeps directly.

## Command

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/sweep_stage2an_fast_global_score_scale.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --prediction-dir outputs/task2/stage2aj_champion_pipeline_with_ai/public_valid_ab_ad_ae_ai/stage2ae \
  --output-json outputs/task2/stage2an_global_score_scale/stage2ae_fast_global_scale_mAP5095.json \
  --output-dir outputs/task2/stage2an_global_score_scale/stage2an_fast_predictions
```

Scale grid:

`{0.5, 0.75, 1.0, 1.25, 1.5}` for:

- class0 phantom;
- class0 animal;
- class1 phantom;
- class1 animal.

## Baseline Check

Identity policy exactly reproduces Stage2AE:

| Split | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| `valid_combined` | 0.21128164745000094 | 0.06327005557430049 |

## Best Policy

Best policy by `valid_combined` mAP50-95:

| Scale | Value |
| --- | ---: |
| class0 phantom | 1.5 |
| class0 animal | 0.5 |
| class1 phantom | 0.5 |
| class1 animal | 1.5 |

Metrics:

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2AE identity | 0.21128164745000094 | 0.06327005557430049 |
| Stage2AN global scale | 0.20037493968130657 | 0.06478975652205522 |

Class-level combined metrics:

| Class | Stage2AE AP50 | Stage2AE AP50-95 | Stage2AN AP50 | Stage2AN AP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1581350491497664 | 0.06750764952433767 | 0.1585570473916838 | 0.06764897883345732 |
| 1 | 0.2644282457502355 | 0.05903246162426329 | 0.24219283197092933 | 0.06193053421065312 |

Split metrics for the selected policy:

| Split | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| `valid_combined` | 0.20037493968130657 | 0.06478975652205522 |
| `valid_phantom` | 0.10462265654206386 | 0.043011390577405134 |
| `valid_animal` | 0.4997355309073923 | 0.14317027939394036 |

## Interpretation

Stage2AN is much safer than Stage2AM:

- it does not collapse combined AP;
- it improves combined mAP50-95 by +0.00151970094775473;
- it reduces combined mAP50 by -0.01090670776869437.

The gain is small and mostly comes from improving class1 high-IoU AP while
sacrificing AP50 ordering. This is not strong enough to replace Stage2AE as the
default candidate, but it is a useful high-IoU calibration variant.

## Decision

Do not replace Stage2AE.

Keep Stage2AN as a minor high-IoU alternative, below Stage2AI in priority:

- default/balanced: Stage2AE;
- high mAP50-95: Stage2AI;
- small global calibration diagnostic: Stage2AN.

Stage2AN should only be considered if official validation rewards high-IoU mAP
and Stage2AI overfits or is too aggressive.

