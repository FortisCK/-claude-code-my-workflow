# Task2 Stage2AI Calibrated Policy Sweep Result

Date: 2026-06-14

## Question

After Stage2AE box calibration, should the score policy still be the Stage2AB domain-aware policy, or should it be re-selected on calibrated boxes?

## Setup

Base run:

`outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`

Box transform:

`outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_1.25_valid_combined.json`

Sweep script:

`scripts/task2/sweep_stage2ae_calibrated_policy.py`

Command:

```bash
python3 scripts/task2/sweep_stage2ae_calibrated_policy.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --box-transform-json outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_1.25_valid_combined.json \
  --output-json outputs/task2/stage2ai_calibrated_policy_sweep/stage2ae_box_policy_sweep.json \
  --prediction-dir outputs/task2/stage2ai_calibrated_policy_sweep/stage2ai_predictions
```

## Selected Policy

Class 0 / normal:

- phantom: `prob_iou75_source_rank_decay_roi`
- animal: `prob_iou75_source_rank_decay_roi`

Class 1 / collision:

- phantom: `pred_iou_source_rank_decay_roi`
- animal: `sqrt_source_roi`

## Result

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2AE | 0.21128164745000094 | 0.06327005557430047 |
| Stage2AI | 0.18439741617609534 | 0.08394956471533466 |

Split metrics:

| Split | Stage2AI mAP50 | Stage2AI mAP50-95 |
| --- | ---: | ---: |
| valid_combined | 0.18439741617609534 | 0.08394956471533466 |
| valid_phantom | 0.09742357618031064 | 0.040369930409670254 |
| valid_animal | 0.48785015325856906 | 0.23876800871618054 |

Interpretation:

- Stage2AI is much stronger on COCO-style mAP50-95: +0.02067950914103419 over Stage2AE.
- Stage2AI is weaker on AP50/mAP50: -0.0268842312739056 versus Stage2AE.
- The gain comes mainly from animal class1 high-IoU ranking, where the calibrated boxes make `sqrt_source_roi` much better than the Stage2AE score mode.
- The phantom split gets worse, so this is not a universal improvement.

## GT-Free Export

`scripts/task2/export_stage2ab_domain_policy_predictions.py` now supports `--score-policy-json`.

GT-free re-export command:

```bash
python3 scripts/task2/export_stage2ab_domain_policy_predictions.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2ai_calibrated_policy_sweep/stage2ai_gt_free_reexport \
  --box-transform-json outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_1.25_valid_combined.json \
  --score-policy-json outputs/task2/stage2ai_calibrated_policy_sweep/stage2ae_box_policy_sweep.json
```

The GT-free re-export exactly matches the sweep export for all public-validation splits:

- `valid_combined`: 40,970 rows, exact match.
- `valid_phantom`: 39,338 rows, exact match.
- `valid_animal`: 1,632 rows, exact match.

## Decision

Do not replace Stage2AE as the default public-validation champion yet.

Keep:

- **Stage2AE** as the balanced/default champion because it preserves mAP50 while improving mAP50-95.
- **Stage2AI** as the high mAP50-95 / COCO-mAP candidate if the official primary metric is confirmed to be mAP50-95 or equivalent.

Rationale:

Stage2AI exposes a real ranking improvement, but it pays for that gain with a large mAP50 drop and weaker phantom metrics. This is useful, but it should be selected only if the official metric rewards high-IoU AP strongly enough.

