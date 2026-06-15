# Task2 Stage2AQ Multi-Source Phantom Class1 Result

Date: 2026-06-14

## Question

Can we use the higher-recall Stage2X multi-source candidate pool only where it
helps, namely phantom class1, while keeping the stable Stage2AE predictions for
class0 and animal?

## Implementation

Script added:

`scripts/task2/sweep_stage2aq_phantom_class1_multisource.py`

Stage2AQ policy:

- keep Stage2AE predictions for all class0 rows;
- keep Stage2AE predictions for animal class1;
- replace only phantom class1 with selected multi-source candidates from
  `outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval`;
- sweep source subset, score mode, top-k, and score scale using phantom class1
  AP50 as the fast objective;
- evaluate the selected policy on `valid_combined`, `valid_phantom`, and
  `valid_animal`.

Command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/sweep_stage2aq_phantom_class1_multisource.py \
  --optimize-split valid_phantom \
  --output-dir outputs/task2/stage2aq_phantom_class1_multisource/stage2ae_stage2x_fast_phantom
```

Output:

`outputs/task2/stage2aq_phantom_class1_multisource/stage2ae_stage2x_fast_phantom`

Clean predictions:

`outputs/task2/stage2aq_phantom_class1_multisource/stage2ae_stage2x_fast_phantom/best_predictions`

## Selected Policy

| Field | Value |
| --- | --- |
| source policy | `yolo_stage2x` |
| sources | `yolo_stage2l`, `stage2x_class1` |
| score mode | `rank_decay_roi` |
| phantom class1 top-k | 5 |
| phantom class1 score scale | 0.75 |

The scale does not affect phantom-only AP, but it affects combined class1
ordering against animal rows. The selected scale was the first best-tie policy.

## Metrics

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2AE | 0.21128164745000094 | 0.06327005557430047 |
| Stage2AI | 0.18439741617609534 | 0.08394956471533466 |
| Stage2AN | 0.20037493968130657 | 0.06478975652205522 |
| Stage2AQ | 0.23661313056264216 | 0.07023557986011882 |

Stage2AQ improves over Stage2AE by:

- `+0.02533148311264122` mAP50;
- `+0.00696552428581835` mAP50-95.

Split metrics for Stage2AQ:

| Split | mAP50 | mAP50-95 | class1 AP50 | class1 AP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `valid_combined` | 0.23661313056264216 | 0.07023557986011882 | 0.3150912119755179 | 0.07296351019589993 |
| `valid_phantom` | 0.12983063980792314 | 0.04994025156087294 | 0.09612959830766027 | 0.03007535842776151 |
| `valid_animal` | 0.4997355309073923 | 0.1431702793939404 | 0.9994710618147846 | 0.28634055878788073 |

Phantom class1 AP50 improved from Stage2AE's `0.045713631775941704` to
`0.09612959830766027`.

## Interpretation

Stage2AQ is the first post-Stage2AE experiment that gives a meaningful balanced
gain. The important difference from Stage2X/Y/AA is scope control:

- Stage2X added useful class1 candidates but hurt AP when used broadly.
- Stage2Y/AA tried to learn or select across the full multi-source pool and
  failed on full validation.
- Stage2AQ uses the multi-source pool only for phantom class1 and keeps the
  stable Stage2AE outputs elsewhere.

This supports the Stage2AP diagnosis: phantom class1 needed both better
candidate coverage and tighter output control, not another global ranker.

## Decision

Promote Stage2AQ as the new balanced/default Task2 candidate.

Keep Stage2AI as the high-IoU alternative because it still has higher
`valid_combined` mAP50-95:

- Stage2AQ: `0.07023557986011882`;
- Stage2AI: `0.08394956471533466`.

Next engineering step:

Integrate Stage2AQ into the official champion export wrapper, or create a
dedicated Stage2AQ hidden-test export wrapper that can reproduce:

1. Stage2AE predictions;
2. Stage2X class1 candidate generation / prediction rows;
3. phantom-class1-only replacement with top-k 5.

## Verification

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/sweep_stage2aq_phantom_class1_multisource.py \
  scripts/task2/diagnose_stage2ap_phantom_class1_hard_negatives.py \
  scripts/task2/run_stage2_champion_pipeline.py
```

Clean schema check passed for:

- `valid_combined_domain_policy_predictions.csv`;
- `valid_phantom_domain_policy_predictions.csv`;
- `valid_animal_domain_policy_predictions.csv`.

Full Task2 regression passed:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 84 passed.
