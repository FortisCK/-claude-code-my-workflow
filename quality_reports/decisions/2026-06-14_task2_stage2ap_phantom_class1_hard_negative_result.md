# Task2 Stage2AP Phantom Class1 Hard-Negative Diagnostic Result

Date: 2026-06-14

## Question

Why does phantom class1 remain weak after Stage2AE/AI/AN, and should the next
effort focus on candidate generation, ranking/verifier training, temporal
smoothing, or box calibration?

## Implementation

Script added:

`scripts/task2/diagnose_stage2ap_phantom_class1_hard_negatives.py`

Command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/diagnose_stage2ap_phantom_class1_hard_negatives.py
```

Inputs:

- run rows:
  `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`
- Stage2AE predictions:
  `outputs/task2/stage2ao_champion_pipeline_with_an/public_valid_ab_ad_ae_ai_an/stage2ae`

Outputs:

- `outputs/task2/stage2ap_phantom_class1_hard_negatives/stage2ae_public_valid/stage2ap_summary.json`
- `outputs/task2/stage2ap_phantom_class1_hard_negatives/stage2ae_public_valid/high_score_false_positives.csv`
- `outputs/task2/stage2ap_phantom_class1_hard_negatives/stage2ae_public_valid/low_rank_true_positive_samples.csv`
- `outputs/task2/stage2ap_phantom_class1_hard_negatives/stage2ae_public_valid/missed_positive_samples.csv`

## Key Numbers

Scope: `valid_phantom`, target class `1`.

| Quantity | Value |
| --- | ---: |
| samples | 824 |
| class1 positive samples | 412 |
| candidate rows / class1 prediction rows | 19,669 |
| oracle IoU50 sample recall | 0.27427184466019416 |
| oracle IoU75 sample recall | 0.14805825242718446 |
| current top-score IoU50 sample recall | 0.1262135922330097 |
| current top-score IoU75 sample recall | 0.03640776699029126 |

Among the 113 positive samples where an IoU50 candidate exists:

| Rank / margin statistic | Value |
| --- | ---: |
| median rank gap before first IoU50 candidate | 1 |
| mean rank gap before first IoU50 candidate | 2.663716814159292 |
| max rank gap before first IoU50 candidate | 27 |
| median top-score minus first-IoU50 score | 0.09752576351165773 |
| mean top-score minus first-IoU50 score | 0.1635556610707988 |
| max top-score minus first-IoU50 score | 0.6709669604262574 |

## Error Types

Row-level class1 prediction distribution:

| Bucket | Count |
| --- | ---: |
| true candidates with IoU >= 0.50 | 330 |
| near misses with 0.25 <= IoU < 0.50 | 1,604 |
| false-positive candidates | 19,339 |

Top 200 high-score false positives:

| Ground-truth class | Count |
| ---: | ---: |
| 0 | 111 |
| 1 | 89 |

This means the high-confidence phantom class1 errors are mixed:

- many are class-confusion errors where normal frames (`gt_class=0`) receive high
  collision scores;
- many are localization errors on true class1 frames, where the box is near but
  below IoU50.

## Interpretation

Stage2AP confirms that phantom class1 has two bottlenecks:

1. Candidate coverage is still too low. Even oracle selection over the current
   Stage2AE yolo-only candidate pool reaches only 27.4% IoU50 recall and 14.8%
   IoU75 recall on class1-positive phantom samples.
2. Ranking is still poor where good candidates exist. Current top-score recall
   is 12.6% at IoU50, and some valid candidates rank more than 20 positions
   below high-score false positives.

Therefore:

- more box calibration alone cannot solve phantom class1;
- global score scaling cannot solve it;
- a verifier-only fine-tune on the same yolo-only candidate pool has limited
  ceiling;
- generic candidate expansion is useful only if paired with better source-aware
  hard-negative ranking.

## Decision

The next useful direction is Stage2AQ:

Use the higher-recall multi-source candidate pool for phantom class1, but train
or design a source-aware hard-negative reranker specifically against:

- high-score class0-as-class1 false positives;
- true class1 near misses with IoU 0.25-0.50;
- valid class1 candidates that are present but ranked below false positives.

The goal should be evaluated by phantom class1 AP50 first, then by
valid_combined mAP50/mAP50-95. A successful next step must improve phantom
class1 AP50 beyond Stage2AE's 0.045713631775941704 without collapsing combined
precision-recall.

## Verification

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/diagnose_stage2ap_phantom_class1_hard_negatives.py
```
