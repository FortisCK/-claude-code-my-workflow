# Task2 Stage2AH Clean Prediction Top-K Diagnostic Result

Date: 2026-06-14

## Question

Does top-k pruning of Stage2AE clean predictions improve public-validation AP by removing false positives?

## Setup

Prediction source:

`outputs/task2/stage2ag_champion_pipeline/public_valid_ab_ad_ae/stage2ae`

Diagnostic script:

`scripts/task2/evaluate_clean_prediction_topk.py`

Command:

```bash
python3 scripts/task2/evaluate_clean_prediction_topk.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --prediction-dir outputs/task2/stage2ag_champion_pipeline/public_valid_ab_ad_ae/stage2ae \
  --output-dir outputs/task2/stage2ah_clean_topk_diagnostic/stage2ae \
  --policy all \
  --policy global_topk \
  --policy animal_topk \
  --k 1 --k 2 --k 3 --k 5 --k 10 --k 20 --k 50
```

Summary:

`outputs/task2/stage2ah_clean_topk_diagnostic/stage2ae/best_summary.json`

## Result

Stage2AE baseline:

| Split | Policy | Rows | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| valid_combined | all | 40,970 | 0.21128164745000094 | 0.06327005557430047 |
| valid_phantom | all | 39,338 | 0.10462265654206386 | 0.04301139057740514 |
| valid_animal | all | 1,632 | 0.4997355309073923 | 0.1431702793939404 |

Best observed rows by split:

| Split | Policy | k | Rows | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| valid_combined | animal_topk | 5 | 40,400 | 0.21128164745000094 | 0.06327012888513178 |
| valid_phantom | all | - | 39,338 | 0.10462265654206386 | 0.04301139057740514 |
| valid_animal | global_topk / animal_topk | 3 | 642 | 0.4997355309073923 | 0.14346445457131815 |

Interpretation:

- Global top-k pruning hurts combined and phantom metrics.
- Animal-only top-k can slightly improve animal mAP50-95 at `k=3`.
- The best combined result, `animal_topk@5`, improves mAP50-95 by only `+0.00000007331083131`, which is effectively negligible and not worth introducing another post-processing branch.

## Decision

Do not promote Stage2AH. Keep Stage2AE as the current Task2 public-validation champion.

Rationale:

- Top-k pruning does not produce a meaningful combined mAP50-95 gain.
- Global pruning clearly hurts phantom.
- Animal-only pruning adds policy complexity for a near-zero combined improvement.

