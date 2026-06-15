# Task2 Stage2V Result: Source-Filtered YOLO-Only Stage2U Evaluation

Date: 2026-06-13
Status: completed

## Question

Does the post-hoc source-filter diagnostic hold when wired into the official Stage2U eval path and rerun from the checkpoint/candidate CSVs?

## Implementation

Added `--valid-source-keep` to:

- `scripts/task2/train_stage2u_quality_ranker.py`

The option filters validation/eval candidates by candidate source before ROI inference and mAP computation. It does not filter training candidates.

Added tests for source parsing/filtering in:

- `tests/test_task2_stage2u_quality_ranker.py`

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile \
  scripts/task2/train_stage2u_quality_ranker.py \
  scripts/task2/evaluate_stage2u_source_filter.py

PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_stage2u_quality_ranker.py \
  tests/test_task2_stage2o_ranker.py
```

Result:

- 8 tests passed.

## Evaluation

Run:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`

Checkpoint:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`

Candidate CSV:

- Stage2Q augmented CSVs, filtered at eval time to `source == yolo_stage2l`.

Candidate counts after filtering:

| Split | Candidates | R@0.50 | R@0.75 |
| --- | ---: | ---: | ---: |
| valid_combined | `20485` | `0.5456` | `0.2170` |
| valid_phantom | `19669` | `0.5049` | `0.2451` |
| valid_animal | `816` | `0.8598` | `0.0000` |

## Results

| Split | Best mode | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: |
| valid_combined | `prob_iou75_source_rank_decay_roi` | `0.1641` | `0.0444` |
| valid_phantom | `prob_iou75_source_rank_decay_roi` | `0.0923` | `0.0390` |
| valid_animal | `prob_iou75_rank_decay_roi` | `0.5000` | `0.0500` |

## Comparison

| Policy | valid_combined mAP50 | valid_combined mAP50-95 | valid_phantom mAP50-95 |
| --- | ---: | ---: | ---: |
| Previous Stage2U original mixed-source best | `0.1875` | `0.0414` | `0.0342` |
| Stage2V YOLO-only source-filtered best | `0.1641` | `0.0444` | `0.0390` |

## Decision

Adopt Stage2V YOLO-only source-filtered evaluation as the current primary-metric best for Task2.

This improves the primary COCO-style mAP50-95:

- valid_combined: `0.0414 -> 0.0444`, +`0.0030`
- valid_phantom: `0.0342 -> 0.0390`, +`0.0048`

The tradeoff is lower AP50:

- valid_combined mAP50: `0.1875 -> 0.1641`

Because the challenge primary metric is mAP, and the internal metric uses mAP50-95 as the stricter target, this is a useful improvement. However, the drop in mAP50 means this is not a universally better detector; it is a high-IoU ranking improvement from reducing noisy candidate sources.

## Next Direction

Do not add more unfiltered sources. The next work should be:

1. source/class/domain audit to understand why `prob_iou75_source_rank_decay_roi` works better on YOLO-only candidates;
2. build a submission/export path that can select the best score mode and source filter explicitly;
3. redesign candidate generation for phantom/high-IoU localization, because YOLO-only improves ranking but does not improve candidate-pool recall.
