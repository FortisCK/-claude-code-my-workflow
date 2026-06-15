# Task2 Stage2R Result: Source-Consistent Refined Ranker

Date: 2026-06-11
Status: completed

## Question

Stage2P refined candidates improve localization upper bound, but the old ranker cannot use them directly. If we include `stage2p_refined` candidates on both train and validation, can the ranker learn to select the refined boxes and improve mAP50-95?

## Implementation

Changes:

- Added `--skip-overlap-check` to `scripts/task2/train_stage2p_box_refiner.py`.
  - Default is still strict no-overlap checking.
  - The bypass is only for eval-only generation of refined candidates on the training split.

Generated train-side refined candidates:

- `outputs/task2/stage2p_box_refiner/convnext_tiny_yolo_geometry_iou20_train_subset_eval/valid_combined_eval_refined_candidates.csv`

Exported expanded train candidate pool:

- `outputs/task2/stage2q_refined_candidate_pool/train_yolo_geometry_stage2p_refined/valid_combined_candidates.csv`

Train pool summary:

- base rows: 163,456
- refined rows: 163,456
- expanded rows: 326,912
- refined rows with IoU >= 0.75: 2,150

Ranker pilot:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2r_refined_source_train_valid256_e4`
- 4 epochs
- best epoch: 1
- valid256 best combined rank_decay mAP50-95: 0.0665

Full eval of best checkpoint on expanded pool:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2r_refined_source_full_eval/eval_metrics.json`

Control eval of the same checkpoint on the original pool:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2r_refined_source_original_pool_full_eval/eval_metrics.json`

## Results

Rank-decay score mode:

| Run | Split | mAP50 | mAP50-95 | loc R@0.50 | loc R@0.75 |
| --- | --- | ---: | ---: | ---: | ---: |
| old ranker, original pool | valid_combined | 0.1887 | 0.0388 | 0.5585 | 0.2213 |
| old ranker, original pool | valid_phantom | 0.1022 | 0.0340 | 0.5073 | 0.2500 |
| old ranker, original pool | valid_animal | 0.4609 | 0.0467 | 0.9533 | 0.0000 |
| old ranker, expanded pool | valid_combined | 0.1519 | 0.0280 | 0.5618 | 0.2513 |
| old ranker, expanded pool | valid_phantom | 0.0663 | 0.0216 | 0.5109 | 0.2840 |
| old ranker, expanded pool | valid_animal | 0.4595 | 0.0464 | 0.9533 | 0.0000 |
| Stage2R ranker, expanded pool | valid_combined | 0.1269 | 0.0203 | 0.5618 | 0.2513 |
| Stage2R ranker, expanded pool | valid_phantom | 0.0374 | 0.0129 | 0.5109 | 0.2840 |
| Stage2R ranker, expanded pool | valid_animal | 0.4748 | 0.0495 | 0.9533 | 0.0000 |
| Stage2R ranker, original pool | valid_combined | 0.1591 | 0.0288 | 0.5585 | 0.2213 |
| Stage2R ranker, original pool | valid_phantom | 0.0698 | 0.0235 | 0.5073 | 0.2500 |
| Stage2R ranker, original pool | valid_animal | 0.4761 | 0.0477 | 0.9533 | 0.0000 |

## Decision

Reject Stage2R as the current Task 2 champion.

The valid256 pilot looked promising, but it did not generalize to full validation. Full-valid combined mAP50-95 dropped from the old best 0.0388 to:

- 0.0203 with the expanded refined candidate pool;
- 0.0288 even when evaluated back on the original YOLO+geometry pool.

This means the failure is not only caused by doubling validation candidates. The retrained ranker itself is worse on phantom, and phantom dominates the combined split.

## Interpretation

Stage2P remains useful diagnostically:

- it increases localization upper bound from valid_combined R@0.75 0.2213 to 0.2513;
- it increases valid_phantom R@0.75 from 0.2500 to 0.2840.

But our current ROI ranker training cannot convert that localization gain into AP. The likely failure modes are:

- candidate-level class imbalance remains extreme;
- full validation distribution is not represented by the first 256-sample pilot;
- refined candidates add many near-positive/ignore boxes that harm AP ranking;
- the ROI classifier optimizes candidate classification, not AP ordering among many boxes per image.

## Current Best

Keep the old Stage2O ranker with the original YOLO+geometry candidate pool as the current best Task 2 pipeline:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/eval_metrics.json`
- valid_combined rank_decay mAP50: 0.1887
- valid_combined rank_decay mAP50-95: 0.0388

## Next Options

Do not continue adding refined boxes to the current ranker unless the ranking objective changes.

More plausible next directions:

- train a per-sample top-k selector/listwise ranker instead of independent ROI classification;
- reduce refined candidates to only high-confidence source candidates rather than appending all refined rows;
- calibrate/threshold candidates per source before AP evaluation;
- revisit proposal generation for animal R@0.75, because current refined candidates still produce 0.0000 animal R@0.75.

## Verification

Passed before training:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/train_stage2p_box_refiner.py scripts/task2/export_stage2p_refined_candidates.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2p_box_refiner.py tests/test_task2_stage2p_export_refined_candidates.py tests/test_task2_stage2o_ranker.py tests/test_task2_stage2o_candidate_pool.py
```

Result:

- 13 tests passed.
