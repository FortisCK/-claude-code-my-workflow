# Task2 Stage2S Result: Top-k Output Control Diagnostic

Date: 2026-06-12
Status: completed

## Question

The current best Task 2 pipeline emits many candidate predictions per frame. Is mAP mainly limited by too many false-positive outputs, and can simple per-frame top-k pruning improve AP?

## Implementation

Added:

- `scripts/task2/evaluate_stage2s_topk_pruning.py`
- `tests/test_task2_stage2s_topk_pruning.py`

The diagnostic reads an existing ranker eval run:

- `*_candidates_used.csv`
- `*_eval_prediction_rows.csv`

and evaluates pruning policies without retraining:

- `all`: no pruning baseline;
- `candidate_top{k}`: keep top-k candidate boxes per frame by max class score;
- `prediction_top{k}`: keep top-k class-specific predictions per frame;
- `class_top{k}`: keep top-k predictions per class per frame;
- `source_candidate_top{k}`: keep top-k candidate boxes per source per frame;
- `source_class_top{k}`: keep top-k class-specific predictions per source/class/frame;
- `oracle_candidate_top{k}`: leakage-only upper bound using GT IoU for candidate selection.

## Inputs

Current best run:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval`

Diagnostic outputs:

- `outputs/task2/stage2s_topk_diagnostic/old_ranker_original_pool_allmodes_combined`
- `outputs/task2/stage2s_topk_diagnostic/old_ranker_original_pool_rank_decay_all_splits`

## Results

### valid_combined, all score modes

Best non-oracle policy:

- score mode: `rank_decay_roi`
- policy: `prediction_top`
- k: 50
- mAP50: 0.1887
- mAP50-95: 0.0388
- loc R@0.75: 0.2213

This is effectively the original no-pruning baseline. No non-oracle top-k policy improves over the current best combined mAP50-95 of 0.0388.

Best oracle policy:

- score mode: `bg_suppressed_roi`
- policy: `oracle_candidate_top`
- k: 1
- mAP50: 0.4629
- mAP50-95: 0.1476

### rank_decay_roi, all splits

| Split | Best non-oracle policy | k | mAP50 | mAP50-95 | loc R@0.75 |
| --- | --- | ---: | ---: | ---: | ---: |
| valid_combined | prediction_top | 50 | 0.1887 | 0.0388 | 0.2213 |
| valid_phantom | prediction_top | 50 | 0.1022 | 0.0340 | 0.2500 |
| valid_animal | source_candidate_top | 5 | 0.5255 | 0.0598 | 0.0000 |

Oracle upper bounds:

| Split | Oracle policy | k | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| valid_combined | oracle_candidate_top | 1 | 0.4261 | 0.1410 |
| valid_phantom | oracle_candidate_top | 1 | 0.3458 | 0.1600 |
| valid_animal | oracle_candidate_top | 1 | 0.7764 | 0.1084 |

## Decision

Simple top-k output control is rejected as a direct improvement.

The current ranker scores are already best when we keep almost all candidates. Reducing each frame to top-k by the existing scores removes good boxes faster than it removes harmful false positives, so combined mAP50-95 does not improve.

However, the oracle result is important: if a selector could choose the best candidate per frame, valid_combined mAP50-95 could rise from 0.0388 to about 0.14. This means the candidate pool contains enough useful boxes for a much stronger result. The missing component is not another independent ROI classifier; it is a per-frame candidate selector/ranker that can identify the correct box among many candidates.

## Interpretation

Stage2S resolves the previous uncertainty:

- The problem is not simply "too many predictions"; naive top-k pruning is not enough.
- The problem is "wrong top-k"; the existing score is not aligned with candidate IoU/AP rank.
- Oracle top-1 is much stronger than all learned scores, so there is real headroom for a listwise or pairwise selector.

Animal behaves differently:

- `source_candidate_top@5` improves animal mAP50-95 from 0.0467 to 0.0598.
- But animal loc R@0.75 remains 0.0000, so this is mostly a low-IoU AP50/class-ranking effect, not precise localization.
- Because animal is a small fraction of combined validation, this does not change the combined decision.

## Current Best

Keep the current champion unchanged:

- old Stage2O ranker;
- original YOLO+geometry candidate pool;
- score mode: `rank_decay_roi`;
- valid_combined mAP50: 0.1887;
- valid_combined mAP50-95: 0.0388.

## Next Direction

Do not continue hand-crafted top-k pruning.

Next meaningful direction:

- train a per-frame/listwise candidate selector using candidate groups from the same frame;
- target should be candidate quality/rank, not independent background/normal/collision classification;
- first prototype can use tabular candidate features before adding image features:
  - source;
  - source rank/confidence;
  - ROI probabilities;
  - class probabilities;
  - candidate geometry;
  - temporal neighborhood features if available.

The selector should output one or a small number of boxes per frame/class and be evaluated directly by mAP.

## Verification

Passed:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/evaluate_stage2s_topk_pruning.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2s_topk_pruning.py tests/test_task2_stage2o_ranker.py tests/test_task2_stage2o_candidate_pool.py
```

Result:

- 11 tests passed.
