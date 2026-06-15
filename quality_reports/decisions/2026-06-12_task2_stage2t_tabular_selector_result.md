# Task2 Stage2T Result: Tabular Listwise Candidate Selector

Date: 2026-06-12
Status: completed

## Question

Can a learned per-frame candidate-quality selector close part of the gap between the current Stage2O ranker and the Stage2S oracle top-1 upper bound?

Stage2S showed:

- current champion, valid_combined: mAP50 0.1887, mAP50-95 0.0388;
- oracle candidate selection, valid_combined: mAP50-95 about 0.14;
- simple top-k pruning by existing scores did not improve the champion.

This implied that the candidate pool has headroom, but the existing score does not reliably pick the right candidate.

## Implementation

Added:

- `scripts/task2/export_stage2o_prediction_rows.py`
- `scripts/task2/train_stage2t_tabular_selector.py`
- `tests/test_task2_stage2t_tabular_selector.py`

Also added a narrow `--skip-overlap-check` flag to `scripts/task2/train_stage2o_candidate_ranker.py`, keeping overlap checks enabled by default.

Stage2T trains a CPU tabular selector using train candidate rows and train-side Stage2O ROI probabilities:

- source metadata;
- source rank/confidence;
- candidate geometry;
- Stage2O ROI probabilities;
- Stage2O score modes;
- per-sample rank features.

The training target is candidate localization quality (`gt_iou` regression and IoU-threshold classifiers). Ground-truth fields are excluded from features.

## Inputs

Current champion full-valid run:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval`

Train-side prediction export:

- `train_eval_candidates_used.csv`
- `train_eval_prediction_rows.csv`
- 142,157 aligned rows

Stage2T output:

- `outputs/task2/stage2t_tabular_selector/histgb_stage2o_trainpred_fullvalid`

## Results

Best valid_combined Stage2T policy:

| Quality | Base score | k | mAP50 | mAP50-95 | loc R@0.75 |
| --- | --- | ---: | ---: | ---: | ---: |
| `prob_iou75` | `rank_decay_roi` | 50 | 0.1551 | 0.0403 | 0.2213 |

Best per split:

| Split | Quality | Base score | k | mAP50 | mAP50-95 | loc R@0.75 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| valid_combined | `prob_iou75` | `rank_decay_roi` | 50 | 0.1551 | 0.0403 | 0.2213 |
| valid_phantom | `prob_iou75` | `rank_decay_roi` | 50 | 0.0810 | 0.0345 | 0.2500 |
| valid_animal | `pred_iou` | `roi` | 20 | 0.4832 | 0.0487 | 0.0000 |

Comparison to previous champion:

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2O `rank_decay_roi` | 0.1887 | 0.0388 |
| Stage2T `prob_iou75 * rank_decay_roi`, k=50 | 0.1551 | 0.0403 |

## Decision

Accept Stage2T as the current mAP50-95 champion, but keep Stage2O as the AP50/robustness baseline.

The improvement is small:

- mAP50-95: +0.0015 absolute over Stage2O;
- mAP50: -0.0336 absolute versus Stage2O.

This means Stage2T is useful only if the official primary metric behaves like COCO-style mAP50-95. If the official scoring emphasizes AP50, Stage2O remains preferable.

## Interpretation

Stage2T did not solve the coarse detection problem in the strong sense. It mainly learns a light quality reweighting that slightly improves higher-IoU AP while preserving the original candidate recall.

Important observations:

- The best policy keeps k=50 candidates per frame, almost equivalent to no pruning.
- Smaller k values generally reduce localization recall and hurt mAP.
- This supports the Stage2S conclusion: the bottleneck is not candidate count alone; it is ranking the correct candidate without throwing away recall.
- Animal still has loc R@0.75 = 0.0000, so high animal AP50 comes from low-IoU matches, not precise localization.

## Next Direction

The next meaningful improvement should not be another hand-tuned top-k rule. Options:

1. Use Stage2T score as a calibration layer for final submission if mAP50-95 is primary.
2. Add image/temporal features to the selector, because tabular features only give a small gain.
3. Keep Stage2O and Stage2T as two heads and choose the final score according to the official metric definition.

## Verification

Passed before full run:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/export_stage2o_prediction_rows.py scripts/task2/train_stage2t_tabular_selector.py scripts/task2/train_stage2o_candidate_ranker.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2t_tabular_selector.py tests/test_task2_stage2o_ranker.py tests/test_task2_stage2s_topk_pruning.py
```

Result:

- 9 tests passed.

GPU note:

- ordinary Codex sandbox did not expose `/dev/nvidia*`;
- sandbox-external `nvidia-smi` saw RTX 6000 Ada, driver 570.195.03, CUDA 12.8;
- train-side ROI prediction export was run sandbox-external with `cardiac-diffusion` to access GPU.
