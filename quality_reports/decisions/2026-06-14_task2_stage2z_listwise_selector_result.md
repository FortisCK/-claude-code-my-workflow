# Task2 Stage2Z Result: Existing Stage2T Listwise Selector on Five-Source Pool

Date: 2026-06-14
Status: completed diagnostic

## Question

Can the existing Stage2T tabular per-frame selector improve the expanded five-source candidate pool by selecting top-k candidates per frame?

## Setup

The selector was trained from existing YOLO+geometry train-side prediction rows:

- train candidates:
  - `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_candidates_used.csv`;
- train predictions:
  - `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_prediction_rows.csv`.

Validation target:

- valid run:
  - `outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval`;
- split:
  - `valid_combined`.

Output:

`outputs/task2/stage2t_tabular_selector/histgb_yologeom_train_stage2x_fivesource_valid_combined`

Command used:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/task2/train_stage2t_tabular_selector.py \
  --train-candidate-csv outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_candidates_used.csv \
  --train-prediction-csv outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_prediction_rows.csv \
  --valid-run-dir outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval \
  --split valid_combined \
  --output-dir outputs/task2/stage2t_tabular_selector \
  --name histgb_yologeom_train_stage2x_fivesource_valid_combined \
  --max-iter 200 \
  --learning-rate 0.06 \
  --l2-regularization 0.02 \
  --score-mode roi \
  --score-mode rank_decay_roi \
  --score-mode source_rank_decay_roi \
  --quality-mode pred_iou \
  --quality-mode prob_iou50 \
  --quality-mode prob_iou75 \
  --quality-mode blend \
  --k 1 --k 2 --k 3 --k 5 --k 10 --k 20 --k 50
```

## Result

Best full `valid_combined` policy:

| Quality | Base score | k | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| `blend` | `roi` | 50 | 0.1645 | 0.0371 |

Comparison:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2T transferred to five-source | 0.1645 | 0.0371 |
| five-source old ranker | 0.1648 | 0.0303 |
| YOLO+geometry mixed-source | 0.1875 | 0.0414 |
| Stage2V champion | 0.2010 | 0.0492 |

## Decision

Do not promote Stage2Z.

The existing Stage2T selector improves over the weak five-source old-ranker score, but it does not beat the simpler YOLO+geometry policy and is well below Stage2V.

This first listwise transfer test failed because the selector was trained only on YOLO+geometry sources, then applied to a five-source pool with unseen source categories. Expanding it would require generating matched train-side prediction rows for all five sources and training a true source-balanced selector.

## Implication

At this point, three attempts to exploit the expanded candidate pool have failed on full validation:

1. Stage2X direct five-source ranker;
2. Stage2Y source-aware residual ranker;
3. Stage2Z transferred tabular/listwise selector.

The current robust path is to harden Stage2V as the Task2 deliverable unless we commit to a larger redesign that creates source-matched train and valid candidate pools for a true listwise selector.
