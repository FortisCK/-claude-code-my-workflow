# Task2 Stage2S Plan: Top-k Output Control Diagnostic

Date: 2026-06-12
Status: completed

## Motivation

The current best Task 2 pipeline is still the old Stage2O ranker with the original YOLO+geometry candidate pool:

- valid_combined rank_decay mAP50: 0.1887
- valid_combined rank_decay mAP50-95: 0.0388

Stage2P/Stage2Q showed that better localization candidates exist, but independent ROI classification does not convert them into AP. The likely structural issue is that each frame emits too many predictions, so false positives and ranking errors dominate mAP.

## Objective

Run a no-training diagnostic over existing prediction rows to test whether per-frame output control improves mAP.

## Inputs

Primary input:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval`

This run contains:

- `valid_*_candidates_used.csv`
- `valid_*_eval_prediction_rows.csv`
- `eval_metrics.json`

## Method

For each split and score mode, evaluate these pruning policies:

- `all`: original no-pruning baseline.
- `candidate_top{k}`: keep top-k candidate boxes per frame by max class score.
- `prediction_top{k}`: keep top-k class-specific predictions per frame.
- `class_top{k}`: keep top-k predictions per class per frame.
- `source_candidate_top{k}`: keep top-k candidate boxes per source per frame.
- `source_class_top{k}`: keep top-k class-specific predictions per source/class/frame.
- `oracle_candidate_top{k}`: leakage-only diagnostic upper bound using GT IoU for candidate selection.

Use k values:

- 1, 2, 3, 5, 10, 20, 50

## Gates

Promote output control into the Task 2 pipeline if any non-oracle policy improves valid_combined mAP50-95 over 0.0388 by at least +0.005.

If only oracle policies improve, then the candidate set is sufficient but learned per-frame selection is the missing component.

If no policy improves, then AP failure is not mainly caused by too many outputs; return to proposal quality or class/domain mismatch.

## Verification

- Add focused tests for top-k selection mechanics.
- Run py_compile and pytest.
- Run Stage2S diagnostic on full-valid old ranker outputs.
- Save JSON/CSV/Markdown report.

## Result

Decision report:

- `quality_reports/decisions/2026-06-12_task2_stage2s_topk_output_control_result.md`

Outcome:

- No non-oracle top-k/source-aware pruning policy improves valid_combined mAP50-95 over the current best 0.0388.
- Best non-oracle combined result is effectively the original baseline:
  - `rank_decay_roi`, `prediction_top@50`, mAP50-95 0.0388.
- Oracle top-1 candidate selection is much stronger:
  - valid_combined mAP50-95 0.1410 to 0.1476 depending on score mode.

Decision:

- reject hand-crafted top-k pruning as a direct pipeline improvement;
- keep old Stage2O ranker + original YOLO/geometry candidate pool as champion;
- next direction should be a learned per-frame/listwise candidate selector, not more independent ROI classification.
