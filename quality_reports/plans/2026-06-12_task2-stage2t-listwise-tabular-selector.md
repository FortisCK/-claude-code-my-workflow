# Task2 Stage2T Plan: Listwise Tabular Candidate Selector

Date: 2026-06-12
Status: completed

## Goal

Improve the current Task 2 champion by learning a per-frame candidate selector over the existing YOLO+Task1-geometry candidate pool.

Current champion:

- run: `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval`
- candidate pool: original Stage2O YOLO+geometry pool
- score mode: `rank_decay_roi`
- valid_combined mAP50: 0.1887
- valid_combined mAP50-95: 0.0388

Stage2S showed that naive top-k pruning does not improve this, but oracle candidate selection raises valid_combined mAP50-95 to about 0.14. The gap suggests the next useful component is a learned candidate selector, not another independent ROI classifier.

## Constraints

- Train only on current train candidate rows and labels.
- Do not use valid labels for model fitting, calibration, feature scaling, threshold selection, or policy selection.
- Use valid only for final reporting and diagnostic comparison against the current champion.
- Preserve the current Stage2O champion unchanged unless Stage2T improves mAP.
- Keep this first selector tabular/CPU-first so it can be debugged quickly.

## Implementation Steps

1. Export train-side Stage2O ROI probabilities.
   - Reuse the existing Stage2O checkpoint.
   - Run eval-only inference on `train_candidates_used.csv`.
   - Add a narrow `--skip-overlap-check` flag to the ranker script so this intentional train-candidate export is possible without weakening default leakage checks.

2. Implement `scripts/task2/train_stage2t_tabular_selector.py`.
   - Inputs: train candidates + train prediction rows; valid candidates + valid prediction rows.
   - Features: source metadata, source rank/confidence, candidate geometry, ROI probabilities, Stage2O score modes, and per-sample ranks.
   - Target: candidate localization quality, primarily `gt_iou` regression and/or `gt_iou >= threshold` classification.
   - Model: scikit-learn histogram gradient boosting or random forest fallback.
   - Output: selected/scored prediction rows and mAP summary.

3. Evaluate direct policies.
   - predicted-quality candidate top-k by frame;
   - class predictions scored as predicted-quality times Stage2O class probabilities;
   - compare against existing `rank_decay_roi` baseline using the same candidate pool and metric.

4. Add focused tests.
   - feature construction does not use `gt_iou` as an input;
   - candidate/prediction CSV alignment is enforced;
   - per-frame selection preserves at most k candidates per sample;
   - metric conversion produces both class 0 and class 1 predictions.

5. Write a decision record.
   - Accept Stage2T only if it improves valid_combined mAP50-95 over 0.0388.
   - Otherwise keep Stage2O as current champion and document the failure mode.

## Verification

Required before completion:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/train_stage2t_tabular_selector.py scripts/task2/train_stage2o_candidate_ranker.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2t_tabular_selector.py tests/test_task2_stage2o_ranker.py tests/test_task2_stage2s_topk_pruning.py
```

Then run a full valid evaluation and compare to the current champion.

## Result

Completed in `quality_reports/decisions/2026-06-12_task2_stage2t_tabular_selector_result.md`.

Best Stage2T policy:

- quality mode: `prob_iou75`;
- base score: `rank_decay_roi`;
- k: 50;
- valid_combined mAP50: 0.1551;
- valid_combined mAP50-95: 0.0403.

This is a small mAP50-95 improvement over the previous Stage2O champion (0.0388), but it reduces mAP50 from 0.1887 to 0.1551. Keep Stage2O as the AP50/robustness baseline.
