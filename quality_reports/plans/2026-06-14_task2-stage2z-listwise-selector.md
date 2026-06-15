# Task2 Stage2Z Listwise Selector Plan

Date: 2026-06-14

## Motivation

Stage2X and Stage2Y show the same pattern:

- expanded candidate pools improve oracle recall;
- independent per-candidate scoring does not convert that recall into mAP;
- sampled validation can be misleading;
- full valid combined remains below the Stage2V champion.

The next hypothesis is that selection should happen per frame/case over a candidate list, not independently per candidate.

## Objective

Evaluate whether the existing Stage2T tabular per-frame selector can improve the five-source full-valid candidate pool by choosing a small number of candidates per frame.

## Initial Diagnostic

Use existing artifacts only:

- train candidate/prediction rows:
  - `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_candidates_used.csv`;
  - `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_prediction_rows.csv`;
- valid run dir:
  - `outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval`.

This tests whether a selector trained on YOLO+geometry can generalize to a larger source set. Unknown sources will have no fitted one-hot column, so this is a conservative first pass.

## Gate

Continue only if full `valid_combined` improves over:

- five-source old ranker: mAP50-95 about `0.0303`;
- YOLO+geometry mixed source: mAP50-95 about `0.0414`.

Promotion requires beating Stage2V:

- Stage2V valid_combined mAP50-95: `0.04924`.

If this diagnostic is below `0.041`, do not expand it; move to Task2 Stage2V hardening/reporting unless a new candidate formulation is introduced.
