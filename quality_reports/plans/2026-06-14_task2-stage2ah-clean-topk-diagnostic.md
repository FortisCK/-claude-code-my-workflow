# Task2 Stage2AH Clean Prediction Top-K Diagnostic Plan

Date: 2026-06-14

## Goal

Evaluate whether pruning Stage2AE clean prediction rows by per-sample/class top-k improves AP by reducing false positives.

## Policies

- `all`: no pruning.
- `global_topk`: keep top-k predictions per sample and class over all domains.
- `animal_topk`: keep all non-animal predictions and only prune animal predictions per sample and class.

## Acceptance Criteria

- Promote only if pruning produces a meaningful valid_combined mAP50-95 improvement without lowering mAP50 or adding fragile validation-only behavior.
- Otherwise record as a negative diagnostic and keep Stage2AE unchanged.

