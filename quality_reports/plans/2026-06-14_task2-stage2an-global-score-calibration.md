# Task2 Stage2AN Global Score Calibration Plan

Date: 2026-06-14

## Goal

Test a conservative global score calibration layer on top of Stage2AE clean
predictions.

Stage2AM showed that per-domain local score optimization can improve phantom
metrics while collapsing `valid_combined` AP. Stage2AN therefore optimizes the
global combined precision-recall curve directly.

## Constraint

Stage2AN must not:

- train or fine-tune a CNN;
- generate new candidates;
- alter boxes;
- replace Stage2AE as fallback unless public-validation evidence is clearly
  better.

It may only multiply existing Stage2AE scores by small class/domain scale
factors and re-evaluate AP.

## Steps

1. Implement a clean-prediction score-scale sweep.
2. Include the identity scale as the baseline, which must reproduce Stage2AE.
3. Search small class/domain scale grids while optimizing `valid_combined`.
4. Export calibrated clean prediction CSVs only if the best policy is useful.
5. Record whether Stage2AN promotes over Stage2AE or remains a negative result.

## Acceptance Criteria

- Identity policy exactly reproduces Stage2AE metrics.
- Selected policy improves `valid_combined` mAP50 and/or mAP50-95 without a
  large tradeoff.
- Report split and class metrics for `valid_combined`, `valid_phantom`, and
  `valid_animal`.

