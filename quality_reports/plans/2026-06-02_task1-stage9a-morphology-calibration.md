# Task 1 Stage 9A Morphology Calibration Plan

Date: 2026-06-02

## Goal

Convert the thin-line tolerance diagnostic into an exact Dice improvement by
post-processing existing hard prediction masks.

The diagnostic showed:

- exact Dice around `0.65`;
- 1px tolerant Dice around `0.80`;
- 2px tolerant Dice around `0.88`;
- class-swap gain near zero.

This suggests that line width, small gaps, and 1-3px alignment errors are a
more important bottleneck than generic backbone capacity or label swapping.

## Input

Use the existing full five-model prediction manifest first:

`outputs/task1/stage5_submission_smoke/predict_full/predictions.csv`

This is close to the current seven-model champion and avoids rerunning full
seven-model prediction writing before we know whether hard-mask calibration is
worthwhile.

## Method

Search a small hard-mask morphology grid:

- no-op baseline;
- label-wise dilation / erosion;
- label-wise opening / closing;
- small connected-component removal;
- foreground closing with class labels restored from the original prediction;
- optional label priority rules for conflicts.

Report:

- mean Dice;
- label_1 Dice;
- label_2 Dice;
- animal Dice;
- phantom Dice;
- delta versus the no-op baseline.

## Stop Criteria

Continue to seven-model migration only if at least one candidate:

- improves full released-eval mean Dice by about `+0.003` or more, or
- improves phantom materially without hurting animal and mean Dice.

If gains are only `+0.000x`, do not spend more time on hard-mask morphology.

## Artifacts

- `scripts/task1/search_morphology_postprocess.py`
- output JSON/CSV under `outputs/task1/stage9a_morphology_grid/`
- decision note under `quality_reports/decisions/`

