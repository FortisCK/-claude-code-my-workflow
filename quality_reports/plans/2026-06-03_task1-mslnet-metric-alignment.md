# Task 1 MSLNet Metric Alignment Plan

Date: 2026-06-03

Status: completed

## Goal

Evaluate our Task 1 predictions using the segmentation metrics described in
the MSLNet paper, so comparisons against Table 2 and Table 5 are as close as
possible to apples-to-apples.

## Scope

- Keep the current released evaluation manifest unchanged.
- Evaluate saved prediction PNGs only; do not retrain or tune models.
- Collapse `label_1` and `label_2` into a binary foreground mask, matching the
  segmentation-style comparison in the MSLNet paper.
- Compute:
  - standard binary Dice;
  - standard binary IoU;
  - distance-tolerance precision / recall / F1 for radii 0 through 4 pixels;
  - average Hausdorff distance (AHD), defined as the average of prediction-to-GT
    and GT-to-prediction nearest-pixel distances.

## Implementation

1. Add reusable MSLNet-style metric helpers under `src/cathaction/metrics/`.
2. Add `scripts/task1/evaluate_mslnet_style.py` for saved prediction CSVs.
3. Add focused tests for exact, one-pixel-shift, and manifest-driven cases.
4. Run the evaluator on the current Stage 9A champion:
   `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`.
5. Save the JSON result under `outputs/task1/diagnostics/`.

## Interpretation Guardrail

MSLNet reports mean and standard deviation over four independent runs. We only
have one trained ensemble, so our result should be compared to the MSLNet mean
as an engineering reference, not as a statistical claim.

## Completed Result

Evaluator:

- `scripts/task1/evaluate_mslnet_style.py`
- reusable metrics: `src/cathaction/metrics/mslnet_style.py`

Current Stage 9A champion result:

- prediction CSV:
  `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`
- output JSON:
  `outputs/task1/diagnostics/stage9a_seven_model_remove_small_min32_mslnet_style/eval_mslnet_style.json`
- per-sample CSV:
  `outputs/task1/diagnostics/stage9a_seven_model_remove_small_min32_mslnet_style/per_sample_mslnet_style.csv`

Sample-mean binary foreground metrics:

- Dice: `0.6215624175077626`
- IoU: `0.4605049121513648`
- AHD: `1.3713229393268282`
- F1 radius 0: `0.6215624175077626`
- F1 radius 1: `0.7607678985630797`
- F1 radius 2: `0.8514899720219553`
- F1 radius 3: `0.911577665104627`
- F1 radius 4: `0.944560505299149`

Compared with MSLNet Table 2 / Table 5 on CathAction segmentation:

- MSLNet Dice: `0.6251`; ours: `0.6216`.
- MSLNet IoU: `0.4658`; ours: `0.4605`.
- MSLNet AHD: `1.5`; ours: `1.37`.
- MSLNet F1 at 3 px: `0.9305`; ours: `0.9116`.
- MSLNet F1 at 4 px: `0.9492`; ours: `0.9446`.

Interpretation: under MSLNet-style binary segmentation metrics, our current
champion is close to but below MSLNet on Dice/IoU/F1, while AHD is slightly
better. The largest gap is the 3-pixel tolerance F1, driven mostly by phantom.
