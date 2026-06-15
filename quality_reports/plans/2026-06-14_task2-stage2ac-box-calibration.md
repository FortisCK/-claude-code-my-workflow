# Task2 Stage2AC Box Calibration Diagnostic Plan

Date: 2026-06-14

## Goal

Test whether the current Task2 Stage2AB bottleneck is partly caused by systematic box geometry bias. Learn a simple domain/class box correction from train-side candidates and evaluate it on public validation using the frozen Stage2AB score policy.

## Rationale

Stage2AB has a strong AP50 signal in some slices but very weak AP50-95, especially where AP drops sharply above IoU 0.50. That pattern can indicate boxes that are ranked reasonably but are consistently shifted or scaled.

## Steps

1. Fit per-domain/per-class box transforms from train candidates and GT boxes.
2. Apply those transforms to validation candidate boxes while keeping the Stage2AB score policy fixed.
3. Compute detection mAP for calibrated and uncalibrated boxes.
4. Decide whether deterministic box calibration is worth promoting or whether the bottleneck is elsewhere.

## Acceptance Criteria

- The fitting step uses train candidate GT only.
- The validation metric computation may use validation GT because this is an internal diagnostic, not a hidden-test export.
- The resulting JSON includes baseline and calibrated metrics plus fitted transforms.
- If calibrated mAP50-95 does not improve meaningfully, reject the route and document why.

