# Task2 Stage2AE Calibration Strength Sweep Plan

Date: 2026-06-14

## Goal

Improve Stage2AD by sweeping the strength of the train-fitted animal-only box transform while keeping the Stage2AB score policy fixed.

## Steps

1. Add a `--transform-strength` option to the calibration diagnostic.
2. Sweep animal-only strengths on `valid_combined`.
3. Evaluate the best strength on `valid_phantom` and `valid_animal`.
4. Export clean predictions using the GT-free Stage2AB exporter and the best transform JSON.
5. Verify schema/policy and record whether the result supersedes Stage2AD.

## Acceptance Criteria

- Stage2AE must keep valid_combined mAP50 equal to Stage2AB/Stage2AD.
- Stage2AE must improve valid_combined mAP50-95 over Stage2AD.
- Phantom metrics must remain unchanged because phantom uses identity transforms.
- Clean CSV export must pass schema/policy verification.

