# Task2 Stage2AU Hidden No-GT Inference Hardening Plan

Date: 2026-06-14

## Goal

Create the missing no-GT inference bridge for the Stage2AQ pipeline.

Stage2AT can orchestrate labeled official-style validation panels. Hidden-test
submission will not have `gt_*` fields, so the current Stage2U eval path is not
enough. Stage2AU should add a small inference entry point that consumes
candidate CSVs without labels / GT boxes and writes the same
`*_eval_prediction_rows.csv` schema expected by the Stage2AQ exporter.

## Constraints

- Do not train or run expensive GPU jobs in this step.
- Preserve existing labeled validation behavior.
- No leakage: no GT fields should be required by the no-GT inference path.
- Keep output schema compatible with `run_stage2_champion_pipeline.py`.

## Steps

1. Add a no-GT candidate loader for Stage2U-compatible candidate rows.
2. Add an eval-only prediction-row writer that does not compute metrics.
3. Add tests with a minimal synthetic candidate CSV lacking `gt_*` fields.
4. Wire Stage2AT manifest notes if needed.
5. Run focused tests and Task2 regression.

## Acceptance Criteria

- A no-GT candidate CSV can be parsed without `gt_class`, `gt_x*`, or
  `candidate_iou`.
- Prediction rows include the score columns consumed by Stage2AQ, especially
  `rank_decay_roi_score_collision`.
- Existing public-validation tests still pass.
