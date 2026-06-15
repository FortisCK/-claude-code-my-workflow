# Task2 Stage2AR Stage2AQ Export Hardening Plan

Date: 2026-06-14

## Goal

Make Stage2AQ reproducible as an exportable Task2 candidate, not just an
experiment directory.

Stage2AQ is currently the best balanced public-validation candidate:

- valid_combined mAP50: `0.23661313056264216`
- valid_combined mAP50-95: `0.07023557986011882`

It improves phantom class1 by replacing only phantom class1 predictions with a
controlled `yolo_stage2l + stage2x_class1` multi-source top-k policy while
keeping Stage2AE elsewhere.

## Constraints

- Do not change Stage2AE/AI/AN behavior.
- Keep clean prediction schema validation.
- Keep the public-validation metric recomputation path.
- Hidden-test support should fail clearly if required Stage2X candidate rows are
  missing, rather than silently producing Stage2AE.

## Steps

1. Inspect the current champion pipeline and Stage2AQ sweep script.
2. Add a reusable Stage2AQ export path:
   - Stage2AE-style baseline export;
   - replacement of phantom class1 from a Stage2X/multi-source run directory;
   - clean CSV schema validation;
   - manifest recording selected policy.
3. Add focused tests for the Stage2AQ replacement helper.
4. Run the pipeline on public validation and verify that Stage2AQ reproduces the
   Stage2AQ sweep metrics.
5. Record the result and update current status.

## Acceptance Criteria

- Stage2AQ export reproduces:
  - valid_combined mAP50 `0.23661313056264216`;
  - valid_combined mAP50-95 `0.07023557986011882`.
- Stage2AQ clean prediction CSVs pass schema checks.
- Existing Stage2AB/AD/AE/AI/AN tests still pass.
- Full Task2 regression tests pass.
