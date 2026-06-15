# Task2 Stage2AY Hidden Raw-Image Wrapper Plan

Date: 2026-06-14

## Goal

Wire raw-image proposal generation into the hidden-test Stage2AQ wrapper.

Stage2AX added hidden-ready no-GT proposal exporters for `yolo_stage2l` and
`stage2x_class1`. Stage2AY should let `run_stage2aq_hidden_pipeline.py` start
from official images directly when requested, while preserving the existing
precomputed-proposal mode.

## Constraints

- Dry-run by default.
- Do not run full GPU inference in this step.
- Preserve precomputed proposal mode.
- Only generate proposal commands for hidden-ready sources.
- Keep `yolo_stage2l + stage2x_class1` as default Stage2AQ hidden source policy.

## Steps

1. Add optional raw-image proposal-generation inputs to the hidden wrapper.
2. Generate YOLO and Stage2X no-GT proposal commands before candidate-pool steps.
3. Ensure downstream source directories point to generated proposal directories.
4. Add command-generation tests.
5. Run Task2 regression and record the result.

## Acceptance Criteria

- Dry-run manifest can start from `--image-dir` or `--image-list-csv`.
- Manifest includes proposal generation -> candidate pool -> Stage2U inference
  -> Stage2AQ export in order.
- Existing precomputed proposal mode still works.
- Full Task2 tests pass.
