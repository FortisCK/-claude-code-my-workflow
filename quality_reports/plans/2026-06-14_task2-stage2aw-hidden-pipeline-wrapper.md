# Task2 Stage2AW Hidden Stage2AQ Pipeline Wrapper Plan

Date: 2026-06-14

## Goal

Add a dry-run-first wrapper for hidden-test style Stage2AQ export.

Stage2AQ hidden inference requires two no-GT branches:

1. yolo-only candidates -> Stage2U no-GT inference -> Stage2AE baseline;
2. multi-source candidates -> Stage2U no-GT inference -> Stage2AQ replacement.

The wrapper should connect Stage2AV, Stage2AU, and the existing champion export
without training or assuming GT.

## Constraints

- Dry-run by default.
- No training in this step.
- No GT fields required.
- Keep the existing public-validation champion pipeline unchanged.
- Support `--execute`, but verify command generation through tests.

## Steps

1. Implement `run_stage2aq_hidden_pipeline.py`.
2. Generate commands for:
   - yolo-only no-GT candidate pool;
   - multi-source no-GT candidate pool;
   - yolo-only Stage2U no-GT inference;
   - multi-source Stage2U no-GT inference;
   - Stage2AE/Stage2AQ champion export.
3. Add dry-run manifest tests.
4. Run focused tests and full Task2 regression.
5. Record result and remaining official result-format gap.

## Acceptance Criteria

- Dry-run manifest lists the five required commands in order.
- The wrapper exposes split/source/checkpoint/output parameters.
- Existing Task2 tests pass.
