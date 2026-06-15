# Task2 Stage2AT Stage2AQ Official-Split Orchestration Plan

Date: 2026-06-14

## Goal

Create a reproducible orchestration entry point for running Stage2AQ on future
official validation / hidden-test splits.

Stage2AQ now has:

- a champion pipeline consumer (`stage2aq`);
- an upstream verifier (`verify_stage2aq_upstream.py`).

Stage2AT should connect those pieces into a command plan that can be dry-run
now and executed later when official split files and proposal rows are present.

## Constraints

- Do not rerun expensive GPU inference/training in this step.
- The orchestration script should default to dry-run.
- It must fail or warn clearly when upstream proposal artifacts are missing.
- It should record all commands and paths in a JSON manifest.
- It must support public-validation reproduction and official/hidden split
  configuration through CLI arguments.

## Steps

1. Inspect saved Stage2X / Stage2O / Stage2U args to identify command
   boundaries.
2. Implement an orchestration script that emits or runs:
   - Stage2O multi-source candidate pool build;
   - Stage2U frozen ranker eval rows;
   - Stage2AQ upstream verification;
   - Stage2AQ champion export.
3. Make public-validation defaults reproduce current artifact paths.
4. Add tests for dry-run command generation.
5. Run dry-run and Task2 regression tests.

## Acceptance Criteria

- Dry-run manifest lists all expected commands in order.
- Existing public-validation artifact paths pass the verifier.
- The script can be run without GPU in dry-run mode.
- Full Task2 regression tests pass.
