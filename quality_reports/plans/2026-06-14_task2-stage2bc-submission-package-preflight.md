# Task2 Stage2BC Submission Package Preflight Plan

Date: 2026-06-14

## Goal

Add a package-level preflight for Task2 submission readiness. The preflight
should verify the files needed by the submission-facing inference entrypoint
without running model inference.

## Scope

Check:

- submission entrypoint script;
- hidden Stage2AQ wrapper;
- no-GT proposal exporters;
- no-GT candidate pool builder;
- no-GT Stage2U inference script;
- champion exporter;
- required trained weights/checkpoints;
- optional input image directory or image-list CSV;
- optional output directory parent.

## Constraints

- Read-only except writing the preflight manifest.
- Do not run training or inference.
- Do not invent the official challenge output schema.
- `--strict` should return nonzero if required checks fail.

## Acceptance Criteria

- A command writes a JSON preflight manifest.
- Required present files pass on the current repository.
- Tests cover both passing and failing checks.
- Full Task2 regression passes.

