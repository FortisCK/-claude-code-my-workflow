# Task2 Stage2AX Proposal Generation Wrapper Plan

Date: 2026-06-14

## Goal

Harden the upstream proposal-generation side of the hidden-test Stage2AQ
pipeline.

Stage2AW can run from precomputed proposal CSVs. Stage2AX should inspect and
codify how proposal CSVs are produced from raw official images, or clearly
separate scripts that are public-validation-only because they require labels.

## Constraints

- No training in this step.
- No expensive full-dataset GPU inference unless explicitly needed later.
- Do not assume hidden-test labels or GT boxes.
- Keep public-validation scripts working.
- Prefer a dry-run manifest wrapper first.

## Steps

1. Inspect existing proposal-generation scripts for label/GT dependencies.
2. Classify each source as hidden-ready, needs no-GT adapter, or validation-only.
3. Implement a dry-run proposal-generation wrapper/manifest for hidden inputs.
4. Add tests for command generation and source classification.
5. Update current status and run Task2 regression.

## Acceptance Criteria

- We know which Stage2AQ proposal sources can be generated from hidden images.
- A script records the exact commands or blockers for producing proposal CSVs.
- Existing Task2 regression remains green.
