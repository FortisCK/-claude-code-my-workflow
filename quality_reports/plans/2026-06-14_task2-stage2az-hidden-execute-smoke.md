# Task2 Stage2AZ Hidden Raw-Image Execute Smoke Plan

Date: 2026-06-14

## Goal

Verify that the Stage2AY raw-image hidden wrapper can execute a small no-GT
pipeline end to end, not just generate a dry-run manifest.

## Constraints

- Keep default full hidden inference behavior unchanged.
- Use a tiny image limit for smoke testing to avoid expensive full inference.
- Do not require ground-truth labels.
- Preserve the existing precomputed-proposal mode and raw-image dry-run mode.

## Steps

1. Add a wrapper-level `--proposal-limit` argument and pass it to YOLO/Stage2X
   no-GT proposal exporters only when set.
2. Add/extend command-generation tests so the limit appears in raw-image
   proposal commands but not in downstream candidate/ranker/export commands.
3. Run focused tests and Task2 regression if the focused checks pass.
4. Execute a tiny raw-image pipeline smoke test and verify that the final
   Stage2AQ prediction CSV exists and has rows.

## Acceptance Criteria

- `run_stage2aq_hidden_pipeline.py --generate-proposals --proposal-limit N`
  includes `--limit N` in both proposal export commands.
- Focused wrapper tests pass.
- Full Task2 tests pass.
- A tiny execute smoke produces:
  - YOLO proposal CSV
  - Stage2X proposal CSV
  - candidate CSVs
  - Stage2U prediction rows
  - final Stage2AQ prediction CSV

