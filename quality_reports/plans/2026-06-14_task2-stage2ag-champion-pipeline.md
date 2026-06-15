# Task2 Stage2AG Champion Pipeline Plan

Date: 2026-06-14

## Goal

Package the current Task2 champion and fallbacks into a repeatable pipeline so public validation, official validation, and hidden-test candidate rows can be exported consistently.

## Variants

- `stage2ab`: domain-aware score policy, no box calibration.
- `stage2ad`: Stage2AB plus animal-only train-fitted box calibration strength 1.0.
- `stage2ae`: Stage2AB plus animal-only box calibration strength 1.25.

## Acceptance Criteria

- The pipeline exports all three variants from the same run directory.
- Each output CSV uses the clean internal schema.
- Metrics are recomputed when GT fields are present.
- Metrics are skipped, not failed, when GT fields are absent.
- Running the pipeline on the current public-validation run reproduces the existing hand-exported AB/AD/AE CSVs exactly.

