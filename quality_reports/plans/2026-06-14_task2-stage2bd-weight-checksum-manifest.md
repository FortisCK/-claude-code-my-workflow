# Task2 Stage2BD Weight Checksum Manifest Plan

Date: 2026-06-14

## Goal

Add checksum tracking for the three trained artifacts required by the Task2
submission inference package.

## Motivation

Stage2BC verifies that required files exist and are non-empty. For submission
packaging, existence is not enough: a copied or Docker-mounted checkpoint can be
truncated or replaced accidentally. Stage2BD should make the required weight
versions explicit and machine-checkable.

## Required Artifacts

- YOLO Stage2L weights
- Stage2X class1 checkpoint
- Stage2U quality-ranker checkpoint

## Steps

1. Add a script that computes sha256 and size for the required Task2 weights.
2. Write the manifest to `quality_reports/decisions/`.
3. Add an optional `--checksum-manifest` path to package preflight.
4. If provided, preflight should verify sha256 and size.
5. Add focused tests for checksum pass/fail behavior.
6. Run full Task2 regression.

## Acceptance Criteria

- A checksum manifest can be generated from the current local weights.
- Package preflight with that manifest passes locally.
- A deliberately mismatched checksum fails under strict mode.
- Full Task2 tests pass.

