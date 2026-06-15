# Task 2 Startup: Inventory, Literature, and Baseline Direction

Date: 2026-06-04

Status: approved by user request

## Objective

Start CATHACTION Task 2 collision detection work after freezing Task 1.
The immediate goal is not to train a model blindly, but to establish a reliable
Task 2 problem definition, data schema, metric interpretation, and first
baseline direction.

## Scope

1. Inspect the local `datasets/collision_detection/` folder.
2. Confirm image/label pairing, label format, class counts, box size
   distribution, video/case identifiers, and any split implications.
3. Re-check the CATHACTION guide and benchmark paper for Task 2 task definition,
   metrics, and published baselines.
4. Search for any newer direct CathAction Task 2 papers or methods.
5. Produce a short decision record with:
   - task definition;
   - data structure;
   - known baselines;
   - recommended first baseline;
   - risks and immediate next steps.

## Non-Goals

- Do not start a long GPU training run in this startup pass.
- Do not create a final Docker pipeline yet.
- Do not tune on any hidden-test feedback.

## Verification

- The data inventory must report counts and class distribution from local files.
- The metric summary must distinguish AP from mAP and identify the primary
  ranking metric.
- The baseline decision must be grounded in local data format and public
  benchmark evidence.
