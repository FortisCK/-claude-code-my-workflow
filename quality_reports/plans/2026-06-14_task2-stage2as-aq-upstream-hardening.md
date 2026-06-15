# Task2 Stage2AS Stage2AQ Upstream Hardening Plan

Date: 2026-06-14

## Goal

Harden the upstream requirements for Stage2AQ so it can be reproduced on future
official validation / hidden-test splits.

Stage2AR made Stage2AQ exportable once these inputs exist:

- Stage2AE-style baseline candidate/prediction rows;
- Stage2X multi-source candidate/prediction rows containing `yolo_stage2l` and
  `stage2x_class1`;
- clean metadata fields: `sample_id`, `domain`, `source`, `source_rank`,
  boxes, and `rank_decay_roi_score_collision`.

Stage2AS should make those requirements explicit and machine-checkable.

## Constraints

- Do not rerun expensive GPU training unless the verifier shows it is necessary.
- Do not assume hidden-test GT fields exist.
- The verifier must work for both public-validation rows with GT and hidden-test
  rows without GT.
- Fail loudly on missing Stage2X rows or source/alignment mismatches.

## Steps

1. Inspect existing Stage2X / candidate-pool / ranker scripts and saved args.
2. Define the minimal input contract for Stage2AQ.
3. Implement a verifier that checks:
   - baseline clean predictions;
   - multi-source candidate and prediction rows;
   - required sources and score columns;
   - row alignment;
   - clean output schema after Stage2AQ replacement.
4. Run the verifier on current public-validation Stage2AR artifacts.
5. Record the result and the commands needed to regenerate upstream artifacts.

## Acceptance Criteria

- A script can verify the current Stage2AQ upstream inputs and exported outputs.
- The script reports source counts, row counts, split coverage, and GT
  availability.
- Current public-validation Stage2AQ artifacts pass the verifier.
- Task2 regression tests still pass.
