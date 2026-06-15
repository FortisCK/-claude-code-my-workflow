# Task2 Stage2AV No-GT Candidate Pool Plan

Date: 2026-06-14

## Goal

Add a hidden-test-safe candidate-pool normalizer that combines precomputed
proposal CSVs from multiple sources into Stage2U-compatible candidate CSVs
without requiring GT fields.

Stage2AU can already run the frozen Stage2U quality ranker on no-GT candidate
rows. Stage2AV should create those candidate rows from proposal sources.

## Constraints

- No training and no expensive GPU inference in this step.
- Do not change the labeled public-validation Stage2O behavior.
- Do not require `gt_class`, `gt_x*`, `candidate_iou`, or labels.
- Preserve enough metadata for downstream Stage2AQ export:
  `sample_id`, `video_id`, `frame_index`, `domain`, image path/size,
  candidate box, `source`, `source_priority`, `source_rank`, `source_conf`.

## Steps

1. Inspect existing proposal-source CSV schemas.
2. Implement a no-GT multi-source candidate normalizer.
3. Add tests for mixed proposal schemas and top-k filtering.
4. Document how it connects to Stage2AU and Stage2AQ export.
5. Run focused tests and Task2 regression.

## Acceptance Criteria

- The normalizer reads multiple `name=proposal_dir` sources and one or more
  split names.
- It writes `{split}_candidates.csv` without GT requirements.
- Output rows can be read by `infer_stage2u_quality_ranker.py`.
- Existing Task2 regression stays green.
