# Task2 Stage2AU Hidden No-GT Inference Hardening Result

Date: 2026-06-14

## Decision

Add a dedicated Stage2U no-GT inference entry point:

`scripts/task2/infer_stage2u_quality_ranker.py`

This script is the hidden-test oriented companion to
`train_stage2u_quality_ranker.py --eval-checkpoint`.

It consumes candidate CSVs that do not contain `gt_class`, `gt_x*`, `gt_iou`,
or `candidate_iou`, and writes the same artifact pair expected by the existing
Stage2AB/AE/AQ export code:

- `{split}_candidates_used.csv`
- `{split}_eval_prediction_rows.csv`

## Why

Stage2AT made the official-style validation path reproducible, but it still
assumed labeled validation splits. Hidden-test submission should not depend on
GT-bearing candidate construction or metric computation.

Stage2AU separates inference from validation:

- input candidate rows can be no-GT;
- model inference still uses the frozen Stage2U checkpoint;
- exported prediction rows still include Stage2AQ score columns such as
  `rank_decay_roi_score_collision`;
- downstream clean-prediction export can remain unchanged.

## Scope

This stage does not generate proposal rows by itself. The hidden-test pipeline
still needs upstream proposal CSVs from the selected proposal sources. Stage2AU
only provides the no-GT quality-ranker bridge once candidate rows exist.

The script writes placeholder GT values in normalized internal
`*_candidates_used.csv` rows only to preserve the shared `CandidateExample`
container. The input does not require GT, and final clean predictions still do
not expose GT fields.

## Verification

Syntax:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m py_compile \
  scripts/task2/infer_stage2u_quality_ranker.py \
  scripts/task2/run_stage2aq_official_pipeline.py
```

Focused tests:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest \
  tests/test_task2_stage2u_nogt_inference.py \
  tests/test_task2_stage2at_orchestration.py
```

Result: 4 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest tests/test_task2_*.py
```

Result: 91 passed.

## Next

The remaining hidden-test gap is upstream proposal generation without labels:

1. create/verify no-GT proposal source CSVs for the official input format;
2. build a no-GT candidate-pool normalizer that does not compute verifier
   labels or oracle IoU;
3. connect proposal generation -> Stage2AU ranker inference -> Stage2AQ export
   into one Docker-safe command.
