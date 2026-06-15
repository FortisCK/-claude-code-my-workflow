# Task2 Stage2AS Stage2AQ Upstream Hardening Result

Date: 2026-06-14

## Question

Can we make Stage2AQ's upstream requirements machine-checkable for future
official validation / hidden-test splits?

## Implementation

Added:

- `scripts/task2/verify_stage2aq_upstream.py`
- `tests/test_task2_stage2aq_upstream_verify.py`

The verifier checks:

- baseline clean predictions exist and match the clean schema;
- multi-source candidate rows exist for each split;
- multi-source prediction rows exist for each split;
- candidate/prediction rows are aligned by `sample_id`, `source`, and
  `source_rank`;
- required Stage2AQ sources exist for the selected source policy;
- required score column exists, default
  `rank_decay_roi_score_collision`;
- Stage2AQ clean prediction CSVs exist and match the clean schema unless
  `--allow-missing-stage2aq` is passed;
- GT field availability is recorded but not required.

## Command

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/verify_stage2aq_upstream.py \
  --output-json outputs/task2/stage2as_aq_upstream_verify/public_valid_verify_summary.json
```

Output:

`outputs/task2/stage2as_aq_upstream_verify/public_valid_verify_summary.json`

## Public-Validation Verification

Verifier passed on the current Stage2AR public-validation artifacts.

Important split summaries:

| Split | candidate rows | prediction rows | Stage2AQ rows | required source rows |
| --- | ---: | ---: | ---: | ---: |
| `valid_combined` | 206,635 | 206,635 | 25,421 | 60,869 |
| `valid_phantom` | 184,469 | 184,469 | 23,789 | 60,869 |
| `valid_animal` | 22,166 | 22,166 | 1,632 | 0 |

The required source policy is `yolo_stage2x`:

- `yolo_stage2l`
- `stage2x_class1`

The verifier confirms that both sources exist in the combined and phantom
multi-source rows. `valid_animal` has no phantom rows, so zero replacement
candidates is expected.

## Regeneration Pointers

Current public-validation Stage2AQ depends on:

1. Stage2AE baseline clean predictions from the champion pipeline.
2. Stage2X five-source candidate pool:
   `outputs/task2/stage2o_candidate_pool/stage2x_five_source_valid_top50`
3. Stage2U eval rows over that candidate pool:
   `outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval`

The saved args are:

- `outputs/task2/stage2o_candidate_pool/stage2x_five_source_valid_top50/args.json`
- `outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval/args.json`

For official/hidden splits, the same two upstream stages must be run on the
official split before calling `stage2aq` in the champion pipeline.

## Decision

Stage2AQ is now hardened at the input-contract level.

Next work should focus on creating a single official-split orchestration command
that runs:

1. proposal generation / collection for Stage2X sources;
2. Stage2O candidate-pool build;
3. Stage2U eval-row generation using the frozen ranker checkpoint;
4. `verify_stage2aq_upstream.py`;
5. `run_stage2_champion_pipeline.py --variant stage2aq`.

## Verification

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/verify_stage2aq_upstream.py \
  scripts/task2/run_stage2_champion_pipeline.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_stage2aq_upstream_verify.py \
  tests/test_task2_champion_pipeline.py
```

Focused result: 8 passed.

Full Task2 regression also passed:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 87 passed.
