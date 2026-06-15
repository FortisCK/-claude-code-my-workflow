# Task2 Stage2AT Stage2AQ Official-Split Orchestration Result

Date: 2026-06-14

## Decision

Stage2AT adds a dry-run-first orchestration entry point for reproducing the
current Stage2AQ pipeline on official-style validation splits:

`scripts/task2/run_stage2aq_official_pipeline.py`

The script does not introduce a new model. It freezes the current best
Stage2AQ path into an ordered command manifest:

1. build the Stage2O multi-source candidate pool;
2. run the frozen Stage2U ConvNeXt quality ranker in eval-only mode;
3. export Stage2AE and Stage2AQ clean prediction CSVs;
4. verify Stage2AQ upstream artifact alignment.

The default mode is dry-run. Actual execution requires `--execute`.

## Public-Validation Dry Run

Command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/run_stage2aq_official_pipeline.py \
  --manifest outputs/task2/stage2at_aq_official_orchestration/public_valid_manifest.json
```

Output:

`outputs/task2/stage2at_aq_official_orchestration/public_valid_manifest.json`

Preflight result:

- missing required inputs: 0
- all default public-validation split files found;
- all default proposal source directories found;
- all `{split}_proposals.csv` files found for `valid_combined`, `valid_phantom`, and `valid_animal`;
- frozen Stage2U checkpoint found;
- Stage2AE baseline run directory found.

## Scope

This stage hardens reproducibility for labeled official validation panels. It
does not yet solve hidden-test no-GT Docker inference. The current Stage2O /
Stage2U public-validation path still expects label-list splits for diagnostics
and candidate verifier labels.

The hidden-test submission path should reuse the same artifact contracts, but it
needs a dedicated no-GT candidate / ranker inference entry point.

## Verification

Syntax:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m py_compile scripts/task2/run_stage2aq_official_pipeline.py
```

Stage2AT tests:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest tests/test_task2_stage2at_orchestration.py
```

Result: 2 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest tests/test_task2_*.py
```

Result: 89 passed.

## Next

The next engineering step should be hidden-test hardening:

- support no-GT sample manifests;
- generate proposal rows without `gt_*` fields;
- run Stage2U ranker inference without metric computation;
- export clean Stage2AQ-format predictions under the same schema.
