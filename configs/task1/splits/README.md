# Task 1 Split Manifests

These manifests describe the local released Task 1 segmentation data. They do
not define the MICCAI hidden test set.

## Files

- `released_train.csv`
  - `animal_train`
  - `phantom_train`
- `released_eval.csv`
  - `animal_test`
  - `phantom_test`
- `human_holdout.csv`
  - `human_train`
- `summary.json`
  - counts by role, collection, domain, and case-id source

## Policy

Use `released_train.csv` for the first baseline training pool. Use
`released_eval.csv` for validation/evaluation on released animal and phantom
data. Keep `human_holdout.csv` separate until the human label semantics and
domain-generalization strategy are confirmed.

The released animal/phantom `*_test` folders include masks, so they are not the
official hidden test set described in the MICCAI 2026 PDF. Do not report
metrics from `released_eval.csv` as hidden-test results.

Animal and phantom filenames currently do not expose a reliable
procedure/case identifier. Internal validation splits inside `animal_train` or
`phantom_train` must therefore not be presented as procedure-level validation
unless a separate case map is obtained.

Human filenames expose a provisional case key as the prefix before `_img-`.
That key is recorded in `case_id`, but human samples are kept out of the first
training pool by policy.

## Regeneration

Run from the repository root after activating the environment:

```bash
conda activate cathaction-task1
python scripts/task1/write_task1_split_manifests.py \
  --data-root datasets \
  --output-dir configs/task1/splits
```

