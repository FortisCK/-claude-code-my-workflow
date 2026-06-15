# Decision Record: Task 1 Initial Split Policy

**Date:** 2026-05-20
**Status:** Accepted for initial baseline
**Scope:** CATHACTION Task 1 segmentation manifests

---

## Context

The extracted local Task 1 data contain released folder-level partitions:

- `animal_train`
- `animal_test`
- `phantom_train`
- `phantom_test`
- `human_train`

The released `animal_test` and `phantom_test` folders include masks, so they are not the MICCAI hidden test set. The 2026-04-22 PDF states that official final evaluation will use a hidden test set, but that hidden test set is not represented by these released mask-containing folders.

Animal and phantom filenames currently do not expose a reliable procedure/case identifier. Human filenames expose a provisional case key as the prefix before `_img-`.

---

## Decision

Use three manifest roles for the first baseline:

| Manifest | Role | Collections | Count | Policy |
| --- | --- | --- | ---: | --- |
| `configs/task1/splits/released_train.csv` | `released_train` | `animal_train`, `phantom_train` | 18,758 | First baseline training pool |
| `configs/task1/splits/released_eval.csv` | `released_eval` | `animal_test`, `phantom_test` | 4,691 | Released validation/evaluation only |
| `configs/task1/splits/human_holdout.csv` | `human_holdout` | `human_train` | 5,283 | Hold out for human-domain analysis until label/split policy is confirmed |

Generated summary:

- `configs/task1/splits/summary.json`

Regeneration command:

```bash
conda run -n cathaction-task1 python scripts/task1/write_task1_split_manifests.py \
  --data-root datasets \
  --output-dir configs/task1/splits
```

---

## Consequences

- Baseline metrics on `released_eval.csv` must be described as released evaluation metrics, not hidden-test metrics.
- We will not create a frame-random internal validation split inside `animal_train` or `phantom_train`.
- Any later procedure-level train/validation split inside released training data requires an official or manually verified case map.
- Human-domain performance will be tracked separately until we decide whether to train on human masks.

---

## Verification

Generated manifests contain:

- `released_train.csv`: 18,758 samples plus header
- `released_eval.csv`: 4,691 samples plus header
- `human_holdout.csv`: 5,283 samples plus header

Unit tests cover:

- split-role assignment;
- manifest writing on a toy dataset;
- manifest sample IDs matching the dataset index.

