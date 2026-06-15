# Decision Record: Task 1 Local Dataset Schema

**Date:** 2026-05-20
**Status:** Accepted for initial baseline
**Scope:** CATHACTION Task 1 segmentation data loaded from local `datasets/`

---

## Context

The Hugging Face CATHACTION dataset no longer provides a single `segmentation.zip`. The local Task 1 data were downloaded as:

- `segmentation_animal_phantom.zip`, extracted to `datasets/segmentation/`
- `segmentation_human_train.zip`, extracted to `datasets/human_dataset_train/`

The 2026-04-22 MICCAI PDF remains the challenge source of truth for task definition and metrics, but the released local folder structure is more specific than the PDF summary. The released `animal_test` and `phantom_test` folders include masks, so they must not be described as the MICCAI hidden test set.

Inventory command:

```bash
conda run -n cathaction-task1 python scripts/task1/inspect_task1_dataset.py \
  --data-root datasets \
  --sample-size 64 \
  --full-shapes \
  --json-out quality_reports/decisions/2026-05-20_task1_dataset_inventory.json \
  --fail-on-pairing-errors
```

JSON inventory:

- `quality_reports/decisions/2026-05-20_task1_dataset_inventory.json`

---

## Decision

Use the following schema for the first Task 1 data pipeline:

| Collection | Domain | Released split | Image path | Mask path | Mask encoding | Count |
| --- | --- | --- | --- | --- | --- | ---: |
| `animal_train` | animal | train | `datasets/segmentation/animal_train/images/*.png` | `datasets/segmentation/animal_train/masks/*.npy` | `uint8`, values observed `0,1,2` | 4,021 |
| `animal_test` | animal | released evaluation | `datasets/segmentation/animal_test/images/*.png` | `datasets/segmentation/animal_test/masks/*.npy` | `uint8`, values observed `0,1,2` | 1,006 |
| `phantom_train` | phantom | train | `datasets/segmentation/phantom_train/images/*.png` | `datasets/segmentation/phantom_train/masks/*.npy` | `uint8`, values observed `0,1,2` | 14,737 |
| `phantom_test` | phantom | released evaluation | `datasets/segmentation/phantom_test/images/*.png` | `datasets/segmentation/phantom_test/masks/*.npy` | `uint8`, values observed `0,1,2` | 3,685 |
| `human_train` | human | train / holdout analysis | `datasets/human_dataset_train/img/*.jpg` | `datasets/human_dataset_train/mask/*_mask.png` | `uint8`, values observed `0,255` | 5,283 |

Pairing rule:

- Animal/phantom images pair with masks by exact stem, e.g. `ani_00000.png` -> `ani_00000.npy`.
- Human images pair with masks by adding `_mask`, e.g. `JFQ_...jpg` -> `JFQ_..._mask.png`.

Initial split rule:

- Use `animal_train` and `phantom_train` as released training data.
- Use `animal_test` and `phantom_test` as released evaluation/validation data.
- Keep `human_train` separate from the first training pool until label semantics and domain-generalization strategy are confirmed.

---

## Label Semantics

Current confirmed facts:

- Animal/phantom masks are `uint8` arrays with values `0`, `1`, and `2`.
- Human masks are `uint8` grayscale images with values `0` and `255`.
- No image/mask pairing errors were detected in the local extracted data.

Unresolved:

- The exact semantic mapping of animal/phantom labels `1` and `2` to catheter vs guidewire is not yet documented in local files inspected so far.
- The human masks appear binary foreground/background; whether foreground merges catheter and guidewire or represents a single tool class needs confirmation from official documentation or visual inspection.

Initial implementation consequence:

- Metric utilities must support both multiclass masks and binary foreground masks.
- The first reported Task 1 metric should include binary foreground DSC/IoU over instrument pixels, and multiclass DSC only after label `1/2` semantics are confirmed.
- Any method-report claim that distinguishes catheter from guidewire must cite the confirmed label mapping.

---

## Image Geometry

Observed geometry:

- Human images and masks are consistently `512x512`.
- Animal and phantom images have heterogeneous dimensions; masks match image height/width within paired samples.
- Phantom images include many unique spatial sizes, so the baseline needs an explicit resize/crop/pad strategy.

Initial implementation consequence:

- Dataset loaders must not assume fixed input size.
- Baseline training config must make resizing explicit and record it as part of result provenance.

---

## Verification

Inventory summary from `inspect_task1_dataset.py`:

- Total images: 28,732
- Total masks: 28,732
- Total paired samples: 28,732
- Pairing issues: 0

This decision should be revisited when:

- official 2026 platform I/O instructions are released;
- label `1/2` semantics are confirmed;
- hidden test submission schema is known;
- we decide whether to include human-domain training in the main Task 1 baseline.
