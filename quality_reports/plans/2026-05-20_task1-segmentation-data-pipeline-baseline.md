# Plan: Task 1 Segmentation Data Pipeline And First Baseline

**Date:** 2026-05-20
**Status:** APPROVED
**Scope:** CATHACTION Task 1 catheter/guidewire segmentation

---

## Goal

Build the first reproducible Task 1 foundation: a verified dataset inventory, split-aware data index, metric tests, and a minimal segmentation baseline path that can later be trained and packaged without violating case/procedure-level split discipline.

This plan intentionally starts with data and metrics before model work. The released folder names and label formats are not identical across domains, so the first implementation step is to make those assumptions explicit and testable.

---

## Current Data Facts Observed Locally

- `datasets/segmentation/animal_train`
  - `images/*.png`: 4,021
  - `masks/*.npy`: 4,021
  - sampled mask values: `0, 1, 2`
- `datasets/segmentation/animal_test`
  - `images/*.png`: 1,006
  - `masks/*.npy`: 1,006
  - sampled mask values: `0, 1, 2`
- `datasets/segmentation/phantom_train`
  - `images/*.png`: 14,737
  - `masks/*.npy`: 14,737
  - sampled mask values: `0, 1, 2`
- `datasets/segmentation/phantom_test`
  - `images/*.png`: 3,685
  - `masks/*.npy`: 3,685
  - sampled mask values: `0, 1, 2`
- `datasets/human_dataset_train`
  - `img/*.jpg`: 5,283
  - `mask/*.png`: 5,283
  - sampled mask values: `0, 255`

Important interpretation guardrail: the released `animal_test` and `phantom_test` folders include masks, so they are not the MICCAI hidden test set. Treat them as released evaluation/validation-style splits until official 2026 platform instructions define the final hidden-test I/O.

---

## Non-Negotiables

- Preserve procedure/case-level split discipline whenever case identifiers can be inferred.
- Do not frame-randomize across a procedure when constructing train/validation splits.
- Do not report released-folder performance as hidden-test performance.
- Implement DSC as the Task 1 primary metric; IoU/Jaccard, mIoU, and pixel accuracy are secondary.
- Keep local data paths configurable and out of git.
- Every generated metric must trace to split file, config, command, and output path.

---

## Proposed Implementation Phases

### Phase 1: Dataset Inventory And Schema

**Files to add or update**

- `scripts/task1/inspect_task1_dataset.py`
- `quality_reports/decisions/YYYY-MM-DD_task1_dataset_schema.md`
- optionally `datasets/README.md` if we want a small tracked note about expected local files

**Work**

- Create a read-only inspection script that reports:
  - required Task 1 directories;
  - image/mask counts;
  - image and mask filename pairing failures;
  - image sizes by domain/split;
  - mask dtype and unique-label summary;
  - approximate class-pixel distribution from a configurable sample or full scan.
- Normalize the observed label conventions into an explicit schema:
  - phantom/animal `.npy`: investigate whether `1/2` map to catheter/guidewire or foreground subclasses;
  - human PNG: binary `255` foreground, unless documentation says otherwise.
- Record unresolved label semantics as a decision/blocker rather than silently guessing.

**Verification**

- Run the inspection script against `datasets/`.
- Confirm counts match the observed extracted dataset.
- Confirm no data files are tracked by git.

### Phase 2: Task 1 Python Package Skeleton

**Files to add**

- `src/cathaction/__init__.py`
- `src/cathaction/data/task1.py`
- `src/cathaction/metrics/segmentation.py`
- `tests/test_task1_dataset_index.py`
- `tests/test_task1_segmentation_metrics.py`

**Work**

- Build a lightweight dataset indexer that produces structured rows:
  - `domain`: `phantom`, `animal`, `human`;
  - `released_split`: `train`, `test`, or `human_train`;
  - `case_id` or inferred procedure key where possible;
  - `image_path`, `mask_path`, image shape, mask encoding.
- Keep image loading lazy; the indexer should be fast enough for CI-style tests.
- Implement metric utilities for binary and multiclass segmentation:
  - Dice / DSC;
  - IoU / Jaccard;
  - mIoU;
  - pixel accuracy.
- Add toy-array tests with known expected values, including empty-mask edge cases.

**Verification**

- `python3 -m py_compile` for new scripts/package files.
- `python3 -m pytest tests/test_task1_dataset_index.py tests/test_task1_segmentation_metrics.py`.

### Phase 3: Split Manifest Strategy

**Files to add**

- `configs/task1/splits/README.md`
- generated split manifests under a gitignored or deliberately tracked path, to be decided before writing them

**Work**

- Decide how to use released folders without leaking:
  - baseline training can use released `phantom_train`, `animal_train`, and optionally `human_train`;
  - released `phantom_test` and `animal_test` can be used as validation/evaluation, but not called hidden test;
  - any internal validation split within `*_train` must be by inferred case/procedure key, not random frame.
- If case IDs cannot be reliably inferred from filenames, document the limitation and avoid over-claiming procedure-level validation.

**Verification**

- Split audit confirms no image stem appears in multiple manifests.
- If case keys are inferred, audit confirms no case key appears in multiple train/validation partitions.

### Phase 4: Minimal Baseline Path

**Files to add after Phases 1-3 are clean**

- `configs/task1/baseline_unet.yaml`
- `scripts/task1/train_baseline.py`
- `scripts/task1/evaluate_baseline.py`
- `scripts/task1/predict_baseline.py`

**Work**

- Use a conservative PyTorch baseline unless we decide to use nnU-Net/MONAI before implementation.
- Start with a small, reproducible U-Net-style model or a wrapper around an established segmentation library.
- Support:
  - configurable input size;
  - binary-vs-multiclass label handling;
  - deterministic seed;
  - checkpoint output;
  - prediction output suitable for metric evaluation.
- Keep first run small enough to smoke test on CPU or a short GPU run.

**Verification**

- Training smoke test on a tiny subset.
- Evaluation smoke test produces DSC/IoU/mIoU/pixel accuracy JSON.
- Re-run evaluation from saved predictions.
- Record command, config, output directory, and residual risks in session log.

---

## Expected Deliverables For This First Task

- A reliable Task 1 data inventory script and report.
- A tested dataset indexer.
- Tested segmentation metric utilities.
- A documented split strategy that does not confuse released test folders with hidden test.
- A minimal baseline scaffold ready for a first short training run.

---

## Approval Gate

Approved on 2026-05-20:

- Use PyTorch as the default first baseline stack.
- Treat released `animal_test` and `phantom_test` as validation/evaluation folders, not hidden test.
- Keep `human_train` separate as a human-domain holdout/analysis domain until label semantics and split strategy are fully confirmed.

Initial implementation starts with Conda environment creation before Phase 1 code.
