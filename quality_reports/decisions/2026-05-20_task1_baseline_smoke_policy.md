# Decision: Task 1 Baseline Smoke Policy

**Date:** 2026-05-20
**Scope:** CATHACTION Task 1 segmentation baseline scaffold
**Status:** Active

---

## Decision

Use a minimal PyTorch `TinyUNet` as the first Task 1 baseline scaffold.

Two CPU smoke configurations are maintained:

- `configs/task1/baseline_unet.yaml`
  - output directory: `outputs/task1/baseline_unet_smoke/`
  - label mode: `binary_foreground`
  - model output channels: 1
  - loss: binary cross entropy with logits
- `configs/task1/baseline_unet_multiclass_smoke.yaml`
  - output directory: `outputs/task1/baseline_unet_multiclass_smoke/`
  - label mode: `multiclass_012`
  - model output channels: 3
  - loss: cross entropy

Both smoke configs use:

- train manifest: `configs/task1/splits/released_train.csv`
- eval manifest: `configs/task1/splits/released_eval.csv`
- input resize: `128x128`
- model width: `base_channels=8`
- smoke train size: 16 samples
- smoke eval size: 8 samples
- smoke epochs: 1

The default smoke result is a pipeline-health check only. It is not a meaningful
segmentation performance claim and must not be reported as method performance.

---

## Rationale

Animal and phantom `.npy` masks currently show labels `0,1,2`. The MICCAI 2026
guide states that Task 1 masks distinguish catheter and guidewire as separate
classes, but the exact numeric `1/2` to catheter/guidewire mapping has not yet
been confirmed from the guide, public website, Hugging Face README, or local
metadata. Therefore:

- `binary_foreground` remains useful for smoke tests and human binary masks;
- `multiclass_012` is the challenge-aligned animal/phantom path, but reports
  must use `label_1` and `label_2` until the class-name mapping is confirmed.

The model is deliberately small so the full training/evaluation/prediction path
can be tested on CPU before investing in a GPU experiment stack.

---

## Current Limitations

- `binary_foreground` collapses labels `1` and `2`; it cannot validate separate
  class quality.
- `multiclass_012` preserves labels `1` and `2`, but class names remain
  unresolved.
- The current resize path is direct resizing, not aspect-ratio preserving pad or
  crop. This is acceptable for smoke testing but should be revisited before real
  experiments.
- The default config evaluates only a tiny subset by design.
- `released_eval` has labels and is not the MICCAI hidden test set.
- Human training data remains a separate holdout/analysis domain until label
  semantics and split strategy are finalized.

---

## Required Next Decisions Before Real Training

- Confirm label semantics for animal/phantom mask values `1` and `2`.
- Decide whether official Task 1 ranking expects binary foreground masks or
  separate class masks.
- Replace direct resize with a documented production transform if needed.
- Establish a case/procedure-level internal validation split if reliable case IDs
  become available.
- Switch from CPU smoke config to a GPU training config once CUDA is healthy.
