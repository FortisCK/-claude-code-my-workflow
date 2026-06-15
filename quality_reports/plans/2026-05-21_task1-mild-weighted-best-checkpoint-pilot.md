# Plan: Task 1 Mild Weighted Pilot With Best Checkpoint

**Date:** 2026-05-21
**Status:** COMPLETED
**Scope:** Improve Task 1 multiclass pilot training/checkpoint behavior

---

## Goal

Address two issues observed in the previous weighted CE run:

1. Only the final checkpoint was saved, even though epoch 6 had the best eval
   DSC.
2. The inverse-frequency class weights over-corrected `label_2`, producing too
   many `label_2` pixels.

---

## Changes

- Save `best_checkpoint.pt` based on `eval.dice`.
- Keep `checkpoint.pt` as the final epoch checkpoint.
- Run a milder weighted CE pilot:
  - class weights: `[0.02, 1.0, 2.0]`
  - train samples: 2,048
  - eval samples: 512
  - epochs: 8
  - image size: `256x256`
  - batch size: 16
  - device: CUDA through `pointdet`

---

## Verification

- Run py_compile and pytest.
- Train the mild weighted GPU pilot.
- Evaluate both final and best checkpoints.
- Export predictions and overlays from the best checkpoint.
- Compare predicted `0/1/2` pixel distribution against GT and prior runs.

---

## Result

Implementation completed:

- `best_checkpoint.pt` is now saved by monitor metric.
- `checkpoint.pt` remains the final epoch checkpoint.
- Mild weighted pilot ran for 8 epochs on CUDA through the `pointdet` env.
- Verification passed:
  - py_compile passed.
  - pytest passed with 19 tests.
  - `git diff --check` passed before training.

Best checkpoint:

- epoch: 3
- eval DSC: 0.46236699014996263
- eval IoU/mIoU: 0.36690955949437576
- `label_1` DSC: 0.5641163915148246
- `label_2` DSC: 0.36061758878510064

Final checkpoint:

- epoch: 8
- eval DSC: 0.35443269816979944
- eval IoU/mIoU: 0.24306551146930994
- `label_1` DSC: 0.5214893140298007
- `label_2` DSC: 0.18737608230979824

Best-checkpoint exported prediction check over 32 PNGs:

- matched predictions: 32/32
- DSC: 0.5282670680660289
- IoU/mIoU: 0.42706114496120234
- `label_1` DSC: 0.605359586343144
- `label_2` DSC: 0.4511745497889139
- predicted pixels: `{0: 2075035, 1: 20465, 2: 1652}`
- resized GT pixels: `{0: 2083026, 1: 11595, 2: 2531}`

Conclusion:

- Best-checkpoint tracking is necessary; the final epoch was worse than epoch 3.
- Mild weights avoid the severe `label_2` over-prediction seen with
  inverse-frequency weights, while still predicting both foreground classes.
