# Plan: Task 1 Weighted Loss Pilot

**Date:** 2026-05-20
**Status:** APPROVED
**Scope:** Improve the Task 1 numeric-label baseline so it predicts `label_2`

---

## Motivation

The first GPU multiclass pilot trained successfully, but exported predictions
over 32 eval samples contained no `label_2` pixels:

- predicted label 0: 2,088,115 pixels
- predicted label 1: 9,037 pixels
- predicted label 2: 0 pixels

The corresponding resized ground truth contained:

- label 0: 2,083,026 pixels
- label 1: 11,595 pixels
- label 2: 2,531 pixels

This points to class imbalance rather than a pipeline failure.

---

## Pilot Change

Add explicit class-weight support to multiclass cross entropy and run a second
GPU pilot with weights computed from the same training subset.

The first 2,048 released-train masks resized to `256x256` have counts:

- label 0: 133,331,142 pixels
- label 1: 677,236 pixels
- label 2: 209,350 pixels

Mean-normalized inverse-frequency weights:

- label 0: 0.003593860466557044
- label 1: 0.7075428952310618
- label 2: 2.288863244302381

---

## Configuration

- config: `configs/task1/baseline_unet_multiclass_weighted_gpu_pilot.yaml`
- train samples: 2,048
- eval samples: 512
- image size: `256x256`
- epochs: 8
- batch size: 16
- device: CUDA via the `pointdet` conda environment

---

## Verification

- Run py_compile and pytest.
- Train the weighted GPU pilot.
- Export 32 eval predictions.
- Re-evaluate saved predictions.
- Generate overlays with `label_1` red and `label_2` blue.
- Compare prediction class distribution against the prior unweighted pilot.
