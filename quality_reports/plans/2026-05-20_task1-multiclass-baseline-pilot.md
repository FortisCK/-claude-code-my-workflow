# Plan: Task 1 Multiclass Baseline Pilot

**Date:** 2026-05-20
**Status:** APPROVED
**Scope:** Run a first numeric-label Task 1 segmentation baseline

---

## Goal

Train the first non-smoke Task 1 baseline using animal/phantom numeric labels
`0/1/2`, without assigning class names to `1` and `2`. Produce both metrics and
qualitative overlays so the numeric classes can be inspected visually.

---

## Hardware Constraint And Resolution

The dedicated `cathaction-task1` environment is CPU-only:

- `nvidia-smi` cannot communicate with the NVIDIA driver.
- `torch.cuda.is_available()` is `False`.
- `environment-task1.yml` includes `cpuonly`.

The physical workstation GPU is healthy when checked outside the Codex sandbox:

- GPU: NVIDIA RTX 6000 Ada Generation
- driver: 570.195.03
- CUDA shown by `nvidia-smi`: 12.8

Existing conda environments with CUDA PyTorch are available. For the first GPU
pilot, use `pointdet`:

- PyTorch: 2.5.1
- CUDA: 12.1
- `torch.cuda.is_available()`: `True`
- device: NVIDIA RTX 6000 Ada Generation

---

## CPU Pilot Configuration

- train manifest: `configs/task1/splits/released_train.csv`
- eval manifest: `configs/task1/splits/released_eval.csv`
- label mode: `multiclass_012`
- image size: `256x256`
- model: `TinyUNet`, `base_channels=16`, 3 output channels
- train subset: first 512 released-train samples
- eval subset: first 128 released-eval samples
- epochs: 3
- batch size: 4
- output directory: `outputs/task1/baseline_unet_multiclass_pilot/`

## GPU Pilot Configuration

- config: `configs/task1/baseline_unet_multiclass_gpu_pilot.yaml`
- train manifest: `configs/task1/splits/released_train.csv`
- eval manifest: `configs/task1/splits/released_eval.csv`
- label mode: `multiclass_012`
- image size: `256x256`
- model: `TinyUNet`, `base_channels=16`, 3 output channels
- train subset: first 2,048 released-train samples
- eval subset: first 512 released-eval samples
- epochs: 5
- batch size: 16
- output directory: `outputs/task1/baseline_unet_multiclass_gpu_pilot/`

---

## Verification

- Run py_compile and pytest after code/config changes.
- Run training, checkpoint evaluation, prediction export, saved-prediction
  evaluation, and overlay generation.
- Record commands and metrics in the session log.

---

## Interpretation Guardrail

Metrics will be reported for numeric `label_1` and `label_2`. Do not call these
catheter or guidewire until the official numeric label map is confirmed.
