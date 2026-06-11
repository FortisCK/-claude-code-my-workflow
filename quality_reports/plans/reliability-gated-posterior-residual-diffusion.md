# Plan: Reliability-Gated Posterior Residual Diffusion

**Date:** 2026-05-21
**Status:** ACTIVE DESIGN NOTE
**Scope:** Record the literature-driven decision after the residual diffusion
pilot and gate-oracle diagnostics. This note supersedes the earlier idea that
diffusion v2 should directly add a sampled residual to the U-Net output.

## Problem Setting

We study single-volume, image-domain, 3D cardiac CT motion artifact correction:

```text
input:  one corrupted 3D cardiac CT volume
target: paired clean ImageCAS volume
mask:   heart mask available for analysis and artifact-local metrics
```

This is not multi-phase or dynamic cardiac CT. TT U-Net-style methods remain
important related work, but they solve a different input setting.

Current controlled data and evaluation:

- ImageCAS paired synthetic clean/corrupted data.
- Fixed split: `data/imagecas/splits/v1.json`, 800 train / 100 val / 100 test.
- Patch training: `128^3`; full-volume eval: sliding window over `192^3`.
- Main metrics: global, heart, boundary, boundary-gradient, PSNR, SSIM, NRMSE.

## Current Evidence

### Strong deterministic baseline

`unet_v1` is a strong supervised residual U-Net baseline:

- Checkpoint: `experiments/checkpoints/unet_v1/epoch_200.pt`
- Full test100:
  - MAE: `38.07 HU`
  - PSNR: `33.23`
  - SSIM: `0.9446`
  - Heart MAE: `43.17 HU`
  - Boundary MAE: `56.06 HU`

### Latent diffusion is not the main path

The latent EDM + VAE model is much worse than U-Net:

- Checkpoint: `experiments/checkpoints/diffusion_v1/epoch_200.pt`
- Full test100:
  - MAE: `72.94 HU`
  - Heart MAE: `47.55 HU`
  - Boundary MAE: `69.36 HU`

Interpretation: latent compression and posterior sampling are poorly aligned
with HU-preserving paired restoration when the deterministic U-Net is strong.

### Voxel residual diffusion is useful but miscalibrated

`diffusion_v2_residual` predicts:

```text
residual = clean - unet(corrupted)
condition = [corrupted, unet(corrupted), corrupted - unet(corrupted)]
```

The 5-epoch pilot trains stably, but direct residual release is too aggressive.

Test5, 16-step, 4-sample posterior mean gate diagnostics:

| Strategy | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary grad L1 HU |
| --- | ---: | ---: | ---: | ---: |
| U-Net init | 40.61 | 48.89 | 64.54 | 62.16 |
| Posterior mean full release | 51.92 | 57.25 | 71.32 | 61.09 |
| Oracle voxel, global | 37.27 | 43.84 | 59.65 | 59.70 |
| Oracle voxel, heart only | 40.30 | 43.84 | 62.36 | 61.14 |
| Oracle voxel, boundary only | 40.35 | 46.98 | 59.65 | 59.75 |
| Oracle block32, global | 40.57 | 48.82 | 64.51 | 62.12 |

Interpretation:

- Full posterior mean release consistently worsens MAE.
- The residual still contains useful local correction directions.
- Voxel oracle improves all five cases.
- Coarse scalar and block32 gates have tiny gains, so useful corrections are
  fine-grained and spatially sparse.
- The main missing component is reliability calibration, not a larger diffusion
  backbone.

Run cards:

- `experiments/runs/2026-05-21_1610_diffusion-v2-residual-gate-case21.md`
- `experiments/runs/2026-05-21_1651_diffusion-v2-residual-gate-test5.md`

## Literature-Driven Interpretation

The literature report supports the following synthesis:

1. **Strong point estimate first**

   In paired restoration with distortion metrics, a supervised U-Net is close
   to the desired conditional mean/MMSE-style estimator. Diffusion samples are
   plausible posterior draws, not necessarily MAE/PSNR/SSIM optimal outputs.

2. **Residual diffusion rather than full-image diffusion**

   Work such as DvSR, ResDiff, ResShift, and UCDIR motivates a predict-and-refine
   framing: a deterministic network restores dominant low-frequency structure,
   while diffusion models remaining high-frequency or ambiguous residuals.

3. **Posterior mean over single sample**

   Multiple posterior samples should be averaged before metric-oriented output.
   Single stochastic samples increase plausible detail but can hurt distortion.

4. **Spatially adaptive reliability is essential**

   The relevant design is not a global residual scale. It is a learned local
   gate or reliability map conditioned on posterior mean, posterior variance,
   image structure, and artifact-local context.

5. **Uncertainty is a feature, not the final answer by itself**

   Current residual std alone is not enough to decide the final correction, but
   it should be an input to a learned reliability gate and an analysis signal.

## Method Decision

Use diffusion refinement, but make it reliability-gated:

```text
x_u = U-Net(corrupted)

r_1, ..., r_K ~ p_theta(clean - x_u | corrupted, x_u, corrupted - x_u)
mu_r = mean(r_1, ..., r_K)
sigma_r = std(r_1, ..., r_K)

g = GateNet(corrupted, x_u, corrupted - x_u, mu_r, sigma_r, |grad x_u|)
x_final = x_u + g * mu_r
```

Gate constraints:

```text
g in [0, g_max]
g_max = 0.25 initially, optionally 0.5 in ablation
initial gate bias should make g close to zero
```

This makes U-Net the default solution and releases diffusion residuals only
where the gate predicts reliable benefit.

## Concrete Code Changes

### Add a gate model

New file:

```text
code/models/residual_gate.py
```

Suggested model:

- lightweight 3D U-Net or shallow residual CNN
- input channels initially:
  - corrupted
  - U-Net output `x_u`
  - corrupted - `x_u`
  - posterior mean residual `mu_r`
  - posterior std `sigma_r`
  - gradient magnitude `|grad x_u|`
- output one-channel gate map
- final activation:

```text
gate = g_max * sigmoid(logits + init_bias)
```

### Add a training config

New file:

```text
code/training/configs/residual_gate_v1.yaml
```

Initial values:

```text
g_max: 0.25
init_bias: -4.0
n_samples_train: 1 or cached posterior mean
n_samples_eval: 4
num_sample_steps_train: 8 or cached posterior mean
num_sample_steps_eval: 16
loss:
  final_l1: 1.0
  heart_l1: 1.0
  boundary_l1: 1.0
  boundary_gradient: 0.1
  gate_l1: 0.01
  gate_tv: 0.01
```

### Add posterior feature caching or online feature generation

Preferred next implementation path:

1. Generate cached posterior features for train/val patches or full volumes:

   ```text
   x_u
   mu_r
   sigma_r
   corrupted - x_u
   clean
   optional masks
   ```

2. Train GateNet cheaply from cached tensors.

Reason: online residual diffusion sampling inside every gate training step is
too expensive and would slow iteration.

If caching is too large, start with a limited train subset and val/test5.

### Add gate training

New file:

```text
code/training/train_residual_gate.py
```

Training target:

```text
x_final = x_u + gate * mu_r
loss = image-domain artifact-aware objective
```

The gate is not trained to imitate voxel oracle directly at first. Direct final
image loss is simpler and avoids overfitting to an alpha grid.

### Add gate inference and evaluation

New files:

```text
code/inference/residual_gate_sliding_window.py
scripts/python/evaluate_residual_gate_full_volume.py
```

Evaluation must report:

- U-Net init metrics
- fixed/shrinkage handcrafted gates
- learned GateNet output
- oracle voxel/block upper bounds from the same posterior features

This keeps learned gate results grounded against the measured upper bound.

## Immediate Experimental Plan

1. Implement residual gate model and training config.
2. Implement posterior feature cache for a small subset first:
   - train subset: 50 to 100 cases or patch cache
   - val subset: 10 cases
   - test subset: existing test5 for quick comparison
3. Train `residual_gate_v1` for a short pilot.
4. Evaluate on test5 against:
   - U-Net init
   - posterior mean full release
   - fixed alpha
   - shrinkage
   - learned gate
   - oracle voxel/block
5. Continue only if learned gate improves heart/boundary metrics while keeping
   global MAE no worse than U-Net.

## Go / No-Go Criteria

Continue toward test20/test100 if test5 learned gate achieves one of:

- global MAE improves by at least `0.3 HU` without worsening heart/boundary, or
- heart MAE improves by at least `1.0 HU` with global MAE delta no worse than
  `+0.2 HU`, or
- boundary MAE or boundary-gradient improves by at least `1.0 HU` with global
  MAE delta no worse than `+0.2 HU`.

Stop and redesign if:

- learned gate collapses to zero everywhere;
- learned gate releases residual broadly and reproduces posterior-mean failure;
- test5 gains come only from one case;
- test20 does not preserve the test5 direction.

## Paper Framing

Do not write this as generic post-processing. The intended framing is:

> Single-volume cardiac CT motion correction is a metric-sensitive inverse
> problem where a deterministic network provides a strong anatomical point
> estimate. We model the remaining local ambiguity as a posterior residual
> distribution and learn a reliability gate that selectively releases
> posterior-mean corrections in artifact-prone regions.

Possible method name:

```text
Reliability-Gated Posterior Residual Diffusion
```

or:

```text
Cardiac Motion Artifact Correction via Reliability-Gated Posterior Residual Diffusion
```
