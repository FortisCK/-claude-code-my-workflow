# Run: diffusion_v2_residual pilot5 test5

**Date:** 2026-05-20
**Goal:** First short-run check for U-Net-conditioned posterior residual EDM.

## Model

- Checkpoint: `experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt`
- Initializer: frozen `unet_v1` EMA, `experiments/checkpoints/unet_v1/epoch_200.pt`
- Target: `clean - unet_pred`
- Condition: `[corrupted, unet_pred, corrupted - unet_pred]`
- Engine: voxel-space residual EDM
- Architecture: no spatial attention, channels `[24, 48, 96, 96]`, 9.47M parameters
- Training: 5 epochs, batch size 1, `128^3` patches, no WandB
- Sampling for eval: 16 deterministic EDM steps, one sample per case

The first attempted voxel-space config used spatial attention and OOMed on
`128^3` patches. The pilot config disables attention; this makes voxel-space
residual diffusion trainable on the available GPU.

## Training Summary

Mean train metrics:

| Epoch | total loss | EDM loss | final L1 | gradient |
|---:|---:|---:|---:|---:|
| 1 | 0.4450 | 0.4412 | 0.0135 | 0.0176 |
| 2 | 0.2999 | 0.2970 | 0.0108 | 0.0124 |
| 3 | 0.2694 | 0.2666 | 0.0103 | 0.0116 |
| 4 | 0.2399 | 0.2372 | 0.0100 | 0.0110 |
| 5 | 0.2438 | 0.2411 | 0.0099 | 0.0108 |

Training is stable and the residual target is learnable. This does not by
itself prove full-volume correction quality.

## Test5 Full-Volume Evaluation

Cases: first 5 held-out test cases from `data/imagecas/splits/v1.json`:
`21, 32, 41, 50, 51`.

### Raw residual addition

Output: `x = unet_pred + 1.0 * residual_sample`

Run directory:
`experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16/`

| Model | MAE HU | PSNR | SSIM | Heart MAE HU | Boundary MAE HU | Boundary grad L1 HU |
|---|---:|---:|---:|---:|---:|---:|
| U-Net init | 40.61 | 32.28 | 0.9357 | 48.89 | 64.54 | 62.16 |
| diffusion_v2 raw | 61.75 | 30.19 | 0.8656 | 68.60 | 80.84 | 67.31 |
| delta vs U-Net | +21.14 | -2.09 | -0.0701 | +19.72 | +16.30 | +5.16 |

Raw residual addition is too aggressive and should not be long-run unchanged.

### Constant residual gate: alpha = 0.25

Output: `x = unet_pred + 0.25 * residual_sample`

Run directory:
`experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16_scale025/`

| Model | MAE HU | PSNR | SSIM | Heart MAE HU | Boundary MAE HU | Boundary grad L1 HU |
|---|---:|---:|---:|---:|---:|---:|
| U-Net init | 40.61 | 32.28 | 0.9357 | 48.89 | 64.54 | 62.16 |
| diffusion_v2 alpha 0.25 | 43.62 | 32.08 | 0.9281 | 50.83 | 66.11 | 61.46 |
| delta vs U-Net | +3.01 | -0.20 | -0.0076 | +1.94 | +1.57 | -0.69 |

The residual hurts MAE but improves boundary-gradient error. This suggests the
residual model is learning edge-like corrections, but needs a learned/local gate.

### Constant residual gate: alpha = 0.10

Output: `x = unet_pred + 0.10 * residual_sample`

Run directory:
`experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16_scale010/`

| Model | MAE HU | PSNR | SSIM | Heart MAE HU | Boundary MAE HU | Boundary grad L1 HU |
|---|---:|---:|---:|---:|---:|---:|
| U-Net init | 40.61 | 32.28 | 0.9357 | 48.89 | 64.54 | 62.16 |
| diffusion_v2 alpha 0.10 | 41.41 | 32.24 | 0.9343 | 49.22 | 64.82 | 61.97 |
| delta vs U-Net | +0.80 | -0.04 | -0.0013 | +0.33 | +0.28 | -0.18 |

Alpha 0.10 nearly preserves U-Net global metrics while retaining a small
boundary-gradient improvement. It still does not beat U-Net on MAE.

## Decision

Do not continue the ungated residual diffusion as the main long run. The next
version should add a learned or uncertainty-conditioned gate:

```text
x_final = x_u + gate(corrupted, x_u, residual_sample, uncertainty) * residual_sample
```

The immediate research value of this pilot is clear:

- voxel-space posterior residual diffusion can train without VAE bottleneck;
- full residual addition is overcorrection;
- small gated residuals preserve U-Net metrics and improve boundary-gradient
  error, suggesting a local refinement/uncertainty story is plausible;
- the next model must make the gate spatial/adaptive rather than use a fixed
  scalar alpha.
