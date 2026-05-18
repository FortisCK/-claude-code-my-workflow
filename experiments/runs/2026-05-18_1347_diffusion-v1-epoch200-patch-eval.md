# Diffusion v1 Epoch 200 Patch Evaluation

**Date:** 2026-05-18  
**Checkpoint:** `experiments/checkpoints/diffusion_v1/epoch_200.pt`  
**VAE:** `experiments/checkpoints/vae_v2/best_val.pt`  
**Script:** `scripts/python/evaluate_diffusion_patches.py`

## Scope

Patch-level evaluation of the trained conditional EDM before implementing
full-volume sliding-window inference.

- Patch size: 128³ center crop
- Cases: first 5 validation + first 5 test cases from `data/imagecas/splits/v1.json`
- Deterministic sampling: 50 steps, `S_churn=0`
- Stochastic posterior: 4 samples, 50 steps
- Metric domain: normalized HU `[-1, 1]`

## Outputs

- Metrics CSV: `experiments/runs/diffusion_v1/eval_epoch200_patches/metrics.csv`
- Summary JSON: `experiments/runs/diffusion_v1/eval_epoch200_patches/summary.json`
- Figures: `experiments/runs/diffusion_v1/eval_epoch200_patches/figures/`

## Quantitative Result

All 10 evaluated patches improved over the corrupted baseline in MAE for both
deterministic and stochastic-posterior-mean predictions.

| Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted input | 0.062249 | 127.46 | 25.82 | 0.8699 | 0.07183 |
| Deterministic 50-step | 0.032334 | 66.20 | 29.77 | 0.9011 | 0.04512 |
| Stochastic mean, 4 × 50-step | 0.026289 | 53.83 | 32.72 | 0.9185 | 0.03272 |

Mean MAE improvement:

- Deterministic 50-step: `-0.029916` norm, about `-61.25 HU`
- Stochastic mean, 4 samples: `-0.035960` norm, about `-73.63 HU`

By split:

| Split | Method | MAE norm | MAE HU | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| val | Corrupted | 0.059957 | 122.76 | 25.77 | 0.8736 |
| val | Deterministic 50-step | 0.034089 | 69.80 | 29.11 | 0.9028 |
| val | Stochastic mean | 0.027905 | 57.14 | 31.67 | 0.9194 |
| test | Corrupted | 0.064541 | 132.15 | 25.87 | 0.8663 |
| test | Deterministic 50-step | 0.030578 | 62.61 | 30.42 | 0.8993 |
| test | Stochastic mean | 0.024674 | 50.52 | 33.77 | 0.9175 |

Posterior uncertainty:

- Mean voxelwise posterior std: `0.01447`
- Std maps are brightest near boundaries and structural transitions in the
  inspected center-slice panels.

## Qualitative Read

The epoch-200 model removes a large fraction of the blur/artifact visible in
the corrupted inputs and recovers the main cardiac structure. Deterministic
samples tend to preserve slightly sharper contrast than the posterior mean,
while the stochastic mean gives the best MAE/PSNR/SSIM among this small sample.

Residual limitations:

- Outputs remain smoother than the clean patches.
- Fine boundaries and small high-contrast details are not fully restored.
- This is still patch-level center-crop evidence; full-volume behavior is not
  established yet.

## Conclusion

`epoch_200.pt` passes the first patch-level correction check. It is now worth
expanding evaluation to more validation/test patches and then implementing
128³ overlapping sliding-window inference for 192³ full-volume correction.
