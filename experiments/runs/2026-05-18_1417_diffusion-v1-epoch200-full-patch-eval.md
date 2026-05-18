# Diffusion v1 Epoch 200 Full Patch Evaluation

**Date:** 2026-05-18  
**Checkpoint:** `experiments/checkpoints/diffusion_v1/epoch_200.pt`  
**VAE:** `experiments/checkpoints/vae_v2/best_val.pt`  
**Script:** `scripts/python/evaluate_diffusion_patches.py`

## Scope

Expanded patch-level evaluation of the trained conditional EDM before
implementing full-volume sliding-window inference.

- Patch size: 128^3 center crop
- Cases: first 100 validation + first 100 test cases from `data/imagecas/splits/v1.json`
- Deterministic sampling: 50 steps, `S_churn=0`
- Stochastic posterior: 4 samples, 50 steps each
- Metric domain: normalized HU `[-1, 1]`

## Outputs

- Metrics CSV: `experiments/runs/diffusion_v1/eval_epoch200_patches_full/metrics.csv`
- Summary JSON: `experiments/runs/diffusion_v1/eval_epoch200_patches_full/summary.json`
- Figures: `experiments/runs/diffusion_v1/eval_epoch200_patches_full/figures/`

## Quantitative Result

All 200 evaluated patches improved over the corrupted baseline in MAE for both
deterministic and stochastic-posterior-mean predictions.

| Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted input | 0.063141 | 129.28 | 25.67 | 0.8683 | 0.08479 |
| Deterministic 50-step | 0.031476 | 64.45 | 29.74 | 0.9005 | 0.05023 |
| Stochastic mean, 4 x 50-step | 0.025802 | 52.83 | 32.62 | 0.9189 | 0.03656 |

Mean MAE improvement:

- Deterministic 50-step: `-0.031665` norm, about `-64.83 HU`
- Stochastic mean, 4 samples: `-0.037339` norm, about `-76.45 HU`

Relative to the corrupted baseline, mean normalized MAE is reduced by about
50.1% for deterministic sampling and 59.1% for the stochastic posterior mean.

By split:

| Split | Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---|---:|---:|---:|---:|---:|
| val | Corrupted | 0.062096 | 127.14 | 25.73 | 0.8699 | 0.08451 |
| val | Deterministic 50-step | 0.031524 | 64.54 | 29.67 | 0.9009 | 0.05104 |
| val | Stochastic mean | 0.025786 | 52.80 | 32.53 | 0.9190 | 0.03715 |
| test | Corrupted | 0.064186 | 131.42 | 25.60 | 0.8668 | 0.08506 |
| test | Deterministic 50-step | 0.031427 | 64.35 | 29.81 | 0.9001 | 0.04943 |
| test | Stochastic mean | 0.025818 | 52.86 | 32.71 | 0.9187 | 0.03597 |

Per-case improvement counts:

- Deterministic 50-step improved MAE on `200 / 200` patches.
- Stochastic posterior mean improved MAE on `200 / 200` patches.
- Weakest deterministic improvement: test case `821`, `0.038677 -> 0.037358` MAE norm.
- Weakest stochastic improvement: test case `821`, `0.038677 -> 0.032239` MAE norm.
- Largest stochastic improvements included test case `68`
  (`0.113289 -> 0.018764`) and val case `487`
  (`0.115651 -> 0.021399`).

Posterior uncertainty:

- Mean voxelwise posterior std: `0.01482`
- Std maps remain highest around boundaries, artifact transitions, and small
  high-contrast structures.

## Qualitative Read

The expanded evaluation matches the earlier 10-case check. The model reliably
removes motion blur from 128^3 center patches and restores the main cardiac
structure. Deterministic samples usually keep slightly sharper contrast, while
the 4-sample posterior mean gives the best aggregate MAE, PSNR, SSIM, and NRMSE.

Additional manual inspection of representative cases `821`, `946`, `68`, and
`487` confirmed the numeric pattern: weak-improvement cases start from already
low corrupted error, while large-improvement cases show clear recovery from
strong blur.

Residual limitations:

- Outputs are still smoother than the clean patches.
- Fine boundaries and small high-contrast details remain imperfect.
- The current evidence is patch-level center-crop evidence; full-volume behavior
  and patch stitching still need to be evaluated.

## Conclusion

`epoch_200.pt` passes the expanded patch-level correction check. The model is
ready for the next validation stage: implement 128^3 overlapping sliding-window
inference for 192^3 full-volume correction, then evaluate whole-volume metrics
and inspect stitching behavior.
