# Diffusion v1 Epoch 200 Full-Volume Evaluation

**Date:** 2026-05-18
**Code SHA:** `dff11c3`
**Checkpoint:** `experiments/checkpoints/diffusion_v1/epoch_200.pt`
**VAE:** `experiments/checkpoints/vae_v2/best_val.pt`
**Script:** `scripts/python/evaluate_diffusion_full_volume.py`

## Scope

First full-volume validation of the trained patch-level conditional EDM.
The model was trained on 128^3 patches, so 192^3 volumes are corrected with
MONAI sliding-window inference.

- Volume size: 192^3
- Sliding-window ROI: 128^3
- Windows per volume: 8
- Overlap: 0.5
- Blend mode: Gaussian, `sigma_scale=0.125`
- `sw_batch_size`: 1
- Cases: first 5 validation + first 5 test cases from `data/imagecas/splits/v1.json`
- Deterministic sampling: 50 steps, `S_churn=0`
- Stochastic posterior: 4 samples, 50 steps each
- Metric domain: normalized HU `[-1, 1]`

## Outputs

- Metrics CSV: `experiments/runs/diffusion_v1/eval_epoch200_full_volume/metrics.csv`
- Summary JSON: `experiments/runs/diffusion_v1/eval_epoch200_full_volume/summary.json`
- Figures: `experiments/runs/diffusion_v1/eval_epoch200_full_volume/figures/`

## Quantitative Result

All 10 evaluated full volumes improved over the corrupted baseline in MAE for
both deterministic and stochastic-posterior-mean predictions.

| Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted input | 0.205573 | 420.91 | 16.61 | 0.8110 | 0.17743 |
| Deterministic 50-step SW | 0.043812 | 89.71 | 26.23 | 0.8673 | 0.06298 |
| Stochastic mean, 4 x 50-step SW | 0.039040 | 79.93 | 27.25 | 0.8795 | 0.05660 |

Mean MAE improvement:

- Deterministic 50-step SW: `-0.161761` norm, about `-331.20 HU`
- Stochastic mean, 4 samples: `-0.166533` norm, about `-340.98 HU`

Relative to the corrupted baseline, mean normalized MAE is reduced by about
78.7% for deterministic sampling and 81.0% for the stochastic posterior mean.

By split:

| Split | Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---|---:|---:|---:|---:|---:|
| val | Corrupted | 0.202694 | 415.02 | 16.70 | 0.8098 | 0.18506 |
| val | Deterministic 50-step SW | 0.048030 | 98.34 | 24.85 | 0.8556 | 0.07334 |
| val | Stochastic mean | 0.043769 | 89.62 | 25.51 | 0.8641 | 0.06797 |
| test | Corrupted | 0.208451 | 426.80 | 16.53 | 0.8122 | 0.16980 |
| test | Deterministic 50-step SW | 0.039595 | 81.07 | 27.61 | 0.8790 | 0.05262 |
| test | Stochastic mean | 0.034311 | 70.25 | 28.98 | 0.8948 | 0.04524 |

Per-case improvement counts:

- Deterministic 50-step SW improved MAE on `10 / 10` full volumes.
- Stochastic posterior mean improved MAE on `10 / 10` full volumes.
- Weakest deterministic improvement: test case `21`,
  `0.180982 -> 0.054406` MAE norm.
- Weakest stochastic improvement: test case `21`,
  `0.180982 -> 0.051147` MAE norm.
- Strongest stochastic improvement among these cases: test case `41`,
  `0.250877 -> 0.018930` MAE norm.

Posterior uncertainty:

- Mean voxelwise posterior std: `0.01583`
- Std maps are highest around structure boundaries and artifact transitions,
  consistent with the patch-level evaluation.

## Qualitative Read

The full-volume sliding-window path works. Center-slice panels show large blur
reduction and recovery of the main cardiac structures. Gaussian overlap did not
show obvious grid seams in the inspected cases, including the weakest-improvement
case `21` and the strongest-improvement case `41`.

The full-volume corrupted baseline is much worse than the center-patch baseline
because the metric now includes the entire 192^3 volume rather than the central
128^3 crop. This is expected and is exactly why full-volume evaluation was
needed after patch-level validation.

Residual limitations:

- Outputs remain smoother than clean CT volumes.
- Small high-contrast details and fine boundaries are still partially blurred.
- Only the center slice was visually audited here; future figures should include
  multiple planes and boundary slices near sliding-window joins.
- This is a 10-case full-volume validation, not yet the full held-out test set.

## Conclusion

Diffusion v1 `epoch_200.pt` passes the first full-volume correction check. The
128^3 overlapping Gaussian sliding-window implementation is functional and gives
large quantitative improvement on full 192^3 volumes. The next step is to scale
this full-volume evaluation to the complete held-out test split and optionally
compare deterministic-only inference against the 4-sample posterior mean for
runtime-quality tradeoff.
