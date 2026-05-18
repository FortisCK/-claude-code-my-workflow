# Diffusion v1 Epoch 200 Test100 Deterministic Full-Volume Evaluation

**Date:** 2026-05-18
**Code SHA:** `6b3f840`
**Checkpoint:** `experiments/checkpoints/diffusion_v1/epoch_200.pt`
**VAE:** `experiments/checkpoints/vae_v2/best_val.pt`
**Script:** `scripts/python/evaluate_diffusion_full_volume.py`

## Scope

Full held-out test split evaluation for Diffusion v1 deterministic full-volume
inference. This extends the initial 10-case full-volume check to all 100 test
cases from the frozen ImageCAS v1 split.

- Split: `test`, 100 cases from `data/imagecas/splits/v1.json`
- Volume size: 192^3
- Sliding-window ROI: 128^3
- Windows per volume: 8
- Overlap: 0.5
- Blend mode: Gaussian, `sigma_scale=0.125`
- `sw_batch_size`: 1
- Sampling: deterministic 50-step EDM, `S_churn=0`
- Metric domain: normalized HU `[-1, 1]`
- Figures: first 20 test cases

## Outputs

- Metrics CSV: `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test100_det50/metrics.csv`
- Summary JSON: `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test100_det50/summary.json`
- Figures: `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test100_det50/figures/`

## Quantitative Result

All 100 held-out test volumes improved over the corrupted baseline in MAE.

| Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted input | 0.213409 | 436.96 | 16.42 | 0.8131 | 0.18942 |
| Deterministic 50-step SW | 0.035623 | 72.94 | 27.95 | 0.8814 | 0.05131 |

Mean MAE improvement:

- Deterministic 50-step SW: `-0.177787` norm, about `-364.02 HU`
- Relative MAE reduction: about 83.3%
- Mean PSNR gain: about `+11.54 dB`
- Mean SSIM gain: about `+0.0683`

Per-case improvement counts:

- Deterministic 50-step SW improved MAE on `100 / 100` test volumes.
- No test volume regressed in MAE.

Weakest improvements by MAE delta:

| Case | Corrupted MAE | Det50 SW MAE | Delta MAE | Det50 SW SSIM |
|---:|---:|---:|---:|---:|
| 768 | 0.181414 | 0.056894 | -0.124521 | 0.8311 |
| 21 | 0.180982 | 0.055249 | -0.125733 | 0.8520 |
| 61 | 0.186230 | 0.055182 | -0.131049 | 0.8473 |
| 298 | 0.181984 | 0.044054 | -0.137930 | 0.8522 |
| 630 | 0.176890 | 0.037441 | -0.139449 | 0.8668 |

Strongest improvements by MAE delta:

| Case | Corrupted MAE | Det50 SW MAE | Delta MAE | Det50 SW SSIM |
|---:|---:|---:|---:|---:|
| 586 | 0.262838 | 0.023826 | -0.239012 | 0.9355 |
| 68 | 0.253665 | 0.018466 | -0.235200 | 0.9567 |
| 186 | 0.259404 | 0.025591 | -0.233813 | 0.9465 |
| 776 | 0.253926 | 0.020949 | -0.232977 | 0.9419 |
| 568 | 0.254101 | 0.022125 | -0.231977 | 0.9435 |

## Qualitative Read

Visual panels for the first 20 test cases show the same pattern as the metrics:
motion blur is strongly reduced and the main cardiac anatomy is restored. The
outputs remain smoother than clean CT, especially around fine myocardial
boundaries and small high-contrast structures.

Additional local qualitative reruns of the weakest official cases `768` and
`547` showed the same failure mode: the deterministic model substantially
reduces blur but leaves residual smoothing and boundary error. No obvious
sliding-window grid seams were visible in the inspected center-slice panels.
Those extra reruns were used only for visual inspection; the official metrics
above are from the single 100-case test run.

## Reproducibility Note

The 100-case evaluation was run in one fixed test-split order with seed `42`.
The full run is reproducible as an ordered run, but isolated single-case reruns
can differ slightly because the deterministic EDM sampler still starts from the
current RNG state before running the zero-churn Heun trajectory.

## Conclusion

Diffusion v1 `epoch_200.pt` passes the full held-out test split deterministic
full-volume evaluation. The deterministic 50-step sliding-window pipeline is
strong enough to serve as the fast/default inference mode. The next useful
comparison is to run stochastic 4-sample posterior mean on either the full test
set or a selected subset of weak/representative cases to quantify the
quality-runtime tradeoff.
