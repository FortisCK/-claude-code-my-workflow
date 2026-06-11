# Diffusion v2 residual gate diagnostic, test5

Date: 2026-05-21

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/python/evaluate_residual_diffusion_gates.py \
  --ckpt experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt \
  --val-cases 0 \
  --test-cases 5 \
  --num-steps 16 \
  --n-samples 4 \
  --fixed-alphas 0.05 0.10 0.25 0.50 \
  --gate-mask-modes none heart \
  --shrinkage-alphas 1.0 \
  --shrinkage-taus 0.005 0.01 0.02 0.05 0.10 \
  --oracle-alphas 0 0.025 0.05 0.10 0.15 0.25 0.50 \
  --oracle-block-size 32 \
  --figure-cases 3 \
  --no-print-summary \
  --out-dir experiments/runs/diffusion_v2_residual/gate_test5_steps16_samples4
```

Cases: test 21, 32, 41, 50, 51

Artifacts:

- Metrics CSV: `experiments/runs/diffusion_v2_residual/gate_test5_steps16_samples4/gate_metrics.csv`
- Summary JSON: `experiments/runs/diffusion_v2_residual/gate_test5_steps16_samples4/summary.json`
- Figures: `experiments/runs/diffusion_v2_residual/gate_test5_steps16_samples4/figures/`

Mean metrics across test5:

| Strategy | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary grad L1 HU | Delta MAE HU |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net init | 40.61 | 48.89 | 64.54 | 62.16 | 0.00 |
| Posterior mean, full release | 51.92 | 57.25 | 71.32 | 61.09 | +11.31 |
| Fixed alpha 0.05, heart only | 40.61 | 48.90 | 64.55 | 62.16 | +0.00 |
| Fixed alpha 0.10, heart only | 40.62 | 48.96 | 64.59 | 62.13 | +0.00 |
| Shrinkage tau 0.005, heart only | 40.61 | 48.89 | 64.55 | 62.10 | -0.00 |
| Shrinkage tau 0.10, heart only | 40.96 | 54.44 | 66.94 | 60.64 | +0.35 |
| Oracle voxel, global | 37.27 | 43.84 | 59.65 | 59.70 | -3.34 |
| Oracle voxel, heart only | 40.30 | 43.84 | 62.36 | 61.14 | -0.32 |
| Oracle voxel, boundary only | 40.35 | 46.98 | 59.65 | 59.75 | -0.27 |
| Oracle block32, global | 40.57 | 48.82 | 64.51 | 62.12 | -0.04 |

Case-wise oracle voxel global deltas vs U-Net:

| Case | Delta MAE HU | Delta heart MAE HU | Delta boundary MAE HU | Delta boundary grad L1 HU |
| --- | ---: | ---: | ---: | ---: |
| 21 | -3.27 | -4.71 | -4.54 | -2.37 |
| 32 | -3.32 | -4.79 | -4.60 | -2.50 |
| 41 | -3.12 | -5.92 | -5.21 | -2.46 |
| 50 | -3.54 | -5.01 | -4.87 | -2.46 |
| 51 | -3.48 | -4.82 | -5.23 | -2.50 |

Interpretation:

- Full posterior mean release is consistently too aggressive and worsens MAE on every case.
- Voxel oracle improves every case by roughly 3 HU global MAE, 5 HU heart MAE, 5 HU boundary MAE, and 2.5 HU boundary-gradient L1.
- Heart-only voxel oracle gives the same heart MAE improvement while largely preserving global MAE.
- Boundary-only voxel oracle gives the same boundary MAE improvement while largely preserving global MAE.
- Block32/scalar oracle gains are tiny, so the useful residual signal is too local for coarse gates.
- Simple uncertainty shrinkage mostly trades a small MAE penalty for boundary-gradient improvement. It is useful as a handcrafted baseline, but not strong enough as the final method.

Next step:

Train a lightweight voxel/local spatial gate with frozen U-Net and frozen residual diffusion posterior features. The gate should start conservative, e.g. low-bias sigmoid with `g_max <= 0.25`, and use ROI/boundary-aware supervision or losses.
