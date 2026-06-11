# Diffusion v2 residual gate diagnostic, case 21

Date: 2026-05-21

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/python/evaluate_residual_diffusion_gates.py \
  --ckpt experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt \
  --case-ids 21 \
  --num-steps 16 \
  --n-samples 4 \
  --fixed-alphas 0.05 0.10 0.25 0.50 \
  --gate-mask-modes none heart \
  --shrinkage-alphas 1.0 \
  --shrinkage-taus 0.005 0.01 0.02 0.05 0.10 \
  --oracle-alphas 0 0.025 0.05 0.10 0.15 0.25 0.50 \
  --oracle-block-size 32 \
  --figure-cases 1 \
  --out-dir experiments/runs/diffusion_v2_residual/gate_case21_steps16_samples4
```

Artifacts:

- Metrics CSV: `experiments/runs/diffusion_v2_residual/gate_case21_steps16_samples4/gate_metrics.csv`
- Summary JSON: `experiments/runs/diffusion_v2_residual/gate_case21_steps16_samples4/summary.json`
- Figure: `experiments/runs/diffusion_v2_residual/gate_case21_steps16_samples4/figures/manual_case_21_posterior_mean.png`

Key case-21 metrics:

| Strategy | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary grad L1 HU |
| --- | ---: | ---: | ---: | ---: |
| U-Net init | 51.35 | 42.18 | 50.91 | 52.14 |
| Posterior mean, full release | 62.59 | 50.11 | 58.44 | 51.51 |
| Fixed alpha 0.05, heart only | 51.35 | 42.20 | 50.92 | 52.14 |
| Fixed alpha 0.10, heart only | 51.36 | 42.27 | 50.94 | 52.11 |
| Oracle voxel, global | 48.08 | 37.48 | 46.04 | 49.77 |
| Oracle block32, global | 51.31 | 42.16 | 50.83 | 52.10 |

Posterior diagnostics:

- Posterior mean residual abs: 22.33 HU
- Posterior residual std mean: 46.26 HU
- Uncertainty/error Pearson correlation: 0.155

Interpretation:

- Full posterior mean still over-releases residuals and worsens distortion metrics.
- Boundary gradient improves slightly under full posterior mean, supporting the current hypothesis that the residual direction carries local high-frequency/boundary information.
- Fixed small heart-only gates mostly preserve U-Net, with tiny boundary-gradient gains at alpha 0.10 to 0.25.
- Voxel oracle has a real upper bound on this case, but scalar and block32 oracles mostly choose the U-Net solution. Useful residual corrections are therefore very local and require fine-grained reliability gating.

Next diagnostic:

- Run this gate ablation on test5 or test20 before implementing a learned gate.
- If voxel-oracle gains persist across cases, train a lightweight spatial gate on frozen U-Net + frozen residual diffusion posterior mean/std.
