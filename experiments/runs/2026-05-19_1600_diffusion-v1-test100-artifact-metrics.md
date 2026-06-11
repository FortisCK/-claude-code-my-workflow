# Run: Diffusion v1 Epoch 200, Test100 Artifact-Aware Metrics

**Date:** 2026-05-19
**Model:** `diffusion_v1`
**Checkpoint:** `experiments/checkpoints/diffusion_v1/epoch_200.pt`
**VAE checkpoint:** `experiments/checkpoints/vae_v2/best_val.pt`
**Sampling:** deterministic, 50 steps
**Purpose:** Re-run frozen diffusion v1 on the full held-out test100 split with
the paper-oriented artifact-aware metric suite.

## Evaluation Protocol

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_diffusion_full_volume.py \
  --val-cases 0 \
  --test-cases 100 \
  --eval-mode deterministic \
  --num-steps 50 \
  --figure-cases 10 \
  --out-dir experiments/runs/diffusion_v1/eval_epoch200_full_volume_test100_det50_artifact_metrics
```

Outputs:

- `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test100_det50_artifact_metrics/metrics.csv`
- `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test100_det50_artifact_metrics/summary.json`
- First 10 qualitative panels under
  `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test100_det50_artifact_metrics/figures/`

## Full Test100 Global Metrics

| Model | N | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted | 100 | 436.96 | 16.42 | 0.8131 | 0.18942 |
| Diffusion v1 det50 | 100 | 72.94 | 27.95 | 0.8814 | 0.05131 |
| U-Net v1 epoch200 EMA | 100 | 38.07 | 33.23 | 0.9446 | 0.02839 |

## Full Test100 Artifact-Aware Metrics

| Model | Heart MAE HU | Heart RMSE HU | Boundary MAE HU | Boundary RMSE HU | Boundary Grad L1 HU |
|---|---:|---:|---:|---:|---:|
| Corrupted | 71.07 | 105.57 | 112.63 | 189.11 | 70.57 |
| Diffusion v1 det50 | 47.55 | 75.46 | 69.36 | 139.21 | 53.44 |
| U-Net v1 epoch200 EMA | 43.17 | 67.08 | 56.06 | 98.28 | 55.49 |

## Severity-Stratified Diffusion Metrics

Severity strata are defined by the corrupted-input heart error rank within this
evaluation run.

| Severity | N | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary Grad L1 HU |
|---|---:|---:|---:|---:|---:|
| Mild | 33 | 69.11 | 41.08 | 60.54 | 46.18 |
| Medium | 34 | 74.98 | 48.31 | 70.39 | 52.81 |
| Severe | 33 | 74.66 | 53.22 | 77.13 | 61.36 |

## Interpretation

The new full test100 artifact-aware run confirms that diffusion v1 is not
competitive with the completed U-Net baseline on deterministic reconstruction
metrics. Diffusion v1 remains much better than corrupted input and is relatively
close on boundary gradient L1, but U-Net v1 epoch200 EMA is stronger on global
MAE/PSNR/SSIM/NRMSE, heart MAE/RMSE, and boundary MAE/RMSE.

This also confirms that the earlier gap was not caused by comparing new U-Net
metrics against an obsolete diffusion evaluation script. The gap is real for
the current v1 models.
