# Diffusion v1 Test5 Artifact-Aware Metric Check

**Date:** 2026-05-18 16:29
**Status:** COMPLETED
**Purpose:** Sanity-check the new paper-oriented heart-region, heart-boundary, and severity-stratified metrics on a small 5-case subset before any full test100 rerun.

## Inputs

- VAE: `experiments/checkpoints/vae_v2/best_val.pt`
- Diffusion: `experiments/checkpoints/diffusion_v1/epoch_200.pt`
- Config: `code/training/configs/diffusion_v1.yaml`
- Split: first 5 held-out test cases from `data/imagecas/splits/v1.json`
- Cases: `21`, `32`, `41`, `50`, `51`
- Inference: deterministic 50-step full-volume sliding window
- ROI: `128^3`, overlap `0.5`, Gaussian blend, `sigma_scale=0.125`
- Boundary metric radius: `3` voxels

## Command

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_diffusion_full_volume.py \
  --val-cases 0 \
  --test-cases 5 \
  --eval-mode deterministic \
  --num-steps 50 \
  --figure-cases 5 \
  --out-dir experiments/runs/diffusion_v1/eval_epoch200_full_volume_test5_det50_artifact_metrics
```

## Outputs

- Metrics: `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test5_det50_artifact_metrics/metrics.csv`
- Summary: `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test5_det50_artifact_metrics/summary.json`
- Figures: `experiments/runs/diffusion_v1/eval_epoch200_full_volume_test5_det50_artifact_metrics/figures/`

## Aggregate Results

| Metric | Corrupted | Diffusion v1 det50 | Direction |
|---|---:|---:|---|
| Global MAE HU | 426.80 | 77.42 | improved |
| Global SSIM | 0.8122 | 0.8791 | improved |
| Heart MAE HU | 80.99 | 53.69 | improved |
| Heart RMSE HU | 120.13 | 96.30 | improved |
| Boundary MAE HU | 122.31 | 75.79 | improved |
| Boundary RMSE HU | 202.11 | 154.82 | improved |
| Boundary gradient L1 HU | 76.07 | 59.14 | improved |

## Per-Case MAE HU

| Case | Severity | Global corrupted | Global det50 | Heart corrupted | Heart det50 | Boundary corrupted | Boundary det50 |
|---|---|---:|---:|---:|---:|---:|---:|
| 21 | mild | 370.56 | 113.12 | 72.68 | 44.42 | 113.23 | 56.95 |
| 32 | mild | 419.56 | 70.42 | 68.75 | 65.56 | 115.61 | 87.14 |
| 41 | severe | 513.67 | 46.16 | 103.56 | 59.10 | 134.17 | 75.77 |
| 50 | severe | 423.91 | 83.21 | 80.97 | 48.57 | 134.33 | 83.09 |
| 51 | medium | 406.33 | 74.19 | 78.97 | 50.80 | 114.23 | 75.99 |

## Interpretation

The new metrics are wired correctly and the direction is consistent with the existing full-volume baseline. Diffusion v1 improves not only global MAE/SSIM but also heart-region and heart-boundary errors on this small subset. This is a sanity check only; do not use it as the final paper table.
