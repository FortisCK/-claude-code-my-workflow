# Run: U-Net v1 Residual Diagnosis on Test100

Date: 2026-05-20  
Purpose: Determine whether the frozen U-Net v1 leaves structured residual
errors that are suitable targets for posterior residual diffusion.

## Command

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/analyze_unet_residuals.py \
  --val-cases 0 \
  --test-cases 100 \
  --figure-cases 12 \
  --out-dir experiments/runs/unet_v1/residual_diagnosis_test100
```

## Inputs

- U-Net checkpoint: `experiments/checkpoints/unet_v1/epoch_200.pt`
- Weights: EMA
- Split: `data/imagecas/splits/v1.json`
- Cases: full test100
- Inference: 128^3 sliding window, 0.5 overlap, gaussian blending
- Residual definition: `r = clean - unet(corrupted)`

## Outputs

- Metrics CSV: `experiments/runs/unet_v1/residual_diagnosis_test100/residual_metrics.csv`
- Summary JSON: `experiments/runs/unet_v1/residual_diagnosis_test100/summary.json`
- Figures: `experiments/runs/unet_v1/residual_diagnosis_test100/figures/`

## Key Results

Overall residual error:

- Global residual MAE: 38.0714 HU
- Heart residual MAE: 43.1719 HU
- Boundary residual MAE: 56.0560 HU
- Global residual p95 absolute error: 123.8227 HU
- Heart residual p95 absolute error: 131.0425 HU
- Boundary residual p95 absolute error: 177.3687 HU

Residual concentration:

- Heart volume fraction: 0.0741
- Heart absolute-error fraction: 0.0850
- Heart error enrichment: 1.1809x
- Boundary volume fraction: 0.0598
- Boundary absolute-error fraction: 0.0895
- Boundary error enrichment: 1.5242x
- Top 5 percent residual voxels contain 40.36 percent of total residual absolute error.

Worst boundary residual cases:

| Case | Global MAE HU | Heart MAE HU | Boundary MAE HU |
| --- | ---: | ---: | ---: |
| 483 | 40.8255 | 72.0074 | 101.7385 |
| 53 | 37.3323 | 69.8056 | 88.6654 |
| 85 | 40.4202 | 75.1529 | 88.0018 |
| 547 | 48.4401 | 77.0415 | 79.5216 |
| 821 | 60.4917 | 47.3176 | 78.8553 |
| 743 | 40.9186 | 44.1167 | 77.3629 |
| 535 | 33.9467 | 54.4609 | 77.2881 |
| 584 | 40.8061 | 50.8288 | 75.1630 |

Highest boundary enrichment cases:

| Case | Boundary Enrichment | Global MAE HU | Heart MAE HU | Boundary MAE HU |
| --- | ---: | ---: | ---: | ---: |
| 696 | 2.5046 | 26.1127 | 58.5327 | 65.4007 |
| 483 | 2.4920 | 40.8255 | 72.0074 | 101.7385 |
| 41 | 2.4865 | 27.6217 | 62.4828 | 68.6810 |
| 53 | 2.3750 | 37.3323 | 69.8056 | 88.6654 |
| 160 | 2.2910 | 26.1873 | 48.9202 | 59.9941 |
| 601 | 2.2796 | 31.3325 | 59.9577 | 71.4260 |
| 535 | 2.2767 | 33.9467 | 54.4609 | 77.2881 |
| 85 | 2.1772 | 40.4202 | 75.1529 | 88.0018 |

## Interpretation

The U-Net residual is not uniformly distributed. Heart and especially boundary
regions carry denser residual error than their voxel fraction. This supports a
v2 formulation that treats U-Net output as a strong deterministic point
estimate and learns the remaining posterior residual.

The residual diagnosis also supports using explicit heart and boundary losses.
A global-only refiner would underweight the clinically relevant residuals
because background and non-boundary anatomy dominate voxel count.

The top residual voxels are not perfectly aligned with the original corrupted
input error. This argues against a blind artifact mask only. v2 should condition
on `corrupted`, `unet_pred`, and `corrupted - unet_pred`, then learn a residual
gate or uncertainty map instead of applying full-volume refinement everywhere.

## Design Consequences

- Keep U-Net v1 as the first frozen initializer for controlled v2 experiments.
- Add a deterministic residual refiner baseline before diffusion.
- Train the refiner on `target = clean - unet_pred`.
- Use input channels `[corrupted, unet_pred, corrupted - unet_pred]`.
- Add heart and boundary weighted losses.
- Add nonartifact consistency or gated fusion before the final diffusion version.

