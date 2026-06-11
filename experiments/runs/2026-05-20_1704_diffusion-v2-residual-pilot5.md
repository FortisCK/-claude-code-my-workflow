# Run: Diffusion v2 Residual EDM Pilot5

Date: 2026-05-20  
Purpose: First voxel-space posterior residual diffusion pilot on top of frozen
U-Net v1.

## Model

- Frozen initializer: U-Net v1 epoch200 EMA
- Target: `clean - unet_pred`
- Condition: `[corrupted, unet_pred, corrupted - unet_pred]`
- Engine: EDM residual sampler
- Denoiser: MONAI `DiffusionModelUNet`
- Final pilot architecture: channels `[24, 48, 96, 96]`, no spatial attention
- Parameters: 9.47M
- Patch size: `128^3`
- Batch size: 1
- Loss: EDM weighted MSE + `0.25` final-image L1 + `0.02` gradient L1

The initial attention-enabled voxel-space config OOMed on 128^3 patches inside
MONAI spatial attention. The pilot therefore uses a no-attention voxel model.

## Training Command

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  -m code.training.train_residual_diffusion \
  --config code/training/configs/diffusion_v2_residual.yaml \
  --epochs 5 \
  --run-slug diffusion_v2_residual_pilot5 \
  --ckpt-dir experiments/checkpoints/diffusion_v2_residual/pilot5 \
  --no-wandb
```

## Training Outputs

- Checkpoint: `experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt`
- Resolved config: `experiments/runs/diffusion_v2_residual_pilot5/config_resolved.yaml`

Mean training metrics:

| Epoch | Total | EDM | Final L1 | Gradient |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.44496 | 0.44122 | 0.01354 | 0.01762 |
| 2 | 0.29991 | 0.29695 | 0.01082 | 0.01244 |
| 3 | 0.26942 | 0.26662 | 0.01029 | 0.01158 |
| 4 | 0.23990 | 0.23718 | 0.00999 | 0.01099 |
| 5 | 0.24380 | 0.24112 | 0.00986 | 0.01081 |

## Main Test5 Evaluation

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_residual_diffusion_full_volume.py \
  --ckpt experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt \
  --weights ema \
  --val-cases 0 \
  --test-cases 5 \
  --roi-size 128 \
  --num-steps 16 \
  --n-samples 1 \
  --figure-cases 5 \
  --out-dir experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16
```

Outputs:

- Metrics: `experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16/metrics.csv`
- Summary: `experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16/summary.json`
- Figures: `experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16/figures/`

Test5 mean results:

| Metric | U-Net init | Diffusion v2 | Delta v2 - U-Net |
| --- | ---: | ---: | ---: |
| Global MAE HU | 40.6124 | 61.7532 | +21.1408 |
| Heart MAE HU | 48.8889 | 68.6047 | +19.7158 |
| Boundary MAE HU | 64.5414 | 80.8390 | +16.2975 |
| Boundary gradient L1 HU | 62.1567 | 67.3124 | +5.1557 |

Per-case global MAE deltas were consistently worse than U-Net:

| Case | U-Net MAE HU | Diffusion v2 MAE HU | Delta |
| ---: | ---: | ---: | ---: |
| 21 | 51.35 | 72.31 | +20.96 |
| 32 | 40.21 | 63.07 | +22.86 |
| 41 | 27.62 | 49.11 | +21.49 |
| 50 | 42.47 | 62.03 | +19.56 |
| 51 | 41.40 | 62.25 | +20.85 |

## Fixed-Gate Test5 Follow-Ups

Two additional test5 runs evaluated constant residual gates:

```text
x_final = unet_pred + alpha * residual_sample
```

| Alpha | Global delta HU | Heart delta HU | Boundary delta HU | Boundary grad delta HU |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | +3.01 | +1.94 | +1.57 | -0.69 |
| 0.10 | +0.80 | +0.33 | +0.28 | -0.18 |

Run directories:

- `experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16_scale025/`
- `experiments/runs/diffusion_v2_residual/eval_epoch005_test5_steps16_scale010/`

The fixed gates show that small residual updates almost preserve U-Net MAE and
can improve boundary-gradient error. They do not yet beat U-Net on MAE.

## Diagnostics

All diagnostics used case 21 with 16-step full-volume inference.

| Variant | Global delta HU | Heart delta HU | Boundary delta HU | Boundary grad delta HU |
| --- | ---: | ---: | ---: | ---: |
| EMA, scale 1.0 | +20.96 | +17.49 | +16.10 | +5.37 |
| Online, scale 1.0 | +16.05 | +18.70 | +14.13 | +3.03 |
| Online, scale 0.5 | +6.24 | +6.50 | +4.68 | -0.72 |
| Online, scale 0.25 | +2.33 | +2.10 | +1.42 | -0.59 |
| Online, scale -0.25 | +2.48 | +1.52 | +1.27 | -0.98 |
| Online, scale 1.0, `sigma_max=0.2` | +15.74 | +16.47 | +13.15 | +2.89 |
| Online, 4 samples, scale 0.25 | +1.31 | +0.80 | +0.65 | -0.11 |

The 4-sample posterior-mean run produced posterior std outputs, but the
single-case uncertainty-error correlation was only `0.05`, so this checkpoint
does not yet support a useful uncertainty claim.

## Interpretation

The first residual EDM implementation is technically working but not yet
competitive with U-Net v1. The important findings are:

- The voxel-space diffusion model can train on 128^3 patches after removing
  spatial attention.
- Patch-level training loss decreases, so the residual target is learnable.
- Raw residual sampling is too aggressive and worsens MAE.
- Online weights are better than EMA at epoch 5, but still behind U-Net.
- Shrinking the residual and using posterior mean reduces damage substantially.
- Negative residual scaling does not reveal a simple sign bug.
- Lowering sampling `sigma_max` alone does not fix the issue.
- Boundary gradient can improve slightly at conservative scales even while MAE
  remains worse, suggesting the model is adding edge/detail changes but lacks
  a reliable gate.

## Next Design Consequence

Do not long-run this exact residual EDM as the final v2. The next v2 revision
should add conservative correction control before another long run:

- learned or deterministic gate around residual application,
- stronger final-image reconstruction loss or distillation-to-clean objective,
- heart/boundary weighting targeted at residual-enriched regions,
- posterior mean as the default correction, not a single raw sample,
- explicit residual magnitude regularization or schedule calibration.
