# Diffusion v1 Pilot

**Date:** 2026-05-17  
**Config:** `code/training/configs/diffusion_v1_pilot.yaml`  
**Purpose:** Run a short 5-epoch conditional EDM pilot before launching the full
`diffusion_v1` schedule.

## Setup

- Frozen VAE: `experiments/checkpoints/vae_v2/best_val.pt`
- VAE config: `code/training/configs/vae_v2_128.yaml`
- Pair mode: precomputed clean/corrupted pairs
- Split file: `data/imagecas/splits/v1.json`
- Train cases: 800
- Patch size: 128³ paired clean/corrupted crops
- Latent shape: 4 × 32³
- EDM `sigma_data`: `0.693282`
- Batch size: 4
- Epochs: 5
- Checkpoint cadence: every epoch
- Checkpoint directory: `experiments/checkpoints/diffusion_v1_pilot`

## Command

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  -m code.training.train_diffusion \
  --config code/training/configs/diffusion_v1_pilot.yaml
```

## Success Criteria

- Training starts without VAE encode OOM.
- Loss remains finite for all 5 epochs.
- Mean epoch loss shows no obvious explosion.
- Checkpoints and EMA weights are written for each epoch.
- Pilot checkpoint is suitable for patch-level sampling sanity.

## Notes

This is a patch-level training pilot. Full-volume correction will still require
patch/sliding-window posterior sampling before full test metrics are meaningful.

## Result

Status: completed.

Runtime:

- Started: 2026-05-17 19:56:10
- Finished: 2026-05-17 20:20:48
- Wall time reported by script: 1476.9 s

Mean training loss:

| Epoch | Mean EDM loss |
|------:|--------------:|
| 1 | 0.7142540369927883 |
| 2 | 0.48739576250314715 |
| 3 | 0.4344237194955349 |
| 4 | 0.41717599138617517 |
| 5 | 0.4075636884570122 |

Checkpoints written:

- `experiments/checkpoints/diffusion_v1_pilot/epoch_001.pt`
- `experiments/checkpoints/diffusion_v1_pilot/epoch_002.pt`
- `experiments/checkpoints/diffusion_v1_pilot/epoch_003.pt`
- `experiments/checkpoints/diffusion_v1_pilot/epoch_004.pt`
- `experiments/checkpoints/diffusion_v1_pilot/epoch_005.pt`

Resolved config:

- `experiments/runs/diffusion_v1_pilot/config_resolved.yaml`

Interpretation:

The pilot passed the basic training-dynamics check. The loss stayed finite,
checkpoints were written every epoch, and the epoch mean fell from `0.7143` to
`0.4076` over five epochs. This is enough to proceed to patch-level sampling
sanity from `epoch_005.pt` before launching a longer run.

## Patch Sampling Sanity

Checkpoint:

- `experiments/checkpoints/diffusion_v1_pilot/epoch_005.pt`

Case:

- Validation case `9`
- 128³ deterministic patch
- `n_samples=1`

Results:

| Sampling | Corrupted MAE | Prediction MAE | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| 10-step deterministic | 0.1502442658 | 0.1603080034 | 19.5158 | 0.6540 | 0.1279 |
| 50-step deterministic | 0.1502442658 | 0.1821469963 | 18.8067 | 0.6514 | 0.1388 |

Interpretation:

The sampling chain is numerically connected and finite, but after only five
epochs it does not yet improve over the corrupted input on this validation
patch. This is expected for a very short pilot; continue training before judging
correction quality.
