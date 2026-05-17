# Diffusion v1 Stage-2 Prep

**Date:** 2026-05-17  
**Purpose:** Prepare conditional latent EDM training after VAE-v2 reconstruction verification.

## References Checked

- `external/GenerativeModels`
  - MONAI LDM path uses latent normalization via `scale_factor = 1 / std(z)`.
  - Decode path divides by the same factor before VAE decoding.
- `external/edm-reference`
  - EDM preconditioning matches the Karras formulas used by `code/models/edm.py`.
  - Loss uses log-normal sigma sampling and weighted clean-target MSE.
  - Sampler uses rho schedule plus optional churn and Heun correction.

## Findings

1. Current EDM math is consistent with the NVlabs EDM reference at the level of
   preconditioning, loss weighting, log-normal sigma sampling, and Heun sampling.
2. Stage-2 config was still pointing at stale VAE-v1 placeholders. It now points
   at the verified VAE-v2 checkpoint:
   - `experiments/checkpoints/vae_v2/best_val.pt`
   - `code/training/configs/vae_v2_128.yaml`
3. Direct 192³ VAE-v2 encoding is not viable. A full-volume latent-stat run OOMed
   at the VAE bottleneck attention step, trying to allocate about 45.6 GiB.
4. Stage-2 training must therefore use paired 128³ patches, matching the feasible
   VAE-v2 training window. Full-volume correction will need a sliding-window or
   tiled inference path rather than direct full-volume VAE encode/decode.
5. `train_diffusion.load_frozen_vae()` now preserves `use_checkpoint` from the
   checkpoint/config payload.

## Latent Statistics

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/estimate_latent_stats.py \
  --batch-size 1 \
  --num-workers 2 \
  --log-every 25 \
  --out-json experiments/runs/vae_v2/latent_stats_train_2026-05-17.json
```

Scope:

- Split: `train`
- Cases: 800
- Patch size: 128³ paired clean/corrupted crops
- Augmentation: diffusion train transform (`p_flip=0.5`, rotation 5 degrees,
  shift 2 voxels)
- AMP: off
- Device: CUDA
- Runtime: 272.872 s

Key result:

- `z_clean` std: `0.6932820917367327`
- `z_clean` mean: `0.005413783367634917`
- `z_cond` std: `0.6519333148936984`
- `z_delta = z_clean - z_cond` std: `0.49714197676688254`

Config update:

- `code/training/configs/diffusion_v1.yaml`
  - `edm.sigma_data: 0.693282`
  - `train.patch_size: [128, 128, 128]`

## One-Batch Training Sanity

GPU sanity command built one batch through:

1. paired dataloader
2. frozen VAE-v2 encode for clean/corrupted patches
3. conditional EDM loss
4. denoiser backward + Adam step

Observed:

- Input volume batch: `(4, 1, 128, 128, 128)`
- Latent batch: `(4, 4, 32, 32, 32)`
- Loss: `1.0282472372055054`
- Peak CUDA memory: `32.85 GiB`

Conclusion:

The current Diffusion v1 training path is launchable as a 128³ paired-patch
prototype on the RTX 6000 Ada 48 GB GPU. It is not yet a complete full-volume
inference pipeline; posterior sampling/evaluation still need a patch/sliding
window path before manuscript-quality full-volume metrics.
