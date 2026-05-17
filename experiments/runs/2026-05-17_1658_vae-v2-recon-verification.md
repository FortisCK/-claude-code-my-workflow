# vae-v2-recon-verification

**Status:** COMPLETED
**Date:** 2026-05-17 16:58-17:16 CEST
**Author:** CMZ / Codex
**Checkpoint:** `experiments/checkpoints/vae_v2/best_val.pt`
**Checkpoint epoch:** 445
**Config:** `code/training/configs/vae_v2_128.yaml`
**Split:** `data/imagecas/splits/v1.json`
**Verification script:** `scripts/python/evaluate_vae_recon.py`

---

## Intent

Verify that the Stage-1 VAE v2 128^3 finetune checkpoint is usable as the
frozen encoder candidate for Stage-2 conditional latent diffusion / flow
matching. This pass checks full validation and test reconstruction metrics and
generates worst-case visual slices.

## Commands

First quick visual subset:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_vae_recon.py \
  --config code/training/configs/vae_v2_128.yaml \
  --ckpt experiments/checkpoints/vae_v2/best_val.pt \
  --max-cases-per-split 8 \
  --figure-cases 4 \
  --amp \
  --out-dir experiments/runs/vae_v2/verify_2026-05-17
```

Full unclamped metric pass, matching the training-log `val/recon` convention:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_vae_recon.py \
  --config code/training/configs/vae_v2_128.yaml \
  --ckpt experiments/checkpoints/vae_v2/best_val.pt \
  --max-cases-per-split 0 \
  --figure-cases 0 \
  --amp \
  --out-dir experiments/runs/vae_v2/verify_2026-05-17_full_unclamped
```

Worst-case visual pass:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_vae_recon.py \
  --config code/training/configs/vae_v2_128.yaml \
  --ckpt experiments/checkpoints/vae_v2/best_val.pt \
  --case-ids 735 179 500 821 547 983 \
  --figure-cases 6 \
  --amp \
  --out-dir experiments/runs/vae_v2/verify_2026-05-17_worst
```

## Outcome

Full unclamped metrics:

| Split | n | MAE norm | MAE HU | Heart MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 100 | 0.011378 | 23.30 | 30.81 | 39.75 | 0.9577 | 0.01297 |
| test | 100 | 0.011191 | 22.91 | 31.15 | 39.75 | 0.9586 | 0.01277 |

The full-val MAE (`0.0113783819`) matches the checkpoint/training log
(`0.0113786144`), so the verification script is aligned with the training
metric path.

Worst cases by whole-volume MAE / SSIM:

| Split | Case | MAE HU | Heart MAE HU | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| val | 735 | 36.6 | 50.2 | 36.84 | 0.9180 |
| val | 179 | 34.4 | 45.1 | 37.23 | 0.9208 |
| val | 500 | 32.6 | 45.7 | 37.32 | 0.9276 |
| test | 821 | 32.0 | 37.0 | 37.92 | 0.9314 |
| test | 547 | 29.8 | 48.3 | 37.85 | 0.9347 |
| test | 983 | 28.5 | 46.7 | 37.27 | 0.9465 |

Visual inspection of representative and worst-case slice grids shows:

- No gross reconstruction collapse.
- Anatomy is spatially aligned.
- The dominant failure mode is mild smoothing / attenuation of thin high-
  contrast structures and edge texture, especially around heart-mask and
  vessel-like boundaries.
- Worst cases still preserve chamber / aorta / gross cardiac geometry, but
  coronary-scale fidelity should not be assumed from these metrics alone.

## Decision

`experiments/checkpoints/vae_v2/best_val.pt` is acceptable as the current frozen
encoder candidate for Stage-2 prototype training.

Before treating Stage-2 results as manuscript-grade:

1. Estimate latent `sigma_data` from VAE-encoded train cases.
2. Update EDM / flow configs to use this VAE v2 checkpoint, not stale v1 paths.
3. Add a coronary/lumen-focused verification pass once the downstream mask or
   proxy metric is ready; whole-volume MAE/SSIM is not sufficient for that claim.

## Artifacts

Tracked lightweight artifacts:

- `experiments/runs/vae_v2/verify_2026-05-17_full_unclamped/metrics.csv`
- `experiments/runs/vae_v2/verify_2026-05-17_full_unclamped/summary.json`
- `experiments/runs/vae_v2/verify_2026-05-17_worst/metrics.csv`
- `experiments/runs/vae_v2/verify_2026-05-17_worst/summary.json`

Local visual artifacts, intentionally not required for commit:

- `experiments/runs/vae_v2/verify_2026-05-17/figures/`
- `experiments/runs/vae_v2/verify_2026-05-17_worst/figures/`
