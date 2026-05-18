# Cardiac Artifact Baseline Summary

**Date:** 2026-05-18
**Status:** baseline frozen
**Primary checkpoint:** `experiments/checkpoints/diffusion_v1/epoch_200.pt`
**VAE checkpoint:** `experiments/checkpoints/vae_v2/best_val.pt`
**Primary inference mode:** deterministic 50-step full-volume sliding window

## Baseline Definition

The current baseline is:

- VAE: `vae_v2/best_val.pt`, epoch 445
- Diffusion: `diffusion_v1/epoch_200.pt`
- Full-volume inference: 192^3 input, 128^3 sliding-window ROI, 0.5 overlap
- Blending: Gaussian, `sigma_scale=0.125`
- Sampling: deterministic EDM, 50 steps, `S_churn=0`
- Held-out test split: `data/imagecas/splits/v1.json`

This is the default/fast baseline for downstream comparison. Stochastic
posterior mean remains available as a quality-mode candidate, but it is not
required to define the baseline.

## Key Artifacts

Run records:

- VAE verification: `experiments/runs/2026-05-17_1658_vae-v2-recon-verification.md`
- Stage-2 prep: `experiments/runs/2026-05-17_1940_diffusion-v1-stage2-prep.md`
- Diffusion pilot: `experiments/runs/2026-05-17_1945_diffusion-v1-pilot.md`
- Diffusion long run: `experiments/runs/2026-05-17_2030_diffusion-v1-longrun.md`
- Patch eval, 10 cases: `experiments/runs/2026-05-18_1347_diffusion-v1-epoch200-patch-eval.md`
- Patch eval, 200 cases: `experiments/runs/2026-05-18_1417_diffusion-v1-epoch200-full-patch-eval.md`
- Full-volume eval, 10 cases: `experiments/runs/2026-05-18_1442_diffusion-v1-epoch200-full-volume-eval.md`
- Full-volume test eval, 100 deterministic cases:
  `experiments/runs/2026-05-18_1517_diffusion-v1-epoch200-test100-det-full-volume-eval.md`

Key commits:

- `4fc9d74` VAE reconstruction verification
- `ba174c7` Diffusion v1 latent patch setup
- `16131ab` Diffusion pilot run record
- `51604f9` Diffusion v1 long run record
- `62fd5a6` Epoch-200 patch eval script and 10-case patch results
- `f786314` Full 200-case patch results
- `dff11c3` Full-volume sliding-window inference code
- `6b3f840` 10-case full-volume results
- `d1e7b0f` 100-case deterministic full-volume test results

## VAE v2 Verification

Checkpoint: `experiments/checkpoints/vae_v2/best_val.pt`

Full unclamped reconstruction metrics:

| Split | n | MAE norm | MAE HU | Heart MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 100 | 0.011378 | 23.30 | 30.81 | 39.75 | 0.9577 | 0.01297 |
| test | 100 | 0.011191 | 22.91 | 31.15 | 39.75 | 0.9586 | 0.01277 |

Decision: the VAE is good enough to serve as the frozen Stage-2 encoder.

## Diffusion Patch Evaluation

Checkpoint: `experiments/checkpoints/diffusion_v1/epoch_200.pt`

Full patch eval scope:

- Cases: 100 val + 100 test
- Input: 128^3 center crop
- Deterministic: 50-step
- Stochastic: 4 samples x 50-step

All 200 patches improved in MAE for both deterministic and stochastic outputs.

| Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted input | 0.063141 | 129.28 | 25.67 | 0.8683 | 0.08479 |
| Deterministic 50-step | 0.031476 | 64.45 | 29.74 | 0.9005 | 0.05023 |
| Stochastic mean, 4 x 50-step | 0.025802 | 52.83 | 32.62 | 0.9189 | 0.03656 |

## Full-Volume Validation

Initial 10-case full-volume validation:

- Cases: 5 val + 5 test
- Inference: 128^3 sliding window over 192^3 volumes
- Deterministic and stochastic modes both evaluated

All 10 full volumes improved in MAE for both deterministic and stochastic
outputs.

| Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted input | 0.205573 | 420.91 | 16.61 | 0.8110 | 0.17743 |
| Deterministic 50-step SW | 0.043812 | 89.71 | 26.23 | 0.8673 | 0.06298 |
| Stochastic mean, 4 x 50-step SW | 0.039040 | 79.93 | 27.25 | 0.8795 | 0.05660 |

## Held-Out Test100 Result

Primary baseline result:

- Split: all 100 held-out test volumes
- Inference: deterministic 50-step full-volume sliding window
- Improvement count: 100 / 100 volumes

| Method | MAE norm | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|---:|
| Corrupted input | 0.213409 | 436.96 | 16.42 | 0.8131 | 0.18942 |
| Deterministic 50-step SW | 0.035623 | 72.94 | 27.95 | 0.8814 | 0.05131 |

Mean deterministic improvement:

- MAE norm: `-0.177787`
- MAE HU: about `-364.02 HU`
- Relative MAE reduction: about 83.3%
- PSNR gain: about `+11.54 dB`
- SSIM gain: about `+0.0683`

Weakest test improvements:

| Case | Corrupted MAE | Det50 SW MAE | Delta MAE | Det50 SW SSIM |
|---:|---:|---:|---:|---:|
| 768 | 0.181414 | 0.056894 | -0.124521 | 0.8311 |
| 21 | 0.180982 | 0.055249 | -0.125733 | 0.8520 |
| 61 | 0.186230 | 0.055182 | -0.131049 | 0.8473 |

Strongest test improvements:

| Case | Corrupted MAE | Det50 SW MAE | Delta MAE | Det50 SW SSIM |
|---:|---:|---:|---:|---:|
| 586 | 0.262838 | 0.023826 | -0.239012 | 0.9355 |
| 68 | 0.253665 | 0.018466 | -0.235200 | 0.9567 |
| 186 | 0.259404 | 0.025591 | -0.233813 | 0.9465 |

## Qualitative Pattern

Across inspected figures, the model consistently removes a large fraction of
the motion blur and restores gross cardiac anatomy. Gaussian sliding-window
stitching did not show obvious grid seams in inspected center-slice panels.

The dominant residual limitation is smoothing: fine boundaries, myocardial
texture, and small high-contrast structures remain less sharp than the clean
volumes. This limitation is visible in both weak and strong cases and should be
treated as the main target for the next model-improvement pass.

## Cleanup Policy

Retain:

- `experiments/checkpoints/vae_v2/best_val.pt`
- `experiments/checkpoints/diffusion_v1/epoch_200.pt`
- tracked run cards, metrics CSVs, and summary JSONs
- source code and configs needed to rerun inference

Safe to delete after baseline freeze:

- intermediate diffusion checkpoints `epoch_010.pt` through `epoch_190.pt`
- pilot diffusion checkpoints
- intermediate VAE epoch and discriminator checkpoints
- local-only figures and smoke/inspection directories when space is needed

## Next Work

The baseline is sufficiently evaluated. Next steps should focus on either:

- model improvement, especially reducing smoothing and improving boundary
  detail, or
- a small stochastic-vs-deterministic comparison on selected weak and strong
  cases to quantify runtime-quality tradeoff.

Full 100-case stochastic evaluation is not required to freeze this baseline.
