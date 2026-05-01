# Plan: 3D Conditional Latent-Diffusion Code Skeleton

**Status:** COMPLETED — all 8 steps green on CPU (2026-05-01)
**Date:** 2026-05-01
**Scope:** Build the `code/models/`, `code/training/`, `code/data/`, `code/inference/`, `code/evaluation/` infrastructure so that, the moment GPU is free, we can flip a switch and start training.

## Completion log (2026-05-01)

| Step | File(s) | Smoke result |
|------|---------|--------------|
| 1. Preprocessing | `code/data/preprocessing.py` | case_1.npz (192³, [-1,1], 620k heart vox) — 1.8s |
| 2. Datasets + transforms | `code/data/imagecas_dataset.py`, `code/data/transforms.py` | clean + paired/online dataloaders both yield (2,1,192³); see `scripts/python/smoke_dataset.py` |
| 3. VAE module | `code/models/vae.py` | (1,1,32³) → latent (1,4,4³); 1 train step ok |
| 4. VAE entry-point | `code/training/train_vae.py`, `configs/vae_v1.yaml` | smoke: 0.85M-param tiny VAE, 4 batches, 219s, ckpt saved |
| 5. Conditional denoiser + EDM | `code/models/conditional_denoiser.py`, `code/models/edm.py` | loss + Heun sample on (1,4,8³) ok |
| 6. Diffusion entry-point | `code/training/train_diffusion.py`, `configs/diffusion_v1.yaml` | smoke: 1.27M-param denoiser, 4 batches, 89s, ckpt saved (online stub, no GPU motion-synth needed) |
| 7. Posterior sampling | `code/inference/posterior_sample.py` | 2 chains × 4 steps on (1,1,192³) decoded, mean+std computed |
| 8. Eval pipeline | `code/evaluation/metrics.py`, `code/evaluation/run_eval.py` | CSV + PNG written; metrics noise-level (expected — untrained tiny model) |

**Gotchas captured during build:**
- AutoencoderKL/DiffusionModelUNet require all `channels` to be multiples of `norm_num_groups` (default 32).
- DiffusionModelUNet's middle block always uses attention regardless of `attention_levels`, so `num_head_channels[-1]` must divide `channels[-1]` even when no other levels use attention.
- VAE/diffusion checkpoints embed their `model_cfg` / `denoiser_cfg`; loaders prefer the embedded payload over the YAML so smoke-arch and full-arch ckpts both load through the same code path.
- Smoke-mode pads case-id pool from train+val so a single preprocessed case still produces enough batches.

**The moment GPU is free**, the path is:
1. Generate ≥1 precomputed pair file per case (or wire a GPU-backed `online_motion_factory` into `train_diffusion.build_paired_dataloader`).
2. `python -m code.training.train_vae --config code/training/configs/vae_v1.yaml`
3. Re-estimate `sigma_data` from VAE-encoded train data; update `configs/diffusion_v1.yaml`.
4. `python -m code.training.train_diffusion --config code/training/configs/diffusion_v1.yaml`
5. `python -m code.evaluation.run_eval --vae-ckpt ... --diff-ckpt ...`

---

## Context (why now)

We are at Week 1-2 transition. ImageCAS data (1000 cases) is on disk; `code/data/motion_synth.py` produces motion-corrupted volumes. **GPU is occupied** (`mcastro` nnUNet 39 GB, vacation-locked) so KL-VAE training (Week 3-4) and conditional-diffusion training (Week 5-6) cannot start. This window is best spent on **code that compiles and smoke-tests on CPU** so that GPU release flips us into immediate full-throughput training.

We are **not** writing this from scratch. We sit on top of two cloned repos that handle ~70% of the load:

- `external/GenerativeModels/tutorials/generative/3d_ldm/` — full MONAI 3D Latent Diffusion tutorial (VAE + DDPM stage 2 + LatentDiffusionInferer)
- `external/GenerativeModels/tutorials/generative/3d_autoencoderkl/` — production-grade 3D KL-VAE training (recon + KL + perceptual + adversarial)
- `external/HM-EDM/diffusion_models/conditional_EDM_3D.py` — Karras-2022 EDM math (sigma scheduling, preconditioning, Heun sampler); MIT license; 555 LoC

MONAI Generative is Apache-2.0; HM-EDM diffusion code is MIT (Phil Wang lucidrains). We comply by citing in module docstrings.

---

## Architecture overview

```
ImageCAS V_clean (HU)                                    Stage 1 (Week 3-4)
     │                                                    ───────────────────
     ▼                                                    Train KL-VAE only,
  code.data.preprocessing.preprocess_volume()             on V_clean.
     │ resample 1mm³ + HU clip [-1024, 3071] +            Loss: L1 + KL + SSIM
     │ z-norm + crop 192³ around heart bbox               (skip adversarial /
     ▼                                                     perceptual for v1).
  code.models.vae.CardiacVAE  (wraps monai.networks.AutoencoderKL)
     │ encode → z (4 ch × 24³)
     ▼
                                                          Stage 2 (Week 5-6)
                                                          ───────────────────
   Pair (V_clean, V_corrupted)  ←── motion_synth.py       Train EDM denoiser
              │                                            with VAE frozen.
              ▼                                            Loss: EDM σ-weighted
       (z_clean, z_cond) = vae.encode(both)                MSE on z_clean.
              │
              ▼
   z_input = concat([z_t = z_clean + σε, z_cond], ch)    Conditioning = concat
              │                                            (user-confirmed,
              ▼                                            HM-EDM convention).
   ε̂ = unet(z_input, σ)  ← monai DiffusionModelUNet
              │
              ▼ EDM Heun sampler (50 steps)
                                                          Inference (Week 7+)
                                                          ───────────────────
   N=16 parallel posterior samples                        Posterior sampling +
              │                                            uncertainty (V2 brief
              ▼                                            §07 novelty 2).
   posterior_mean (V_corrected), posterior_std (V_uncertainty)
              │
              ▼ vae.decode → V_corrected (HU)
```

### Key design decisions (already user-approved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Latent factor | 8× spatial (192³ → 24³, 4 ch) | MONAI default works; ~120 KB / latent vs 50 MB / pixel volume |
| Conditioning | **Channel concat** (z_t ⨁ z_cond) | Simpler, HM-EDM convention, well-supported by MONAI inferer's `mode="concat"` |
| Diffusion framework | **EDM (Karras 2022)** | HM-EDM precedent in medical CT; 50-step sampling vs DDPM 1000 (30× faster N-sample posterior) |
| VAE loss (v1) | L1 + KL + SSIM only | Skip perceptual+adversarial for v1 (faster to debug; can add v2 if recon quality suffers) |
| UNet | MONAI `DiffusionModelUNet` 3D | Maintained, integrates with MONAI inferer; we adapt HM-EDM's EDM math layer to wrap it |

---

## File-by-file structure (12 files + 2 YAML)

All paths are absolute under `/home/mingzhang/cardiac-artifacts/`.

### Data layer

**`code/data/preprocessing.py`** (~150 LoC, new)
- Function `preprocess_volume(v_path, label_path, out_dir, target_spacing=(1.0,1.0,1.0), hu_clip=(-1024, 3071), crop_size=(192,192,192))` → preprocessed `.npz` with `volume`, `heart_mask`, `metadata`
- Reuses: `code.data.imagecas_loader` for path resolution
- Uses MONAI transforms: `Spacing`, `ScaleIntensityRange`, `CropForeground` (with TotalSegmentator-derived bbox)
- Smoke test: 1 case end-to-end produces `.npz` with shape (192,192,192)

**`code/data/imagecas_dataset.py`** (~150 LoC, new)
- Class `ImageCASPairedDataset(monai.data.Dataset)`: yields `(z_clean, z_cond)` pairs OR `(V_clean,)` alone (mode flag for VAE vs diffusion training)
- Class `ImageCASCleanDataset` for VAE training (single volume per item)
- Reuses: motion_synth.synthesize_motion_artifact for on-the-fly corrupted generation OR loads pre-generated pairs from disk
- Modes: `mode="vae"` (single clean), `mode="diffusion"` (paired clean + corrupted)
- Smoke test: dataloader yields tensors with correct shapes

**`code/data/transforms.py`** (~80 LoC, new)
- MONAI Compose pipelines for VAE training and for diffusion training
- Augmentation: random flip (3 axes), random rotate (small, ±5°), random intensity shift
- Reuses: MONAI transforms; no custom code

### Model layer

**`code/models/vae.py`** (~50 LoC, new)
- Class `CardiacVAE(monai.networks.nets.AutoencoderKL)`: thin wrapper specializing constructor to cardiac-CT defaults
  - `spatial_dims=3, in_channels=1, out_channels=1, num_channels=(32, 64, 128), latent_channels=4, num_res_blocks=2, attention_levels=(False, False, True)`
- Methods: `encode_to_latent(v) → z` (deterministic μ), `decode_from_latent(z) → v`
- Adapted from: `external/GenerativeModels/tutorials/generative/3d_autoencoderkl/3d_autoencoderkl_tutorial.py` (constructor lines 183-192)

**`code/models/conditional_denoiser.py`** (~80 LoC, new)
- Class `ConditionalDenoiser(monai.networks.nets.DiffusionModelUNet)`: 3D UNet with `in_channels=8` (4 z_t + 4 z_cond)
  - `out_channels=4, num_channels=(64, 128, 256, 256), num_res_blocks=2, attention_levels=(False, False, True, True), num_head_channels=(0, 0, 64, 64)`
- Forward signature: `forward(z_input, sigma) → ε̂` where `z_input = cat([z_t, z_cond], dim=1)`
- Sigma is passed via `timesteps` kwarg (verify MONAI accepts float sigma; if not, cast via `c_noise(sigma)` per EDM preconditioning)
- Adapted from: MONAI 3D LDM tutorial `DiffusionModelUNet` constructor

**`code/models/edm.py`** (~200 LoC, port + adapt from HM-EDM)
- Class `EDM(nn.Module)`: ports `external/HM-EDM/diffusion_models/conditional_EDM_3D.py`
- Constructor: takes `denoiser: ConditionalDenoiser`, `image_size=(24, 24, 24)`, EDM hyperparams
- Method `loss(z_clean, z_cond) → scalar`: EDM σ-weighted MSE
- Method `sample(z_cond, n_samples=1, num_steps=50) → z_pred` (Heun sampler, deterministic OR stochastic via S_churn)
- Removes HM-EDM dependencies on lucidrains UNet attributes (`model.channels`, `model.conditional_diffusion`)
- Hyperparams (HM-EDM defaults): σ_min=0.002, σ_max=80, σ_data=0.5 (TBD: re-tune for latent), ρ=7, P_mean=-1.2, P_std=1.2, S_churn=80

### Training layer

**`code/training/train_vae.py`** (~150 LoC, new entry-point)
- Header docstring: cites `experiments/runs/<slug>` per python-code-conventions §7
- Loads YAML config via OmegaConf
- Trains `CardiacVAE` with L1 + KL + SSIM loss
- Mixed precision (`torch.autocast`)
- WandB logging
- Saves checkpoint + `config_resolved.yaml`
- Adapted from: `3d_autoencoderkl_tutorial.py` lines 222-320 (loss = recon + KL only; we drop the adversarial+perceptual blocks for v1)

**`code/training/configs/vae_v1.yaml`** (~50 lines)
- Defines: model architecture, optimizer (Adam lr=1e-4), batch_size=2, n_epochs=100, val_interval=5, KL weight 1e-6, SSIM weight 0.1
- Latent shape recorded for downstream consumption

**`code/training/train_diffusion.py`** (~200 LoC, new entry-point)
- Header docstring per conventions
- Loads VAE checkpoint (frozen, eval mode, no_grad encoding)
- Trains `EDM(ConditionalDenoiser)` on `(z_clean, z_cond)` pairs
- Loss = `edm.loss(z_clean, z_cond)`
- Saves checkpoint + EMA copy
- Adapted from: HM-EDM `step2_train.ipynb` Trainer class + MONAI 3D LDM tutorial training loop

**`code/training/configs/diffusion_v1.yaml`** (~50 lines)
- Defines: UNet architecture, EDM hyperparams (σ_min/σ_max/etc.), optimizer, batch_size=4, n_epochs=200, EMA decay 0.999

### Inference & Evaluation

**`code/inference/posterior_sample.py`** (~150 LoC, new)
- Function `posterior_sample(v_corrupted, vae, edm, n_samples=16, num_steps=50, return_uncertainty=True) → (V_mean, V_std)`
- N parallel Heun sampling chains starting from independent Gaussian noise
- Computes posterior mean (V_corrected) + voxel-wise std (V_uncertainty)
- Adapted from: HM-EDM `step3_predict.ipynb` `Sampler.sample_3D_w_trained_model`

**`code/evaluation/metrics.py`** (~100 LoC, new)
- Functions: `psnr(a, b, data_range)`, `ssim_3d(a, b, data_range)`, `nrmse(a, b)`, `dice_lumen(a, b, hu_threshold=200)`
- Reuses: MONAI metrics where available

**`code/evaluation/run_eval.py`** (~150 LoC, new entry-point)
- Loads VAE + EDM checkpoints
- Iterates test split, runs posterior sampling, computes metrics, saves CSV + summary plot
- Outputs: `experiments/runs/<slug>/eval_metrics.csv` + 5 example PNG visualizations

---

## Implementation order (8 steps with smoke-test gates)

Each step ends with a smoke test that **must pass on CPU before proceeding**.

### Step 1 — Data preprocessing (`preprocessing.py`)
- Implement `preprocess_volume()`
- **Smoke test**: run on case 1 → `.npz` with shape (192, 192, 192), spacing (1, 1, 1), HU range [-1024, 3071]
- Does NOT require GPU (SimpleITK + numpy only)
- Output goes to `data/imagecas/processed/v1/case_<id>.npz`

### Step 2 — Datasets + transforms
- Implement `ImageCASCleanDataset` and `ImageCASPairedDataset`, `transforms.py`
- **Smoke test**: dataloader iter yields tensor batch with right shapes (2, 1, 192, 192, 192) for VAE mode, (2, 1, 192, 192, 192) × 2 for diffusion mode
- Pre-generate motion pairs for first 5 cases (via existing motion_synth.py on CPU) for testing

### Step 3 — VAE module + smoke train (1 step)
- Implement `code/models/vae.py`
- **Smoke test**: instantiate `CardiacVAE`, forward 1 random tensor (1, 1, 32, 32, 32) on CPU, check output shape
- Then 1 mini-batch training step on CPU (loss computes, backward pass succeeds)
- Ensures architecture is wired correctly before scaling

### Step 4 — VAE training entry-point + config
- Implement `train_vae.py` + `vae_v1.yaml`
- **Smoke test**: launch training with `--smoke` flag (1 epoch × 4 batches) on CPU, check checkpoint save + WandB log entry exists
- Real training at scale defers to GPU

### Step 5 — Conditional denoiser + EDM module
- Implement `code/models/conditional_denoiser.py` + `code/models/edm.py`
- **Smoke test**: instantiate both, run `edm.loss(z_clean, z_cond)` on (1, 4, 24, 24, 24) random tensors on CPU; backward pass succeeds
- Run `edm.sample(z_cond, n_samples=2, num_steps=10)` on CPU, check output shape

### Step 6 — Diffusion training entry-point + config
- Implement `train_diffusion.py` + `diffusion_v1.yaml`
- **Smoke test**: launch with `--smoke` flag (1 epoch × 4 batches, 10 sampling steps) on CPU, check VAE-frozen pattern correct (no grad in encoder), checkpoint save

### Step 7 — Posterior sampling
- Implement `code/inference/posterior_sample.py`
- **Smoke test**: forward N=4 parallel chains on (1, 1, 32, 32, 32) random "V_corrupted", compute mean + std, check shapes; on CPU 5 sampling steps

### Step 8 — Evaluation pipeline
- Implement `code/evaluation/metrics.py` + `run_eval.py`
- **Smoke test**: compute metrics on synthetic toy pair (V_clean, V_clean+noise) → PSNR > 20 dB, SSIM > 0.8 (sanity)

---

## Critical risks + mitigations

| Risk | Mitigation |
|------|------------|
| **MONAI `DiffusionModelUNet` rejects float `timesteps`** (expects int) | Step 5 smoke test will catch. Fallback: feed `c_noise(sigma) * 1000` cast to long; or write a thin wrapper module |
| **EDM `sigma_data=0.5` wrong for our latent** (latent stats differ from pixel `[-1, 1]`) | After Step 4 finishes, encode a sample of training data and compute std; update `sigma_data` in `diffusion_v1.yaml` before Step 6 real run |
| **VAE on 192³ exceeds 48 GB GPU memory at batch_size=2** | Start with crop_size=(160,160,160) or batch_size=1 + grad accumulation; tune at training time |
| **Concat conditioning UNet 8 ch input doesn't match `latent_channels=4`** | Step 5 smoke test catches; arithmetic is `2 × latent_channels = 8` |
| **`AutoencoderKL` API in MONAI 1.5 vs monai-generative 0.2.3** drift | Verify imports in Step 3; pin versions in pyproject.toml |
| **GPU release while we're mid-implementation** | Each step's smoke test is CPU-runnable. We can pause at any step and run real training the moment GPU is free |

---

## What we are NOT doing in this plan

- **Actual training runs** (zero GPU cycles consumed by this plan)
- **Run-card discipline implementation** (run cards come at first real training, Step 4-onwards has the slug placeholder)
- **Hyperparameter tuning beyond initial defaults** (defaults are educated guesses from MONAI tutorial + HM-EDM; sweep happens after first full GPU run)
- **Adversarial / perceptual VAE loss** (defer to v2 if v1 recon insufficient)
- **DPS guidance / classifier-free guidance** (mentioned in V2 brief §07 — defer to evaluation phase)
- **Downstream lumen segmentation evaluation** (Week 9+; we only stub `dice_lumen` in metrics.py)

---

## Verification (end-to-end on CPU)

After all 8 steps complete, the following should all pass on CPU:

```bash
# Smoke train VAE for 4 mini-batches
python -m code.training.train_vae --config code/training/configs/vae_v1.yaml --smoke

# Smoke train diffusion for 4 mini-batches (uses VAE smoke checkpoint)
python -m code.training.train_diffusion --config code/training/configs/diffusion_v1.yaml --smoke

# Smoke posterior sample (uses both smoke checkpoints)
python -m code.inference.posterior_sample --vae <ckpt> --edm <ckpt> --case 1 --n-samples 4 --num-steps 10

# Smoke eval (1 case)
python -m code.evaluation.run_eval --vae <ckpt> --edm <ckpt> --case-ids 1
```

If all four green, **the moment GPU is free**, the plan is to:
1. Switch `--smoke` flag off
2. Run `train_vae` real (~3-5 days on 48 GB A6000)
3. Re-tune `sigma_data` from VAE-encoded training-data statistics
4. Run `train_diffusion` real (~5-7 days)
5. Run `run_eval` on test split

---

## Critical files reused (DO NOT reimplement)

- `external/GenerativeModels/tutorials/generative/3d_autoencoderkl/3d_autoencoderkl_tutorial.py` — VAE training scaffold
- `external/GenerativeModels/tutorials/generative/3d_ldm/3d_ldm_tutorial.py` — Stage 2 latent diffusion training scaffold
- `external/HM-EDM/diffusion_models/conditional_EDM_3D.py` — EDM math (port to `code/models/edm.py`)
- `code/data/paths.py` — system-path registry (already exists, do NOT modify)
- `code/data/imagecas_loader.py` — already exists, do NOT modify
- `code/data/motion_synth.py` — already exists, called by `ImageCASPairedDataset` for on-the-fly pair generation

---

## Estimated effort

- **Code writing**: ~1380 LoC + 2 YAML configs (per Phase 1 estimate)
- **Wall time at 1 person**: 2-3 working days for skeleton + smoke tests
- **Smoke-test compute**: ~30-60 min CPU total (all 8 step smoke tests combined)
- **GPU time when ready**: 5-12 days for full training (Stage 1 + Stage 2)
