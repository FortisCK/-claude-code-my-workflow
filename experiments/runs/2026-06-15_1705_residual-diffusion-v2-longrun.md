# Residual Diffusion v2 — Long Run (convergence diagnostic)

Date: 2026-06-15 17:05 CEST
Author: MZ
Status: DONE
Run completed: 2026-06-15 23:09 CEST (80 epochs, 21665.4 s; eval/oracle 23:38)
Git SHA: 52529ca
Config: `code/training/configs/diffusion_v2_residual.yaml`
Seed: 42
Dataset version: ImageCAS-v1, train=800 precomputed pairs (split `data/imagecas/splits/v1.json`)
Planned GPU-hours: ~11 (80 epochs × ~501 s/epoch from pilot timing). Runs
**concurrently** with the cathaction RT-DETRv2 task (~6.9 GB); ~42 GB free on the
48 GB GPU, residual EDM footprint is small (batch 1, 128³, no attention).

## Purpose

The residual diffusion that feeds EVERY gate experiment so far is only the
**5-epoch pilot** (`diffusion_v2_residual/pilot5/epoch_005.pt`). The pilot loss
was still dropping fast at epoch 5 (total 0.445→0.244), i.e. far from converged.
The pilot run card itself said: "do not long-run this exact EDM as final v2;
add conservative correction control first." That control (the reliability gate)
is now built and validated, so the deferred long run is unblocked.

This run trains the residual EDM toward convergence to test whether a properly
trained diffusion gives (a) a better posterior mean and (b) a higher
voxel-oracle ceiling than the pilot — the prerequisite for any real performance
edge from the gate.

## Intent (falsifiable) + early-stop

**Expectation:** with convergence, the ungated test5 posterior mean should move
substantially from the pilot's 61.75 HU **toward the U-Net's 40.61 HU**, and the
test5 voxel-oracle ceiling should rise above the pilot's -3.34 HU global.

**Early-stop / falsification (decision gate at epoch ~30, ~4 h in):** evaluate
the epoch_030 ungated posterior mean (n-samples 4, 16 steps) on test5.
- CONTINUE to epoch 80 if test5 posterior-mean MAE ≤ ~48 HU (clear movement
  toward U-Net).
- ABORT if it is still > ~52 HU (stalled near the pilot's 61.75 HU) → the
  diffusion is not becoming a competitive conditional-mean estimator; the
  performance ceiling is structural; pivot the paper spine to UQ + downstream
  and keep U-Net + gate v1 as the "competitive + consistent" performance line.
- 48–52 HU is a grey zone → let it finish but temper expectations.

Rationale for skepticism (documented up front so this is not post-hoc): a
supervised L1 U-Net already approximates the conditional mean, which is exactly
what the diffusion posterior mean also estimates, so the diffusion mean should at
best *match* the U-Net. The v3a null result (adding heart_mask/SNR channels did
not help the gate) is evidence the bottleneck is per-voxel reliability
*predictability*, not residual quality — which a longer diffusion may not fix.
Prior on a strong performance win: ~25–35%; on a small region-localized edge: ~60%.

## Hyperparameters (delta from pilot)

- epochs: 5 → **80** (fresh run, not resumed; constant lr 1e-4, no schedule)
- everything else identical to `diffusion_v2_residual.yaml`: denoiser channels
  [24,48,96,96], no spatial attention, batch 1, patch 128³, EDM
  (sigma_max 1.0, sigma_data 0.05, P_mean -3.0), loss EDM-MSE + 0.25 final-L1 +
  0.02 grad-L1, EMA 0.999, ckpt every 5 epochs.

## Commands

Training (background):

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  -m code.training.train_residual_diffusion \
  --config code/training/configs/diffusion_v2_residual.yaml \
  --epochs 80 \
  --run-slug diffusion_v2_residual_longrun \
  --ckpt-dir experiments/checkpoints/diffusion_v2_residual/longrun \
  --no-wandb
```

Early-read eval (auto-run by a watcher once epoch_030.pt exists; also re-run on
epoch_080.pt at the end):

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_residual_diffusion_full_volume.py \
  --ckpt experiments/checkpoints/diffusion_v2_residual/longrun/epoch_030.pt \
  --weights ema --val-cases 0 --test-cases 5 \
  --roi-size 128 --num-steps 16 --n-samples 4 --figure-cases 0 \
  --out-dir experiments/runs/diffusion_v2_residual/eval_longrun_e030_test5
```

## Reference numbers to beat

| Quantity (test5) | pilot (epoch5) | U-Net | oracle ceiling |
|---|---:|---:|---:|
| ungated posterior-mean MAE HU | 61.75 (n=1) / ~? (n=4) | 40.61 | — |
| voxel-oracle global delta HU | — | — | -3.34 |

## Final metrics

### Epoch-30 early read (decision gate) — CONTINUE

test5, 4-sample posterior mean, 16 steps (`eval_longrun_e030_test5/`):

| | MAE HU | vs U-Net |
|---|---:|---:|
| U-Net init | 40.6124 | — |
| diffusion ungated posterior mean (e030) | **43.6600** | +3.0476 |
| pilot e005 (n=1) reference | 61.75 | +21.14 |

Training loss still dropping (e5 total 0.244 → e33 0.188). The ungated posterior
mean fell to within +3.05 HU of U-Net, **below the 48 HU CONTINUE threshold** →
**decision: CONTINUE to epoch 80.** The diffusion is becoming a competitive
conditional-mean estimator; the residual proposal is much better than the pilot's,
so the gate has better material and the oracle ceiling has likely risen.

### Epoch-80 final + oracle re-measurement

Training DONE in 21665.4 s (80 epochs, loss plateaued ~0.17 from pilot's 0.244).
test5, same 5 cases as the pilot oracle (21/32/41/50/51).

Ungated posterior-mean MAE trajectory:

| | MAE HU | vs U-Net (40.61) |
|---|---:|---:|
| pilot e5 (n=1) | 61.75 | +21.14 |
| e030 (n=4) | 43.66 | +3.05 |
| **e080 (n=4)** | **41.79** | **+1.19** |

Voxel-oracle ceiling (test5), converged diffusion vs pilot — **headroom SHRANK**:

| oracle Δ vs U-Net | pilot | **e080** |
|---|---:|---:|
| global | -3.34 | **-2.83** |
| heart | -5.09 | **-3.81** |
| boundary | -4.90 | **-4.38** |

Cross-check: the (mismatched) v1 gate applied to the converged-diffusion cache
still gives global Δ -0.0751 HU — same as v1 on the pilot. Convergence did not
unlock gate gain.

## What happened

The diffusion converged cleanly and the earlier "diffusion is far worse than
U-Net" was largely a 5-epoch training artifact: the ungated posterior mean fell
from 61.75 → 41.79 HU, now only +1.19 HU above U-Net. **But it never beats U-Net
ungated**, exactly as predicted — a supervised L1 U-Net already approximates the
conditional mean, which is the same quantity the diffusion posterior mean
estimates, so the posterior mean asymptotes *to* U-Net, not below it.

The decisive number is the **oracle ceiling, which went the wrong way**: global
-3.34 → -2.83, heart -5.09 → -3.81, boundary -4.90 → -4.38. A *better* diffusion
gives the gate *less* exploitable residual, not more — because as the posterior
mean converges toward U-Net's estimate, the residual it proposes shrinks and
overlaps more with what U-Net already captured. Both facts are the same coin:
raw release becomes less harmful AND there is less to gain from gating.

Implication for performance: even a perfect gate now caps at -2.83 HU global /
-3.81 HU heart, and the learned gate historically captures only ~2-15% of the
oracle, so retraining the gate on the converged diffusion is expected to land at
≈ v1 (-0.076 global / -0.65 heart) or slightly worse. No meaningful performance
win is available from this formulation. (test5, n=5 — small, but the trajectory
and oracle direction are consistent and mechanistically expected.)

## Decision

**KEEP** the converged checkpoint `diffusion_v2_residual/longrun/epoch_080.pt`
as the definitive residual posterior. **DISCARD the performance-via-refinement
path as a paper headline:** the gate's achievable gain is structurally capped and
convergence shrinks it, so chasing a gate retrain for accuracy is low-EV.

This is a clean, characterized negative result, not a dead end: it rigorously
answers "can diffusion refinement beat a strong supervised U-Net on paired
HU-preserving cardiac-CT restoration?" → no, with an oracle-headroom
characterization of why. Two payoffs remain:
1. The converged posterior has meaningful per-voxel uncertainty (σ_r) for the
   first time (pilot uncertainty-error corr was 0.05-0.155, undertrained) →
   evaluate UQ calibration (novelty #2) on this checkpoint. **This is the
   high-EV next step.**
2. Performance framing becomes honest-and-defensible: "matches U-Net, with a
   small consistent conservative-gated improvement, plus free calibrated
   uncertainty U-Net cannot provide" — not "beats U-Net."

Next: do not re-cache/retrain the gate for accuracy. Instead evaluate posterior
std calibration (reliability diagram / ECE / coverage / uncertainty-error
correlation, artifact localization) on `epoch_080.pt`.

## Cross-references

- Predecessor: `2026-05-20_1704_diffusion-v2-residual-pilot5.md`
- Consumers (if KEEP): re-cache `residual_diffusion_features`, retrain `residual_gate_v1`, re-eval test100
- Plan: `quality_reports/plans/reliability-gated-posterior-residual-diffusion.md`,
  `quality_reports/plans/aaai-first-cvpr-fallback-roadmap.md`
