# OOD motion generalization probe — is the U-Net robust or did it overfit the sim?

Date: 2026-06-16 15:45 CEST
Author: MZ
Status: DONE
Git SHA: be17e6b (working tree: ood probe + motion_synth dvf_fn injection uncommitted)
Config: code/training/configs/diffusion_v2_residual.yaml (initializer = unet_v1/epoch_200.pt EMA)
Seed: 42
Dataset version: ImageCAS-v1, test split (clean volumes + on-the-fly OOD motion)
Plan: quality_reports/plans/2026-06-16_coronary-lumen-winning-result.md (root-cause branch)

## Purpose

The training pairs varied ONLY motion_strength~U(0.5,1.3) and period~U(600,1000);
the four amplitudes and the scan geometry/recon-phase were FIXED. So the
corruption family is ~1-2D. Does the frozen U-Net (unet_v1) genuinely correct
motion, or did it memorize the specific synthetic forward operator? Decisive for
whether Path A (image-domain paired supervision) has any room left for a
generative method, or whether we must pivot.

## Method

`scripts/python/ood_motion_probe.py`. For each test case, regenerate corrupted
under regimes the U-Net never saw, keeping the forward operator
(warp -> cone-beam project -> Parker FBP -> HU calibrate) byte-identical and
changing ONLY the motion. The structurally-different motion is injected via a
new `dvf_fn` hook in `synthesize_motion_artifact` (default = parametric_dvf):
`make_random_smooth_dvf_fn` produces a non-anatomical smooth random DVF (no
contraction/twist/long-axis), severity-matched via peak_mm=12 so the comparison
isolates "different motion MODEL" from "harder motion".

Regimes: control (parametric, in-dist), ood_amp (amps bumped: twist 22°,
contraction/long-axis 14mm — training never varied amps), ood_strength
(strength 2.2 > 1.3 max), ood_randdvf (random smooth motion model, peak 12mm).
Note: reconstruction_phase_pct is a dead param in motion_synth (no-op) and
n_views 500 vs 1000 both oversample a 192³ recon (no-op), so operator-knob OOD
is uninformative — the motion-MODEL OOD (ood_randdvf) is the hard test.

## Results (test5, U-Net; HU; "removed" = artifact reduction vs corrupted)

| regime | corrupt MAE | unet MAE | removed | heart MAE | lumen MAE |
|---|---:|---:|---:|---:|---:|
| control | 426.9 | 40.6 | 90% | 49.4 | 84.5 |
| ood_amp | 428.2 | 42.6 | 90% | 58.9 | 103.5 |
| ood_strength | 430.6 | 50.9 | 88% | 78.8 | 128.8 |
| ood_randdvf (matched severity) | 424.6 | 41.2 | 90% | 56.3 | 65.2 |

Per-case ood_randdvf vs control global MAE delta: -0.1, +1.6, -0.1, -0.4, +1.8
(corrupt severities matched within ~3 HU). Consistent, no case collapses.

## What happened

The U-Net degrades GRACEFULLY, not catastrophically, across every OOD regime,
and on the hardest test — a structurally-different non-anatomical random motion
model at MATCHED severity — its global MAE is essentially unchanged (41.2 vs
40.6) and it still removes 90% of the artifact. A model that had memorized the
parametric forward operator would collapse here (removed% crashing); it does not.
heart degrades modestly (+14%); lumen is actually lower (the random field lacks
the parametric model's heavy apex-twist that most disturbs the coronaries).

## Decision

**KEEP as the generalization evidence. Verdict: the U-Net genuinely learned motion
de-blurring, not the synthesis recipe — it is a strong, robust baseline.**

Strategic consequence: Path A (image-domain, paired-supervised, conditional
diffusion refinement) is now fully closed for a "diffusion beats the baseline"
result — accuracy is capped (diffusion mean = conditional mean = U-Net),
perceptual/lumen does not win (2026-06-16_1110 lumen eval), AND robustness is not
a differentiator (U-Net is already robust here). The genuine-win paths are:
(B) Path B — unconditional clean-image prior + posterior sampling against the
known physics (DPS), where the prior is mathematically irreplaceable; or
(C) a UQ-led + negative-result-characterization paper.

Caveat: this is synthetic→synthetic-different-model. Sim-to-REAL generalization
(real clinical motion) remains the ultimate test and needs unpaired real data.

## Cross-references

- Lumen/perceptual eval (also no diffusion win): `2026-06-16_1110_lumen-eval-e080.md`
- Diffusion convergence / capped accuracy: `2026-06-15_1705_residual-diffusion-v2-longrun.md`
- Code: `scripts/python/ood_motion_probe.py`, `code/data/motion_synth.py` (dvf_fn + make_random_smooth_dvf_fn)
