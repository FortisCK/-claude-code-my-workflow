# Plan: Enable DPS on Native Real Cardiac Motion via Operator Estimation

**Status:** DRAFT (awaiting approval)
**Date:** 2026-06-19
**Author:** design-workflow synthesis (9 agents, verified vs logs + source)
**Branch:** `cardiac-artifacts`
**Prior art:** DPS-MC pilot (`experiments/runs/2026-06-17_dps-mc-stage1-pilot.md`),
external validity (`experiments/runs/2026-06-19_dps-real-leo-external-validity.md`),
blind-DPS native failure (`experiments/runs/2026-06-19_blind-dps-native-058.md`)

---

## 0. The decisive fact that reshapes the problem (verified against logs)

| Source | U-Net | DPS true-op | DPS constant (pop-mean) | gap-closure of constant |
|---|---|---|---|---|
| test100 known vs popmean | 0.626 | 0.815 (+0.189) | 0.789 (+0.163) | **f = 0.86** |
| varamp test20 (amps already varied) | 0.606 | 0.782 (+0.176) | 0.728 (+0.122) | **f = 0.69** |
| wrongamp ctrl (2×) | 0.632 | — | 0.617 (−0.015) | collapses ✓ |

**Two consequences:**
1. **A CONSTANT operator already recovers 69-86% of the known-operator gain.** So the
   worry was inverted: it's not that estimation looks trivially good — it's that there's
   **almost no headroom for any estimator to beat the trivial constant operator**,
   because `A()` at full views + per-operator HU refit is largely insensitive to motion
   amplitude/period (amplitude scales artifact *severity*; DPS DC is forgiving of severity).
2. **The only operator DoF that genuinely RE-SHAPES the artifact (not just its magnitude)
   is the cardiac PHASE OFFSET** — which angular views see peak-systolic deformation.
   Currently **fixed at 0** in the synthesizer; also the **least identifiable** param
   (Parker-redundant views → true ambiguity).

The vise — *the identifiable params don't move DPS; the param that moves DPS isn't
reliably identifiable* — is the actual research question.

**Verified code facts (kill 3 of 4 candidate strategies):**
- `tomosipo.to_autograd` makes `A` differentiable ONLY w.r.t. the image volume, NOT motion
  params. DVFs are precomputed eagerly (`dps_forward.py:92`). Period/phase enter via
  `.long()` binning (`motion_synth.py:499`, `dps_forward.py:71`) → **zero gradient**.
  → learned param/DVF regressors (Strategies A, C) cannot get the gradient they assume.
- Amplitudes ARE differentiable (scalar multipliers at `parametric_dvf`).
- `reconstruction_phase_pct` is **dead code** (set `:125`, never read). `motion_strength`
  is degenerate with amplitudes.

---

## 1. RECOMMENDED APPROACH

**Per-case analysis-by-synthesis (no learned regressor in v1)** with 3 amendments:
- **A1 — estimate ONLY what the measurement constrains:** 3 amplitudes (Adam, they're
  differentiable) + period + phase_offset (discrete grid, they're non-diff). Drop
  motion_strength / recon_phase_pct / bulk_translation.
- **A2 — headline metric is "estimated vs CONSTANT operator," NOT vs U-Net** (vs-U-Net is
  pre-rigged: constant already clears 0.70).
- **A3 — primary Gap-1 regime is phase-offset + non-parametric residual** (the only regime
  where the constant operator is not already near-optimal, and the one resembling real motion).

Reuses existing differentiable `A`; **no training, no new checkpoint** in v1.

---

## STAGE 0 — "is there a prize?" probe (~1 GPU-day, do FIRST)

Add `phase_offset` to the synthesizer; sweep it; measure how much DPS-true beats
DPS-constant as a function of phase-offset error.
**If wrong phase-offset does NOT materially degrade DPS → no operator estimator is
load-bearing → deploy the constant operator and STOP** (clean publishable result:
"data-consistency DPS is robust to operator misspecification within a realistic motion
family"). This gate must pass before anything else is built.

---

## 2. WIDENED SYNTHETIC DISTRIBUTION (linchpin of an honest test)

`motion_strength` FIXED at 1.0 (severity carried by amplitudes — removes degeneracy).

| Param | Range |
|---|---|
| contraction_amp_mm | U(5,16) |
| twist_amp_deg | U(5,28) |
| long_axis_amp_mm | U(5,16) |
| cardiac_period_ms | U(550,1050) |
| **phase_offset_frac** | **U(0,1) — NEW, the honesty linchpin** |
| mask_smooth_sigma_mm | U(3,8) |

`phase_offset_frac`: shift time before profile, `t_eff=(t_ms+offset*period)%period`, wired
into BOTH `synthesize_motion_artifact` AND `dps_forward` (shared via `parametric_dvf`).

**Non-parametric tail (TIER-2 eval only):** reuse `make_random_smooth_dvf_fn` (Gaussian
field, n_ctrl∈{4,5,6}, peak_mm~U(2,6), independent temporal profile) — motion the
parametric family CANNOT represent (mimics real).

**Severity match:** reject samples with corrupted-vs-clean heart-MAE outside [40,160] HU.

**Self-check gate:** on wide TIER-1, the constant operator's gap-closure must DROP to
**f_pop ≤ 0.40** (from 0.69). If not, widening has no teeth — widen further before trusting
any estimator number.

---

## 3. GAP-1 PROTOCOL

Held-out: 100-case ImageCAS test split, fresh wide draw seeded `hash(case_id)`/`gap1_eval`.
Frozen prior + U-Net.

**Tiers:** TIER-1 (parametric, sanity) / **TIER-2 (param + non-param + arbitrary phase = HEADLINE)**.

**Arms (same y per case):** (1) U-Net floor, (2) DPS known op = ceiling, (3) DPS constant
op = the baseline to beat, (4) DPS estimated op = candidate, (5) anti-cheat 2× wrong amp
(must collapse), (6) subtle anti-cheat phase shifted 0.25 cycle (if it scores same as
correct → phase is a free nuisance, premise falsified — publishable either way).

**Metrics:** TS `coronary_arteries` Dice-RCA (gold) + `_LEGACY` (anti-cheat) + heart-MAE +
per-param error.

**Make-or-break (vs CONSTANT, not U-Net):**
`f_marginal = (Dice_est − Dice_const) / (Dice_known − Dice_const)`

| Outcome | Condition (TIER-2) |
|---|---|
| **GO** | f_marginal ≥ 0.50 AND Dice_est − Dice_const ≥ +0.03 abs, paired Wilcoxon p<0.05, no case regresses >0.03 below U-Net, both anti-cheats collapse |
| **MARGINAL** | 0.25 ≤ f_marginal < 0.50 → iterate grid / phase marginalization |
| **NO-GO** | f_marginal < 0.25 OR est≈const OR arm-6 doesn't collapse → kill the line (publishable negative) |

---

## 4. BUILD

**Reuse unchanged:** `dps_sample.py`, `DifferentiableMotionForward` (final fast path),
residual-EDM prior + U-Net, coronary Dice-RCA harness, `make_random_smooth_dvf_fn`.

| File | Action |
|---|---|
| `code/data/motion_synth.py` | Modify: add `phase_offset_frac` to `MotionParams`; apply in time→phase path (shared synth+forward). ONLY operator-path change. |
| `scripts/python/sample_motion_params_wide.py` | New: §2 wide sampler + non-param hook + severity rejection. |
| `code/inference/abs_operator_fit.py` | New: per-case ABS. (a) HU-affine from no-motion baseline, FROZEN; (b) Stage-A grid-screen (period×phase) at view_stride=8, x̂=U-Net, keep top-2 by heart-masked ‖A_θ(x̂)−y‖²; Stage-B refine 3 amplitudes by Adam (lr3e-2, 20 steps, unit-RMS clamp) at stride=4, recompute only per-phase DVFs per step (amplitudes differentiable). x̂ stays on diffusion manifold. |
| `scripts/python/eval_gap1_abs.py` | New: §3 protocol (6 arms, f_marginal, Wilcoxon, bootstrap CI, anti-cheat gates). |
| `scripts/python/probe_phaseoffset_prize.py` | New: Stage-0 gate. |

No model training for v1. (Learned phase warm-start = v2, only if Stage-1 MARGINAL.)
Eval run cards per stage.

---

## 5. GPU / SERIALIZATION

Single A6000 48GB; user process always ~22GB; full-view A fwd+bwd ~22.5GB. **Never two GPU
jobs** (lockfile + Monitor until-loop, no foreground sleep).

| Job | Peak | Lever |
|---|---|---|
| wide synth | <22GB | one at a time, resumable |
| ABS Stage-A grid | 6-8GB | stride=8, x̂ frozen |
| ABS Stage-B amp | 8-10GB | stride=4, DVF-only recompute |
| final DPS | 22.5GB | stride=1, runs ALONE |
| TS seg | separate | after DPS |

~6-9 min/case → full 100×4-arm+TS ≈ 1.5-2 GPU-days, gated behind Stage 0 + self-check.
Verify ABS θ̂ stride-invariance (refit stride-1 on ~5 cases) before trusting headline.

---

## 6. PATH TO REAL LEO058 (only after Gap-1 GO)

- **R0 — forward-model-mismatch BUDGET (do FIRST, can kill the goal):** on real cases, no
  GT, quantify how well our A re-synthesizes a no-motion clinical recon (‖A_nomotion(x̂)−y_real‖).
  If irreducible scanner/recon-shape mismatch dominates the motion signal → ABS launders
  scanner mismatch into spurious "motion" → method can't work on real regardless of θ. Report bound first.
- **R1 — resolution:** resample real y to 1mm/192³; fold resampling into R0 budget; no sub-mm claims with a 1mm prior.
- **R2 — ABS on real y:** frozen HU-affine; OOD guard (θ̂ at range boundary → flag fail);
  real-data anti-cheat (scrambled operator → if residual similar, fit is laundering mismatch → stop).
- **R3 — endpoints (no-reference sharpness BANNED as primary — it's the documented metric trap):**
  per-view sinogram-domain DC residual vs feed-forward; downstream TS coronary continuity/centerline
  (+ LEGACY agreement); hallucination guards (voxel ratio ≤1.5, radiologist read). Pre-register all three.

---

## 7. HONEST RISK REGISTER

| Risk | P | What failure teaches |
|---|---|---|
| Stage-0 no prize (wrong phase doesn't hurt DPS) | 35% | Operator robust → deploy constant op, abandon estimation. Publishable. |
| Gap-1 FAILS (f_marginal<0.25, TIER-2) | 40% (cond. on prize) | Phase unrecoverable from static recon (identifiability wall) → need sinogram/multi-phase data. Negative result. |
| Gap-1 MARGINAL | 25% | Justifies v2 hybrid (learned phase warm-start). |
| Gap-1 GO but Gap-2 (real transfer) FAILS | 70% (cond. on GO) | Most likely real outcome; R0/R2 catch it before over-claiming → bottleneck is forward-model fidelity, not estimation. |
| Gap-1 GO and Gap-2 GO | ~10% overall | The win: real de-artifacting on native motion. |

**Bottom line: ~25-30% chance of a positive real-data result; ~70-75% chance of a clean
honest negative/scoping result. Every branch is publishable.** Spend effort in increasing
order of cost to learn which branch we're on.

### Staged go/no-go (least effort first)
```
STAGE 0  (~1 GPU-day)  phase-offset prize probe   → no prize: STOP, deploy constant op
STAGE 1a (~0.5 day)    wide synth + self-check (f_pop≤0.40?)  → fails: widen/STOP
STAGE 1b (~2 days)     Gap-1 ABS eval TIER-1→TIER-2  → NO-GO: STOP; MARGINAL: v2; GO: continue
STAGE R0 (~0.5 day)    real forward-model-mismatch budget  → dominates: STOP
STAGE R1-R3 (~2 days)  real Leo058 ABS + downstream + visual, falsification-gated
```
Full ~7 GPU-days only if every gate passes. First two gates (~1.5 days) decide if the
direction is real at all.

## Rejected (with reason)
- Strategies A/C (learned estimators): depend on a θ-gradient that does not exist in code;
  would learn synthesizer fingerprints (optimistic Gap-1, useless Gap-2).
- Estimating motion_strength / recon_phase_pct / bulk_translation: degenerate / dead / nuisance.
- Recovery-over-U-Net headline: pre-rigged. No-reference sharpness on real: banned (metric trap).
