# Diffusion-v2 innovation shortlist (vetted) — pivot after DPS-on-real died

Status: DRAFT (direction-setting; awaiting user pick of main line)
Date: 2026-06-19
Source: 11-agent ideation+vetting workflow (wf_cfae7d9a), grounded in this session's findings
Supersedes the "beat U-Net on a metric" framing — that door is closed (see Verified facts).

## Verified facts that discipline everything (checked against run cards this session)

- DISTORTION floor: U-Net (L1) = MMSE; nothing beats it on MAE/PSNR.
- DOWNSTREAM (Dice-RCA), feed-forward: U-Net 0.656; diff_mean 0.654 (ties); diff_sample
  **0.619 (LOSES)** — run card 2026-06-16_1630. The sample's sharp texture HURTS the
  segmenter. => feed-forward diffusion does NOT beat U-Net on downstream either.
- DPS (KNOWN operator) beats downstream (+0.19 synth, +0.129 real-anatomy+injected) but is
  inverse-crime; R0: real native motion NOT representable (+0.0% vs +18.9%/+37.5% pos-control)
  => operator-dependent real-data methods are dead.
- σ_r (posterior std) uncertainty-error corr: **CONVERGED model = 0.446** (verified, run card
  2026-06-19_uq-calibration-killgate, n=10, range 0.26-0.55). The 0.05-0.155 was the UNDERTRAINED
  pilot; the plan-doc "0.42" is confirmed. => uncertainty dimension is REAL (Direction #2 alive).
  Still TODO for publication-grade: full ECE/coverage/sparsification(AUSE) suite.

## The 3 survivors (pivot from "win on accuracy" to "characterize boundary + act on uncertainty")

### #1 Operator-Fidelity Budget (novelty 8, PURSUE)
Promote R0 into a general, transferable GT-free **admissibility test**: before deploying any
operator-dependent (DPS/data-consistency/unrolled) cardiac MoCo on real data, report the
budget decomposing ‖A_nomotion(x̂)−y‖_heart into (i) scanner/recon mismatch (ii) parametric-
motion-explainable signal. Claim: real cardiac motion is **model-class non-representable** →
field-wide practice of validating MoCo on y=A(clean) overstates real gains. First experiment
(~1 GPU-day, forward-only): scale r0_forward_mismatch.py to the full ranked Leo native-motion
cohort; per-case budget + bootstrap CIs + positive-control power table; tie DPS Dice gain on
injected vs ~0 on native at matched severity. Operator-free? No. Real-data? Yes.
Risk: must cleanly separate static round-trip (~60-120 HU, dominant) from motion signal; scope
to operator FAMILY not "estimation dead in general".

### #2 Hallucination-Controlled Restoration via Posterior-Variance Gating (novelty 6, PURSUE)
Operator-free, real-applicable. x_final = x_u + sigmoid(a − b·σ_r)·(x_sample − x_u): trust
generative detail where chains agree, fall back to MMSE U-Net where they disagree. Real leg
(no GT): σ_r flags moving coronary segments / OOD; use as ABSTENTION ("declines to invent");
report fraction of correction suppressed in high-σ voxels. Upgrades the UQ leg from "measure
ECE" to "a method that acts on its uncertainty". **KILL-GATE (cheap, do first): UQ calibration
on synthetic test100 (has GT) — ECE/coverage/σ-vs-error.** Measured corr is 0.05-0.155 → this
is a coin-flip. Risk: σ rises for two reasons (epistemic + artifact magnitude), inseparable on
real without GT; accuracy upside likely ~0 (best op point may be g≈0) → frame as safety, not a Dice win.

### #3 The Sharpness Trap: reference-free structural-correctness audit (novelty 7, PURSUE as a section)
Controlled demo that no-reference sharpness/perceptual quality and downstream STRUCTURAL
correctness move in OPPOSITE directions for diffusion restoration of OOD real cardiac CT;
deliver a reference-free hallucination guard = two-segmenter agreement gap |Dice_TS − Dice_LEGACY|
+ coronary voxel-count ratio. First experiment (~1-2 GPU-days, mostly analysis): sweep no-ref
sharpness vs Dice-RCA-vs-clean for {U-Net, diff-mean, diff-sample, gate, DPS} on synthetic
test100 (GT colors true-vs-hallucinated), calibrate guard ROC, apply on real Leo.
Risk: (1) honest exemplar is blind-DPS-058 (51.8→77.6 hallucinated), NOT the degenerate
0.656→0.000 training failure; (2) document TS vs LEGACY independence or the guard is blind when
hallucination is most convincing.

## Paper spine (recommended)

**"When does cardiac-CT motion correction actually work on real data, and how do you know
without ground truth?"** — diffusion-central, honesty-by-construction characterization+method
paper. Best fit IEEE TMI or MICCAI eval/analysis track (NOT a CVPR SOTA slot).
- Act I — boundary (#1): real motion is model-class non-representable → measurement-consistency
  wins are inverse-crime artifacts of synthetic eval → licenses going operator-free.
- Act II — trap (#2 #3): on OOD real input the prior paints sharp texture that fools every
  no-reference metric while destroying segmentable structure → reference-free guard is the antidote.
- Act III — deployable answer (#2): operator-free posterior-variance-gated estimator that releases
  detail only where confident and ABSTAINS otherwise; validated by synthetic ECE + real abstention-localization.
Every numeric claim traces to a run card; diffusion is central (hallucinator in II, uncertainty
source in III); real-data leg never fabricates GT (validates by localization/calibration/abstention).
Absorbs the in-repo nulls (sample loses) AS evidence for the thesis.

## Graveyard (do NOT revisit)

- Task-PMRF: novelty pre-empted (arXiv:2508.12640, MICCAI'25 3D medical PMRF) + transports toward
  clean texture = the thing that makes the segmenter worse.
- "Sample beats mean" / best-of-N / MBR consensus: refuted in-repo (diff_sample 0.619 < U-Net).
- DiSTiL-MoCo (distill DPS→operator-free student): inverse-crime laundering; feed-forward-on-real
  already smooths/hallucinates.
- Generative motion augmentation: OOD probe shows U-Net already generalizes across motion models
  (MAE 41.2 vs 40.6); real-vs-sim residual is anatomy+resolution gap, not motion.
- PhysReg (forward-consistency regularizer): incremental, capped by distortion floor, same OOD wall.
- CORE-Bench standalone: single-institution/single-operator; demote its anti-cheat+oracle+admissibility
  protocol to the spine's eval scaffolding.

## Honest odds

~25% strong method result (variance gate, hinges on σ_r clearing the ECE kill-gate = coin-flip),
~70% solid characterization/eval paper, ~5% null. Bet on the characterization spine; treat a
variance-gated win as bonus.

## Cheapest highest-value next experiment

UQ calibration kill-gate (#2a): ECE/coverage/σ-vs-error on synthetic test100 (operator-free, has
GT, reuses residual_diffusion_sliding_window mean+σ). Decides if the only method-win leg is alive,
AND is needed for the uncertainty leg regardless. ~0.5 GPU-day.
