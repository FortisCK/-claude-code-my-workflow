# UQ calibration kill-gate — does σ_r buy anything over U-Net? (converged model)

Date: 2026-06-19 CEST
Author: MZ
Status: DONE — PASS (σ_r informative on converged model; uncertainty dimension is real)
Git SHA: be17e6b (working tree: phase_offset + probes uncommitted)
Config: diffusion_v2_residual.yaml, ckpt longrun/epoch_080.pt (ema), n_samples=6
Seed: 42; Dataset: ImageCAS-v1 test split, first 10 cases
Plan: quality_reports/plans/2026-06-19_diffusion-v2-innovation-shortlist.md (Direction #2 kill-gate)

## Intent — the decisive "is diffusion worth more than U-Net here?" gate

As a point estimator diffusion CANNOT beat U-Net (MMSE law; this run: diff MAE 38.7 ≈
U-Net 38.1). The ONLY diffusion-unique capability is the posterior (uncertainty). Gate:
does σ_r (posterior std) correlate with actual error on the CONVERGED model? The
undertrained pilot was 0.05-0.155; a plan doc claimed 0.42 (untraced). Resolve it.

## Result — PASS

uncertainty-error correlation (Pearson σ_r vs |pred−clean|, per case):
**mean 0.446**, range [0.259, 0.553], n=10, ALL positive.
Per case: 0.259, 0.462, 0.525, 0.486, 0.476, 0.514, 0.492, 0.274, 0.419, 0.553.
(diff_v2 MAE 38.68 ≈ U-Net 38.13 — ties on distortion, as expected.)

The plan-doc "0.42" is CONFIRMED (0.446); the 0.05-0.155 was the undertrained pilot.
=> σ_r is a genuine, moderate calibration signal U-Net cannot provide. The
diffusion-unique positive contribution (calibrated UQ + variance-gated abstention) has
a real basis.

## Full calibration suite (uq_calibration_suite.py, CPU on saved npz, n=10, heart-masked)

| metric | value | reading |
|---|---|---|
| Pearson σ vs \|error\| | **0.348** | informative but MODERATE (the 0.446 single-run number was optimistic) |
| coverage @50% nominal | 0.362 | under-covered |
| coverage @90% nominal | **0.706** | OVER-CONFIDENT (intervals too narrow) -> needs temperature scaling |
| AUSE / random | **0.646** | high-σ abstention only ~35% of the way random->oracle: WEAK ranking |
| mean σ 36.9 HU vs mean \|err\| 46.7 HU | — | σ underestimates error (consistent w/ under-coverage) |

Figure: experiments/runs/uq_calibration_killgate/calibration.png (reliability + sparsification).

**Honest conclusion:** σ_r is a REAL, diffusion-unique signal (informative, better than random)
but MODERATE-TO-WEAK: under-calibrated (temperature scaling can fix coverage) and only weakly
ranks errors (AUSE 0.65 — scaling cannot fix the ranking). => an HONEST SUPPORTING leg for the
abstention story, NOT a paper-carrying standalone result. The cheap error_corr gate oversold it;
the full suite is the truth.

## Anti-cheat correction (verified on disk this session)

TS coronary_arteries = Dataset509, coronary_arteries_LEGACY = Dataset507 — BOTH
`coronary_arteries_cm_nativ_400subj`, same nnU-Net + same 400-subj cohort, differ only in
trainer. They are NOT independent -> "LEGACY anti-cheat" only catches gross collapse, not
shared-blind-spot gaming. The genuinely independent judge is the GT-lumen GEOMETRIC Dice
(no nnU-Net). All anti-cheat claims (here + DPS run cards) must gate on GT-lumen, not LEGACY.

## Decision

KEEP — uncertainty dimension validated as the diffusion-unique win. Proceed: (1) full
calibration suite (ECE/coverage/AUSE) to make it publication-grade; (2) variance-gated
abstention method (Act III) now justified to build. Resolves the σ_r integrity flag.

## Cross-references

- Shortlist/plan: quality_reports/plans/2026-06-19_diffusion-v2-innovation-shortlist.md
- MICCAI panel (TMI as home; baselines + reader study still needed): area-chair synthesis
- Outputs: experiments/runs/uq_calibration_killgate/{metrics.csv,summary.json,uncertainty/}
