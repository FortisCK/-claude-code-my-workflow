# R0 cohort — operator-fidelity budget on the real native-motion cohort (Act-I headline)

Date: 2026-06-21 CEST
Author: MZ
Git SHA: b0cf6d4 (working tree: r0_cohort.py uncommitted)
Config: diffusion_v2_residual.yaml U-Net init; forward-only (no DPS, no training, no TS-Dice)
Dataset: Leo_anon worst-N native-motion cases (lowest in-coronary sharpness, leo_motion_screen/ranking.csv)
Planned GPU-hours: ~1 (forward-only, ~13 grid evals x2 per case; coexists w/ user GPU job)
Status: DONE — Act-I CONFIRMED (n=20; real +0.6% vs synth +21.2%, CIs disjoint)

## Intent (Act-I headline, falsifiable)

Scale the single-case R0 (058) into a STATISTICAL claim: across the real native-motion
cohort, can ANY parametric motion operator explain the real artifact better than no-motion?
Each real case is PAIRED with a matched-severity positive control on the SAME anatomy
(inject known synthetic motion at ~the same artifact magnitude). Expectation (locked):
- REAL native motion: residual reduction ~0% (operator non-representable), bootstrap CI excludes a meaningful effect.
- SYNTHETIC matched-severity (same anatomy): large reduction (test is sensitive at this difficulty).
=> real cardiac motion is MODEL-CLASS non-representable, controlling for difficulty. This is
the GT-free admissibility-test contribution; it also explains why DPS (already shown +0.13
on injected vs +0.0 on native) cannot transfer — NO re-run of DPS here, it is cited.

## Method (per case, forward-only)

real y -> 1mm 192^3 heart-crop (TS heart); x_hat = U-Net(y). no-motion residual
||A_nomotion(x_hat)-y||_heart (HU-affine calibrated). grid phase{0,.25,.5,.75} x amp{.5,1,1.5}
-> best residual -> red_real = (r_nomo - r_best)/r_nomo. Positive control: inject synthetic
motion onto x_hat scaled to ~match the real artifact magnitude -> same grid -> red_synth.
Aggregate mean +/- bootstrap 95% CI over the cohort.

## LOCKED reading

mean red_real < 8% with CI upper-bound < red_synth lower-bound -> real motion not
representable (Act-I confirmed). If red_real ~ red_synth -> the single-case 058 was not
representative; reassess.

## Result (n=20 worst-motion Leo cases)

| arm | mean residual reduction | bootstrap 95% CI |
|---|---:|---|
| REAL native motion | **+0.6%** | [-0.8, +1.9] |
| SYNTH matched-severity (same anatomy) | **+21.2%** | [+16.5, +25.9] |

20/20 cases: real ~0% (no parametric operator beats no-motion), synth large. CIs DISJOINT
(real upper +1.9 < synth lower +16.5). Mean artifact magnitude real 70 HU vs synth 86 HU —
the synthetic control is even HARDER (larger) yet representable, so difficulty is ruled out
as the confound (conservative). Generalizes the single-case 058 (+0.0%) to a statistical claim.

## Decision

KEEP — Act-I headline established. Real cardiac motion is MODEL-CLASS non-representable by
the parametric cone-beam+DVF forward operator, controlling for difficulty. GT-free; explains
why operator-dependent methods (DPS et al.) cannot transfer to real native motion (no re-run;
cited from 2026-06-19 runs). This is the paper's strongest, most novel contribution.

CAVEAT: one-shot severity match landed synth slightly higher (86 vs 70 HU) — conservative, but
the final figure should report per-case magnitudes + ideally tighten the match. n=20 -> can
extend to the full ranked cohort for the camera-ready.

## Cross-references

- Single-case R0 + positive control: experiments/runs/2026-06-19_r0-forward-model-mismatch.md
- DPS injected-vs-native contrast (cited, not re-run): 2026-06-19_dps-real-leo-external-validity.md, _blind-dps-native-058.md
- Code: scripts/python/r0_cohort.py


## Cross-references

- Single-case R0: experiments/runs/2026-06-19_r0-forward-model-mismatch.md (058, +0.0%; pos-control +18.9/37.5%)
- Plan: quality_reports/plans/2026-06-19_dps-native-motion-operator-estimation.md
- Code: scripts/python/r0_cohort.py (reuses r0_forward_mismatch helpers)
