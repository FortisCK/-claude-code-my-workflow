# Stage 0 — sparse-view sinogram sim + FBP/SIRT sanity (pivot line)

Date: 2026-07-03 CEST
Author: MZ
Git SHA: (cherry-picked plan e9e1c99 onto cardiac-artifacts)
Config: reuse ScanParams cone-beam geometry (sid540/sdd950, det 320x800); tomosipo+astra
Dataset: ImageCAS-v1 test split (clean CCTA volumes + GT lumen masks)
Planned GPU-hours: ~0.5
Status: DONE — GATE PASSED (classical recon destroys coronary lumen at ultra-sparse)
Plan: quality_reports/plans/2026-07-03_sparse-view-ccta-reconstruction-pivot.md (Stage 0)

## Intent (falsifiable, gate for the whole pivot)

Confirm the ULTRA-SPARSE regime is genuinely hard for classical recon (so a diffusion prior
is warranted) and FIX the operating view count. Forward-project clean CCTA through the real
cone-beam operator (full 2pi), subsample to N views, reconstruct with FBP + SIRT, and measure
how badly the CORONARY LUMEN degrades vs full clean.
GATE: at the chosen N, FBP+SIRT visibly destroy coronary lumen (heart-MAE high, lumen-MAE high,
streaks) -> classical fails -> a prior has headroom. Pick the N where lumen is destroyed but
the body is still roughly located (the regime where a prior can plausibly help).

## Method

Cone-beam A_N (2pi, N evenly-spaced views) via tomosipo ts.operator. FBP = ramp-filter sinogram
+ backproject (HU-affine calibrated to clean). SIRT = 40-iter algebraic (C A^T R (y - A x)).
Sweep N in {12,16,24,32,48,64,96}. Metrics vs clean: heart-MAE, in-GT-lumen MAE, lumen sharpness.
Render clean | FBP@N | SIRT@N at a coronary slice.

## Result (n=2 cases; cone mag 1.5, det 400x400, SIRT 40it)

clean in-lumen sharpness ref = 181 HU/vox.

| views | FBP heartMAE | FBP lumMAE | SIRT heartMAE | SIRT lumMAE | SIRT lumSharp |
|---:|---:|---:|---:|---:|---:|
| 12 | 140.8 | 145.3 | 95.8 | 160.1 | 46.4 |
| 16 | 127.5 | 139.3 | 83.7 | 135.2 | 52.2 |
| 24 | 111.8 | 118.6 | 67.3 | 136.4 | 56.7 |
| 32 | 96.2 | 104.0 | 58.1 | 125.7 | 57.7 |
| 48 | 74.3 | 82.1 | 47.8 | 120.5 | 58.5 |
| 64 | 61.7 | 72.9 | 43.0 | 114.1 | 59.4 |
| 96 | 44.3 | 56.7 | 38.9 | 107.6 | 61.0 |

Visual (case 21 panel): FBP = severe sparse-view streaks (destroyed <=24 views); SIRT = streaks
suppressed but coronary lumen washed to a smooth blob even at 64 views. In-lumen sharpness
46-61 vs clean 181 (3-4x loss); lumen-MAE stays 107-160 HU across the whole sweep while heart-MAE
drops with views -> the LUMEN (thin, high-freq) is the hard part sparse-view cannot recover.

## Decision

GATE PASSED. The ultra-sparse regime genuinely destroys coronary lumen for classical recon ->
a diffusion prior has real headroom. Operating point: PRIMARY = 24 views (clearly ultra-sparse,
lumen destroyed, body located); run Stage-1 gate at {16, 24, 32} to bracket.
CAVEATS (carry to Stage 1): SIRT 40it under-converged; the real baseline to beat is ASD-POCS/TV
(edge-preserving); prior may RECOVER or HALLUCINATE the lumen -> that is the Stage-1 + trust-layer test.
NEXT (Stage 1): needs a CLEAN cardiac image prior p(x) for DPS (current diffusion is a residual
model, not usable) -> train an unconditional 3D cardiac EDM; build ASD-POCS baseline + DPS-recon
sampler; gate = DPS beats ASD-POCS on coronary Dice-RCA at the operating point.

## Cross-references

- Plan: quality_reports/plans/2026-07-03_sparse-view-ccta-reconstruction-pivot.md
- Reused geometry: code/inference/dps_forward.py (ScanParams cone builder)
- Code: scripts/python/sparse_view_sim.py
