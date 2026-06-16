# Plan — Path (a): Downstream-task-aware generative coronary-CTA motion correction

Status: ACTIVE (supersedes 2026-06-16_coronary-lumen-winning-result.md as the project spine)
Date: 2026-06-16
Data constraint (the organizing principle): ONLY ImageCAS (1000 cases, public,
with coronary GT) + physics-based synthetic motion (owned forward operator). NO
real paired clinical motion data. Therefore every paper claim must be fully
demonstrable on ImageCAS — which rules Path B (DPS, whose win needs real/OOD data
we lack) out as the spine, and selects Path (a).

## Decision (why this path)

Established this session (all verified): a strong supervised U-Net (38 HU MAE) is
near-optimal on distortion AND robust to OOD motion; a distortion-trained
diffusion (conditional latent v1; residual-EDM+gate, converged) TIES it on MAE and
on the gold-standard frozen-segmenter coronary Dice-RCA (sample 0.619 < U-Net
0.656; mean 0.654 ties). BUT the gold-standard revealed a LARGE uncaptured prize:
U-Net restores coronary segmentability only to 0.656 vs clean 1.000 — a 0.344 gap
that MAE hid entirely. The only mechanism to capture it is to stop optimizing
distortion and train a GENERATIVE model for the downstream/perceptual objective
(Blau-Michaeli). Generation stays the contribution; the U-Net is an initializer.

## Paper spine — "Downstream-task-aware generative motion correction" (3 legs)

- C1 (method; novelty #1 evolved): a task-aware GENERATIVE refiner that recovers
  coronary lumen segmentability the strong supervised baseline misses. Pipeline:
  corrupted -> frozen U-Net (initializer) -> task-aware generative refiner -> x_final.
  Win measured on Dice-RCA. Generation (flow/diffusion + downstream/perceptual
  loss) is the source of the win, not the U-Net.
- C2 (evaluation; novelty #3, result-independent): the downstream-task-aware
  evaluation protocol (frozen-segmenter coronary Dice-RCA + lumen-region fidelity/
  sharpness/CNR) AND the finding that distortion metrics (MAE) massively understate
  downstream degradation (U-Net ~90% on MAE = only 0.656 Dice-RCA vs clean).
- C3 (uncertainty; novelty #2): calibrated posterior uncertainty from the
  generative model (uncertainty-error corr 0.42 measured; reliability/ECE/coverage
  to formalize).

Retired-to-ablation: conditional latent diffusion v1, residual-EDM + reliability
gate (shown not to beat U-Net) — kept as ablations + UQ source, not the method.

Venue: MICCAI (natural fit for rigorous-synthetic + downstream-task-aware). CVPR/
AAAI a stretch without real-data or a flashier result.

## DONE (assets + findings)

- ImageCAS preprocessed 192³ + synthetic motion pairs + heart masks.
- Aligned coronary lumen GT (192³), verified bit-exact (`generate_lumen_masks.py`).
- Frozen VAE v2; strong U-Net baseline (38 HU, robust); latent diffusion v1;
  converged residual diffusion (`longrun/epoch_080.pt`) + reliability gate.
- Lumen / sharpness / CNR metrics (`metrics.py`, `artifact_metrics.py`); coherent
  single-sample sliding-window; `--save-volumes-dir`.
- Gold-standard Dice-RCA harness (`coronary_dice_rca.py`, TS coronary_arteries
  licensed) — de-risked + pilot (n=5) run.
- OOD motion probe (`ood_motion_probe.py` + `motion_synth` dvf_fn + random DVF):
  U-Net is robust -> not a sim-memorizer.
- Findings: distortion-diffusion ties U-Net (MAE + Dice-RCA); large downstream
  prize (0.656->1.0); UQ corr 0.42; distortion-perception framing.

## TO-DO

### C1 — the method (critical path; paper strength hinges here)
- [ ] Generate lumen GT for train(800)+val(100) (`generate_lumen_masks.py --split train/val`).
- [ ] Differentiable task loss (avoid non-differentiable TS): lumen-aware soft-Dice
      of a differentiable contrast/vesselness response inside GT lumen + edge term
      + small L1 fidelity leash; optional 3D PatchGAN (reuse `train_vae_v2.py`).
- [ ] Choose generative form: PMRF (rectified flow on U-Net posterior mean) vs
      adversarial + lumen-aware diffusion/refiner. (Decide in Step-2 plan.)
- [ ] Step-2 PILOT (test5) with PRE-SET bar: Dice-RCA(refiner) - Dice-RCA(unet)
      >= +0.03, MAE not materially worse, no hallucination -> KILL or scale.
- [ ] If pass: full train + test100 eval (Dice-RCA + MAE + lumen fidelity).
- [ ] Hallucination guardrails: held-out segmenter (coronary_arteries_LEGACY)
      anti-cheat; UQ-based hallucination flag; report distortion-perception Pareto.

### C2 — evaluation protocol (mostly done; scale + harden)
- [ ] Add per-case incremental save to `evaluate_residual_diffusion_full_volume.py`
      (the test100 lumen eval was lost at 53/100 — never again).
- [ ] Dice-RCA on full test100 (pilot was n=5) for U-Net + refiner + ablations.
- [ ] Formalize "MAE hides downstream" with test100 numbers; lumen fidelity table.

### C3 — uncertainty (partial; formalize)
- [ ] UQ calibration on the converged posterior: reliability diagram, ECE,
      coverage, uncertainty-error corr on test100 (adapt the gate calibration script).
- [ ] Tie UQ to the refiner (does sigma flag unreliable / potential-hallucination regions).

### Cross-cutting / manuscript
- [ ] Unified ablation table: corrupted / U-Net / latent-diff-v1 / residual-diff+gate
      / task-aware refiner x {MAE, Dice-RCA, lumen fidelity, UQ}.
- [ ] OOD robustness section (have the probe; optionally extend).
- [ ] Real-data qualitative demo (FOR/LIRS, no-reference) IF any real motion CCTA
      obtainable — PENDING user's data answer.
- [ ] Figures: pipeline; main downstream result; qualitative lumen; UQ calibration; OOD.
- [ ] MICCAI manuscript draft; update decision records to the evolved novelty framing.

### Housekeeping
- [ ] Commit the checkpoint (large body of verified uncommitted work).
- [ ] Update CLAUDE.md state table + MEMORY.md with the re-examination conclusions.

## Critical path / risk

The entire paper strength hinges on the C1 Step-2 pilot: does the task-aware
generative refiner actually capture a meaningful slice of the 0.344 Dice-RCA gap
without hallucinating? If YES -> strong-result paper (C1 headline + C2 + C3). If
TIE -> C2 (eval protocol + MAE-hides-downstream) + C3 (UQ) + the distortion-
perception characterization still carry a solid methodological paper. Either way
the project is publishable; the pilot decides the ceiling.

## Immediate next steps (order)

1. Commit checkpoint.
2. Write the Step-2 method plan (generative form + differentiable task loss + bar).
3. Generate train/val lumen GT.
4. Build loss + refiner; run Step-2 pilot (test5) against the bar.
5. (then) scale C2/C3 on test100; assemble ablations + figures; draft.
