# Decision Record — Research direction (V2, 2026-04-29)

**Status:** ACTIVE — focus 层
**Companion document:** [`2026-04-29_research-direction-v2.html`](2026-04-29_research-direction-v2.html)
  *(the full strategic brief, Chinese narrative + English terms, ~1300 lines, was originally `simple_plan.html` at the repo root)*
**Predecessor (landscape 底盘):** [`2026-04-29_research-direction-v1.md`](2026-04-29_research-direction-v1.md) → [`v1.pdf`](2026-04-29_research-direction-v1.pdf) — V1 是综合调研报告,V2 是其上的 focus 层。两者互补不替代,一起构成完整决策档案。
**Author:** CMZ
**Reviewers:** internal (LTSI / Pascal & Carlos pending)

---

## TL;DR

A 3D conditional latent-diffusion model for **single-phase cardiac CT
motion-artifact correction**, with native posterior-sampling uncertainty
and TAVI-relevant downstream evaluation. Target venues: MICCAI 2027 (primary),
CVPR 2027 (parallel), IEEE TMI (journal-extension fallback). First-submission
horizon: 5–7 months from 2026-04-29.

## What changed from V1 (the headline diffs)

The V2 brief was written after full-text reads of the three core papers
(TT U-Net, HM-EDM, ProDM) plus C2F-MC (the cardiac-MRI Path-B reference). Key
revisions vs V1:

1. **HM-EDM downgraded** from "primary threat" to "workshop-quality
   proof-of-concept, baseline template". Brain SSIM 0.17 → 0.51 on n=100;
   5-case reader study; 100 cases of data.
2. **ProDM scope-cut more precise.** They do non-contrast COCA Agatston scoring;
   their motion engine moves only calcium points. Method does not transfer
   to contrast CCTA + lumen/stenosis.
3. **C2F-MC added** as Path-B-in-cardiac (MRI) reference. Linear k-space
   forward operator handle; CT image-domain has no equivalent — graceful
   scope-cut paragraph in the manuscript intro.
4. **TT U-Net real numbers** captured: SSIM-Local 0.84, Dice-RCA 0.81;
   without temporal information drops to 73–81% / 73–76%. This is the
   single-frame baseline floor.
5. **Methodological borrowing made concrete.** ProDM's task-driven differentiable
   loss → our lumen-aware loss; HM-EDM's patch-wise 3D-EDM template → directly
   reproducible on our A6000.
6. **Risk matrix re-balanced.** Scooping risk slightly down (HM-EDM weaker
   than expected, ProDM scope-disjoint); sim-to-real gap up (a domain-wide
   limitation, not unique to us).
7. **Speed window tightened** to 4–5 months (MGH and Northwestern groups
   both plausibly working on cardiac extensions).

## Three orthogonal novelty claims (each independently defensible)

1. **gated CCTA + conditional latent LDM + lumen-aware loss** — methodological.
2. **Theoretically grounded posterior-sampling uncertainty** — probabilistic.
3. **Downstream-task-aware evaluation** (stenosis grading, TAVI landmark accuracy)
   — evaluation-methodological.

The reasoning for the three-claim structure is in §07 of the HTML brief: a
single novelty claim is fragile against one skeptical reviewer; three
orthogonal claims create redundancy that survives partial rejection.

## Key constraints (decisions locked in)

- **Compute:** single A6000, 48 GB. Latent-space diffusion is the default; pixel-space cropped 128×128×96 ROI is the explicit fallback if VAE fidelity fails.
- **Data:** ImageCAS (1000 contrast CCTA, Apache-2.0, Kaggle) + PAD motion-simulation pipeline (Deng et al., open) + XCAT phantom (license pending; Pascal/Carlos ask first).
- **Stack:** Python ≥ 3.10, PyTorch, MONAI `GenerativeModels`. No Apex.
- **Working language:** English in `.claude/` and shipped artefacts; 中文 + English terms in user-authored internal artefacts.
- **Path B exclusion:** k-space-style posterior sampling on raw projections is out of scope (no measurement-domain handle in the CT image-domain dataset class we have). Image-domain pseudo-forward becomes an ablation subsection at most.

## Timeline (from §09 of the brief)

| Month | Milestone |
| ----- | --------- |
| 2026-04 | (now) Lit anchors locked; data plan defined; this configuration adaptation. |
| 2026-05 | 3D KL-VAE pretrain on ImageCAS; HU-fidelity sanity check (go/no-go gate). |
| 2026-06–07 | Conditional latent-diffusion training; condition-injection ablations. |
| 2026-08 | Posterior sampling pipeline; UQ calibration vs MC-dropout / TTA. |
| 2026-09 | Downstream task evaluation (Dice, stenosis, TAVI landmark — killer figure). |
| 2026-10 | Baseline reproductions (TT U-Net, HM-EDM); manuscript drafting. |
| 2026-11 | Polish + rebuttal materials; MICCAI 2027 submission (Jan deadline); CVPR 2027 (Nov deadline) if pacing allows. |

## Active risks (from §08 of the brief)

| Risk                                  | Severity   | Fallback                                                                                       |
| ------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------- |
| VAE reconstruction infidelity         | High       | Retreat to pixel-space cropped ROI on cardiac mask                                             |
| HM-EDM "generalisation" framing       | Low–Medium | Reproduce HM-EDM cardiac-side; emphasise rigid-vs-non-rigid + UQ + downstream + lumen-aware    |
| PAD pipeline reproducibility (MATLAB) | Medium     | Simpler DVF-warp + forward-projection; or contact Deng lab                                     |
| Scooping (MGH / Northwestern)         | Medium     | Speed: 4–5 month arXiv occupy; UQ + TAVI-landmark legs survive even partial scoop              |
| Synth-to-real generalisation          | Low        | Domain-wide limitation; ProDM reader study sat at Likert ~3 too — Likert ~3 is publishable     |

## How this decision feeds the workflow

- Plans in `quality_reports/plans/` reference this document for project context.
- `MEMORY.md` carries `[LEARN]` entries that condense the lock-ins
  (latent-first-with-pixel-fallback; three-novelty-claim structure;
  C2F-MC scope cut; bilingual policy; run-card discipline).
- `Bibliography_base.bib` is seeded with the six anchor references called
  out in the brief.
- The `python-reviewer` and `experiments-protocol.md` rule encode the
  reproducibility expectations the brief implies.

## Amendment process

When the situation evolves enough that this V2 stops being load-bearing —
a paper gets scooped, a baseline lands surprisingly close, the VAE-fidelity
gate fails — author a V3 record alongside this one. Don't edit V2.
