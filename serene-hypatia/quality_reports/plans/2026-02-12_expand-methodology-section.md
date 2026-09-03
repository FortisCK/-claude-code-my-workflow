# Plan: Expand Methodology Section of TRUST Paper

**Status:** DRAFT
**Date:** 2026-02-12

## Context

The Methodology section of `paper/main.tex` currently has 3.1 Problem Formulation and 3.2 Semi-Supervised Setting (both complete), plus 3.3 Framework Overview (3 high-level paragraphs, no detailed math). Need to expand into full formal subsections with equations for all three TRUST components, loss functions, and training procedure.

## Approach

Keep 3.1 and 3.2 unchanged. Replace/expand 3.3 into five subsections:

| § | Title | Key Equations |
|---|-------|---------------|
| 3.3 | Framework Overview | Thesis statement: one uncertainty, three consumers |
| 3.4 | Mean Teacher + Uncertainty Estimation | EMA, TTA variance $u_k$, reliability weight $w_k$ |
| 3.5 | UG-HCO (core novelty) | Heat equation → DCT solution → uncertainty-conditioned $k(\omega, \mathcal{U})$ |
| 3.6 | GCN Refinement | Graph construction, uncertainty-gated message passing, residual correction |
| 3.7 | Loss Functions + Training | $\mathcal{L}_{sup}$, $\mathcal{L}_{con}$, $\mathcal{L}_{topo}$, $\mathcal{L}_{total}$, ramp-up, training algorithm |

## Design Decisions

- TTA variance for uncertainty (not MC Dropout)
- Simple multiplicative diffusivity modulation for UG-HCO
- Fixed anatomical graph topology for GCN
- Coordinate refinement loss only for $\mathcal{L}_{topo}$

## Files

- `paper/main.tex` — expand lines 69-120
- `Bibliography_base.bib` — add `wang2025building`, `Kipf2017_gcn`
- `CLAUDE.md` — update paper state

## Verification

- pdflatex 3-pass + bibtex compiles cleanly
- No undefined citations
- Proofreader agent on Method section
