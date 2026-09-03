# Research Specification: TRUST

**Date:** 2026-02-12
**Researcher:** [Author, Universite de Rennes]
**Interviewer:** Claude Code (interview-me skill)

---

## Research Question

How can we build a unified, uncertainty-aware semi-supervised framework that accurately detects all 10 clinically relevant aortic root landmarks from 3D CT under label scarcity, enabling automated coronary safety assessment for TAVI planning?

## Paper Framing

**Narrative type:** "Clinical problem solved with smart engineering" — but the weight falls on *"we had to invent this method to solve this clinical gap."*

The clinical problem (coronary safety) is the **motivation**; the TRUST framework (unified uncertainty-driven SSL) is the **contribution**. Neither stands alone — the method is justified by the clinical need, and the clinical value is enabled by the method.

## Four Gaps (Introduction Framing)

### Gap 1: Clinical — No unified 10-landmark system targeting coronary safety

Prior studies have detected coronary ostia for TAVI measurements (e.g., Astudillo et al.). However, most pipelines focus on sizing-centric landmarks and do not present a unified 10-landmark system that (i) explicitly targets coronary safety metrics (coronary height) as a first-class output and (ii) is designed for low-label settings with reliability controls.

**Positioning:** NOT "the first to ever detect ostia." Rather: an end-to-end planning-oriented landmark set with coronary safety evaluation, trained under label scarcity.

### Gap 2: Methodological — SSL + anatomical constraints untailored for TAVI landmarks

Semi-supervised landmark detection exists (e.g., shape-regulated self-training), and spatial-configuration priors have been explored in supervised settings. The gap: these ideas have not been tailored to structured aortic-root/TAVI landmarking with explicit anatomical constraints and failure modes (calcification, low contrast). SSL successes in segmentation do not directly transfer to landmark tasks where "one wrong point" breaks downstream clinical measurements.

### Gap 3: Architectural (Main Technical Novelty) — No unified uncertainty control signal

While graph/shape-based landmark refinement is known (topology-adapting graph learning, learnable skeleton GCNs), and uncertainty/SSL heuristics exist separately, no prior method uses uncertainty as a **single unified control signal** that simultaneously: (i) conditions feature extraction via uncertainty-modulated diffusivity (UG-HCO), (ii) filters pseudo-supervision in the consistency loss, and (iii) gates topological refinement in GCN message passing. TRUST estimates uncertainty once and reuses it across all modules — a coherent system rather than a collection of add-ons.

### Gap 4: Reliability — No systematic prevention of anatomically impossible configurations

Prior works report average landmark error but do not systematically prevent or quantify anatomically impossible configurations under calcification/low-contrast, nor provide a reliability signal for clinical triage. TRUST addresses this with uncertainty-aware learning + topology constraints, evaluated via hard-case stratification and uncertainty-error correlation analysis.

**Framing level for Gap 4:** Option (c) — show uncertainty-error correlation plot (proving the signal is meaningful) but stop short of a full clinical triage study. Frame as: *"TRUST provides per-landmark uncertainty estimates that correlate with prediction error, offering a potential pathway for clinical triage."*

## The TRUST Framework

### Architecture Overview

**TRUST** = Topological Reasoning and Uncertainty-aware Semi-supervised Teaching

Three components, unified by a single uncertainty signal:

1. **Mean Teacher (Semi-supervised backbone):** Teacher-Student with EMA. Teacher produces pseudo-labels for 600 unlabeled volumes. Uncertainty $u_k$ estimated via stochastic teacher predictions (TTA or MC Dropout).

2. **UG-HCO (Core novelty):** Uncertainty-Guided Heat Conduction Operator. Diffusivity $k(\omega, \mathcal{U})$ is modulated by epistemic uncertainty — global context flows more aggressively into low-confidence regions (calcified areas, ambiguous P7) and less into confident regions. Bridges appearance learning and structure-aware refinement.

3. **GCN Refinement (Topological reasoning):** Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ over 10 landmarks. Enforces anatomically plausible spatial relationships. Uses accurate landmarks (P0-P2, P6) as anchors to constrain drifting landmarks (P7).

### Unified Uncertainty Signal

One source (stochastic teacher) → one set of per-landmark scores $u_k$ → three consumers:
- **UG-HCO:** Modulates diffusivity — more context injection in uncertain regions
- **Consistency loss:** $w_k = \exp(-\alpha \cdot u_k)$ dampens unreliable pseudo-labels
- **GCN message passing:** Same $w_k$ prevents unreliable nodes from corrupting neighbors

### Key Technical Claim

If a reviewer asks "what is the single most novel element?": **UG-HCO** — an uncertainty-conditioned heat conduction operator where diffusivity is modulated by epistemic uncertainty, adaptively injecting global context specifically in low-confidence regions.

## The 10 Landmarks

| ID | Name | Clinical Use | Detection Difficulty |
|----|------|-------------|---------------------|
| P0-P2 | Hinge Points (NCC, LCC, RCC) | Annular plane → sizing | Low-moderate |
| P3-P5 | Commissures (N-L, L-R, R-N) | Rotational alignment | Low-moderate |
| P6 | Center | Valve area, catheter path | Low (1.93mm) |
| P7 | MS Point | MSL → pacemaker risk (NOCD) | **Hardest** (3.46mm; calcification) |
| P8 | Left Coronary Ostium | Coronary height → obstruction risk | Moderate (2.26mm) |
| P9 | Right Coronary Ostium | Coronary height → obstruction risk | Low (1.72mm) |

**Clinical derived parameters:**
- Coronary Height: H = Dist(P8/P9, Annular Plane from P0-P2). <10mm = high risk, <12mm = caution.
- MSL: Dist(P7, Annular Plane). Short MSL → pacemaker risk.

## Data

- **Labeled:** 150 CT volumes with expert-annotated 10-landmark coordinates
- **Unlabeled:** 600 CT volumes (same clinical pipeline, no annotations)
- **Split/protocol:** To be finalized. Preliminary MRE 2.42mm (10 points).
- **Baseline comparison:** Ma et al. (2023): 8 landmarks, fully supervised, MRE 2.23mm

## Planned Evaluation

### Metrics
- MRE (mm) — primary accuracy
- SDR@2/3/4mm — clinically relevant thresholds
- Per-landmark analysis (highlighting P7, P8/P9)
- Coronary height error + risk stratification (sensitivity/specificity)
- Uncertainty-error correlation plot

### Ablation (6 rows)
1. Supervised only (two-stage CNN)
2. + SSL (Mean Teacher)
3. + SSL + Topology/Geometry (GCN)
4. + SSL + Topology + HCO
5. + Uncertainty (SSL consistency weighting only)
6. Full TRUST (+ uncertainty in Topo gating)

### Hard-case stratification
- Heavy calcification
- Motion artifacts / poor image quality
- Anatomical variants (e.g., bicuspid aortic valve)

## Contribution Statement (Draft)

The contributions of this paper are:
1. A unified 10-landmark detection framework for TAVI planning that extends beyond valve sizing to provide automated coronary safety assessment (coronary height measurement) as a first-class clinical output.
2. TRUST, a semi-supervised learning framework where a single epistemic uncertainty estimate simultaneously conditions feature extraction (UG-HCO), filters pseudo-labels, and gates topological refinement — forming a coherent system rather than independent heuristics.
3. UG-HCO, an uncertainty-conditioned heat conduction operator that adaptively propagates global context into low-confidence regions, addressing calcification-induced feature degradation.
4. Comprehensive evaluation including per-landmark analysis, hard-case stratification, and uncertainty-error correlation, demonstrating that the unified uncertainty signal improves robustness on the most clinically challenging landmarks.

## Open Questions

1. **Paper title:** Still under discussion. Current working title: "Uncertainty-Aware Physio-Topological Semi-Supervised Learning for TAVI Landmark Detection and Coronary Safety Assessment." May evolve.
2. **Experimental protocol:** Data split, cross-validation strategy, and final numbers not yet finalized. Focus on Introduction + Method first.
3. **GCN experiments:** Not yet run. Architecture is defined; training pending.
4. **Target venue:** IEEE TMI (primary), with MICCAI/MedIA as alternatives.
5. **Prior art nuance:** Need to verify exact claims about Astudillo et al. and other ostia-detecting methods to ensure Gap 1 framing is accurate.

## Interview Insights

- The paper must avoid "first ever" claims — prior work detects ostia, does SSL for landmarks, uses GCN for refinement. TRUST's novelty is the **unified uncertainty-driven integration**.
- P7 difficulty is a **shared challenge** (Ma et al. also at 3.38mm), not a TRUST-specific failure. Frame as anatomical, not methodological.
- Uncertainty-error correlation (option c) is the right level of reliability evidence — supportive but not overreaching.
- The "one wrong point breaks everything" argument is powerful for explaining why SSL for landmarks differs from SSL for segmentation.
