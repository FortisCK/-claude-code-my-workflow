---
paths:
  - "paper/**/*.tex"
  - "figs/**/*"
---

# Medical Imaging Domain Knowledge: TRUST

## TAVI Procedure Context

Transcatheter aortic valve implantation (TAVI) is a minimally invasive procedure to replace a stenotic aortic valve. Pre-procedural planning from CT angiography requires accurate identification of aortic root landmarks to:
- Size the prosthetic valve
- Assess coronary ostia height (risk of coronary obstruction)
- Plan the C-arm angulation for deployment

## The 10 Aortic Root Landmarks

| ID | Name | Anatomical Location | Clinical Use |
|----|------|---------------------|-------------|
| P0 | NCC Hinge | Nadir of non-coronary cusp | Annular plane (sizing) |
| P1 | RCC Hinge | Nadir of right coronary cusp | Annular plane (sizing) |
| P2 | LCC Hinge | Nadir of left coronary cusp | Annular plane (sizing) |
| P3 | N-R Commissure | NCC/RCC junction | Cusp-wedge sector boundary |
| P4 | R-L Commissure | RCC/LCC junction | Cusp-wedge sector boundary |
| P5 | L-N Commissure | LCC/NCC junction | Cusp-wedge sector boundary |
| P6 | Center | Geometric center of valve coaptation | Valve area, catheter path |
| P7 | MS Point | Lower part of membranous septum (RCC/NCC junction, deep) | MSL measurement, pacemaker risk (NOCD) |
| P8 | LCO | Left coronary ostium (LCC sinus wall) | Coronary height, obstruction risk |
| P9 | RCO | Right coronary ostium (RCC sinus wall) | Coronary height, obstruction risk |

**Key clinical derived parameters:**
- **Coronary Height:** H = Dist(P8 or P9, Annular Plane fitted by P0-P2). Threshold: <12mm flagged risk per SCCT 2019 consensus (Blanke et al.); some older literature uses <10mm.
- **MSL (Membranous Septum Length):** Dist(P7, Annular Plane). Predicts pacemaker implantation risk.
- **P7 is the hardest landmark** to detect (3.38mm in Ma et al., 3.46mm in current model) due to severe calcification and low CT contrast at the membranous septum.

## CT Imaging Conventions

- **Modality:** CT angiography (CTA) of the aortic root
- **Voxel spacing:** typically 0.3-0.8 mm in-plane, 0.5-1.0 mm slice thickness
- **Hounsfield units:** contrast-enhanced aorta typically 200-500 HU
- **Orientation:** standard radiological convention (patient's left on viewer's right)
- **Field of view:** cardiac/thoracic, cropped to aortic root region for processing

## Standard Evaluation Metrics

| Metric | Definition | Use |
|--------|-----------|-----|
| MRE (mm) | Mean Radial Error: average Euclidean distance from predicted to ground truth landmark | Primary accuracy metric |
| SDR@r (%) | Success Detection Rate: percentage of predictions within r mm of ground truth | Clinically relevant threshold metric |
| SDR@2mm | Clinical precision threshold | Strict accuracy |
| SDR@3mm | Common benchmark threshold | Standard comparison |
| SDR@4mm | Relaxed threshold | Broader context |

## Semi-Supervised Learning Context

- **Labeled data:** CT scans with expert-annotated landmark coordinates (expensive to acquire)
- **Unlabeled data:** CT scans without annotations (abundant in clinical databases)
- **Mean Teacher:** exponential moving average of student weights; produces stable pseudo-labels
- **Consistency regularization:** enforce prediction agreement between student and teacher under perturbations
- **Pseudo-labels:** teacher predictions used as supervision for unlabeled data
- **Uncertainty estimation:** quantify prediction confidence to filter noisy pseudo-labels

## TRUST Method Components

| Component | Full Name | Purpose |
|-----------|-----------|---------|
| UG-HCO | Uncertainty-Guided Heat Conduction for physio-perception | Uncertainty-conditioned diffusion for global-context propagation under calcification/low-contrast |
| GCN Refinement | Topological Reasoning via Graph Refinement | Enforce anatomically plausible spatial relationships via landmark graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ |
| Mean Teacher | Semi-supervised teaching with EMA | Consistency regularization on unlabeled data with uncertainty-weighted pseudo-labels |

## Key Related Work

When citing or comparing, use the correct attribution:
- **Mean Teacher:** Tarvainen & Valpola (2017)
- Verify all method attributions against `Bibliography_base.bib`
- Distinguish clearly between our contributions and prior work
