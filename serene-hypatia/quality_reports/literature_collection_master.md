# Master Reference List for TRUST Paper

**Date:** 2026-02-15
**Sources mined:** G2LCPS (Ren 2024), Tang 2025, CASEMark (Huang 2025), Torosdagli 2023, H3DE-Net (Huang 2025)
**Already in bib:** 14 entries (marked with ✅)

---

## Proposed Related Work Structure

Our Related Work needs 4-5 subsections aligned with our contributions:

1. **TAVI Planning and Aortic Root Landmarking** (Gap 1: Clinical)
2. **Anatomical Landmark Detection** (supervised methods — context)
3. **Semi-Supervised Learning in Medical Imaging** (Gap 2: SSL)
4. **Graph/Topology-Based Structural Reasoning** (Gap 3: Architecture)
5. **Uncertainty Estimation in Medical Imaging** (Gap 4: Reliability)

---

## Category 1: TAVI Planning and Aortic Root Landmarking

These are our direct domain references. Most come from our existing knowledge + Torosdagli.

| Priority | Authors | Year | Key Contribution | Status |
|----------|---------|------|------------------|--------|
| **HIGH** | Ma, Lemarchand, et al. | 2023 | Two-stage CNN for 8 aortic valve landmarks, MRE 2.23mm | ✅ In bib |
| **HIGH** | Lalys et al. | 2018 | Automatic aortic root segmentation + landmarks for TAVI | ❌ Need to add |
| **HIGH** | Astudillo et al. | 2023 | (Coronary ostia detection — need to verify from master_supporting_docs) | ❌ Need to verify |
| MED | (TAVI clinical guidelines / ACC/AHA) | — | Clinical context for coronary obstruction risk thresholds | ❌ Need clinical refs |

---

## Category 2: Anatomical Landmark Detection (Supervised)

### 2a. Heatmap-based methods

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| **HIGH** | Payer, Stern, Bischof, Urschler | 2019 | SCN: spatial configuration + heatmap regression | G2LCPS, CASEMark, Tang, H3DE | ✅ In bib |
| **HIGH** | Sun et al. (HRNet) | 2019 | High-resolution representation learning for pose estimation | G2LCPS, H3DE | ❌ Need to add |
| MED | Newell et al. | 2016 | Stacked Hourglass multi-stage architecture | G2LCPS | ❌ Need to add |
| MED | Ao & Wu | 2023 | FARNet: feature aggregation + refinement for anatomical landmarks | G2LCPS, CASEMark | ❌ Need to add |
| LOW | Wei et al. | 2016 | Convolutional Pose Machines | G2LCPS | ❌ Optional |
| LOW | Cai et al. | 2020 | Delicate local representations for multi-person pose | G2LCPS | ❌ Optional |

### 2b. Regression-based methods

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| MED | Toshev & Szegedy | 2014 | DeepPose: cascaded DNN regressor for pose estimation | G2LCPS, Tang | ❌ Need to add |
| MED | Luvizon et al. | 2019 | Soft-argmax for end-to-end joint coordinate regression | G2LCPS | ❌ Need to add |
| LOW | Li et al. (RLE) | 2021 | Regression achieves heatmap parity via reparameterized likelihood | G2LCPS | ❌ Optional |

### 2c. Coarse-to-fine / Two-stage methods

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| **HIGH** | Zheng et al. | 2015 | 3D deep learning for landmark detection (volumetric, two-step) | Torosdagli | ❌ Need to add |
| MED | Chen et al. | 2021 | Fast 3D landmark detection via 3D Faster R-CNN (coarse-to-fine) | Torosdagli | ❌ Need to add |
| MED | Ghesu et al. | 2017 | Robust multi-scale anatomical landmark detection in incomplete 3D-CT | Torosdagli | ❌ Need to add |

### 2d. 3D medical landmark detection (general)

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| MED | Zhang et al. | 2020 | Context-guided FCN for joint CMF segmentation + landmarking | Tang, Torosdagli | ❌ Need to add |
| LOW | Nishimoto et al. | 2023 | Multi-phased regression for 3D craniofacial landmarks in CT slices | Tang | ❌ Optional |

---

## Category 3: Semi-Supervised Learning in Medical Imaging

### 3a. Consistency regularization / Teacher-Student

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| **HIGH** | Tarvainen & Valpola | 2017 | Mean Teacher: EMA consistency targets | G2LCPS, Tang | ✅ In bib |
| **HIGH** | Chen et al. | 2021 | CPS: Cross Pseudo Supervision for segmentation | G2LCPS | ❌ Need to add |
| **HIGH** | Sohn et al. | 2020 | FixMatch: consistency + confidence thresholding | G2LCPS | ✅ In bib |
| MED | Laine & Aila | 2016 | Temporal ensembling for SSL | Tang | ❌ Need to add |
| MED | Ouali et al. | 2020 | Cross-consistency training (feature perturbation) | G2LCPS | ❌ Need to add |
| MED | Liu et al. | 2022 | Perturbed and strict mean teachers for segmentation | G2LCPS | ❌ Need to add |
| LOW | Xu et al. (Soft Teacher) | 2021 | Soft teacher for semi-supervised object detection | Tang | ❌ Optional |

### 3b. Semi-supervised landmark detection

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| **HIGH** | Ren et al. | 2024 | G2LCPS: global-to-local CPS for landmark prediction | Direct | ✅ In bib |
| **HIGH** | Dong et al. (TS3) | 2019 | Teacher-student filtering for SSL facial landmarks | G2LCPS | ❌ Need to add |
| MED | Honari et al. | 2018 | SSL landmark detection with auxiliary tasks | G2LCPS | ❌ Need to add |
| MED | Moskvyak et al. | 2021 | SSL keypoint localization with semantic representations | G2LCPS | ❌ Need to add |
| MED | Springstein et al. | 2022 | SSL human pose estimation in art-historical images | Tang | ❌ Optional |
| MED | Seoungyoon et al. (HybridMatch) | 2023 | SSL facial landmarks via hybrid heatmap | G2LCPS | ❌ Need to add |

---

## Category 4: Graph / Topology-Based Structural Reasoning

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| **HIGH** | Kipf & Welling | 2017 | GCN: Semi-supervised classification with graph convolutions | Direct | ✅ In bib |
| **HIGH** | Li et al. | 2020 | Structured landmark detection via topology-adapting deep graph learning | Direct | ✅ In bib |
| **HIGH** | Scarselli et al. | 2009 | The GNN model | Torosdagli | ❌ Need to add |
| **HIGH** | Lang et al. | 2020 | Local attention-based GCN for CMF landmark localization | Torosdagli | ❌ Need to add |
| MED | Battaglia et al. | 2018 | Relational inductive biases, deep learning, and graph networks | Torosdagli | ❌ Need to add |
| MED | Santoro et al. | 2017 | Simple neural network module for relational reasoning | Torosdagli | ❌ Need to add |
| MED | Clough et al. | 2020 | Topological loss for deep learning segmentation | Direct | ✅ In bib |
| LOW | Li et al. | 2015 | Gated graph sequence neural networks | Torosdagli | ❌ Optional |

---

## Category 5: Uncertainty Estimation in Medical Imaging

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| **HIGH** | Gal & Ghahramani | 2016 | Dropout as Bayesian approximation (MC Dropout) | Direct | ✅ In bib |
| **HIGH** | Yu et al. | 2019 | Uncertainty-aware self-ensembling for 3D segmentation (SSL + uncertainty) | Tang | ✅ In bib |
| **HIGH** | Wang et al. (U2PL) | 2022 | SSL segmentation using unreliable pseudo-labels with uncertainty | G2LCPS | ❌ Need to add |
| MED | Kendall & Gal | 2017 | What uncertainties do we need in Bayesian deep learning? (aleatoric vs epistemic) | Well-known | ❌ Need to add |
| MED | Lakshminarayanan et al. | 2017 | Deep ensembles for predictive uncertainty estimation | Well-known | ❌ Need to add |

---

## Category 6: Deep Learning Backbones (cited as needed)

| Priority | Authors | Year | Key Contribution | Source | Status |
|----------|---------|------|------------------|--------|--------|
| HIGH | Ronneberger et al. | 2015 | U-Net | Direct | ✅ In bib |
| HIGH | He et al. | 2016 | ResNet | Direct | ✅ In bib |
| HIGH | Wang et al. | 2025 | Building vision models upon heat conduction | Direct | ✅ In bib |
| MED | Cicek et al. | 2016 | 3D U-Net | H3DE | ❌ Need to add |
| MED | Milletari et al. | 2016 | V-Net | Tang, H3DE | ❌ Need to add |

---

## Summary: References to Add

### HIGH priority (must-have for paper): ~15 new entries

1. **Lalys et al., 2018** — TAVI aortic root segmentation + landmarks
2. **Chen et al., 2021 (CPS)** — Cross Pseudo Supervision (core CPS paper)
3. **Laine & Aila, 2016** — Temporal ensembling
4. **Sun et al., 2019 (HRNet)** — High-resolution representation learning
5. **Zheng et al., 2015** — 3D deep learning for landmark detection
6. **Dong et al. (TS3), 2019** — Teacher-student SSL facial landmarks
7. **Scarselli et al., 2009** — Graph Neural Network model
8. **Lang et al., 2020** — Attention-based GCN for landmark localization
9. **Battaglia et al., 2018** — Relational inductive biases / graph networks
10. **Wang et al. (U2PL), 2022** — SSL segmentation with unreliable pseudo-labels
11. **Kendall & Gal, 2017** — Aleatoric vs epistemic uncertainty
12. **Seoungyoon et al. (HybridMatch), 2023** — SSL facial landmarks
13. **Honari et al., 2018** — SSL landmarks with auxiliary tasks
14. **Ouali et al., 2020** — Cross-consistency training

### MEDIUM priority (strongly recommended): ~10-12 entries

15. **Newell et al., 2016** — Stacked Hourglass
16. **Toshev & Szegedy, 2014** — DeepPose
17. **Luvizon et al., 2019** — Soft-argmax regression
18. **Ao & Wu, 2023** — FARNet
19. **Chen et al., 2021 (CMF)** — 3D Faster R-CNN for CMF landmarks
20. **Ghesu et al., 2017** — Multi-scale 3D landmark detection
21. **Zhang et al., 2020** — Context-guided FCN for joint segmentation + landmarks
22. **Liu et al., 2022** — Perturbed Mean Teachers
23. **Moskvyak et al., 2021** — SSL keypoint localization
24. **Santoro et al., 2017** — Relational reasoning module
25. **Lakshminarayanan et al., 2017** — Deep ensembles
26. **Milletari et al., 2016** — V-Net

### LOW priority (optional, if space permits): ~5-8 entries

- Wei 2016, Cai 2020, Li 2021 (RLE), Nishimoto 2023, Springstein 2022, Li 2015 (Gated GNN), Xu 2021 (Soft Teacher), Cicek 2016

---

## Notes

- **Gap not yet filled:** We still need 1-2 clinical TAVI references for coronary obstruction risk thresholds (e.g., Ribeiro et al. or Dvir et al. on coronary obstruction). These don't appear in the mined papers because they are domain-specific.
- **Astudillo et al.** needs verification from master_supporting_docs — is it in our collection?
- The **CPS paper (Chen et al., 2021)** is critically important — it's the methodological foundation of our Stage 2.
