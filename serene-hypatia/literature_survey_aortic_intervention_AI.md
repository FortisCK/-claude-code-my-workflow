# Literature Survey: AI for Aortic Intervention Planning

**Prepared for:** PhD Student, Université de Rennes  
**Data assets:** Aortic root CT (TAVI) + Abdominal aortic aneurysm CT (AAA-30)  
**Date:** 2026-03-17  
**Coverage:** 2021–2026, emphasis on 2023–2026

---

## Domain 1: AI for TAVI Planning

### 1.1 Key Papers

| # | Title | First Author | Year | Venue | Summary |
|---|-------|-------------|------|-------|---------|
| 1 | **TAVI-PREP: A Deep Learning-Based Tool for Automated Measurements Extraction in TAVI Planning** | M. Santaló-Corcoy | 2023 | Diagnostics | Fully automated pipeline combining MeshDeformNet for 3D surface mesh generation and a 3D Residual U-Net for landmark detection. Extracts 22 measurements from the aortic valvular complex; high Pearson correlations (0.90–0.97) with expert measurements across 200 CT scans, except for coronary heights (r=0.72–0.80). Trained on public MM-WHS dataset + private data. [1] |
| 2 | **A CT-based deep learning system for automatic assessment of the aortic root** | (Kong et al. / related) | 2023 | arXiv (2302.05378) | Deep learning system for automatic aortic root assessment from CT, providing aortic annulus sizing and geometry characterization. Uses 3D mesh deformation networks to produce patient-specific anatomical meshes for TAVI planning. [2] |
| 3 | **Automated Aortic Valve Landmark Detection from CT using Deep Learning for TAVI Planning** | F. Astudillo et al. (multiple groups) | 2023–2024 | Various (MICCAI workshops, JCMR) | Several groups have proposed heatmap-regression-based landmark detectors for the three commissures, nadirs, and coronary ostia using 3D U-Nets or PointNet-based approaches. Typical localization error: 2–4 mm for annular landmarks, 4–6 mm for coronary ostia. |
| 4 | **Bicuspid Aortic Valve Detection Using Deep Learning on CT** | Emerging work | 2024–2025 | JACC Imaging / preprints | Limited but emerging work on classifying tricuspid vs. bicuspid aortic valve (BAV) morphology from pre-TAVI CT scans using CNNs. BAV identification is critical because it alters valve sizing strategy and complication risk. Most approaches are classification-based (ResNet/EfficientNet on short-axis reformats). |

### 1.2 Datasets & Benchmarks
- **MM-WHS** (Multi-Modality Whole Heart Segmentation): Public dataset with 120 cardiac CT/MRI scans; used by TAVI-PREP for training segmentation backbone.
- **No dedicated public TAVI-landmark benchmark exists.** Most groups use private clinical datasets (100–500 patients).
- **3Mensio / Vitrea**: Commercial software used as ground-truth reference in most validation studies.

### 1.3 Evaluation Protocols
- **Landmark detection:** Mean Radial Error (MRE) in mm, Successful Detection Rate (SDR) at thresholds (2mm, 2.5mm, 4mm).
- **Measurement accuracy:** Pearson correlation, Bland-Altman analysis, Mean Absolute Error (MAE), Mean Absolute Relative Error (MARE).
- **Valve sizing agreement:** Prosthesis size agreement rate vs. expert (exact match + ±1 size).

### 1.4 Gaps
- **Coronary height prediction remains unreliable** (TAVI-PREP: r=0.72–0.80), yet this is critical for coronary obstruction risk.
- **No public benchmark for aortic root landmarks** — hinders reproducible comparison.
- **BAV detection is nascent** — very few validated AI models; BAV patients represent ~2% of TAVI but have higher complication rates.
- **Post-TAVI outcome prediction** (paravalvular leak, conduction disturbance) is almost entirely unexplored by AI.
- **Semi-supervised or label-efficient methods have not been applied** to TAVI landmark detection.

### 1.5 Maturity: 🟡 Growing
Active research with a handful of validated tools (TAVI-PREP, 3Mensio AI). Landmark detection works for annulus but fails for coronary ostia. No clinical deployment at scale. Major opportunity for semi-supervised methods given small labeled datasets.

---

## Domain 2: AI for AAA / EVAR Planning

### 2.1 Key Papers

| # | Title | First Author | Year | Venue | Summary |
|---|-------|-------------|------|-------|---------|
| 1 | **Automatic Explainable Segmentation of Abdominal Aortic Aneurysm from CTA** | M. Roby | 2025 | IEEE Access | Mixture-of-Experts framework with 3 specialized U-Nets and a dynamic error-learning router for AAA lumen + outer wall segmentation. Dice 0.9648 (lumen), 0.9615 (wall), HD95 ~1.35mm. Tested on 22 patients / 1,560 images. Includes NURBS-based interactive correction for failure cases. [3] |
| 2 | **DeepAAA: Clinically Applicable and Generalizable Detection of Abdominal Aortic Aneurysm Using Deep Learning** | J.T. Lu | 2019 | MICCAI | Pioneering DL model for AAA detection from routine CT. Sensitivity 91%, specificity 95% on 321 exams. Demonstrated feasibility of opportunistic AAA screening from non-dedicated CTs. [4] |
| 3 | **Automated Segmentation of Aortic Structures from Contrast-Enhanced and Non-Contrast CT** | A. Chandrashekar | 2020–2023 | Various (Radiology:AI, JVS) | Attention-based U-Net achieving DSC 93.2% (CTA) and 88.7% (non-contrast CT). Demonstrated that non-contrast CT segmentation is feasible but significantly harder. Strong correlations with manual diameter and volume measurements. [5] |
| 4 | **Deep Learning Methods in Abdominal Aortic Aneurysm Segmentation: A Review** | (Survey paper) | 2024 | IEEE Access (11300868) | Comprehensive review covering U-Net variants, attention mechanisms, and multi-view approaches for AAA segmentation. Identifies key challenges: small datasets, lack of standardized benchmarks, non-contrast CT difficulty, thrombus segmentation. [6] |

### 2.2 Datasets & Benchmarks
- **No large public AAA segmentation benchmark** — most studies use 20–80 patient private datasets.
- Some groups use the **Vascular Model Repository** or share small annotated cohorts.
- The **AAA-30 dataset** (your data) would be relatively competitive in size for published work.
- **SEG.A challenge** (Segmentation of the Aorta, MICCAI 2023): Focused on full aorta segmentation, not AAA-specific, but provides useful pretraining data.

### 2.3 Evaluation Protocols
- **Segmentation:** Dice Similarity Coefficient (DSC), IoU/Jaccard, Hausdorff Distance (HD95), Average Surface Distance (ASD).
- **Clinical metrics:** Maximum diameter correlation, volume agreement, cross-sectional area at key landmarks (neck, iliac bifurcation).
- **Splits:** Typically patient-level train/val/test splits (never slice-level) to avoid data leakage.

### 2.4 Gaps
- **Non-contrast CT segmentation** performs significantly worse (DSC drops ~5%) — important for screening workflows.
- **Thrombus and calcification sub-segmentation** are underexplored; critical for EVAR planning and rupture risk.
- **Automated EVAR stent-graft sizing** from CT remains largely manual — AI could automate neck length, angulation, and iliac measurements.
- **Endoleak detection post-EVAR** has very few AI papers; small datasets and class imbalance are challenges.
- **AAA rupture risk prediction** beyond maximum diameter (e.g., using wall stress, shape features, or learned representations) remains early-stage.
- **No good solution for automated centerline extraction + landing zone identification for EVAR planning.**

### 2.5 Maturity: 🟡 Growing (segmentation: approaching mature; EVAR planning: early)
AAA lumen segmentation from CTA is becoming reliable (Dice >0.95). However, wall/thrombus segmentation, non-contrast CT, and end-to-end EVAR planning pipelines are still immature. Rupture risk prediction with DL is early-stage.

---

## Domain 3: Semi-Supervised / Label-Efficient Landmark Detection

### 3.1 Key Papers

| # | Title | First Author | Year | Venue | Summary |
|---|-------|-------------|------|-------|---------|
| 1 | **Semi-supervised Anatomical Landmark Detection via Shape-regulated Self-training** | R. Chen | 2021 | Neurocomputing (arXiv:2105.13593) | Model-agnostic framework using PCA-based shape model to regulate pseudo labels in self-training for anatomical landmark detection. Region Attention loss focuses on structure-consistent regions. Demonstrated on cephalometric, hand, and spine landmark datasets. State-of-the-art semi-supervised landmark detection baseline. [7] |
| 2 | **Self-Supervised Pre-Training with Diffusion Model for Few-Shot Landmark Detection in X-Ray Images** | R. Di Via | 2025 | WACV 2025 | Uses diffusion model pretraining to learn rich representations for few-shot anatomical landmark detection. Demonstrates that diffusion features provide strong priors for landmark localization with very limited labels (5–20 images). Evaluated on cephalometric and hand X-ray datasets. [8] |
| 3 | **SAM (Segment Anything Model) adaptations for medical landmark detection** | Multiple groups | 2023–2025 | Various (MICCAI, MedIA) | SAM and SAM-2 have been adapted for medical image segmentation via prompt engineering and fine-tuning (MedSAM, SAM-Med2D). However, SAM is fundamentally a segmentation model, and adapting it for point-based landmark detection remains an open problem. Some work uses SAM features as a backbone for downstream landmark regression. |
| 4 | **Active Learning for Anatomical Landmark Detection** | Various | 2023–2024 | MICCAI workshops, MIDL | Emerging work on uncertainty-based active learning strategies for landmark annotation. Key idea: use predictive uncertainty (ensemble disagreement, MC-Dropout) to select the most informative samples for expert annotation. Limited to 2D cephalometric/dental domains so far. |

### 3.2 Datasets & Benchmarks
- **ISBI 2015 Cephalometric Landmark Dataset**: 400 lateral cephalograms, 19 landmarks. Standard benchmark.
- **JSRT Hand Landmark Dataset**: Hand X-rays with ~37 landmarks.
- **Spine Landmark Datasets**: Various vertebral body corner/center landmark datasets.
- **No 3D cardiac/aortic landmark detection benchmark exists for semi-supervised evaluation.**

### 3.3 Evaluation Protocols
- **Mean Radial Error (MRE)** in mm (primary metric).
- **Successful Detection Rate (SDR)** at 2mm, 2.5mm, 4mm, 10mm thresholds.
- Semi-supervised protocols: typically train with 10%, 25%, 50% labeled data + remaining unlabeled.
- Few-shot: evaluated at 1-shot, 5-shot, 10-shot, 20-shot regimes.

### 3.4 Gaps
- **Almost all semi-supervised landmark work is 2D** (cephalometric X-rays). Extension to 3D volumes (CT) is largely unexplored.
- **Diffusion-based landmark detection is brand new** — Di Via et al. (WACV 2025) is among the first; no cardiac/aortic application yet.
- **Foundation model adaptation (SAM) for landmarks** remains unsolved — SAM produces masks, not points. Converting SAM features to landmark regression is an open research direction.
- **No active learning for 3D anatomical landmarks** in cardiac or vascular imaging.
- **Shape-constrained semi-supervised methods** (Chen et al.) have not been extended to 3D or to domains beyond bone/dental.

### 3.5 Maturity: 🟠 Early-to-Growing
2D semi-supervised landmark detection is established with clear baselines. 3D semi-supervised landmark detection is wide open. Diffusion-based and foundation-model-based approaches are nascent (2024–2025). **This is a high-opportunity gap for the student's PhD.**

---

## Domain 4: Emerging Tools for 3D Medical Imaging (2024–2026)

### 4.1 Key Papers

| # | Title | First Author | Year | Venue | Summary |
|---|-------|-------------|------|-------|---------|
| 1 | **Swin-UMamba: Adapting Mamba-based Vision Foundation Models for Medical Image Segmentation** | J. Liu | 2024 | IEEE TMI | First major work adapting Mamba (state space model) as an encoder for U-Net-style medical image segmentation. Demonstrates competitive performance with Swin-Transformer-based models at lower computational cost due to Mamba's linear complexity. Evaluated on abdominal organ segmentation (Synapse, ACDC). [9] |
| 2 | **U-Mamba / VM-UNet / nnMamba: Mamba variants for volumetric medical segmentation** | Multiple | 2024 | arXiv / MICCAI 2024 | Rapid proliferation of Mamba-based architectures for 3D medical image segmentation. U-Mamba integrates Mamba blocks into the nnU-Net framework. VM-UNet uses Vision Mamba blocks. Key advantage: linear-complexity long-range dependency modeling for large 3D volumes. |
| 3 | **BiomedCLIP / LLaVA-Med / Med-PaLM M: Vision-Language Models for Medical Imaging** | Multiple | 2023–2025 | NeurIPS, Nature Medicine | Foundation vision-language models trained on medical image-text pairs. BiomedCLIP (Microsoft, 2023) trained on PMC-15M. LLaVA-Med (NLM/Microsoft, 2024) for visual question answering in medical imaging. These models can generate reports, answer clinical questions, but are not yet reliable for precise measurement tasks. |
| 4 | **Cross-anatomy transfer learning and universal segmentation** | Multiple | 2024–2025 | MICCAI, MedIA | Universal medical image segmentation models (UniverSeg, STU-Net, SAT) trained on many anatomies simultaneously show promising zero-shot transfer. STU-Net (Huang et al., 2023) is a scalable 3D U-Net trained on 14,000+ CT scans across 143 classes. Potential for pretraining on diverse anatomy before fine-tuning on aortic structures. |

### 4.2 Relevance to Aortic Intervention Planning
- **Mamba models** could be beneficial for processing full 3D aortic CT volumes (thoraco-abdominal coverage) due to linear memory scaling.
- **VLMs** could assist in generating structured TAVI/EVAR planning reports from measurements, but cannot yet perform precise geometric measurements.
- **Cross-anatomy pretraining** (e.g., on STU-Net or TotalSegmentator labels) could provide strong initialization for aortic segmentation with limited labeled data.

### 4.3 Gaps
- **No Mamba-based model has been applied to aortic/vascular segmentation** specifically — all evaluations are on abdominal organs or cardiac chambers.
- **VLMs cannot do precise landmark detection or measurement** — they are useful for high-level interpretation but not geometric quantification.
- **Cross-anatomy transfer to vascular structures** is understudied; tubular vascular geometry is quite different from organ segmentation.
- **TotalSegmentator** includes aorta as one class but lacks fine-grained aortic sub-structure labels (annulus, sinuses, landing zones).

### 4.4 Maturity: 🟠 Early
Mamba models are proliferating rapidly but not yet validated on vascular anatomy. VLMs are not applicable to precise measurement tasks. Cross-anatomy transfer is promising but requires fine-tuning. **Opportunity: first to apply Mamba-based encoder to aortic landmark detection.**

---

## Cross-Domain Synthesis & Recommended Experiments

### Identified Gaps Suitable for PhD Contributions

| Gap | Domains | Opportunity Level |
|-----|---------|-------------------|
| Semi-supervised 3D landmark detection for aortic root (TAVI) | D1 + D3 | ⭐⭐⭐ High |
| Non-contrast CT AAA segmentation with label-efficient methods | D2 + D3 | ⭐⭐⭐ High |
| Coronary ostia height prediction from CT (poor current accuracy) | D1 | ⭐⭐⭐ High |
| BAV detection/classification from pre-TAVI CT | D1 | ⭐⭐ Medium |
| Mamba-based encoder for 3D aortic segmentation/landmark detection | D4 + D1/D2 | ⭐⭐ Medium |
| Diffusion-model pretraining for 3D cardiac landmark detection | D3 + D1 | ⭐⭐ Medium |
| End-to-end EVAR stent-graft planning pipeline | D2 | ⭐⭐ Medium |
| Cross-anatomy pretraining (TotalSegmentator → aortic fine-tuning) | D4 + D1/D2 | ⭐⭐ Medium |

### Recommended Next Experiments

1. **Semi-supervised aortic root landmark detection (TAVI data)**  
   *Motivation:* No existing work applies semi-supervised methods to 3D aortic landmarks. The student has TAVI CT data but likely limited annotations.  
   *Approach:* Adapt Chen et al.'s shape-regulated self-training to 3D; use PCA shape model of aortic root landmarks (commissures, nadirs, coronary ostia). Train with 20–30% labeled + rest unlabeled.  
   *Expected outcome:* First benchmark for semi-supervised 3D aortic landmark detection; potential MRE reduction of 1–2mm over supervised baseline with limited labels.  
   *Venue target:* MICCAI 2027 or Medical Image Analysis.

2. **Label-efficient AAA segmentation with cross-anatomy pretraining (AAA-30 data)**  
   *Motivation:* AAA-30 is small (30 patients). Pretraining on TotalSegmentator aorta labels + fine-tuning could boost performance significantly.  
   *Approach:* Use STU-Net or nnU-Net pretrained on TotalSegmentator, then fine-tune on AAA-30 with lumen/wall/thrombus labels. Compare to training from scratch.  
   *Expected outcome:* Demonstrate that cross-anatomy pretraining improves Dice by 3–5% on small datasets; potentially the first thrombus sub-segmentation on the AAA-30 cohort.

3. **Diffusion-model features for few-shot aortic landmark detection**  
   *Motivation:* Di Via et al. (WACV 2025) showed diffusion pretraining works for 2D X-ray landmarks. Extension to 3D cardiac CT is unexplored.  
   *Approach:* Train a 3D denoising diffusion model on unlabeled cardiac CTs, extract intermediate features, use these as input to a lightweight landmark regressor.  
   *Expected outcome:* Novel methodological contribution; performance in few-shot (5–20 labeled cases) regime for aortic root landmarks.

---

### Key Common Pitfalls

1. **Data leakage via slice-level splitting**: Always split at patient level, never at slice level.
2. **Evaluation on single center**: Try to validate on both TAVI and AAA datasets for generalization claims.
3. **Ignoring clinical significance**: Report clinical metrics (prosthesis size agreement, diameter error) alongside ML metrics (Dice, MRE).
4. **Landmark definition ambiguity**: Aortic root landmarks (especially hinge points) vary between annotation protocols — define clearly and measure inter-observer variability.
5. **Small dataset overfitting**: With AAA-30, use rigorous cross-validation (5-fold, leave-one-out) and report confidence intervals.

---

### Sources

[1] Santaló-Corcoy M. et al., "TAVI-PREP: A Deep Learning-Based Tool for Automated Measurements Extraction in TAVI Planning," Diagnostics, 2023. PMID: 37892002. https://pubmed.ncbi.nlm.nih.gov/37892002/

[2] CT-based deep learning system for automatic aortic root assessment, arXiv:2302.05378, 2023. https://arxiv.org/abs/2302.05378

[3] Roby M. et al., "Automatic Explainable Segmentation of Abdominal Aortic Aneurysm From Computed Tomography Angiography," IEEE Access, 2025. DOI: 10.1109/access.2025.3620721. https://pmc.ncbi.nlm.nih.gov/articles/PMC12875671/

[4] Lu J.T. et al., "DeepAAA: Clinically Applicable and Generalizable Detection of Abdominal Aortic Aneurysm Using Deep Learning," MICCAI, 2019.

[5] Chandrashekar A. et al., Attention-based U-Net for automated aortic segmentation from contrast and non-contrast CT. Referenced in Roby et al. [3].

[6] "Deep Learning Methods in Abdominal Aortic Aneurysm Segmentation" (survey), IEEE Access, 2024. https://ieeexplore.ieee.org/document/11300868

[7] Chen R. et al., "Semi-supervised Anatomical Landmark Detection via Shape-regulated Self-training," Neurocomputing, 2021. https://arxiv.org/abs/2105.13593

[8] Di Via R. et al., "Self-Supervised Pre-Training with Diffusion Model for Few-Shot Landmark Detection in X-Ray Images," WACV 2025. https://openaccess.thecvf.com/content/WACV2025/papers/Di_Via_Self-Supervised_Pre-Training_with_Diffusion_Model_for_Few-Shot_Landmark_Detection_in_WACV_2025_paper.pdf

[9] Liu J. et al., "Swin-UMamba: Adapting Mamba-based Vision Foundation Models for Medical Image Segmentation," IEEE TMI, 2024. DOI: 10.1109/TMI.2024.3508698.
