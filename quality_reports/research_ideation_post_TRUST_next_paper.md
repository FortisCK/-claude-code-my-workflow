# Research Ideation: Post-TRUST Follow-up Studies

**Date:** 2026-09-03
**Scope:** Broad scan of recent methods, public datasets, and clinical questions. Carlos's phase/angle work and ImageCAS are treated as examples, not constraints.

## Starting Assets

- 150 labeled pre-TAVI CT volumes with 10 aortic-root landmarks.
- A growing same-center pool of more than 1,000 unlabeled CT volumes.
- Three annotation sets on cases 101--150: Jean and two sessions from Leo.
- Existing deterministic measurements for MSL, coronary heights, annular plane, three-hinge diameter/area, C-arm angles, and DLZ calcification.
- An established semi-supervised landmark pipeline and a completed first manuscript.

## Decision After Initial Discussion

The retained candidate is **CT-based dense aortic-root reconstruction for TAVI planning**. The model should recover patient-specific contours and surfaces from the original CT, with P0--P9 serving as anatomical anchors and topology supervision. It should not attempt to infer the full anatomy from ten points alone.

Directions whose main endpoint is a marginal improvement in landmark MRE, annotation efficiency, uncertainty calibration, or model-quality validation are deprioritized. Inter-reader disagreement of approximately 2 mm makes further average landmark-accuracy optimization a weak standalone clinical story.

Candidate auxiliary datasets have distinct roles:

| Dataset | Potential role in the retained direction | Important limitation | Priority |
|---|---|---|---|
| M-WHS-100 | Expert voxel-wise supervision near the basal ring, aortic valve/root, sinuses, STJ, and calcification | Sparse public documentation and unclear exact class semantics/access split require a file-level audit | High, pending audit |
| TAVRP-PL | Large-scale initialization for aorta, root, valve, LV, annulus, and iliac structures | All 578 labels are automatically derived pseudo-labels from TotalSegmentator data, not clinical TAVI ground truth | Medium |
| ImageCAS-X | Coronary lumen, segment, centerline, and mesh priors; potentially useful for ostium localization and joining the coronary tree to the reconstructed root | Does not annotate the annulus, valve, or root and is CCTA rather than a TAVI cohort | Medium for a coronary extension; low for the core root model |

The 2D projection-angle work should initially be treated as an application or later branch, not mixed into the core reconstruction claim. A dense 3D model can derive annular and commissural orientation and render a candidate cusp-overlap projection, but patient-specific 3D-to-2D validation requires paired preoperative CT, intraoperative fluoroscopy, and recorded C-arm geometry. Multiphase angle analysis instead requires ECG-gated phase-resolved CT; the three datasets above do not supply that temporal supervision.

### Relevant Prior Experience: CathAction

**Current status:** Parked as a longer-term, data-conditional direction rather than an immediate follow-up project.

The user has previously worked with CathAction, so intraoperative video understanding is not a greenfield direction. The original CathAction release contains approximately 500,000 annotated frames for catheterization action recognition/anticipation and collision detection, plus approximately 25,000 catheter/guidewire masks. Its labels cover catheter and guidewire advancement/retraction and rotation, and its benchmark includes segmentation, collision detection, action recognition, action anticipation, and cross-domain transfer. The 2026 MICCAI challenge expands the setting to more than 600,000 frames across silicon phantom, animal fluoroscopy, and real human procedures.

This experience makes three follow-up tasks materially more plausible:

1. **Collision anticipation rather than collision detection:** predict an impending catheter/guidewire--wall collision several frames before contact and recommend a corrective action.
2. **Anatomy-conditioned action anticipation:** combine the tool trajectory and action history with a vessel or aortic-root map so that the model predicts actions and risks relative to patient anatomy, not from appearance alone.
3. **CT-conditioned TAVI fluoroscopic guidance:** transfer a preoperative annular/root plan into the intraoperative view, track alignment and deployment depth, and quantify deviations from the plan.

CathAction alone does not provide TAVI root landmarks, optimal C-arm angles, or paired preoperative CT. It supplies the temporal and interventional perception component; a TAVI-specific navigation study would still require paired local CT--fluoroscopy cases and C-arm metadata. A separate, less TAVI-specific paper could instead combine CathAction with simulated or CT-derived vessel geometries to study early collision warning and corrective action.

**Novelty boundary:** patient-specific CT--fluoroscopy fusion and anatomical overlay during TAVI have already been demonstrated clinically. A follow-up should therefore not claim novelty from overlay alone. The stronger target is **CT-plan-conditioned temporal guidance**: automatically register the anatomy, track prosthesis depth and rotation, recognize the current action, anticipate unsafe deviation from the plan, and recommend a specific correction.

**Minimum data hierarchy:** paired preoperative CT and intraoperative fluoroscopy are essential for patient-specific validation. Raw XA DICOM with primary/secondary C-arm angles and projection geometry is strongly preferred. If geometry tags are absent but paired raw images and equipment calibration are available, geometry may be estimated jointly with registration at reduced reliability. Exported video alone supports 2D tool/action analysis and relative tracking, but generally not metrically calibrated 3D overlay. Completely unpaired CT and fluoroscopy can support synthetic pretraining or component studies, not a validated patient-specific navigation claim.

**Conditions for reactivation:** confirm that the hospital can retrieve same-patient preoperative CT and raw intraoperative XA DICOM or fluoroscopy video; recover C-arm primary/secondary angles and projection geometry from DICOM, equipment logs, or calibration; and obtain valve type/size, deployment depth, procedure timestamps, and clinician-selected final projection angles. Without these assets, retain CathAction for standalone action/collision studies rather than claiming complete TAVI navigation.

## Earlier Surveyed Directions

### 1. Clinically Measurement-Aware Active Annotation

**Research question:** Can a model select the small subset of unlabeled TAVI scans whose annotation most improves clinically relevant measurements, rather than merely reducing average landmark error?

**Hypothesis:** Selecting cases using a combination of representation diversity, landmark uncertainty, and propagated measurement uncertainty will reach the same MSL, coronary-height, annular-plane, and calcification accuracy with substantially fewer newly annotated scans than random sampling or image-level uncertainty alone.

**Why it is a real new paper:** Recent 3D active-learning work combines self-supervised features, uncertainty, and diversity, but TRUST can make the selection objective measurement-aware and anatomy-aware. The central outcome becomes annotation efficiency for TAVI planning, not another unlabeled-data scaling point.

**Data and experiment:**

1. Simulate active-learning rounds within the existing 100 labeled training cases, keeping validation and test cases untouched.
2. Pretrain or extract representations from the expanded unlabeled archive.
3. Compare random, diversity-only, landmark-uncertainty, and proposed measurement-aware selection.
4. Prospectively annotate a small selected subset from the new archive to confirm the retrospective simulation.
5. Report label budget versus landmark MRE and downstream measurement error, including performance on rare or difficult anatomy.

**Main risk:** A simulation-only active-learning paper is less convincing. At least one real clinician annotation round is highly desirable.

**Updated verdict:** Deprioritized after discussion because its principal endpoint remains improved landmark accuracy or annotation efficiency.

### 2. Sparse Landmarks to Dense Aortic-Root Geometry

**Research question:** Can heterogeneous supervision from sparse landmarks, limited contours, and public segmentation data recover dense patient-specific annular and aortic-root geometry?

**Hypothesis:** A shared point-surface model with anatomical and topological constraints will preserve TRUST's landmark accuracy while producing more faithful area-, perimeter-, sinus-, and STJ-based measurements than the current three-hinge circumcircle approximation.

**Why it matters:** This directly resolves the most important geometric limitation of TRUST. The output becomes closer to an actual planning model rather than a set of ten coordinates.

**Useful data:**

- Existing TAVI landmark CTs and available annular polygons.
- A newly annotated subset with dense annulus/root/leaflet contours, accelerated by an interactive 3D model.
- TAVRP-PL for weak TAVR-related pseudo-segmentation supervision.
- WHS++ or AortaSeg24 for broader cardiac/aortic shape pretraining.

**Main risk:** Public labels do not share exactly the same anatomy definitions or clinical quality as the local TAVI annotations. A small expert-reviewed dense reference set remains necessary.

**Updated verdict:** Retained as the leading candidate because it changes the output from sparse points to planning-relevant patient-specific anatomy.

### 3. Calibrated Landmark and Measurement Confidence

**Research question:** Can a model provide finite-sample calibrated confidence regions for correlated landmarks and valid confidence intervals for nonlinear TAVI measurements?

**Hypothesis:** Structured calibration over the ten-landmark graph, followed by uncertainty propagation through the measurement functions, will produce better coverage and more useful failure triage than TTA variance, heatmap confidence, or independent per-point intervals.

**Required distinction from recent work:** Conformal prediction for anatomical landmark regions already exists. The novelty must be structured multi-landmark calibration, downstream measurement coverage, and an explicit expert-review rule, not merely adding conformal prediction to TRUST.

**Data and experiment:** Use a strictly separate calibration split or cross-conformal design; assess empirical coverage, interval width, failure-detection AUROC, and the proportion of scans safely auto-accepted. Use the multi-reader data to distinguish model error from plausible annotation variation.

**Main risk:** The present validation set of 20 cases is too small for fine-grained coverage claims. More calibration cases or repeated cross-validation are needed.

**Updated verdict:** Deprioritized because uncertainty calibration and model validation are not desired as the next paper's main contribution.

### 4. External-Domain Robustness and Single-Case Adaptation

**Research question:** Can TAVI landmark detection remain reliable across centers, scanners, reconstruction kernels, and contrast protocols without target-domain labels?

**Hypothesis:** task-specific self-supervised pretraining plus quality-gated test-time adaptation will reduce domain-shift error while avoiding harmful adaptation on individual volumes.

**Why it matters:** This attacks the current paper's single-center limitation directly. Recent work on single-volume test-time adaptation makes the setting methodologically current.

**Data and experiment:** Public cardiac/aortic CT can supply unlabeled target domains and auxiliary segmentation tasks, but the decisive test still requires a small externally annotated TAVI cohort using the same P0--P9 definitions.

**Main risk:** Without external P0--P9 ground truth, this becomes a pretraining study rather than credible external validation.

**Verdict:** Potentially the strongest evidence paper, conditional on obtaining an external clinical collaborator.

### 5. Anatomy-Grounded TAVI Outcome Prediction

**Research question:** Do learned CT features add predictive value beyond interpretable TRUST geometry and standard clinical variables for a specific post-TAVI outcome?

**Candidate endpoints:** new conduction disturbance or pacemaker implantation, paravalvular leak, coronary obstruction, or one-year adverse outcomes.

**Hypothesis:** A model combining automated geometry, calcification distribution, CT representation, device/procedure variables, and clinical covariates will outperform tabular clinical baselines while retaining interpretable anatomical contributions.

**Main requirement:** The expanded archive must be linkable to reliable outcomes, device information, implantation depth, and clinical variables. A single prespecified endpoint is preferable to a broad multi-outcome fishing exercise.

**Main risk:** TAVR outcome prediction is already crowded and includes large externally validated cohorts. A small retrospective model without external or temporal validation will not be competitive.

**Verdict:** High clinical upside if outcome records are available at scale; otherwise defer.

### 6. Generalist or Promptable Anatomical Landmark Model

**Research question:** Can topology and semantic landmark descriptions support few-shot transfer to new 3D anatomical landmark sets?

**Hypothesis:** A universal 3D encoder with graph-conditioned landmark queries will adapt to a new landmark ontology with fewer labels than task-specific training.

**Why it is current:** Multi-dataset and foundation-model approaches are moving landmark detection toward generalist and few-shot systems.

**Main risk:** This direction is becoming crowded, and assembling genuinely compatible 3D landmark datasets is laborious. A backbone swap or generic multi-dataset pretraining result would be too incremental.

**Verdict:** Suitable for a methods-focused MICCAI-style project, but less directly connected to TAVI clinical translation.

### 7. Synthetic Rare-Anatomy and Counterfactual Stress Testing

**Research question:** Can anatomy-conditioned 3D generation create controlled rare TAVI geometries for tail-performance testing or augmentation?

**Hypothesis:** Synthetic cases conditioned on low coronary height, unusual root geometry, or severe calcification can expose and reduce rare-case failure better than ordinary augmentation.

**Main risk:** Image realism is not equivalent to clinical or geometric validity. This requires expert review, distributional checks, and real rare-case validation.

**Verdict:** Interesting high-risk project; not the first follow-up to start.

## Public Data by Role

| Resource | What it can contribute | What it cannot establish |
|---|---|---|
| TAVRP-PL | 578 CTs with TAVR-related pseudo-segmentation labels | Clinical-quality TAVI landmarks or external validation |
| WHS++ | 104 CT and 102 MRI volumes from six centers for whole-heart domain diversity | Fine aortic-root and coronary-ostium definitions |
| AortaSeg24 | 100 CTA volumes with 23 aortic branches/zones | Annular hinge and commissural landmarks |
| ImageCAS-X | Coronary lumen, segment labels, centerlines, and meshes on 800 CCTA scans | Direct equivalence to pre-TAVI root anatomy or local landmark definitions |
| orCaScore | Paired noncontrast/CTA scans from four hospitals/vendors with expert coronary-calcium labels | Aortic-valve calcification or TAVI planning ground truth |
| CT-RATE | Large-scale 3D CT-report pretraining data | Contrast-enhanced TAVI anatomy or planning labels |
| Public TAVI fluoroscopy | 2D root masks and four landmarks for intraoperative guidance | 3D CT planning measurements |

Public datasets should be treated as auxiliary supervision, pretraining, or domain-shift resources. None is a substitute for a small, consistently annotated external TAVI test set.

## What Not to Make the Next Paper

1. Adding only a `1000+` endpoint to the current `0/100/300/620` curve.
2. Replacing the TRUST encoder with a foundation model and reporting a small MRE gain.
3. Attaching an LLM that merely turns fixed measurements into prose.
4. Training a broad TAVI outcome classifier without a prespecified endpoint and external or temporal validation.
5. Claiming multi-center generalization from public datasets whose labels and acquisition populations do not match the P0--P9 task.

## Updated Recommended Sequence

1. Audit M-WHS-100, TAVRP-PL, the local polygons, and local CT acquisition phases against a precise dense-root annotation ontology.
2. Ask clinicians to define the minimum annulus/root/leaflet structures required for a clinically meaningful planning model and estimate the effort for a small expert-reviewed reference set.
3. Build a feasibility baseline for joint CT segmentation and P0--P9 localization, using landmarks as anchors rather than reconstructing anatomy from points alone.
4. Audit whether the archive contains linkable device, implantation-depth, ECG, fluoroscopy, and outcome data; these determine which clinical application can become the principal endpoint.
5. Keep active learning, uncertainty calibration, the LLM agent, multiphase analysis, CT-to-fluoroscopy, and synthetic generation as optional branches rather than the central claim.

## Key Recent Evidence

- CSAL-3D, MICCAI 2025: https://papers.miccai.org/miccai-2025/0198-Paper2315.html
- VISTA3D, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/He_VISTA3D_A_Unified_Segmentation_Foundation_Model_For_3D_Medical_Imaging_CVPR_2025_paper.html
- MedSapiens, Medical Image Analysis 2026: https://doi.org/10.1016/j.media.2026.104015
- Multi-output conformal landmark uncertainty, Medical Image Analysis 2026: https://doi.org/10.1016/j.media.2026.103953
- Single-image test-time adaptation, MICCAI 2025: https://papers.miccai.org/miccai-2025/0834-Paper2927.html
- Versatile segmentation from partially labeled datasets, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Versatile_Medical_Image_Segmentation_Learned_from_Multi-Source_Datasets_via_Model_CVPR_2024_paper.pdf
- MAISI synthetic 3D CT, WACV 2025: https://openaccess.thecvf.com/content/WACV2025/html/Guo_MAISI_Medical_AI_for_Synthetic_Imaging_WACV_2025_paper.html
- CT plus clinical outcome prediction after TAVR, 2025: https://doi.org/10.1016/j.pcad.2025.04.007
- TAVRP-PL: https://doi.org/10.5281/zenodo.16274176
- WHS++: https://www.zmic.org.cn/care_2024/track5/
- AortaSeg24: https://arxiv.org/abs/2502.05330
- ImageCAS-X: https://arxiv.org/abs/2608.30404
- orCaScore: https://orcascore.grand-challenge.org/Data/
- CT-RATE: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
- Public TAVI fluoroscopy data: https://doi.org/10.5281/zenodo.19219901
