# Mini Literature Review: Expanded Unlabeled CT Pool and TRUST Follow-up Work

**Date:** 2026-09-03

## Bottom Line

The current `0/100/300/620` experiment already answers the narrow question posed in the TRUST manuscript: whether unlabeled data materially improve performance under a fixed labeled set. Adding a post-hoc `1000+` point would test further scaling or saturation, but it would not address the manuscript's main residual weaknesses: single-center generalization, inference-time uncertainty calibration, reader-dependent ground truth, and sparse landmark-only annular modeling.

The expanded pool is better used in a pre-specified follow-up study with a new hypothesis. Data are not "used up" by an experiment, so reserving them is not by itself a valid reason. The defensible reason is scope and experimental consistency: replacing the 620-scan model would require a coherent refresh of the principal comparison, ablations, per-landmark results, and downstream analyses, whereas a single extra endpoint would provide limited evidence about the current method.

## Recent Directions

### 1. Task-specific 3D self-supervised pretraining

- VoCo uses contextual position priors for self-supervised pretraining of 3D medical volumes and reports transfer gains across six downstream tasks.
- OpenMind benchmarks 3D self-supervised methods at much larger scale and shows that performance depends strongly on architecture, pretraining setup, and the downstream task. This cautions against assuming that more unlabeled volumes automatically improve a model.
- CABLD demonstrates a recent consistency-based, label-efficient direction specifically for 3D anatomical landmarks and evaluates generalization across imaging contrasts.

**Opportunity for TRUST:** pretrain a TAVI-specific 3D encoder on the expanded CT pool, then compare from-scratch TRUST, self-supervised pretraining, and CPS-based semi-supervision under fixed label budgets and scanner/protocol shifts. This is a new representation-learning question, not merely another point on the present scaling curve.

### 2. Calibrated inference-time uncertainty

- Jonkers et al. introduce multi-output conformal prediction for 2D/3D anatomical landmarks, producing prediction regions with finite-sample coverage guarantees.
- This direction directly addresses TRUST's current limitation that pooled TTA uncertainty saturates after convergence and is therefore weak for inference-time quality assurance.

**Opportunity for TRUST:** calibrate per-landmark 3D confidence regions, propagate them to MSL, coronary-height, and annular-plane intervals, and define an expert-review trigger for unreliable cases.

### 3. Multi-reader-aware landmark learning

- Salari et al. show that landmark annotation averaging can be inferior to random sampling or ensembles and explicitly connect inter-reader variability with model uncertainty.
- TRUST now has a valuable seed dataset: 50 cases with Jean and two Leo annotation sessions. The current paper uses these labels for evaluation, but does not learn reader-dependent distributions.

**Opportunity for TRUST:** train a reader-aware probabilistic landmark model, distinguish annotation ambiguity from model uncertainty, and evaluate against individual readers and an adjudicated reference.

### 4. Uncertainty-guided active annotation

- CSAL-3D combines self-supervised features, uncertainty, and diversity to select informative 3D medical volumes for annotation.

**Opportunity for TRUST:** rank the `1000+` unlabeled TAVI scans by uncertainty and representation diversity, then ask clinicians to annotate a small, enriched subset. Selection could deliberately target rare low-coronary-height anatomy, severe calcification, bicuspid valves, and scanner/protocol outliers. This would test whether uncertainty reduces annotation burden, which is more consequential than showing a marginal benefit from several hundred additional pseudo-labeled scans.

### 5. From sparse landmarks to complete TAVI planning and outcomes

- Recent automated TAVI planning systems increasingly predict full annular contours and direct measurements rather than relying only on three hinge points.
- Lemarchand et al. (2026) analyze 501 TAVI patients and show that implantation depth relative to patient-specific MSL is associated with conduction outcomes. This supports linking automated CT anatomy to procedural variables and outcomes rather than treating a single geometric cutoff as a clinical decision rule.

**Opportunity for TRUST:** jointly predict landmarks, dense annular contour, sinus/STJ geometry, and calcification; then validate derived planning measurements and, where available, post-TAVI conduction outcomes.

### 6. Local tool-using LLM agent for TAVI planning

**Status:** longer-term, high-risk research track; record now, but do not make it the immediate follow-up priority.

The useful role of a local LLM is not to regress landmarks from raw CT or merely rewrite measurements as prose. It would act as an auditable planning controller that interprets a clinician's request, selects and invokes specialist tools, checks their outputs, chooses recovery actions, retrieves device-specific rules, and presents a structured plan with explicit evidence and uncertainty.

Candidate tools include landmark detection, aortic-root and annular-contour segmentation, deterministic annular and coronary measurements, calcification quantification, C-arm angle calculation, uncertainty and out-of-distribution checks, device-IFU retrieval, and planning-view rendering. An agent is scientifically justified only if it demonstrates genuinely conditional behavior, such as selecting tools according to anatomy, retrying failed localization, requesting missing information, switching to a fallback model, or abstaining when confidence is insufficient. A fixed sequence of identical calls is better described as a conventional pipeline.

This direction does not initially require training an LLM from scratch. A staged implementation could use an existing local instruction model with constrained structured tool calls, retrieval, and deterministic validation; parameter-efficient fine-tuning would be considered only after sufficient clinician-authored planning traces are available. The main barriers are local inference infrastructure, robust DICOM/model tool interfaces, hallucination containment, device-rule maintenance, privacy and audit requirements, and the lack of case-linked expert planning decisions and clinical outcomes.

**Minimum evidence for a paper:** comparison against a fixed automated pipeline and expert planning; tool-selection and recovery accuracy; factual and numerical error rates; abstention calibration; plan completeness; reader agreement; time saved; and external or temporal validation. Until these data and evaluation conditions are available, this track should remain a documented exploratory route rather than the central next-paper commitment.

## Recommended Follow-up Study

**Working question:** Can uncertainty and multi-reader variability turn a large unlabeled TAVI CT archive into an efficient, calibrated, and generalizable landmark-planning system?

1. Self-supervised pretrain the 3D encoder on the expanded CT archive.
2. Use uncertainty plus feature diversity to select a small annotation subset enriched for difficult anatomy and acquisition shifts.
3. Incorporate the three-reader annotations to learn landmark distributions rather than a single fixed coordinate.
4. Apply conformal calibration to produce per-landmark confidence regions and downstream measurement intervals.
5. Validate temporally or externally, ideally with adjudicated references and clinical outcomes.

This combines assets already available to the project and directly attacks the limitations of the current manuscript. It is a stronger basis for a separate paper than a simple `620` versus `1000+` pseudo-label scaling comparison.

## Suggested Rationale for the Current Study

The present manuscript uses a fixed experimental design to establish the contribution of unlabeled data through a controlled `0/100/300/620` ablation. Extending the pool after completion would either add a single post-hoc saturation point or require rerunning all model-dependent analyses for consistency. Because the additional same-center unlabeled scans do not provide external validation or new ground truth, their marginal value for the current claims is limited. We therefore propose using the expanded archive in a dedicated follow-up study on self-supervised pretraining, uncertainty-guided annotation, reader-aware learning, and calibrated inference.

## Key References

- Wu, L., Zhuang, J., and Chen, H. "VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis." CVPR, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.html
- Wald, T. et al. "An OpenMind for 3D Medical Vision Self-supervised Learning." ICCV, 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Wald_An_OpenMind_for_3D_Medical_Vision_Self-supervised_Learning_ICCV_2025_paper.html
- Salari, S. et al. "CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization." ICCV, 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Salari_CABLD_Contrast-Agnostic_Brain_Landmark_Detection_with_Consistency-Based_Regularization_ICCV_2025_paper.html
- Jonkers, J. et al. "Reliable Uncertainty Quantification for 2D/3D Anatomical Landmark Localization Using Multi-output Conformal Prediction." Medical Image Analysis, 2026, 103953. https://doi.org/10.1016/j.media.2026.103953
- Salari, S., Rivaz, H., and Xiao, Y. "Reliability of Deep Learning Models for Anatomical Landmark Detection: The Role of Inter-rater Variability." arXiv:2411.17850, 2024. https://arxiv.org/abs/2411.17850
- Zhu, N. et al. "CSAL-3D: Cold-start Active Learning for 3D Medical Image Segmentation via SSL-driven Uncertainty-Reinforced Diversity Sampling." MICCAI, 2025. https://papers.miccai.org/miccai-2025/0198-Paper2315.html
- Lemarchand, L. et al. "Influence of the Cusp-Overlap and Three Cusps Coplanar Techniques on New-Onset Conduction Disturbances Following Transcatheter Aortic Valve Implantation." Cardiovascular Intervention and Therapeutics, 2026. https://doi.org/10.1007/s12928-026-01276-0

## BibTeX Entries

```bibtex
@inproceedings{Wu2024VoCo,
  author    = {Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
  title     = {VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
  pages     = {22873--22882}
}

@inproceedings{Wald2025OpenMind,
  author    = {Wald, Tassilo and Ulrich, Constantin and Suprijadi, Jonathan and Ziegler, Sebastian and Nohel, Michal and Peretzke, Robin and K{\"o}hler, Gregor and Maier-Hein, Klaus},
  title     = {An OpenMind for 3D Medical Vision Self-supervised Learning},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2025},
  pages     = {23839--23879}
}

@inproceedings{Salari2025CABLD,
  author    = {Salari, Soorena and Harirpoush, Arash and Rivaz, Hassan and Xiao, Yiming},
  title     = {{CABLD}: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2025},
  pages     = {20991--21002}
}

@article{Jonkers2026ConformalLandmarks,
  author  = {Jonkers, Jef and Coopman, Frank and Duchateau, Luc and Van Wallendael, Glenn and Van Hoecke, Sofie},
  title   = {Reliable Uncertainty Quantification for 2D/3D Anatomical Landmark Localization Using Multi-output Conformal Prediction},
  journal = {Medical Image Analysis},
  volume  = {110},
  pages   = {103953},
  year    = {2026},
  doi     = {10.1016/j.media.2026.103953}
}

@article{Salari2024InterRaterLandmarks,
  author  = {Salari, Soorena and Rivaz, Hassan and Xiao, Yiming},
  title   = {Reliability of Deep Learning Models for Anatomical Landmark Detection: The Role of Inter-rater Variability},
  journal = {arXiv preprint arXiv:2411.17850},
  year    = {2024}
}

@inproceedings{Zhu2025CSAL3D,
  author    = {Zhu, Ning and Ye, Ping and Zhong, Lanfeng and Yue, Qiang and Zhang, Shaoting and Wang, Guotai},
  title     = {{CSAL-3D}: Cold-start Active Learning for 3D Medical Image Segmentation via SSL-driven Uncertainty-Reinforced Diversity Sampling},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year      = {2025},
  volume    = {LNCS 15961},
  pages     = {120--130},
  doi       = {10.1007/978-3-032-04937-7_12}
}

@article{Lemarchand2026CuspOverlap,
  author  = {Lemarchand, L{\'e}o and Grolleau, Raphael and Boulmier, Dominique and Leurent, Guillaume and Tomasi, Jacques and others},
  title   = {Influence of the Cusp-Overlap and Three Cusps Coplanar Techniques on New-Onset Conduction Disturbances Following Transcatheter Aortic Valve Implantation},
  journal = {Cardiovascular Intervention and Therapeutics},
  year    = {2026},
  doi     = {10.1007/s12928-026-01276-0}
}
```
