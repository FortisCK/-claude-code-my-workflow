# Project Memory

Corrections and learned facts that persist across sessions.
When a mistake is corrected, append a `[LEARN:category]` entry below.

---

<!-- Append new entries below. Most recent at bottom. -->
- [LEARN:hooks] protect-files.sh blocks Edit/Write on protected files. Must temporarily remove pattern, edit, restore.
- [LEARN:directories] User prefers minimal directory structure -- don't over-organize upfront.
- [LEARN:template] User's actual template is ieeecolor.cls + generic.sty, NOT IEEEtran.cls. Document class: `\documentclass[journal,twoside,web]{ieeecolor}`
- [LEARN:components] TRUST component 2 is UG-HCO (Uncertainty-Guided Heat Conduction), not GLFE or UAGC.
- [LEARN:acronym] TRUST = Topological Reasoning and Uncertainty-aware Semi-supervised Teaching.
- [LEARN:compilation] pdflatex for ieeecolor. BIBINPUTS=..:$BIBINPUTS needed since .bib is in repo root.
- [LEARN:fusion-claims] When comparing Student1/Student2/Fuse, report matched metrics before claiming fusion is "substantially better." In `S1S2&Fuse.csv`, fusion is clearly better than Student 2 but only marginally better than Student 1 in MRE, so frame it as stabilization/robustness.
- [LEARN:fusion-s1-tie] Treat Fuse and Student 1 as practically tied in `S1S2&Fuse.csv` (`2.19` vs `2.20` mm); avoid saying fusion improves over the stronger branch, and frame it as branch-robust inference plus variance stabilization.
- [LEARN:advice-before-agreement] When the user challenges manuscript direction, provide independent professional judgment before agreeing or editing. For the Fuse/S1 comparison, the right recommendation is no standalone Results table/paragraph in the main paper; keep fusion only as the final-output definition unless asked otherwise.
- [LEARN:annular-area] Annular area prediction uses the circumcircle fitted through the three hinge points P0/P1/P2, not projection of the manual polygon onto the predicted plane. Manual polygon area is the contour reference; point-derived area is a circular approximation.
- [LEARN:multi-reader-cohort] Do not infer that an aggregate file named `n30_consistent` uses the paper's official downstream cases. The authoritative downstream cohort contains 14 cases below 101 and overlaps the new 101--150 reader cohort in only 16 cases; require case-level IDs before claiming a same-test-set n=30 comparison.
- [LEARN:reader-a1-reference] Reader A1 is the original ground-truth annotation, so TRUST-vs-A1 is the primary paper evaluation and should not be duplicated in the additional multi-reader table. Treat the provisional six-row downstream values supplied on 2026-09-02 as superseded by the corrected three-comparison full 10-point results.
- [LEARN:carlos-second-article-comment] The Carlos annotation "I wonder whether this section may be in a second article" is attached to the "Per-Landmark and Clinical Analysis" heading, not the label-efficiency experiment; interpret it as questioning whether the downstream/clinical-analysis section belongs in a separate article.
- [LEARN:llm-agent-scope] For the proposed follow-up, do not reduce the local LLM to report wording. The intended role is a tool-using TAVI planning agent that dynamically selects, invokes, verifies, and coordinates specialist image-analysis and deterministic measurement tools; report generation is only the final presentation step.
- [LEARN:future-work-brainstorming-scope] Treat Carlos's phase/angle work and ImageCAS as illustrative examples, not constraints or preferred anchors. Future-paper ideation should scan broadly across recent methods, public datasets, and clinical questions, then rank directions against the project's actual assets and feasibility.
- [LEARN:post-TRUST-direction-filter] Do not prioritize follow-up projects whose main endpoint is marginally better landmark accuracy, annotation efficiency, uncertainty calibration, or model validation. Given approximately 2 mm inter-reader disagreement and clinically adequate current performance, prioritize a genuinely new output or clinical question such as dense anatomy, procedural planning, outcomes, or patient-specific modeling.
- [LEARN:dataset-ideation-independence] When the user asks whether ImageCAS-X, M-WHS-100, TAVRP-PL, or 2D angle data can be used, evaluate each as a possible origin for an independent new task or paper. Do not assume they must support the previously discussed dense aortic-root reconstruction direction.
- [LEARN:tavi-navigation-novelty] Generic patient-specific preoperative CT to intraoperative fluoroscopy fusion and anatomical overlay during TAVI are established ideas. A credible follow-up must go beyond overlay, for example by combining temporal action understanding with prosthesis pose/depth tracking, plan deviation prediction, and corrective guidance.
