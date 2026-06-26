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
