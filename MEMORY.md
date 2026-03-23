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
