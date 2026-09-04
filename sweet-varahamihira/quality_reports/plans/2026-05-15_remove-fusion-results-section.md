# Remove Fusion Results Section

Date: 2026-05-15

Goal: Remove the standalone inference-time fusion results paragraph and table from the main paper, because Fuse is practically tied with Student 1 and should not be elevated into an independent results story.

Planned edits:

1. [x] Remove the `Inference-time cross-student fusion` paragraph, Table `tab:fusion`, and its interpretation paragraph from `paper/main.tex`.
2. [x] Update the contribution bullet that currently advertises `inference-time fusion analysis`.
3. [x] Compile the paper and check references, warnings, and page count.

Rationale:

- Fuse has nearly identical MRE to Student 1 (`2.19` vs `2.20` mm), and Student 1 is slightly better at tight SDR thresholds.
- A standalone table risks overemphasizing a marginal effect and distracting from the stronger TRUST story: unified uncertainty consumed by CPS, UG-HCO, and Topo-GCN, plus clinical downstream evaluation.
- The fusion rule can remain in the Methods as the definition of the final output, but it should not be framed as a separate accuracy contribution.

## Result

Implemented in `paper/main.tex`.

- Removed the standalone `Inference-time cross-student fusion` Results paragraph, Table `tab:fusion`, and interpretation paragraph.
- Replaced `inference-time fusion analysis` in the contribution list with `ablation studies`.
- Marked the previous scatter-to-fusion plan as superseded.

Verification:

- Ran `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `paper/`.
- Output is 18 pages.
- No undefined references/citations or stale `tab:fusion` references.
- Only two tiny overfull boxes remain (`0.61pt`, `0.55pt`), both below the 10pt quality threshold.
