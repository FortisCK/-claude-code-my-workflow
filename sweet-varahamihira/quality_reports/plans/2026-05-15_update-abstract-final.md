# Update Abstract Final

Date: 2026-05-15

Goal: Bring the abstract into alignment with the corrected retraining results and current clinical-downstream story.

Planned edits:

1. [x] Update stale result numbers: `2.16` -> `2.19`, `22.6%` -> `21.5%`.
2. [x] Update downstream scope from five to six TAVI planning measurements.
3. [x] Replace the old LCO/RCO sentence with a balanced clinical closing that matches the Conclusion: preserved clinically consequential stratifications, strong per-cusp calcification agreement, and explicit validation targets for annular-area bias and cohort-limited per-ostium sensitivity.
4. [x] Compile the paper and verify references, warnings, and page count.

Draft direction:

Write one flowing abstract paragraph. Keep the main contribution focused on unified uncertainty as a training-time signal consumed by CPS, UG-HCO, and Topo-GCN. Do not mention the removed fusion analysis.

## Result

Implemented in `paper/main.tex`.

- Abstract now reports `2.19` mm and `21.5%`.
- Clinical downstream scope now says six TAVI planning measurements.
- Closing sentence no longer says the left ostium is supported while the right ostium is underpowered; it now frames annular-area under-prediction and cohort-limited per-ostium sensitivity as validation targets.
- Abstract word count: 247 words.

Verification:

- Ran `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `paper/`.
- Output remains 18 pages.
- No stale `2.16`, `22.6%`, `five measurements`, `tab:fusion`, or old RCO-underpowered abstract language remains in `paper/main.tex`.
- No undefined references/citations.
- Only two tiny overfull boxes remain (`0.61pt`, `0.55pt`), both below the 10pt quality threshold.
