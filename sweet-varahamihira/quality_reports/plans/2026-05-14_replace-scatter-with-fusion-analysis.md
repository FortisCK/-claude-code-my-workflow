# Replace Uncertainty Scatter With Fusion Analysis

Date: 2026-05-14

Status: Superseded on 2026-05-15 by `quality_reports/plans/2026-05-15_remove-fusion-results-section.md`. The uncertainty scatter was still removed, but the standalone fusion Results paragraph/table was removed from the main paper because Fuse is practically tied with Student 1 and should not be elevated into an independent results story.

Goal: Remove the stale uncertainty-error scatter paragraph/figure and replace it with a controlled inference-time fusion analysis using `S1S2&Fuse.csv`.

Rationale:

- The current scatter plot and Pearson `r = 0.908` are based on old uncertainty/model outputs and are inconsistent with the corrected retraining results.
- `S1S2&Fuse.csv` provides matched 30-case errors for Student 1, Student 2, and the fused output.
- The fusion result has the lowest MRE and standard deviation, but it is practically indistinguishable from Student 1 in mean error; the narrative must therefore emphasize stabilization/robustness rather than a large standalone gain.

Planned edits:

1. [x] Remove `Uncertainty calibration` paragraph and Figure `fig:uncertainty_calibration`.
2. [x] Insert `Inference-time cross-student fusion` paragraph and table.
3. [x] Update the contribution sentence that currently mentions uncertainty-error calibration.
4. [x] Compile and verify references.

Computed metrics from the 30 case rows, excluding each section's aggregate mean row:

| Output | MRE ± std | SDR@2.0 | SDR@2.5 | SDR@3.0 | SDR@4.0 |
|--------|----------:|--------:|--------:|--------:|--------:|
| Student 1 | 2.20 ± 1.27 | 51.67 | 67.67 | 78.00 | 91.00 |
| Student 2 | 2.44 ± 1.36 | 43.33 | 57.67 | 70.33 | 88.33 |
| Fused | 2.19 ± 1.22 | 50.67 | 67.00 | 77.33 | 92.33 |

Verification target:

- `pdflatex`/`bibtex` compile completes.
- No undefined refs/citations introduced.
- No overfull `hbox` > 10 pt.

## Result

Implemented in `paper/main.tex`.

Interpretation used in the paper:

- Fusion is best in standard deviation (`1.22`) and SDR@4.0 (`92.33`), and its rounded MRE (`2.19`) is practically tied with Student 1 (`2.20`).
- Fusion is substantially better than Student 2 (`2.44 -> 2.19`) but effectively tied with Student 1, and Student 1 is slightly better at tight SDR thresholds.
- Therefore the narrative frames fusion as a branch-robust inference rule and stabilization mechanism rather than a large standalone ensemble gain.

Verification:

- Ran `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `paper/`.
- Output remains 19 pages.
- No undefined references/citations.
- Only two tiny overfull boxes (`0.61pt`, `0.55pt`), both below the 10pt quality threshold.
