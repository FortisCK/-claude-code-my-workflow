---
paths:
  - "paper/**/*.tex"
  - "figs/**/*"
---

# Task Completion Verification Protocol

**At the end of EVERY task, Claude MUST verify the output works correctly.** This is non-negotiable.

## For LaTeX Paper (.tex):
1. Compile with pdflatex 3-pass + bibtex:
   ```bash
   cd paper && pdflatex -interaction=nonstopmode main.tex
   BIBINPUTS=..:$BIBINPUTS bibtex main
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode main.tex
   ```
2. Check for errors in the log output
3. Grep for `Overfull \\hbox` warnings
4. Grep for `undefined citations` or `Label(s) may have changed`
5. Verify PDF was produced with correct page count
6. Open PDF for visual verification: `open paper/main.pdf`

## For Figures:
1. Verify all `\includegraphics` references point to existing files in `figs/`
2. Check figure files are non-zero size
3. Check format is appropriate (PDF for vector, PNG for raster)
4. Verify figures are referenced in the paper text

## For Python Scripts (.py):
1. Run `python3 scripts/python/filename.py`
2. Verify output figures were created in `figs/` with non-zero size
3. Check for correct format and resolution

## For Bibliography:
1. Verify all `\cite{}` keys in paper have entries in `Bibliography_base.bib`
2. Check for orphan bibliography entries (entries not cited in paper)
3. Verify no "undefined citation" warnings in LaTeX log

## Common Pitfalls:
- **Wrong compiler**: use `pdflatex` (not xelatex) for ieeecolor/IEEEtran
- **Missing BIBINPUTS**: bibtex needs `BIBINPUTS=..:$BIBINPUTS` since `.bib` is in repo root
- **Figure path mismatch**: `\includegraphics` paths must be relative to `paper/` directory
- **Assuming success**: always verify output files exist AND contain correct content

## Verification Checklist:
```
[ ] Paper compiles without errors
[ ] No overfull hbox warnings > 10pt
[ ] All citations resolve
[ ] All figure references resolve
[ ] All cross-references resolve
[ ] PDF produced at expected page count
[ ] Opened in viewer to confirm visual appearance
[ ] Reported results to user
```
