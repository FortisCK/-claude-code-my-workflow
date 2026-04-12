---
paths:
  - "Thesis/**/*.tex"
  - "quality_reports/**"
---

# Task Completion Verification Protocol

**At the end of EVERY task, Claude MUST verify the output is correct.** This is non-negotiable.

## For Thesis Evaluation Reports:
1. Verify the report covers all required evaluation dimensions
2. Check that specific thesis locations (chapter, section, page, equation) are cited for each issue
3. Verify severity classifications are consistent (CRITICAL / MAJOR / MINOR)
4. Confirm constructive suggestions accompany every criticism
5. Save to `quality_reports/thesis_evaluation/`

## For Thesis Compilation Checks:
1. Run `cd Thesis && latexmk -xelatex -interaction=nonstopmode main.tex` (or the appropriate entry file)
2. Check for compilation errors and undefined references
3. Verify bibliography resolves (no missing citations)
4. Report any overfull hbox warnings

## For Proofreading Reports:
1. Ensure every issue has: location, current text, suggested fix, category
2. Verify suggestions are grammatically correct themselves
3. Report saved to `quality_reports/`

## For Literature Analysis:
1. Verify cited papers exist and are correctly attributed
2. Cross-reference with `Bibliography_base.bib` and `master_supporting_docs/`
3. Spot-check DOIs or titles for accuracy

## For Code/Script Reviews (if thesis includes code):
1. Run scripts to verify they execute without errors
2. Check reproducibility (set.seed, fixed random states)
3. Verify output files are created at expected paths

## Common Pitfalls:
- **Fabricating details**: Never claim a thesis says something without reading it
- **Vague feedback**: Always cite specific locations (Ch. 3, Sec. 3.2, Eq. 7, Table 4)
- **Missing context**: Read surrounding sections before flagging an issue
- **Overlooking strengths**: Every review should acknowledge what the thesis does well

## Verification Checklist:
```
[ ] Output report is complete and saved
[ ] Every issue cites a specific thesis location
[ ] Severity levels are appropriate and consistent
[ ] Constructive suggestions provided
[ ] No fabricated claims or hallucinated content
[ ] Reported results to user
```
