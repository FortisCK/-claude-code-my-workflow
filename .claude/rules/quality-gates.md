---
paths:
  - "paper/**/*.tex"
  - "scripts/python/**/*.py"
---

# Quality Gates & Scoring Rubrics

## Thresholds

- **80/100 = Commit** -- good enough to save
- **90/100 = PR** -- ready for review
- **95/100 = Excellence** -- aspirational

## Paper (.tex)

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Compilation failure | -100 |
| Critical | Undefined citation | -15 |
| Critical | Broken figure reference | -15 |
| Critical | Equation error / typo in math | -10 |
| Major | Overfull hbox > 10pt | -5 |
| Major | Notation inconsistency | -3 |
| Major | Inconsistent terminology | -3 |
| Major | Missing figure caption | -3 |
| Minor | Table formatting issue | -1 |
| Minor | Long lines (>100 chars) | -1 (EXCEPT math formulas) |
| Minor | Minor style deviation from IEEE TMI | -1 |

## Python Scripts (.py)

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Syntax error | -100 |
| Critical | Missing imports | -10 |
| Major | No reproducibility seed | -5 |
| Major | Hardcoded absolute paths | -5 |
| Major | Missing figure output | -5 |
| Minor | Missing type hints | -1 |
| Minor | Style violations (PEP 8) | -1 |

## Enforcement

- **Score < 80:** Block commit. List blocking issues.
- **Score < 90:** Allow commit, warn. List recommendations.
- User can override with justification.

## Quality Reports

Generated **only at merge time**. Use `templates/quality-report.md` for format.
Save to `quality_reports/merges/YYYY-MM-DD_[branch-name].md`.

## Tolerance Thresholds (Landmark Detection)

| Quantity | Tolerance | Rationale |
|----------|-----------|-----------|
| MRE (mm) | Report to 2 decimal places | Standard in landmark detection |
| SDR (%) | Report to 1 decimal place | Percentage precision |
| Training loss | Convergence trend, not exact value | Stochastic optimization |
| Inference time (ms) | +/- 5% | Hardware variability |
