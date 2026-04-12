---
paths:
  - "Thesis/**/*.tex"
  - "quality_reports/**"
---

# Quality Gates & Scoring Rubrics

## Thresholds

- **80/100 = Commit** -- evaluation report good enough to save
- **90/100 = PR** -- ready for advisor review / sharing with student
- **95/100 = Excellence** -- publishable-quality feedback

## Thesis Evaluation Report Quality

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Missing entire evaluation dimension | -20 |
| Critical | Fabricated or hallucinated claim about thesis | -30 |
| Critical | Wrong attribution (crediting wrong paper) | -15 |
| Major | Vague criticism without specific location | -5 |
| Major | Missing constructive suggestion for a raised issue | -5 |
| Major | Inconsistent severity classification | -3 |
| Minor | Redundant points across sections | -1 |
| Minor | Missing acknowledgment of thesis strengths | -2 |

## Thesis Content Quality (what we evaluate in the thesis)

| Dimension | Weight | Key Criteria |
|-----------|--------|-------------|
| Problem & Motivation | 15% | Clear research question, well-motivated, novel |
| Technical Approach | 25% | Sound methodology, correct proofs/algorithms |
| Experimental Design | 25% | Fair baselines, proper metrics, reproducibility |
| Literature Review | 15% | Complete, accurate, well-positioned |
| Writing Quality | 10% | Clear, consistent notation, no errors |
| Presentation | 10% | Figures, tables, formatting, length |

## Enforcement

- **Score < 80:** Block commit. List blocking issues in the evaluation report.
- **Score < 90:** Allow commit, warn. List recommendations.
- User can override with justification.

## Quality Reports

Generated **only at merge time**. Use `templates/quality-report.md` for format.
Save to `quality_reports/merges/YYYY-MM-DD_[branch-name].md`.
