---
paths:
  - "Thesis/**/*.tex"
  - "quality_reports/**"
---

# Proofreading Protocol for Thesis Evaluation

**Proofreading produces an evaluation report — it does NOT directly edit the thesis.**

## What to Check

1. **Grammar** -- subject-verb agreement, missing articles, wrong prepositions, tense consistency
2. **Typos** -- misspellings, duplicated words, search-and-replace artifacts
3. **Notation** -- consistent use of symbols, variable names, mathematical conventions throughout
4. **Citations** -- consistent citation style, correct use of \cite/\citet/\citep, all references resolved
5. **Academic quality** -- informal language, missing hedging, overclaiming, vague statements
6. **Formatting** -- overfull hbox, orphaned headings, broken cross-references, table/figure numbering
7. **LaTeX quality** -- undefined commands, package conflicts, compilation warnings

## Workflow

### Phase 1: Review & Report (NO EDITS to thesis)

1. Read the entire thesis file (or specified chapter)
2. Produce a **proofreading report** with every proposed correction:
   - Location (chapter, section, page, line if possible)
   - Current text (quoted)
   - Proposed fix
   - Category (grammar / typo / notation / citation / formatting / style)
3. Save report to `quality_reports/thesis_evaluation/proofreading_[chapter].md`
4. **Do NOT modify any thesis source files**

### Phase 2: Advisor Review

The advisor reviews the proofreading report and decides:
- Which corrections to share with the student
- Which are stylistic preferences vs. genuine errors
- Priority ordering for the student

### Phase 3: Apply Fixes (ONLY with explicit permission)

If the advisor explicitly authorizes direct edits:
- Apply only approved changes
- Use Edit tool with care
- Verify each edit succeeded
- Report completion summary
