---
name: proofread
description: Run the proofreading protocol on paper sections. Checks grammar, typos, overflow, consistency, and academic writing quality. Produces a report without editing files.
disable-model-invocation: true
argument-hint: "[filename or 'all']"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Task"]
---

# Proofread Paper Sections

Run the mandatory proofreading protocol on paper files. This produces a report of all issues found WITHOUT editing any source files.

## Steps

1. **Identify files to review:**
   - If `$ARGUMENTS` is a specific filename: review that file only
   - If `$ARGUMENTS` is "all": review all `.tex` files in `paper/`

2. **For each file, launch the proofreader agent** that checks for:

   **GRAMMAR:** Subject-verb agreement, articles (a/an/the), prepositions, tense consistency
   **TYPOS:** Misspellings, search-and-replace artifacts, duplicated words
   **OVERFLOW:** Overfull hbox (LaTeX), content exceeding column boundaries
   **CONSISTENCY:** Citation format (`\cite{}`), notation, terminology (see knowledge-base-template.md)
   **ACADEMIC QUALITY:** Informal language, missing words, awkward constructions
   **IEEE TMI STYLE:** Proper use of `\IEEEPARstart`, section numbering, reference format

3. **Produce a detailed report** for each file listing every finding with:
   - Location (line number or section)
   - Current text (what's wrong)
   - Proposed fix (what it should be)
   - Category and severity

4. **Save each report** to `quality_reports/FILENAME_report.md`

5. **IMPORTANT: Do NOT edit any source files.**
   Only produce the report. Fixes are applied separately after user review.

6. **Present summary** to the user:
   - Total issues found per file
   - Breakdown by category
   - Most critical issues highlighted
