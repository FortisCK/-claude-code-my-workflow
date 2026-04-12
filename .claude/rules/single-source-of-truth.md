---
paths:
  - "Thesis/**/*.tex"
  - "quality_reports/**"
---

# Single Source of Truth: Thesis Evaluation Protocol

**The student's thesis `.tex` files in `Thesis/` are READ-ONLY.** Our output is evaluation reports, not edits.

## The Source Chain

```
Thesis/*.tex (STUDENT'S SOURCE — READ-ONLY)
  ├── quality_reports/thesis_evaluation/  (our evaluation output)
  ├── quality_reports/session_logs/       (session tracking)
  ├── Bibliography_base.bib              (reference bibliography)
  └── master_supporting_docs/            (reference papers)

NEVER edit thesis source files without explicit advisor permission.
ALWAYS produce evaluation reports as separate documents.
```

---

## Read-Only Enforcement

1. Thesis .tex files are protected by `.claude/hooks/protect-files.sh`
2. If a source edit is needed (e.g., to test a fix), ask the advisor first
3. Evaluation reports go to `quality_reports/thesis_evaluation/`
4. Proofreading reports go to `quality_reports/`

---

## Evaluation Report Standards

Each evaluation report must:
- Reference specific thesis locations (chapter, section, page, equation number)
- Classify issues by severity (CRITICAL / MAJOR / MINOR)
- Provide constructive suggestions for every issue raised
- Acknowledge thesis strengths, not just weaknesses
- Be suitable for sharing with the student

---

## Citation Cross-Reference

When checking thesis citations:
1. Read the thesis bibliography (.bib file in `Thesis/`)
2. Cross-reference with `Bibliography_base.bib` (our reference copy)
3. Check papers in `master_supporting_docs/supporting_papers/` for accuracy
4. Flag missing key references, miscited results, or attribution errors
