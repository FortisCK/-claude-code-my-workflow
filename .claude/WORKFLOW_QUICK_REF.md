# Workflow Quick Reference

**Project:** CATHACTION, MICCAI 2026 participant submission
**Institution:** Université de Rennes
**Mode:** Plan first, then contractor execution after approval

---

## The Loop

```
Task request
    ↓
[PLAN] for non-trivial work -> save to quality_reports/plans/ -> user approval
    ↓
[EXECUTE] implement autonomously, with early-session check-ins
    ↓
[VERIFY] tests / metrics / Docker / render / visual checks
    ↓
[REPORT] concise summary, evidence, residual risks
```

---

## I Ask You When

- **Scientific ambiguity:** metric definition, label interpretation, split protocol, external data policy.
- **Design fork:** fast baseline vs. robust challenge-grade implementation.
- **Clinical-safety claim:** wording could overstate what fluoroscopy-only inference supports.
- **Submission risk:** Docker or output-schema choice could conflict with official platform instructions.
- **Scope change:** a refactor, new model family, or new experiment would move beyond the approved plan.

---

## I Just Execute When

- Applying approved plan steps.
- Running verification, smoke tests, and quality scans.
- Updating logs, specs, decision records, and memory.
- Fixing obvious documentation drift or placeholder text.
- Polishing visuals to the established standard.

---

## Quality Thresholds

| Score | Action |
|-------|--------|
| >= 95 | Publication/spotlight-ready |
| >= 90 | Submission/PR-ready |
| >= 80 | Commit-ready |
| < 80 | Fix blocking issues before saving as complete |

---

## Non-Negotiables

- **Paths:** repository-relative paths only; no hardcoded `/Users/...`, `~`, or machine-specific dataset paths.
- **Splits:** train/validation/test splits are case/procedure-level; no frame-level leakage across cases.
- **Metrics:** Task 1 ranks by DSC; Task 2 ranks by mAP. Secondary metrics must not be presented as primary unless explicitly stated.
- **External data:** only challenge data plus publicly available data/pretrained models allowed; no private/proprietary clinical data.
- **Submission:** inference must be automatable without user interaction and packaged toward Docker.
- **Hidden test:** never hand-tune, reverse-engineer, or manually curate hidden-test predictions.
- **Visuals:** paper and presentation visuals must be publication-ready, labeled, reproducible, and not misleading about domain or sample.
- **Memory:** corrections become `[LEARN:category] wrong -> right` entries in `MEMORY.md`.

---

## Preferences

**Visual:** polished, publication-ready, vector-first where possible; diagrams should carry clinical and metric meaning, not decoration.
**Reporting:** precise, structured, and rigorous; concise summaries with enough evidence to audit.
**Check-ins:** more frequent during the first sessions so the workflow is transparent.
**Session logs:** always after plan approval, incrementally after decisions, and at wrap-up.
**Reproducibility:** strict; flag near-misses, split drift, non-determinism, and missing provenance early.

---

## Exploration Mode

For experimental model ideas, use the fast-track workflow only inside `explorations/`:

- State the research value first.
- Keep artifacts clearly marked as sandbox outputs.
- Do not promote results into the method report without reproduction and split checks.
- See `.claude/rules/exploration-fast-track.md`.
