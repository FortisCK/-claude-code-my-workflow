# Workflow Quick Reference

**Model:** Contractor (you direct, Claude orchestrates)
**Project:** Master's Thesis Evaluation — CS/AI

---

## The Loop

```
Your instruction
    ↓
[PLAN] (if multi-chapter or unclear) → Show plan → Your approval
    ↓
[EXECUTE] Read thesis, evaluate, produce report
    ↓
[REPORT] Summary + findings + next steps
    ↓
Repeat
```

---

## I Ask You When

- **Evaluation scope:** "Full thesis or specific chapter?"
- **Ambiguous content:** "The student claims X but it's unclear if Y. Flag it or investigate?"
- **Severity judgment:** "This could be CRITICAL or MAJOR — how strict for blind review?"
- **Scope question:** "Also check the bibliography in depth, or focus on methodology?"
- **Student communication:** "Should this feedback be direct or softened?"

---

## I Just Execute When

- Reading and analyzing thesis content
- Producing evaluation reports
- Proofreading (grammar, typos, notation)
- Citation cross-referencing
- Literature gap analysis
- Session logging and documentation

---

## Quality Gates (No Exceptions)

| Score | Action |
|-------|--------|
| >= 80 | Evaluation report ready to commit |
| < 80  | Fix gaps in evaluation report |

---

## Non-Negotiables

- **Read-only thesis source** — never edit `Thesis/*.tex` without explicit permission
- **Specific locations** — every issue cites chapter, section, page, or equation number
- **Constructive tone** — every criticism includes a concrete, actionable suggestion
- **No fabrication** — never claim the thesis says something without reading it
- **Acknowledge strengths** — every review includes positive findings
- **Blind-review calibration** — feedback targets what a reviewer at a top CS/AI venue would flag

---

## Preferences

**Reporting:** Structured markdown reports with severity levels; concise summaries at top
**Session logs:** Always (post-plan, incremental, end-of-session)
**Evaluation depth:** Thorough — better to be too detailed than too shallow
**Tone:** Professional, constructive, calibrated for a master's thesis (not a journal paper)

---

## Exploration Mode

For experimental analysis (e.g., reproducing thesis results, testing claims):
- Work in `explorations/` folder
- 60/100 quality threshold (vs. 80/100 for production)
- No plan needed — just a research value check
- See `.claude/rules/exploration-fast-track.md`

---

## Next Step

You provide task → I plan (if needed) → Your approval → Read thesis → Evaluate → Report → Done.
