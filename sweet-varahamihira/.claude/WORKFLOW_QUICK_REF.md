# Workflow Quick Reference

**Model:** Contractor (you direct, Claude orchestrates)

---

## The Loop

```
Your instruction
    |
[PLAN] (if multi-file or unclear) -> Show plan -> Your approval
    |
[EXECUTE] Implement, verify, done
    |
[REPORT] Summary + what's ready
    |
Repeat
```

---

## I Ask You When

- **Design forks:** "Option A (fast) vs. Option B (robust). Which?"
- **Paper ambiguity:** "Spec unclear on X. Assume Y?"
- **Metric edge case:** "SDR threshold mismatch. Investigate?"
- **Scope question:** "Also refactor Y while here, or focus on X?"

---

## I Just Execute When

- Code fix is obvious (bug, pattern application)
- Verification (compilation, figure checks, bib validation)
- Documentation (logs, commits)
- Figure generation (per established standards)

---

## Quality Gates (No Exceptions)

| Score | Action |
|-------|--------|
| >= 80 | Ready to commit |
| < 80  | Fix blocking issues |

---

## Non-Negotiables

- **Paths:** Relative paths from repo root for all files; `../Bibliography_base.bib` from `paper/`
- **Seeds:** Random seeds set explicitly in Python scripts for reproducibility (`random.seed()`, `np.random.seed()`, `torch.manual_seed()`)
- **Figure standards:** 300 DPI minimum, PDF/EPS for vector, PNG for raster, `bbox_inches='tight'`
- **Color palette:** Colorblind-safe palette for all figures (use `seaborn` colorblind or custom accessible palette)
- **Tolerance thresholds:** MRE in mm (report to 2 decimal places), SDR at 2mm/3mm/4mm thresholds

---

## Preferences

**Visual:** Publication-ready; IEEE TMI single-column (3.5in) or double-column (7.16in) widths
**Reporting:** Concise bullets, details on request
**Session logs:** Always (post-plan, incremental, end-of-session)
**Reproducibility:** Strict -- all results must be traceable to specific configurations

---

## Exploration Mode

For experimental work, use the **Fast-Track** workflow:
- Work in `explorations/` folder
- 60/100 quality threshold (vs. 80/100 for production)
- No plan needed -- just a research value check
- See `.claude/rules/exploration-fast-track.md`

---

## Next Step

You provide task -> I plan (if needed) -> Your approval -> Execute -> Done.
