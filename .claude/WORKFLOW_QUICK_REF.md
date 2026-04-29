# Workflow Quick Reference

**Model:** Contractor (you direct, Claude orchestrates)

---

## The Loop

```
Your instruction
    ↓
[PLAN] (if multi-file or unclear) → Show plan → Your approval
    ↓
[EXECUTE] Implement, verify, done
    ↓
[REPORT] Summary + what's ready
    ↓
Repeat
```

---

## I Ask You When

- **Design forks:** "Option A (fast) vs. Option B (robust). Which?"
- **Code ambiguity:** "Spec unclear on X. Assume Y?"
- **Replication edge case:** "Just missed tolerance. Investigate?"
- **Scope question:** "Also refactor Y while here, or focus on X?"

---

## I Just Execute When

- Code fix is obvious (bug, pattern application)
- Verification (tolerance checks, tests, compilation)
- Documentation (logs, commits)
- Plotting (per established standards)
- Deployment (after you approve, I ship automatically)

---

## Quality Gates (No Exceptions)

| Score | Action |
|-------|--------|
| >= 80 | Ready to commit |
| < 80  | Fix blocking issues |

---

## Non-Negotiables

- **Paths.** Relative-from-repo-root in Python (no `/Users/…` in code).
  XeLaTeX uses `TEXINPUTS=../../Preambles` from `manuscript/<target>/`.
  Data paths read from `data/.paths.local` via `code/data/paths.py`.
- **Seeds.** Single `monai.utils.set_determinism(seed=...)` at the top of
  every entry-point script. Recorded in the run card. No reseeding in loops.
- **Run cards.** Every training / evaluation invocation has a Markdown card
  under `experiments/runs/YYYY-MM-DD_HHMM_<slug>.md` written *before* the run
  starts and updated *after* it ends. No card → not a real run.
- **Figures.** 300 DPI minimum, white background (publication-ready).
  Project palette: TBD — to be locked in alongside the first paper-figure
  draft (deferred from this adaptation).
- **Tolerance thresholds.** TBD — to be locked in after the first VAE
  HU-fidelity sanity check. Current anchor: ProDM-style ε=1e-7 for FP32,
  1e-4 for FP16; PSNR/SSIM target deltas TBD.
- **Working language.** English in `.claude/`, `CLAUDE.md`, `manuscript/`,
  code docstrings. 中文 + English terms welcome in user-authored internal
  artefacts.

---

## Preferences

- **Visual:** publication-ready always. Drafts use the same theme + DPI as
  the final, so figure quality never has to be "fixed up later".
- **Reporting:** concise bullets first; expand to prose only if asked.
  No trailing summary paragraph after a successful task — the diff and
  log entry already say what changed.
- **Session logs:** always (post-plan, incremental at decision points,
  end-of-session). Bilingual register OK.
- **Replication tolerance:** strict — flag any near-miss above 1e-6 on
  point estimates. Defer to `.claude/rules/replication-protocol.md` once a
  numeric pipeline exists.
- **Decision cadence:** for the first ~5 sessions, check in more often so
  the user can learn the workflow's rhythm. After that, lean autonomous —
  surface only ambiguities and design forks, not routine confirmations.

---

## Exploration Mode

For experimental work, use the **Fast-Track** workflow:
- Work in `explorations/` folder
- 60/100 quality threshold (vs. 80/100 for production)
- No plan needed — just a research value check (2 min)
- See `.claude/rules/exploration-fast-track.md`

---

## Next Step

You provide task → I plan (if needed) → Your approval → Execute → Done.
