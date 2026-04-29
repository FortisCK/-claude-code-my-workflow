# <slug — 2-4 word kebab-case>

**Status:** PLANNED | RUNNING | DONE | ABORTED
**Date:** YYYY-MM-DD HH:MM <timezone>
**Author:** <initials>
**Git SHA:** <`git rev-parse HEAD` at run start>
**Config:** `code/training/configs/<config>.yaml`(or `n/a`)
**Seed:** <integer passed to `set_determinism`>
**Dataset version:** e.g., `ImageCAS-v1 (n=1000)`
**Planned GPU-hours:** <rough estimate>

---

## Intent

One paragraph: **what hypothesis does this run test, and what observation
would falsify it?** If you cannot articulate a falsifiable expectation,
mark this run `EXPLORATORY` (a single line of intent is enough,
but the card must still exist).

## Hyperparameters (delta from previous best)

- batch size: ...
- learning rate: ...
- epochs / steps: ...
- ...

(Don't restate the whole config — point at it via `Config` field above and
list only what's different from prior runs.)

---

## Outcome (filled after run)

**Actual GPU-hours:** <from WandB or `time` output>
**Final metrics:** PSNR x.x, SSIM 0.xx, ... (test fold)
**WandB:** <link>

### What happened

2–3 paragraphs.  Be honest about negative or null results.

### Decision

**KEEP / DISCARD / ITERATE** — and why.

### Cross-references

- Manuscript claims that depend on this run: <§ / Table / Figure>
- Predecessor run: `<timestamp_slug.md>` (or `n/a`)
- Successor (if any): `<timestamp_slug.md>` (or `TBD`)
- Related decision record: `<path>` (or `n/a`)
