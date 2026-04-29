---
paths:
  - "code/training/**/*.py"
  - "code/evaluation/**/*.py"
  - "experiments/**/*.md"
---

# Experiments Protocol — Run-card discipline

Every training run produces a Markdown **run card** under `experiments/runs/`.
The card is created **before** training starts (intent) and updated **after**
training ends (outcome). A run without a card is not a valid experiment and
its results may not appear in the manuscript.

This rule is the experimental analogue of the manuscript-level preregistration
in [`../skills/preregister/SKILL.md`](../skills/preregister/SKILL.md).

---

## When this fires

- A new file appears under `code/training/*.py` or `code/evaluation/*.py`,
  or an existing one is modified in a way that changes its behaviour.
- A new file appears under `experiments/runs/`.
- The user invokes `/review-python` on a training or evaluation script.

The `python-reviewer` agent reads this rule and flags training scripts whose
top-of-file docstring does not reference a run-card path.

## Filename convention

```
experiments/runs/YYYY-MM-DD_HHMM_<short-slug>.md
```

The slug is 2–4 words, kebab-case, summarising the run. The timestamp keeps
`ls -1` chronological. Examples:

- `experiments/runs/2026-05-04_1630_vae-baseline-imagecas.md`
- `experiments/runs/2026-06-12_0900_ldm-channelconcat-128.md`
- `experiments/runs/2026-08-03_1415_posterior-sampling-N16.md`

## Schema (required fields)

| Field                | When filled    | Notes                                                                    |
| -------------------- | -------------- | ------------------------------------------------------------------------ |
| Date                 | Before         | YYYY-MM-DD HH:MM with timezone                                           |
| Author               | Before         | initials                                                                  |
| Git SHA              | Before         | output of `git rev-parse HEAD` at run start                              |
| Config               | Before         | path to the YAML config used                                              |
| Seed                 | Before         | the integer passed to `set_determinism`                                  |
| Dataset version      | Before         | e.g., `ImageCAS-v1 (n=1000)` — bumps when preprocessing changes          |
| Planned GPU-hours    | Before         | rough estimate; informs whether to interrupt other work                  |
| Status               | Both           | `PLANNED` → `RUNNING` → `DONE`/`ABORTED`                                  |
| Intent (one para)    | Before         | falsifiable expectation. If you can't write one, mark the run "EXPLORATORY" |
| Hyperparameters      | Before         | delta from previous best (don't restate the whole config)                |
| Actual GPU-hours     | After          | from WandB or `time` output                                              |
| Final metrics        | After          | the headline numbers (PSNR / SSIM / ECE / downstream Dice)               |
| WandB link           | After          | so future-you can check learning curves                                  |
| What happened        | After          | 2–3 paragraphs, honest                                                    |
| Decision             | After          | `KEEP` / `DISCARD` / `ITERATE` and why                                   |
| Cross-references     | After          | manuscript sections + tables that depend on this run; predecessor / successor runs |

A boilerplate template is in [`../../experiments/README.md`](../../experiments/README.md).

## Why this discipline

Without a written intent, post-hoc rationalisation is unavoidable. The
falsification criterion has to be locked in before you see the loss curve.

This is the same logic as preregistration, applied at experiment granularity.
For one-off proof-of-concept runs (e.g., "does this dataloader even
produce paired tensors?"), mark the card `EXPLORATORY` — a single line of
intent is enough, but the card must still exist.

## What counts as a "run"

| Counts                                            | Doesn't count                          |
| ------------------------------------------------- | -------------------------------------- |
| Full training of any model, even short            | Smoke tests `< 10 seconds`             |
| Hyperparameter sweeps (one card per leaf)         | Static unit tests                      |
| Eval pipelines that produce paper-cited numbers   | Notebook scratch                       |
| Inference / posterior-sampling pipelines          | Pure refactors that run the same logic |
| `python-reviewer` audit runs                       | One-shot data-stats prints              |

When in doubt, write the card.

## Cross-references

- [`cross-artifact-review.md`](cross-artifact-review.md) — every numeric
  manuscript claim traces to a run card via a `% source:` comment.
- [`replication-protocol.md`](replication-protocol.md) — the tolerance
  contract that `/audit-reproducibility` enforces.
- [`python-code-conventions.md`](python-code-conventions.md) — code-side
  reproducibility rules that pair with the run-card discipline.
- [`../skills/preregister/SKILL.md`](../skills/preregister/SKILL.md) —
  paper-level preregistration analogue.
