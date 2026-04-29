# `experiments/` — Training runs ledger

The single ledger of training runs. Every training run has a Markdown run
card under `experiments/runs/`, written **before** training starts (intent)
and updated **after** training ends (outcome).

> 一条训练 run 没有对应的 run card,就不算 run 过 — 它的结果不能进 paper。

This rule is enforced by [`.claude/rules/experiments-protocol.md`](../.claude/rules/experiments-protocol.md)
and surfaced by `python-reviewer` when reviewing training scripts.

## Run-card filename

```
experiments/runs/YYYY-MM-DD_HHMM_<short-slug>.md
```

Examples:
- `experiments/runs/2026-05-04_1630_vae-baseline-imagecas.md`
- `experiments/runs/2026-06-12_0900_ldm-channelconcat-128.md`

The slug is a 2–4 word summary; the timestamp keeps `ls` chronological.

## Run-card schema

A run card has two sections: **Intent** (filled before training) and
**Outcome** (filled after). Both fields are required at their respective times.

```markdown
# <slug>

**Date:** 2026-05-04 16:30 CET
**Author:** CMZ
**Git SHA:** <hash at training start>
**Config:** `code/training/configs/<config>.yaml`
**Seed:** 42
**Dataset version:** ImageCAS-v1 (n=1000)
**Planned GPU-hours:** ~14h on A6000
**Status:** PLANNED | RUNNING | DONE | ABORTED

## Intent

One paragraph: what hypothesis does this run test, and what observation
would falsify it? If you cannot articulate a falsifiable expectation,
the run is exploratory — say so explicitly.

## Hyperparameters (delta from previous best)

- batch size: 4 (was 2)
- learning rate: 2e-4 (unchanged)
- ...

## Outcome

**Actual GPU-hours:** 13.7h
**Final metrics:** PSNR 30.2, SSIM 0.81, LPIPS 0.09 (test fold A)
**WandB:** <link>

### What happened

2–3 paragraphs. Be honest about negative or null results.

### Decision

KEEP | DISCARD | ITERATE — and why.

### Cross-references

- Manuscript claims that depend on this run: §4.2 Table 1 row "Ours-LDM".
- Predecessor run: `2026-05-02_..._vae-pixelspace-baseline.md`
- Successor (if any): TBD.
```

A template version of this schema lives at
[`../templates/experiment-run-card.md`](../templates/experiment-run-card.md)
once it's added (TODO — not in the current adaptation plan).

## Why this discipline

Without a written intent, you can't tell post-hoc whether a run "worked"
without rationalising the result. With a written intent, the falsification
criterion is locked in before you see the loss curve. Same logic as
preregistration, but at the experiment granularity. See
[`.claude/skills/preregister/SKILL.md`](../.claude/skills/preregister/SKILL.md)
for the paper-level analogue.

## Cross-references

- [`../.claude/rules/experiments-protocol.md`](../.claude/rules/experiments-protocol.md) — the rule.
- [`../.claude/rules/cross-artifact-review.md`](../.claude/rules/cross-artifact-review.md) — manuscript claims must trace back to a run card.
- [`../code/README.md`](../code/README.md) — training-script conventions.
