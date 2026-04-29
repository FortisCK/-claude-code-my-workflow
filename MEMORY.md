# Project memory — Cardiac CT Motion-Artifact Correction

Cross-session learnings (corrections, validated approaches, locked-in
decisions). When something is corrected or a non-obvious approach is
validated, append a `[LEARN:category]` entry below.

The upstream **template** development log (v1.5.x → v1.8.0 cycle lessons)
lives at [`.claude/archive/MEMORY-template.md`](.claude/archive/MEMORY-template.md).
It's reference material only — not load-bearing for this project.

> **Bilingual note.** Entries may be in English or 中文 + English terms.
> Pick whichever expresses the lesson more precisely. Anything that ships
> externally stays in English; learnings are internal.

---

<!-- Append new entries below. Most recent at the bottom. -->

## Project framing & strategic constraints

[LEARN:project] **单 A6000 算力预算 (48 GB)。** Latent-space diffusion 是默认路径
(VAE 把 256³ volume 压到 ~32×32×16);pixel-space cropped ROI (128×128×96)
is the explicit fallback if VAE reconstruction fidelity fails the HU-preservation
sanity check. Why: `simple_plan` §08 risk row 1 — VAE reconstruction error
exceeding motion-artifact magnitude collapses the whole latent route.
How to apply: every architecture-level decision should preserve the option
to retreat to pixel-space ROI; never lock in VAE-specific assumptions
(e.g., specific latent shapes hard-coded into downstream losses).

[LEARN:framing] **Three orthogonal novelty claims, each independently
defensible.** (1) gated-CCTA + latent-LDM + lumen-aware loss; (2) theoretically
grounded posterior-sampling UQ; (3) downstream-task-aware evaluation
(stenosis grading, TAVI landmark accuracy). Why: a single novelty claim is
fragile against one skeptical reviewer; three orthogonal claims create
redundancy. How to apply: when ablating or simplifying, never collapse two
novelty claims into one — the redundancy IS the contribution.

[LEARN:scope] **C2F-MC (Wang & Tamir 2025) is Path B; we are Path A.** They
exploit k-space linear forward operator (`y = M·F·S·Φ·x`) for non-rigid
cardiac MRI motion correction via DPS. CT image-domain has no equivalent
linear handle; ImageCAS provides only reconstructed DICOM. Why: this is the
elegant scope-cut paragraph that protects the manuscript intro from
"why don't you do posterior sampling on raw measurements?" reviews.
How to apply: cite C2F-MC explicitly in the intro; offer Path-B-lite
(image-domain pseudo-forward) as an ablation subsection; never claim our
image-domain method is theoretically optimal — only that it's the
realistic CT image-domain approach for this dataset class.

[LEARN:scope] **HM-EDM is a baseline template, not an unbeatable SOTA.**
Brain SSIM 0.17 → 0.51 with 100 cases and a 5-case reader study —
workshop-quality proof-of-concept. Their 128×128×50 patch + EDM σ-schedule
+ channel-concat conditioning runs on A100 32 GB; our A6000 48 GB has
headroom. Why: avoids over-emphasising HM-EDM as a generalisation gate.
How to apply: reproduce HM-EDM in cardiac as a method-level baseline
(showing it fails on cardiac geometry); do not credit it with cardiac
generalisation we haven't seen.

[LEARN:scope] **ProDM (Gong et al., Dec 2025) does NOT overlap our scope.**
They do non-contrast COCA Agatston scoring; their motion engine *only moves
calcium points* (inpaint → reinsert per projection angle → Radon). We do
contrast CCTA + lumen/stenosis with PAD whole-heart motion. Why: ProDM
appearing in arXiv created scooping panic — full-text read showed
methodological non-overlap. How to apply: cite ProDM, borrow their
task-driven differentiable-loss philosophy (Agatston-surrogate → our
lumen-geometry-aware loss), but do not concede scope overlap.

## Workflow conventions

[LEARN:workflow] **Bilingual policy.** English in `.claude/`, `CLAUDE.md`,
`manuscript/`, code docstrings, and external artefacts. 中文 + English terms
welcome in user-authored specs / plans / session logs / `[LEARN]` entries
/ run-card intent paragraphs / decision records. Why: matches the user's
natural register (per `simple_plan.html`) without breaking the workflow's
portability. How to apply: when authoring an internal artefact, default
to whichever language expresses the idea most precisely.

[LEARN:reproducibility] **No run card → not a run.** Every training or
evaluation invocation produces a Markdown run card under
`experiments/runs/YYYY-MM-DD_HHMM_<slug>.md`, written *before* training
starts (intent), updated *after* (outcome). Why: forces falsifiable
expectations to be locked in before seeing the loss curve — same logic as
preregistration. How to apply: see
[`.claude/rules/experiments-protocol.md`](.claude/rules/experiments-protocol.md);
`python-reviewer` flags training scripts whose docstrings don't reference a
run-card path.

[LEARN:workflow] **Plan-first for non-trivial tasks.** Multi-file changes,
ambiguous requirements, anything > 1 hour → enter plan mode → save plan to
`quality_reports/plans/YYYY-MM-DD_<short>.md` → wait for approval → execute.
Why: catches mid-plan pivots before they cost real time. How to apply:
covered in [`.claude/rules/plan-first-workflow.md`](.claude/rules/plan-first-workflow.md);
the very first session of this project (workflow adaptation) followed it.

## Tooling expectations

[LEARN:tools] **Active surface is ~16 skills + 7 agents.** Lecture-/R-flavoured
template surface lives under `.claude/archive/`. Restoring an archived
skill / agent / rule is a single `git mv`; see
[`.claude/archive/README.md`](.claude/archive/README.md).
How to apply: if a missing capability surfaces during a task, check
`.claude/archive/` before authoring something new from scratch.
