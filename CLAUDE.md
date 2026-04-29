# CLAUDE.md — Cardiac CT Motion-Artifact Correction

**Project:** 3D Conditional Latent-Diffusion Cardiac CT Motion-Artifact Correction
**Institution:** LTSI — Université de Rennes
**Branch:** `cardiac-artifacts`
**Strategic brief:** [`quality_reports/decisions/2026-04-29_research-direction-v2.md`](quality_reports/decisions/2026-04-29_research-direction-v2.md) (V2 of the simple_plan)
**Target venues:** MICCAI 2027 (primary), CVPR 2027 (parallel), IEEE TMI (journal-extension fallback)

---

## What this project is

A 3D conditional latent-diffusion model for **single-phase cardiac CT motion-artifact
correction**, with native posterior-sampling uncertainty and TAVI-relevant
downstream evaluation. The model is conditioned on motion-corrupted CCTA volumes
and produces (a) a posterior-mean point estimate, (b) a per-voxel uncertainty
map, and (c) a downstream-task-aware, lumen-geometry-aware evaluation protocol.

The three novelty claims, each independently defensible, are documented in
[`quality_reports/decisions/2026-04-29_research-direction-v2.md`](quality_reports/decisions/2026-04-29_research-direction-v2.md) §07.

---

## Core principles

- **Plan first.** Enter plan mode before any non-trivial task; save the plan
  to `quality_reports/plans/`. See [`.claude/rules/plan-first-workflow.md`](.claude/rules/plan-first-workflow.md).
- **Verify after.** Compile / render / sanity-check at the end of every task.
  See [`.claude/rules/verification-protocol.md`](.claude/rules/verification-protocol.md).
- **Single source of truth.** The manuscript `.tex` is authoritative for all
  numeric claims. Each numeric claim must trace to a specific
  `experiments/runs/<...>.md` run card via a `% source:` comment. See
  [`.claude/rules/cross-artifact-review.md`](.claude/rules/cross-artifact-review.md)
  and [`.claude/rules/experiments-protocol.md`](.claude/rules/experiments-protocol.md).
- **Quality gates.** Nothing ships below 80/100 (advisory; enforced by `/commit`).
  See [`.claude/rules/quality-gates.md`](.claude/rules/quality-gates.md).
- **Reproducibility is hard.** Every training run gets a run card. No card → not a real run.
- **`[LEARN]` tags.** When corrected or when an approach is validated, append
  `[LEARN:category]` to [`MEMORY.md`](MEMORY.md). See `.claude/rules/meta-governance.md`.

Cross-session context lives in [`MEMORY.md`](MEMORY.md); past plans, specs,
session logs, decision records, and run-card cross-references are in
[`quality_reports/`](quality_reports/).

---

## Working language

- **English** in `.claude/`, `CLAUDE.md`, `manuscript/`, code docstrings, and
  anything that ships externally.
- **中文 (with English technical terms)** is welcome in user-authored
  artefacts: specs, plans, session logs, MEMORY.md `[LEARN]` entries, run-card
  intent paragraphs, decision records.
- The `simple_plan.html` style — Chinese narrative, English for terms like
  *latent diffusion*, *posterior sampling*, *Dice*, *SSIM*, *ablation* — is the
  canonical bilingual register for internal docs.

---

## Folder structure

```
cardiac-artifacts/
├── CLAUDE.md                    # this file
├── MEMORY.md                    # cross-session [LEARN] entries
├── README.md                    # template-inherited (rewrite deferred)
├── Bibliography_base.bib        # shared bib for all manuscript subprojects
├── manuscript/                  # paper drafts (one folder per submission target)
│   ├── miccai2027/
│   ├── cvpr2027/
│   └── tmi-extension/
├── code/                        # PyTorch + MONAI source
│   ├── data/                    #   ImageCAS loaders, PAD wrappers, transforms
│   ├── models/                  #   VAE-KL, latent denoiser, condition modules
│   ├── training/                #   train scripts + configs
│   ├── evaluation/              #   PSNR/SSIM/UQ/downstream pipelines
│   ├── inference/               #   posterior sampling, N-sample ensembling
│   └── notebooks/               #   exploratory only
├── experiments/                 # run-card ledger
│   └── runs/                    #   one .md per training run
├── data/                        # dataset metadata only (content gitignored)
├── Figures/                     # paper figures
├── Preambles/                   # shared LaTeX header (for manuscripts)
├── Slides/                      # DORMANT (defense talks etc.)
├── Quarto/                      # DORMANT (theme-template.scss kept)
├── master_supporting_docs/      # anchor PDFs (TT U-Net, HM-EDM, ProDM, …)
├── quality_reports/
│   ├── plans/                   # plan-first artefacts
│   ├── specs/                   # spec-then-plan artefacts
│   ├── session_logs/            # post-plan + incremental + end-of-session
│   ├── decisions/               # project-defining decision records
│   ├── checkpoints/             # /checkpoint snapshots
│   ├── merges/                  # quality reports at merge time
│   └── preregistrations/        # /preregister artefacts
├── explorations/                # research sandbox (60/100 quality bar)
├── scripts/                     # repo-level utility scripts
│   └── python/                  # project-specific helpers
├── templates/                   # session-log, spec, decision-record templates
└── .claude/
    ├── archive/                 # dormant template surface (lecture / R)
    ├── skills/                  # ~16 active skills
    ├── agents/                  # 7 active agents
    ├── rules/                   # active rules
    ├── hooks/                   # pre-compact, post-compact, log-reminder, …
    ├── references/              # journal profiles, audit pet peeves, …
    ├── scripts/                 # statusline, etc.
    └── settings.json            # permissions, hooks
```

Dormant lecture / R surface (Beamer compilation, TikZ tooling, Quarto deploy,
R-pipeline analysis, slide-quality reviewers) lives under
[`.claude/archive/`](.claude/archive/) — recoverable with `git mv` if the
project ever needs it (defense talk, R reader-study).

---

## Commands

> **Note.** The project is in pre-implementation state — most commands below
> are forward-looking conventions. They activate as the corresponding
> infrastructure lands (Python install, first manuscript draft, etc.).

```bash
# --- Python (planned) ---
# Environment will be pinned via pyproject.toml + lock at first install:
python -m venv .venv && source .venv/bin/activate
pip install -e .                                # editable install of the project package
python -m code.training.train_vae --config code/training/configs/vae_v1.yaml
python -m code.evaluation.run_eval  --config code/evaluation/configs/baseline.yaml
python -m code.inference.posterior_sample      --checkpoint <path> --n-samples 16

# --- Manuscript LaTeX (XeLaTeX, 3-pass + bibtex) ---
cd manuscript/miccai2027
TEXINPUTS=../../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode main.tex
BIBINPUTS=../..:$BIBINPUTS bibtex main
TEXINPUTS=../../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode main.tex
TEXINPUTS=../../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode main.tex

# --- Run-card lifecycle ---
# Before training:
echo "experiments/runs/$(date +%Y-%m-%d_%H%M)_<slug>.md created? (required)"
# During training: status PLANNED → RUNNING → DONE/ABORTED
# After training: outcome metrics, WandB link, decision (KEEP/DISCARD/ITERATE)

# --- Repo-level utilities (kept from template) ---
python scripts/check-skill-integrity.py
bash scripts/validate-setup.sh
```

---

## Quality thresholds (advisory)

| Score | Checkpoint | Meaning                            |
| ----- | ---------- | ---------------------------------- |
| 80    | Commit     | Good enough to save                |
| 90    | PR         | Ready for arXiv / submission       |
| 95    | Excellence | Aspirational                       |

Enforced by `/commit` (halts + asks for override); not enforced by a git
pre-commit hook.

---

## Skills quick reference (active)

| Command                       | What it does                                                                |
| ----------------------------- | --------------------------------------------------------------------------- |
| `/lit-review [topic]`         | Literature search + synthesis with BibTeX-ready citations                   |
| `/research-ideation [topic]`  | Generate research questions, hypotheses, candidate empirical strategies     |
| `/interview-me [topic]`       | Interactive interview that formalises a fuzzy idea into a structured spec   |
| `/preregister [--style …]`    | Draft a preregistration document (OSF / AsPredicted / AEA RCT)              |
| `/review-paper [file]`        | Manuscript review: single-pass / `--adversarial` / `--peer <venue>`         |
| `/seven-pass-review`          | Seven-pass parallel adversarial review (forked subagents)                   |
| `/respond-to-referees [r] [m]`| R&R cross-reference + response draft                                        |
| `/verify-claims [file]`       | Chain-of-Verification (forked verifier, fresh context)                      |
| `/audit-reproducibility`      | Cross-check numeric claims in manuscript ↔ code outputs                     |
| `/proofread [file]`           | Grammar / typo / overflow / consistency over `.tex` or `.qmd`               |
| `/review-python [target]`     | Python code review (PyTorch + MONAI flavour, run-card discipline)           |
| `/checkpoint [topic]`         | Save a structured state snapshot before stopping or handing off             |
| `/learn [skill-name]`         | Extract a discovery into a persistent skill                                 |
| `/context-status`             | Show session health + context usage                                         |
| `/deep-audit`                 | Repository-wide consistency audit                                           |
| `/permission-check`           | Diagnose permission layers when prompts fire unexpectedly                   |
| `/commit [msg]`               | Stage, commit, PR, merge (quality-gate aware)                               |

Archived skills (Beamer / Quarto / R-pipeline / TikZ-aware): see
[`.claude/archive/README.md`](.claude/archive/README.md).

---

## Conventions quick reference

| Topic                              | Where the rule lives                                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Python / PyTorch / MONAI standards | [`.claude/rules/python-code-conventions.md`](.claude/rules/python-code-conventions.md)                     |
| Run-card discipline                | [`.claude/rules/experiments-protocol.md`](.claude/rules/experiments-protocol.md)                           |
| Plan-first workflow                | [`.claude/rules/plan-first-workflow.md`](.claude/rules/plan-first-workflow.md)                             |
| Spec-then-plan                     | [`.claude/rules/plan-first-workflow.md`](.claude/rules/plan-first-workflow.md) + [`templates/requirements-spec.md`](templates/requirements-spec.md) |
| Session logging                    | [`.claude/rules/session-logging.md`](.claude/rules/session-logging.md)                                     |
| Manuscript ↔ code traceability     | [`.claude/rules/cross-artifact-review.md`](.claude/rules/cross-artifact-review.md)                         |
| Replication tolerance              | [`.claude/rules/replication-protocol.md`](.claude/rules/replication-protocol.md)                           |
| Post-flight verification (CoVe)    | [`.claude/rules/post-flight-verification.md`](.claude/rules/post-flight-verification.md)                   |
| Quality gates                      | [`.claude/rules/quality-gates.md`](.claude/rules/quality-gates.md)                                         |
| Meta-governance                    | [`.claude/rules/meta-governance.md`](.claude/rules/meta-governance.md)                                     |

---

## Current project state

| Artefact slot                | Status        | Notes                                                                  |
| ---------------------------- | ------------- | ---------------------------------------------------------------------- |
| Workflow configuration       | **Active**    | This adaptation, 2026-04-29 — plan `wondrous-honking-gray.md`          |
| Bibliography (anchor refs)   | Seeded        | 6 entries (TT U-Net, HM-EDM, ProDM, C2F-MC, DPS, baselines)            |
| ImageCAS preprocessing       | Not started   | First task once data is downloaded                                     |
| 3D KL-VAE checkpoint         | Not started   | Planned May 2026                                                       |
| Conditional latent denoiser  | Not started   | Planned June–July 2026                                                 |
| Posterior sampling pipeline  | Not started   | Planned August 2026                                                    |
| Downstream-task evaluation   | Not started   | Planned September 2026                                                 |
| Manuscript: MICCAI 2027      | Not started   | First draft November 2026                                              |
| Manuscript: CVPR 2027        | Not started   | Parallel draft if pacing allows                                        |

The full timeline lives in
[`quality_reports/decisions/2026-04-29_research-direction-v2.md`](quality_reports/decisions/2026-04-29_research-direction-v2.md) §09.

---

## How to start the next task

1. **Trivial?** Just do it. (Single-line edit, typo, obvious fix.)
2. **Non-trivial?** Enter plan mode. Plans live in `quality_reports/plans/`.
3. **Ambiguous?** Spec-then-plan. Use `templates/requirements-spec.md`.
4. **Numeric / experimental?** Open the run card *first* under
   `experiments/runs/`, then write the code.
5. **Citation-bearing draft?** Run `/verify-claims` before declaring done.

---

## Pointers for new sessions

If you're an LLM joining mid-flow, read in order:

1. `CLAUDE.md` (this file)
2. `MEMORY.md`
3. The most recent file in `quality_reports/plans/`
4. The most recent file in `quality_reports/session_logs/`
5. `git log --oneline -10` and `git diff`

Then state your understanding of the current task before acting.
