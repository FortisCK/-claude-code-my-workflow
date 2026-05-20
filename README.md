# CATHACTION Participant Repository

**Project:** CATHACTION: Endovascular Intervention Tool Segmentation and Collision Detection
**Institution:** Université de Rennes
**Role:** MICCAI 2026 challenge participant repository

This repository is the working home for our CATHACTION challenge submission: code, experiments, Docker packaging, publication-ready figures, and the final MICCAI method description or paper.

The collaboration model is deliberately rigorous. Non-trivial work starts with a saved plan in `quality_reports/plans/`, implementation proceeds autonomously after approval, and each task ends with concrete verification.

## Challenge Summary

CATHACTION is a MICCAI 2026 challenge for endovascular surgical AI in X-ray fluoroscopy. The participant-facing work in this repository targets two tasks:

| Task | Goal | Primary Metric | Secondary Metrics |
|------|------|----------------|-------------------|
| Task 1 | Catheter and guidewire segmentation | Dice Similarity Coefficient (DSC) | IoU/Jaccard, mIoU, pixel-wise accuracy |
| Task 2 | Collision detection | mean Average Precision (mAP) | AP |

Working challenge facts from the 2026-04-22 PDF:

- 650 fluoroscopy videos/cases.
- Three domains: silicon vascular phantom, preclinical animal X-ray, and real human X-ray procedures.
- About 40,000 annotated segmentation frames.
- About 600,000 annotated collision-label frames.
- 70% train, 15% validation, 15% hidden test, split at procedure/case level.
- Submission is expected to include a Docker container, predefined result file, and short method description/report.
- Submission deadline: 2026-08-23.

The public website at <https://airvlab.github.io/cathaction/> currently contains older dataset counts. Until an official 2026 challenge platform page supersedes the PDF, the PDF is the source of truth for this repo.

## Repository Layout

| Path | Purpose |
|------|---------|
| `CLAUDE.md`, `AGENTS.md` | Agent-facing project instructions |
| `.claude/`, `.codex/`, `.agents/` | Rules, skills, agents, and hooks |
| `scripts/` | Validation scripts and analysis helpers |
| `Figures/` | Publication figures and visual outputs |
| `Slides/`, `Quarto/` | Optional presentation decks |
| `quality_reports/` | Plans, specs, session logs, decisions, reviews, checkpoints |
| `master_supporting_docs/` | Supporting papers, slides, and challenge documents |
| `explorations/` | Sandbox experiments that are not yet production claims |

## Workflow

For non-trivial tasks:

1. Create or update a plan in `quality_reports/plans/`.
2. Wait for user approval.
3. Implement in contractor mode.
4. Verify with the strongest available check.
5. Log decisions, findings, and residual risks.

Key rules:

- No frame-level leakage across cases.
- No private/proprietary clinical data.
- No hidden-test hand tuning or manual prediction curation.
- Every reported metric must trace to code, config, split, checkpoint, command, and output.
- Figures and diagrams should be paper-ready, labeled, and reproducible.

## Setup Checks

Run:

```bash
./scripts/validate-setup.sh
```

Minimum required tools for the workflow scaffold are `git` and `python3`. Docker becomes required once the submission package exists. XeLaTeX and Quarto are optional unless we build slides or web documentation.

Useful checks:

```bash
python3 -m py_compile scripts/check-surface-sync.py scripts/check-palette-sync.py scripts/quality_score.py
./scripts/check-palette-sync.sh
./scripts/check-surface-sync.sh
```

When code and tests exist, add:

```bash
python3 -m pytest tests
docker build -t cathaction-submission .
```

## Quality Targets

| Score | Meaning |
|-------|---------|
| 80 | Commit-ready |
| 90 | Submission/PR-ready |
| 95 | Publication-ready / spotlight-ready |

For this challenge, quality includes metric fidelity, split integrity, reproducibility, Docker viability, method-report consistency, and visual polish.

## Current Status

- Workflow configuration is being adapted from the academic scaffold to CATHACTION.
- Baseline model, experiment registry, official submission I/O schema, and Docker path are still to be defined.
- The next project step should choose the implementation stack and create the first reproducible baseline plan.

## Workflow Foundation

This repo contains an adapted version of an academic agent workflow scaffold with plan-first execution, specialized reviewers, verification hooks, quality scoring, and persistent memory. The CATHACTION configuration in this repository is now participant-focused rather than a generic template.
