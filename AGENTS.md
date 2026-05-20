# AGENTS.md -- CATHACTION Project Workflow

**Project:** CATHACTION: Endovascular Intervention Tool Segmentation and Collision Detection
**Institution:** Université de Rennes
**Role:** MICCAI 2026 challenge participant repository
**Branch:** main

---

## Core Principles

- **Plan first** -- non-trivial work starts with a saved plan in `quality_reports/plans/` and user approval.
- **Verify after** -- every task ends with concrete verification: tests, renders, metric checks, Docker smoke tests, or documented blockers.
- **Challenge source of truth** -- the 2026-04-22 CATHACTION PDF is authoritative until the official 2026 website supersedes it.
- **Participant discipline** -- this repo contains all challenge code plus the final method paper/report; keep experiments reproducible and submission-ready.
- **No leakage** -- split and evaluate at procedure/case level; never frame-randomize across cases or tune on hidden-test feedback.
- **Publication-ready visuals** -- figures, diagrams, tables, and slides must be polished enough for a MICCAI paper or spotlight presentation.
- **[LEARN] tags** -- when corrected, save `[LEARN:category] wrong -> right` to [MEMORY.md](MEMORY.md).

Cross-session context lives in [MEMORY.md](MEMORY.md); specs, plans, logs, decisions, and checkpoints live in [quality_reports/](quality_reports/).

---

## CATHACTION Facts

| Item | Working Fact |
| --- | --- |
| Task 1 | Catheter and guidewire segmentation in X-ray fluoroscopy |
| Task 1 metrics | DSC primary; IoU/Jaccard, mIoU, pixel accuracy secondary |
| Task 2 | Collision detection in endovascular intervention |
| Task 2 metrics | mAP primary; AP secondary |
| Domains | Silicon phantom, preclinical animal X-ray, real human X-ray procedures |
| Dataset scale | 650 videos/cases; about 40k segmentation frames; about 600k collision-label frames |
| Split | 70% train, 15% validation, 15% hidden test at case/procedure level |
| Submission | Docker container, predefined result file, short method description/report |
| Data policy | Challenge data plus public external data/pretrained models only; no private/proprietary clinical data |
| Deadline | Submission deadline: 2026-08-23 |

The current public website still shows older counts (about 500k frames / 25k masks, updated 2024-11-28). Do not silently mix those with the MICCAI 2026 PDF counts.

---

## Folder Structure

```
cathaction/
├── CLAUDE.md                    # Claude-facing workflow instructions
├── AGENTS.md                    # Codex-facing workflow instructions
├── .claude/                     # Claude rules, skills, agents, hooks
├── .codex/                      # Codex hooks and agent mirrors
├── .agents/                     # Codex skill mirrors
├── MEMORY.md                    # Cross-session learned corrections
├── Bibliography_base.bib        # Shared bibliography when present
├── Figures/                     # Publication figures, diagrams, visual outputs
├── Preambles/                   # Shared LaTeX/Beamer preamble
├── Slides/                      # Beamer decks when needed
├── Quarto/                      # RevealJS decks and theme when needed
├── docs/                        # GitHub Pages / generated documentation
├── guide/                       # Source documentation
├── scripts/                     # Utilities, analysis helpers, validation scripts
├── scripts/R/                   # R helpers if used
├── quality_reports/             # Plans, specs, logs, decisions, review reports
├── explorations/                # Sandbox experiments
├── templates/                   # Reusable report/spec/decision templates
└── master_supporting_docs/      # Challenge PDF, papers, and supporting material
```

---

## Working Commands

```bash
# Environment and workflow checks
./scripts/validate-setup.sh
python3 -m py_compile scripts/check-surface-sync.py scripts/check-palette-sync.py scripts/quality_score.py

# Quality and documentation checks
./scripts/check-palette-sync.sh
./scripts/check-surface-sync.sh

# Slides, if used
cd Slides && TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode file.tex
./scripts/sync_to_docs.sh LectureN
python3 scripts/quality_score.py Quarto/file.qmd

# Challenge implementation, once code exists
python3 -m pytest tests
docker build -t cathaction-submission .
```

Only run commands that match existing files. If a test suite or Dockerfile does not exist yet, record that as a setup gap rather than inventing a passing check.

---

## Quality Thresholds

| Score | Checkpoint | Meaning |
|-------|------------|---------|
| 80 | Commit | Good enough to save |
| 90 | Submission/PR | Ready for serious external review |
| 95 | Excellence | Publication-ready / spotlight-ready |

For challenge work, "quality" includes reproducibility, split integrity, metric fidelity, documentation clarity, Docker viability, and visual polish.

---

## Skills Quick Reference

| Command | What It Does |
|---------|-------------|
| `/data-analysis [dataset]` | End-to-end analysis or experiment pipeline |
| `/review-r [file]` | R code review if R scripts are used |
| `/review-paper [file]` | Manuscript/method report review |
| `/respond-to-referees [report] [manuscript]` | Response drafting for reviews |
| `/lit-review [topic]` | Literature search + synthesis |
| `/research-ideation [topic]` | Research questions + strategies |
| `/verify-claims [file]` | Chain-of-Verification fact-check |
| `/audit-reproducibility [paper]` | Paper/code numeric consistency audit |
| `/deep-audit` | Repository-wide consistency audit |
| `/checkpoint [topic]` | Save a structured handoff snapshot |
| `/compile-latex [file]` | Compile Beamer slides when used |
| `/deploy [LectureN]` | Render Quarto slides when used |
| `/slide-excellence [file]` | Multi-agent slide review when used |
| `/commit [msg]` | Stage, commit, PR, and merge workflow |

---

## Current Project State

| Artifact | Path | Status | Notes |
| --- | --- | --- | --- |
| Challenge specification | external PDF in Downloads | Source of truth | 2026-04-22 CATHACTION MICCAI challenge description |
| Workflow configuration | `.claude/`, `.codex/`, `AGENTS.md`, `CLAUDE.md` | Being adapted | Participant-focused CATHACTION workflow |
| Codebase | TBD | Not yet implemented | Future Python/ML stack should be recorded before major coding |
| Method paper/report | TBD | Planned | Final MICCAI method description/submission report |
| Slides/visuals | `Slides/`, `Quarto/`, `Figures/` | Optional | Use for polished presentation and publication assets |
