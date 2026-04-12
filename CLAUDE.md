# CLAUDE.MD -- Master's Thesis Evaluation with Claude Code

**Project:** Master's Thesis Evaluation — CS/AI
**Institution:** TBD
**Branch:** main

---

## Core Principles

- **Plan first** -- enter plan mode before non-trivial tasks; save plans to `quality_reports/plans/`
- **Verify after** -- check evaluation reports for completeness and accuracy at the end of every task
- **Read-only by default** -- the student's thesis .tex files are the source of truth; produce evaluation reports, never edit source files without explicit permission
- **Quality gates** -- no evaluation report ships below 80/100
- **[LEARN] tags** -- when corrected, save `[LEARN:category] wrong → right` to MEMORY.md

---

## Folder Structure

```
claude-code-my-workflow/
├── CLAUDE.MD                    # This file
├── .claude/                     # Rules, skills, agents, hooks
├── Thesis/                      # Student's LaTeX source (READ-ONLY)
├── Bibliography_base.bib        # Reference bibliography
├── Figures/                     # Extracted figures for analysis
├── master_supporting_docs/      # Reference papers, guidelines, rubrics
│   ├── supporting_papers/       # Related papers for citation checking
│   └── supporting_slides/       # Defense slides (if applicable)
├── quality_reports/             # Plans, session logs, evaluation outputs
│   ├── plans/                   # Saved plans
│   ├── session_logs/            # Session logs
│   ├── thesis_evaluation/       # Evaluation reports (our output)
│   └── merges/                  # Merge reports
├── explorations/                # Research sandbox (analysis, experiments)
├── scripts/                     # Utility scripts + analysis code
│   └── R/                       # R scripts for data/figure analysis
├── templates/                   # Session log, quality report templates
└── docs/                        # Generated outputs
```

---

## Commands

```bash
# Compile thesis (verify it builds)
cd Thesis && latexmk -xelatex -interaction=nonstopmode main.tex

# Run quality score on evaluation report
python3 scripts/quality_score.py quality_reports/thesis_evaluation/report.md

# Check bibliography
cd Thesis && bibtex main
```

---

## Quality Thresholds

| Score | Gate | Meaning |
|-------|------|---------|
| 80 | Commit | Evaluation report good enough to save |
| 90 | PR | Ready for advisor review |
| 95 | Excellence | Publishable-quality feedback |

---

## Skills Quick Reference

| Command | What It Does |
|---------|-------------|
| `/review-paper [file]` | **Primary** — Full thesis/chapter evaluation |
| `/proofread [file]` | Grammar/typo/notation review of thesis |
| `/validate-bib` | Cross-reference thesis citations |
| `/lit-review [topic]` | Literature search for missing references |
| `/research-ideation [topic]` | Generate research improvement suggestions |
| `/interview-me [topic]` | Interactive deep-dive on thesis topics |
| `/compile-latex [file]` | Verify thesis compiles |
| `/review-r [file]` | Review thesis R/Python code |
| `/data-analysis [dataset]` | Analyze thesis datasets |
| `/commit [msg]` | Stage, commit, push evaluation work |
| `/devils-advocate` | Challenge thesis arguments |

---

## Evaluation Dimensions (CS/AI Thesis)

| Dimension | Key Questions |
|-----------|--------------|
| Problem & Motivation | Clear research question? Well-motivated? Novel contribution? |
| Technical Approach | Sound methodology? Correct proofs/algorithms? Appropriate complexity? |
| Experimental Design | Fair baselines? Proper metrics? Statistical significance? Reproducible? |
| Literature Review | Complete coverage? Accurate characterization? Clear positioning? |
| Writing Quality | Clear prose? Consistent notation? Logical flow? No typos? |
| Presentation | Good figures/tables? Self-contained captions? Appropriate length? |

---

## Current Project State

| Chapter | File | Status | Key Content |
|---------|------|--------|-------------|
| *Thesis not yet imported* | -- | Pending | Awaiting upload to `Thesis/` |
