# TRUST: Topology-Guided Semi-Supervised Aortic Root Landmark Detection

Paper-writing workspace for **TRUST** — a semi-supervised framework for detecting 10 aortic root anatomical landmarks from 3D CT scans, targeting TAVI planning. Built with a Claude Code academic workflow for structured, reproducible paper development.

**Target venue:** IEEE Transactions on Medical Imaging (TMI)
**Institution:** Universite de Rennes

---

## Repository Structure

```
arotic-landmarks/
├── CLAUDE.md                    # Project config for Claude Code
├── .claude/                     # Rules, skills, agents, hooks
├── Bibliography_base.bib        # Centralized bibliography
├── paper/                       # LaTeX paper source (IEEEtran)
├── figs/                        # Publication figures (300 DPI)
├── scripts/                     # Utility and visualization scripts
│   └── python/                  # Python figure generation
├── quality_reports/             # Plans, session logs, reviews
├── explorations/                # Research sandbox
├── templates/                   # Session log, quality report templates
└── master_supporting_docs/      # Reference papers
```

## Workflow

This repo uses the [Claude Code academic workflow](https://github.com/pedrohcgs/claude-code-my-workflow) adapted for paper writing:

- **Plan-first** — enter plan mode before non-trivial tasks
- **Contractor mode** — after plan approval, Claude implements autonomously
- **Quality gates** — 80 (commit), 90 (PR), 95 (excellence)
- **Specialized agents** — proofreader, domain reviewer, paper flow reviewer, verifier
- **Protected files** — bibliography, settings, and IEEEtran class files are guarded by hooks

## Key Commands

```bash
# Compile paper (3-pass pdflatex + bibtex)
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Quality score
python3 scripts/quality_score.py paper/main.tex

# Run figure generation scripts
python3 scripts/python/script_name.py
```

## Prerequisites

| Tool | Required For |
|------|-------------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Workflow automation |
| pdflatex | Paper compilation ([TeX Live](https://tug.org/texlive/)) |
| Python 3.9+ | Figure generation, quality scoring |
| [gh CLI](https://cli.github.com/) | PR workflow |

## Origin

Adapted from [pedrohcgs/claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow) — an academic workflow template for Claude Code.

## License

MIT License.
