# AGENTS.md -- TRUST Project Configuration

**Project:** TRUST: Topological Reasoning and Uncertainty-aware Semi-supervised Teaching for TAVI Landmark Detection
**Institution:** Universite de Rennes
**Branch:** main

---

## Core Principles

- **Plan first** -- enter plan mode before non-trivial tasks; save plans to `quality_reports/plans/`
- **Verify after** -- compile/render and confirm output at the end of every task
- **Single source of truth** -- paper `.tex` files are authoritative
- **Quality gates** -- nothing ships below 80/100
- **[LEARN] tags** -- when corrected, save `[LEARN:category] wrong -> right` to MEMORY.md

---

## Folder Structure

```
arotic-landmarks/
├── AGENTS.md                    # This file
├── .Codex/                     # Rules, skills, agents, hooks
├── Bibliography_base.bib        # Centralized bibliography
├── paper/                       # LaTeX paper source (ieeecolor)
├── figs/                        # Publication figures
├── scripts/                     # Utility scripts
│   └── python/                  # Python visualization/figure scripts
├── quality_reports/             # Plans, session logs, merge reports
├── explorations/                # Research sandbox (see rules)
├── templates/                   # Session log, quality report templates
└── master_supporting_docs/      # Reference papers
```

---

## Commands

```bash
# LaTeX paper compilation (3-pass, pdflatex for ieeecolor)
cd paper && pdflatex -interaction=nonstopmode main.tex
BIBINPUTS=..:$BIBINPUTS bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Quality score
python3 scripts/quality_score.py paper/main.tex
```

---

## Quality Thresholds

| Score | Gate | Meaning |
|-------|------|---------|
| 80 | Commit | Good enough to save |
| 90 | PR | Ready for deployment |
| 95 | Excellence | Aspirational |

---

## Skills Quick Reference

| Command | What It Does |
|---------|-------------|
| `/compile-paper [file]` | 3-pass pdflatex + bibtex |
| `/proofread [file]` | Grammar/typo/overflow review |
| `/check-figures` | Verify figure references and quality |
| `/review-paper [file]` | Comprehensive manuscript review |
| `/review-python [file]` | Python code quality review |
| `/validate-bib` | Cross-reference citations |
| `/devils-advocate` | Challenge paper design |
| `/commit [msg]` | Stage, commit, PR, merge |
| `/lit-review [topic]` | Literature search + synthesis |
| `/research-ideation [topic]` | Research questions + strategies |
| `/interview-me [topic]` | Interactive research interview |
| `/figure-generation [dataset]` | Python figure generation workflow |

---

## TRUST Notation Registry

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Input 3D CT volume |
| $\{l_i\}_{i=1}^{K}$ | Set of K anatomical landmarks (K=10) |
| $f_\theta$ | Student network (parameters $\theta$) |
| $f_{\theta'}$ | Teacher network (EMA parameters $\theta'$) |
| $\mathcal{L}_{sup}$ | Supervised loss (labeled data) |
| $\mathcal{L}_{con}$ | Consistency loss (unlabeled data) |
| $\mathcal{L}_{topo}$ | Topology-guided graph refinement loss |
| $u_i$ | Uncertainty estimate for landmark $i$ |
| MRE | Mean Radial Error (mm) |
| SDR | Success Detection Rate at threshold $r$ |

---

## Current Paper State

| Section | File | Status | Key Content |
|---------|------|--------|-------------|
| Main | `paper/main.tex` | Skeleton | Document root (ieeecolor class) |
| Abstract | (in main.tex) | TODO | TRUST framework summary (150-250 words) |
| Introduction | (in main.tex) | TODO | Clinical motivation + contributions |
| Related Work | (in main.tex) | TODO | SSL, landmark detection, graph reasoning, uncertainty |
| Methodology | (in main.tex) | Draft | Mean-Teacher + UG-HCO + GCN Refinement |
| Experiments | (in main.tex) | TODO | 150L+600U dataset, MRE/SDR, ablation |
| Discussion | (in main.tex) | TODO | P7 bottleneck, coronary safety, limitations |
| Conclusion | (in main.tex) | TODO | Summary + future work |
