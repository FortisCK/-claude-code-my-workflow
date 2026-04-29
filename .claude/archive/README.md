# `.claude/archive/` — Dormant template surface

This directory holds parts of the upstream
[`pedrohcgs/claude-code-my-workflow`](https://github.com/pedrohcgs/claude-code-my-workflow)
template that are not active for the **Cardiac CT Motion-Artifact Correction**
project. They are kept (not deleted) so they can be promoted back if the
project ever needs them — e.g., a thesis-defense slide deck, a clinical
reader-study analysis in R, or a TikZ figure for the manuscript.

## Why archive instead of delete

- Forking the workflow template again later is cheaper if we keep the
  upstream surface intact.
- Some lecture/R surface might genuinely come back (defense talks, an R
  reader-study, a TikZ schematic for the paper).
- `git log --follow` still works on archived files.

## What's here

| Path                                | Was active for                                          |
| ----------------------------------- | ------------------------------------------------------- |
| `skills/compile-latex/`             | LaTeX 3-pass compilation for Beamer decks               |
| `skills/deploy/`                    | Quarto → GitHub Pages render+sync                       |
| `skills/extract-tikz/`              | TikZ source → SVG extraction                            |
| `skills/new-diagram/`               | Scaffold a TikZ diagram from the gallery                |
| `skills/qa-quarto/`                 | Adversarial Beamer ↔ Quarto parity                      |
| `skills/translate-to-quarto/`       | Beamer `.tex` → Quarto `.qmd` translation               |
| `skills/validate-bib/`              | BibTeX cross-reference (lecture-flavoured)              |
| `skills/devils-advocate/`           | Adversarial pedagogy challenge                          |
| `skills/create-lecture/`            | Full-lecture creation orchestrator                      |
| `skills/visual-audit/`              | Slide-layout audit                                      |
| `skills/pedagogy-review/`           | Holistic pedagogical review                             |
| `skills/slide-excellence/`          | Multi-agent slide review                                |
| `skills/data-analysis/`             | End-to-end **R** data analysis pipeline                 |
| `skills/review-r/`                  | R-code review                                           |
| `agents/beamer-translator.md`       | Beamer → Quarto translator subagent                     |
| `agents/pedagogy-reviewer.md`       | Pedagogy reviewer for slides                            |
| `agents/quarto-critic.md`           | Quarto-vs-Beamer parity critic                          |
| `agents/quarto-fixer.md`            | Applies fixes from quarto-critic                        |
| `agents/slide-auditor.md`           | Visual slide auditor                                    |
| `agents/tikz-reviewer.md`           | TikZ devil's-advocate reviewer                          |
| `agents/r-reviewer.md`              | R-code reviewer subagent                                |
| `agents/domain-reviewer.md`         | Domain reviewer for slides                              |
| `rules/beamer-quarto-sync.md`       | Beamer ↔ Quarto single-source-of-truth                  |
| `rules/no-pause-beamer.md`          | Beamer `\pause` ban                                     |
| `rules/single-source-of-truth.md`   | Beamer-canonical SSOT                                   |
| `rules/tikz-*.md`                   | TikZ prevention / measurement / visual-quality          |
| `rules/proofreading-protocol.md`    | Slide proofreading three-phase protocol                 |
| `rules/pdf-processing.md`           | PDF extraction conventions                              |
| `rules/r-code-conventions.md`       | R-code conventions (replaced by `python-code-conventions.md`) |
| `rules/knowledge-base-template.md`  | Lecture KB template                                     |
| `scripts-R/`                        | The 00–05 R analysis pipeline (`run_all`, `load`, `clean`, `analyze`, `tables`, `figures`) |
| `samples/HelloWorld.tex`            | Sample Beamer deck                                      |
| `samples/HelloWorld.qmd`            | Sample Quarto deck                                      |
| `MEMORY-template.md`                | Original upstream MEMORY.md (template development log)  |

## Promoting an item back to active

```bash
# Skill — restore to .claude/skills/
git mv .claude/archive/skills/compile-latex .claude/skills/compile-latex

# Agent — restore to .claude/agents/
git mv .claude/archive/agents/beamer-translator.md .claude/agents/beamer-translator.md

# Rule — restore to .claude/rules/
git mv .claude/archive/rules/no-pause-beamer.md .claude/rules/no-pause-beamer.md
```

After promotion, update `CLAUDE.md`'s **Skills Quick Reference** table so the
restored item is discoverable.

## What was deleted vs archived

Nothing was deleted in the v1 → cardiac-artifacts adaptation. Everything that
became inactive lives here.
