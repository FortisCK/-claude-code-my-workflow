# `manuscript/` — Paper drafts

One subdirectory per submission target. Each subdirectory is a self-contained
LaTeX project that imports figures from `../Figures/` and citations from
`../../Bibliography_base.bib`.

## Layout

```
manuscript/
├── miccai2027/      # MICCAI 2027 main conference submission (primary target)
├── cvpr2027/        # CVPR 2027 submission (parallel target if pacing allows)
└── tmi-extension/   # IEEE TMI journal-extension fallback
```

## Per-subdirectory layout (when populated)

```
miccai2027/
├── main.tex          # Top-level document
├── sections/         # \input{sections/intro.tex} etc.
├── figures.tex       # Single source of figure environments (optional)
├── supplementary.tex # Supplementary material
├── miccai.cls        # Venue style file (kept in-tree to pin formatting)
└── Makefile          # `make`, `make clean`, `make supplementary`
```

## Conventions

- **Bibliography:** all subdirectories share `../../Bibliography_base.bib`.
  Add `\bibliography{../../Bibliography_base}` in `main.tex`.
- **Figures:** import from `../../Figures/`. Naming: `fig_<short-name>.{pdf|png}`.
  Vector preferred; raster only when the source is genuinely raster (e.g., CT slices).
- **Cite-keys:** `Author<Year>_keyword` (e.g., `Deng2023_TTUNet`).
  See [`Bibliography_base.bib`](../Bibliography_base.bib) for examples.
- **LaTeX engine:** XeLaTeX, 3-pass + bibtex. Each `Makefile` defines this.
  Header includes from `../../Preambles/header.tex` are optional —
  conference style files usually override most of it.
- **Authoritative artifact:** the manuscript `.tex` is the single source of
  truth for all numeric claims. Each numeric claim must trace to a specific
  `experiments/runs/<run>.md` card via a `% source:` comment, per
  [`.claude/rules/cross-artifact-review.md`](../.claude/rules/cross-artifact-review.md)
  and [`.claude/rules/experiments-protocol.md`](../.claude/rules/experiments-protocol.md).
- **Track changes:** drafts shipped to co-authors should use
  `latexdiff` against the previous tagged version. Commit the diff PDF to
  `manuscript/<target>/diffs/`.

## Working in two languages

The manuscript itself is **English**. Internal commentary (`% TODO:` notes,
draft fragments in `sections/_drafts/`, side notes for Pascal / Carlos) may be
中文 — but anything that ends up in the submitted PDF is English.

## Cross-references

- [`../experiments/README.md`](../experiments/README.md) — run-card discipline that
  manuscript claims must trace back to.
- [`../code/README.md`](../code/README.md) — code repo released alongside the paper.
- [`../.claude/skills/review-paper/SKILL.md`](../.claude/skills/review-paper/SKILL.md) —
  how to run a manuscript review (single-pass, adversarial, or peer-review pipeline).
- [`../.claude/skills/seven-pass-review/SKILL.md`](../.claude/skills/seven-pass-review/SKILL.md) —
  seven-pass parallel adversarial review for late-stage drafts.
