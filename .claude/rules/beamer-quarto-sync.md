---
paths:
  - "Slides/**/*.tex"
  - "Quarto/**/*.qmd"
---

# Beamer -> Quarto Sync Rule

This repository is primarily a CATHACTION participant submission repo. Beamer and Quarto are optional support surfaces for presentations, explanations, and publication visuals.

When a Beamer `.tex` deck has a Quarto `.qmd` mirror, every content edit to the Beamer file must be propagated to the Quarto file in the same task. Do not let challenge facts, metric definitions, equations, or claims drift between formats.

## Deck Mapping

| Artifact | Beamer | Quarto | Status |
|---------|--------|--------|--------|
| Setup demo | `Slides/HelloWorld.tex` | `Quarto/HelloWorld.qmd` | Demo only; remove or ignore for real challenge work |

Add real CATHACTION decks here when created.

## Workflow

1. Apply content fix to Beamer `.tex`.
2. Apply equivalent content fix to Quarto `.qmd` if a mirror exists.
3. Compile Beamer.
4. Render Quarto with `./scripts/sync_to_docs.sh LectureN`.
5. Report both verification results.

## Translation Reference

| Beamer | Quarto Equivalent |
| ------ | ----------------- |
| `\muted{text}` | `[text]{style="color: #525252;"}` |
| `\key{text}` | `[**text**]{.hi-gold}` |
| `\textcolor{positive}{text}` | `[text]{.positive}` |
| `\textcolor{negative}{text}` | `[text]{.negative}` |
| `\item text` | `- text` |
| `$formula$` | `$formula$` |

## When Not To Sync

- No Quarto mirror exists.
- The change is LaTeX-only infrastructure.
- The user explicitly asks to skip Quarto sync.
- The file is a demo or scratch deck outside the approved task.

## Precedence

Beamer remains authoritative for mirrored slide content. Quarto-only presentation decorations are allowed only when they do not alter scientific facts, metrics, or claims.
