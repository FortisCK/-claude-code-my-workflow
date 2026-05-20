# Preambles

Shared LaTeX/Beamer preamble for CATHACTION presentations, method-report figures, and optional slide decks.

## Usage in a deck

```latex
\documentclass{beamer}
\input{header}   % resolves via TEXINPUTS=../Preambles:$TEXINPUTS

\title{CATHACTION Method}
\author{You}
\date{\today}

\begin{document}
\frame{\titlepage}
% ...
\end{document}
```

Compile with `/compile-latex <file>` -- the skill sets `TEXINPUTS` for you. For manual compilation:

```bash
cd Slides
TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode YourLecture.tex
```

## The palette contract

Color names in `header.tex` **must** match the SCSS variable names in [`../Quarto/theme-template.scss`](../Quarto/theme-template.scss) so Beamer and Quarto renderings use the same palette.

The `scripts/check-palette-sync.sh` script greps both files and reports any divergence:

```bash
./scripts/check-palette-sync.sh
```

It's also invoked (non-blocking) from `./scripts/validate-setup.sh`.

When you customize the palette:

1. Edit HEX values in both `Preambles/header.tex` (LaTeX) **and** `Quarto/theme-template.scss` (SCSS).
2. Keep the names aligned: `primary-blue`, `primary-gold`, `highlight-yellow`, `light-bg`, `jet`, `positive`, `negative`, `neutral`, `hi-slate`, `hi-green`, `hi-red`.
3. Run `./scripts/check-palette-sync.sh` — it should report "in sync".

## What's inside

- **Palette** — 11 named colors matching the SCSS.
- **Beamer theme assignments** -- structure, titles, itemize, alert, blocks, minimal footer.
- **TikZ libraries** -- `arrows.meta, positioning, calc, decorations.pathreplacing, fit, shapes.geometric, backgrounds`.
- **Shared TikZ styles** -- reusable diagram styles for method schematics and result summaries.
- **Convenience macros** -- `\muted{...}`, `\key{...}`, `\good{...}`, `\bad{...}`, `\transitionslide{...}`.

## Extending

Add packages a specific deck needs *after* `\input{header}`, not in this file. Keep this preamble small and auditable. Only add to `header.tex` if every CATHACTION deck or figure source needs it.

For a deck-specific preamble (rare), create `Preambles/[deck]-addon.tex` and `\input` it after `header.tex`.
