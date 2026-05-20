---
paths:
  - "Figures/**/*"
  - "Quarto/**/*.qmd"
  - "Slides/**/*.tex"
  - "scripts/**/*"
  - "src/**/*"
  - "*.tex"
  - "*.md"
---

# Single Source of Truth: CATHACTION

Different artifact families have different authoritative sources. Do not edit derived artifacts as if they were primary.

## Challenge Facts

```
2026-04-22 CATHACTION MICCAI PDF (SOURCE OF TRUTH)
  ├── README / guide / docs summaries
  ├── method report background
  ├── slide decks
  └── reviewer knowledge base
```

If the official 2026 challenge platform or website later supersedes the PDF, record that decision in `quality_reports/decisions/` and update all summaries together.

## Code And Results

```
Source code + config + split file + checkpoint (SOURCE OF RESULT)
  ├── prediction files
  ├── metric tables
  ├── paper numbers
  └── figures
```

Every number in the method report must be reproducible from a named code path and config. Do not manually edit derived CSVs, tables, or predictions to make a result look better.

## Submission Package

```
Dockerfile + inference entrypoint + model artifacts (SOURCE OF SUBMISSION)
  ├── platform container
  ├── result file in challenge schema
  └── submission description
```

Container behavior must match the method report. If a local script differs from the Docker path, treat that as a bug until reconciled.

## Slides And Quarto (When Used)

```
Beamer .tex (slide SOURCE OF TRUTH)
  ├── extract_tikz.tex -> PDF -> SVGs
  ├── Quarto .qmd -> HTML
  ├── Bibliography_base.bib
  └── Figures/* outputs
```

For mirrored slide decks, propagate content changes from Beamer to Quarto. Presentation-only divergence in Quarto is allowed only when it does not change facts, notation, metrics, or claims.

## Required Checks Before Reporting Completion

- Challenge facts match the PDF or a documented later source.
- Reported metrics trace to code/config/split/checkpoint.
- Predictions and tables are generated, not hand-edited.
- Docker submission path is aligned with local inference path when packaging exists.
- Slide mirrors remain synchronized when both Beamer and Quarto versions exist.
