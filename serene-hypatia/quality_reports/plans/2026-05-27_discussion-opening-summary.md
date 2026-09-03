# Discussion Opening Summary Plan

Date: 2026-05-27

## Reviewer Comment

Carlos suggested that the Discussion should start with a paragraph summarising the main contributions, and that the first paragraph of the Conclusion can be used.

## Rationale

The current Discussion begins directly with the ablation interpretation. This is technically valid, but it is abrupt: readers enter the Discussion without a short synthesis of what the paper has shown. Adding a concise opening paragraph will make the Discussion read more like a manuscript Discussion section rather than an extended Results interpretation.

## Planned Edits

- Add a short opening paragraph after `\section{Discussion}` that summarizes:
  - TRUST as a semi-supervised 10-landmark framework for comprehensive TAVI planning.
  - Unified uncertainty as the central methodological contribution.
  - The main numerical result (2.19 mm, 21.5% error reduction).
  - Clinical downstream coverage and key preserved risk stratifications.
- Keep the Conclusion paragraph intact for now, but avoid verbatim duplication in Discussion.
- Sync the same opening in `paper/main_zh.md`.
- Compile the paper with the project pdflatex/bibtex/pdflatex/pdflatex workflow and check warnings/page count.

## Success Criteria

- Discussion starts with a contribution/major-finding summary before ablation details.
- No stale or duplicated phrasing that makes Discussion and Conclusion feel copy-pasted.
- `paper/main.tex` compiles successfully.
- No new citation/reference warnings.
