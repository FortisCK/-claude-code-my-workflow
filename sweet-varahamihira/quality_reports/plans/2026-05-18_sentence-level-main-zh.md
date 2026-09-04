# Sentence-Level Chinese Mirror of Current Paper

Date: 2026-05-18

Goal: Convert `paper/main_zh.md` from a paragraph-level Chinese mirror into a sentence-level Chinese review draft of the current `paper/main.tex`.

Rationale:

- The user wants to review the manuscript primarily from the Chinese version.
- A condensed or loosely mirrored Chinese draft is not enough for this workflow.
- The Chinese file should preserve sentence order, caveats, quantitative claims, formulas, table values, figure-caption content, clinical interpretations, limitations, and conclusion as closely as possible while remaining readable in Chinese Markdown.

Planned edits:

1. [x] Use `paper/main.tex` as the only authoritative source.
2. [x] Rewrite `paper/main_zh.md` so that each English paragraph is represented by near sentence-level Chinese translation rather than summary.
3. [x] Preserve all current final decisions: no standalone fusion-results story, no stale uncertainty scatter, `2.19` mm, `21.5%`, six downstream measurements, annular-area under-prediction, and cohort-limited coronary sensitivity.
4. [x] Verify stale terms and key current terms with `rg`.

Result:

- Replaced the 673-line paragraph-level Chinese mirror with a 1466-line sentence-level Chinese review draft.
- Preserved section order, formulas, major citations, all figure-caption content, table values, clinical measurement definitions, result interpretations, discussion, limitations, and conclusion.
- Verified stale terms/numbers (`2.28`, `18.3%`, `22.6`, `2.16`, `0.908`, `scatter`, `逐病例均值`, `per-case mean`) are absent.
- Verified current key terms/numbers (`2.19`, `21.5%`, `六项`, `46.7%`, `75.0%`, `n$_+$=1`, `9.4%`, `43.1`, `6.7`, `4.8`, `-0.27`, `-0.36`, `2.36±1.62`, `2.34±1.48`, `3.86`, `3.34`, `41.2%`, `298/300`, `6.74±5.12`) are present.
- `paper/main.tex` was not modified in this step.
