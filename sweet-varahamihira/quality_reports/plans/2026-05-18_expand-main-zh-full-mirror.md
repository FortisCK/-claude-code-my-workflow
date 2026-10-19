# Expand Chinese Mirror to Full Paragraph-Level Version

Date: 2026-05-18

Goal: Replace the condensed `paper/main_zh.md` reading summary with a fuller paragraph-level Chinese mirror of the current `paper/main.tex`.

Rationale:

- The 2026-05-16 version was numerically and narratively aligned, but it was intentionally condensed.
- The user expects `main_zh.md` to be closer to the English manuscript, not a shortened review aid.
- `paper/main.tex` remains the source of truth; `main_zh.md` should mirror its scientific content for Chinese review.

Planned edits:

1. [x] Expand `paper/main_zh.md` from a concise reading稿 into a paragraph-level Chinese mirror.
2. [x] Preserve the current final numbers and decisions: `2.19`, `21.5%`, six downstream measurements, no standalone fusion-results story, no stale uncertainty scatter.
3. [x] Include the main formulas, tables, figure-caption summaries, clinical measurement definitions, result interpretations, discussion, limitations, and conclusion.
4. [x] Verify stale terms and key current terms with grep after editing.

Result:

- Replaced the 354-line condensed Chinese reading稿 with a 673-line paragraph-level Chinese mirror of the current English manuscript.
- Preserved the current detection, ablation, label-efficiency, per-landmark, and six-measurement clinical downstream results.
- Verified that stale terms/numbers such as `2.28`, `18.3%`, `0.908`, `scatter`, `逐病例均值`, and `per-case mean` are absent from `paper/main_zh.md`.
- Verified that current key terms/numbers such as `2.19`, `21.5%`, `六项`, `46.7%`, `75.0%`, `n+=1`, `9.4%`, `43.1`, `6.7`, `4.8`, `-0.27`, `-0.36`, `2.36±1.62`, `2.34±1.48`, `3.86`, `3.34`, `41.2%`, `298/300`, and `6.74±5.12` are present.
