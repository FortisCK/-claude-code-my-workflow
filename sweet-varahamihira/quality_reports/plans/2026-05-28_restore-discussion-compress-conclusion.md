# Restore Discussion and Compress Conclusion Plan

Date: 2026-05-28

## Correction

The reviewer comment was intended for the Conclusion, not the Discussion. The previous edit shortened the Discussion, which should be restored.

## Planned Edits

- Restore the English Discussion body to the pre-shortening version.
- Preserve the already-approved limitation ordering:
  1. single-center/single-annotator and limited high-risk cases,
  2. inference-time uncertainty saturation,
  3. Stage 1/Stage 2 cascade crop failure,
  4. sparse landmarks vs full annular contour.
- Compress the Conclusion into a shorter final paragraph that retains:
  - TRUST and unified uncertainty,
  - 2.19 mm / 21.5% result,
  - six downstream measures and key risk-stratification preservation,
  - concise limitations/future work.
- Sync `paper/main_zh.md`.
- Compile with pdflatex/bibtex/pdflatex/pdflatex and check warnings/page count.

## Success Criteria

- Discussion is restored, not shortened.
- Conclusion is shorter and less repetitive.
- English and Chinese versions are aligned.
- Paper compiles without new citation/reference warnings.
