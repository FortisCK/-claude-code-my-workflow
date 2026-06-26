# Session Log: 2026-05-13 -- Post-Retrain Paper Update (Phase 1 + Phase 2 + Phase 2.5)

**Status:** Phase 1, 2.1, 2.2, 2.4, and 2.5 completed; uncertainty scatter/fusion cleanup completed; remaining items are housekeeping or external confirmations.

## Objective

After Leo delivered new annotations (improved P0-P7 for cases 151-221, coronary
ostia P8/P9 for cases 111-151, annulus polygons for cases 001-100+151+153,
plus 456 fix), the model was retrained. New test-set per-landmark MREs delivered
in `final_result.csv` (30 cases, with last row being column means).

Phase 1 scope: update every TRUST-row hard number in paper (tables, abstract,
§4.3-§4.4 inline numbers). Defer to Phase 2: §4.5 per-landmark narrative,
§4.4 downstream, §5 limitations, Abstract "RCO underpowered" sentence, Figure 4/5
regeneration, Table VII r/ū, Figure 3 uncertainty scatter.

## Headline Number Changes

| Metric | Old | New |
|--------|----:|----:|
| Overall MRE (mm) | 2.28 +/- 1.33 | **2.16 +/- 1.24** |
| Reduction vs Ma (supervised) | 18.3% | **22.6%** |
| SDR@2mm | 53.00% | 52.00% (slight) |
| SDR@2.5mm | 61.67% | 67.00% |
| SDR@3mm | 75.00% | 77.33% |
| SDR@4mm | 89.33% | **92.00%** |
| Failure rate (>20mm) | -- | **0.00%** (0/300) |

## Per-Landmark Highlights

| Landmark | Old TRUST | New TRUST | Note |
|----------|----------:|----------:|------|
| P0 NCC Hinge | 2.38 | 2.55 | slight regression |
| P1 RCC Hinge | 3.01 | 2.73 | improved |
| P2 LCC Hinge | 2.27 | 1.79 | now an improvement vs supervised |
| P5 LCC-NCC | 1.97 | 1.76 | improved |
| P6 Center | 1.59 | **1.50** | still best |
| P7 MS Point | 2.94 | 3.09 | slight regression (still hardest) |
| P8 LCO | 2.24 | **1.95** | improved |
| P9 RCO | 2.23 | **1.97** | dramatic vs supervised 3.57 |

## Changes Made

| File | Change | Reason |
|------|--------|--------|
| `paper/main.tex` Abstract | MRE 2.28->2.16, 18.3%->22.6% | New result |
| `paper/main.tex` Table I (Comparison) TRUST 10-pt + 8-pt rows | Full SDR row update | New result |
| `paper/main.tex` §4.3 main TRUST paragraph | MRE/SDR/CI/std/% all aligned; SCN 26.9%->30.8%; CPS 10.6%->15.3% | Recomputed against unchanged baselines |
| `paper/main.tex` §4.3 8-pt sanity check paragraph | Reframed as "comparable to Ma" since new gain only 0.9% | Avoid claiming larger improvement than data shows |
| `paper/main.tex` Table III (Cumulative ablation) last row | -0.18->-0.30, 53->52, 89.33->92.00 | New result |
| `paper/main.tex` Table III commentary | "factor of two"->"three-fold margin"; -0.25->-0.34 | Match new delta |
| `paper/main.tex` Table IV (Consumer sub-ablation) last row | Same updates | Same |
| `paper/main.tex` Table IV commentary | "at least as large"->"substantially exceeds"; -0.16 sum still holds | Match new delta |
| `paper/main.tex` Table V (Per-landmark) TRUST column | All 10 rows + Overall | New result |
| `paper/main.tex` Table VI (Label efficiency) TRUST 100 row | 2.28->2.16, -0.51->-0.63 | New result |
| `paper/main.tex` §4.4 label efficiency commentary | -0.51->-0.63 | Match table |
| `paper/main.tex` Table VII (Uncertainty decomp) Pooled row | MRE/SDR updated; r/u kept (no new uncertainty data) | Partial: full update pending |
| `paper/main.tex` §4.4 uncertainty decomp commentary | 2.28->2.16, 1.33->1.24 | Match table |

## Design Decisions

| Decision | Alternatives | Rationale |
|----------|--------------|-----------|
| Phase split (numbers now, narrative later) | All in one round | User explicit ask "数字直接更新修改好" then narrative phase |
| Update narrative inline numbers (e.g. -0.30 in §4.3 commentary) | Defer all narrative | These are direct numerical claims tied to tables, not interpretive prose |
| Reframe 8-pt sanity check as "comparable" | Drop section entirely; or claim 0.9% reduction | New 8-pt MRE 2.21 vs Ma 2.23 is essentially tied; honest reframing avoids overclaim while not deleting evidence |
| Keep ablation row deltas as-is when row not retrained | Annotate or recompute synthetic deltas | User explicitly said "Ablation先不着急改" but full TRUST row updated; created inconsistency is acknowledged and Phase 2 will fix |
| Keep r/u in Table VII at old values | Drop the columns; or recompute | We have MRE only, not new uncertainty distributions; flagged for Phase 2 |

## Incremental Work Log

**Leo data inventory (earlier this session):**
- 41 ostia annotations (P8, P9) cases 111-151
- 63 P0-P7 cases 151-221 (gaps; likely from unlabeled pool joining labeled)
- 102 annulus polygons (.pf with 4x3 transform matrix; 2D verts + 3D anchor) cases 001-100 + 151 + 153
- 1 case 456 P0-P7 fix
- Polygons identified as new downstream eval opportunity (annulus area, plane orientation)

**CSV correction:** Initially counted 31 cases. User flagged: last row is column means. Recomputed with N=30.

**Compile verification:** 19 pages, 0 errors, 0 undefined refs, 0 overfull >10pt.

## Learnings & Corrections

- [LEARN:csv-trailing-mean-row] Some CSV outputs append a column-mean
  row as the last data row, not as a separate footer. Always check whether
  last row is data or aggregate before computing summary statistics.

- [LEARN:ablation-partial-retrain] When retraining only the full model
  while keeping ablation baselines fixed, the row-to-row deltas become
  apples-to-oranges. Honest options: (a) flag explicitly in caption,
  (b) retrain ablations, (c) drop delta column. Current paper accepts
  inconsistency temporarily (user choice) with intent to revisit.

- [LEARN:8pt-vs-10pt-tradeoff] New model traded slight 8-pt regression
  (2.07 -> 2.21) for big 10-pt gain (2.28 -> 2.16) by reallocating
  capacity toward P8/P9. Honest reporting requires reframing 8-pt
  sanity check rather than continuing to claim large improvement.

## Verification Results

| Check | Result | Status |
|-------|--------|--------|
| pdflatex 3-pass + bibtex compile | 19 pages | PASS |
| Undefined citations | 0 | PASS |
| Undefined references | 0 | PASS |
| Overfull \\hbox > 10pt | 0 | PASS |
| All TRUST-row table updates applied | 7 tables verified | PASS |
| All inline numerical claims updated | abstract + 4 paragraphs | PASS |

## Open Questions / Blockers

- [ ] Does user have new per-landmark **uncertainty data** for Table VII (r, u) and Figure 3 (uncertainty-error scatter)?
- [ ] Does user have new **downstream measurements** (MSL, coronary heights, calcification, C-arm, DLZ)?
- [ ] Were Stage 1 numbers (6.74 mm, 298/300 ROI containment) recomputed? Currently kept as-is.
- [ ] Are |D_L|=20, 50 TRUST runs also retrained with new data, or only |D_L|=100?
- [ ] How to handle the **annulus polygons** Leo sent? Could enable 6th downstream measurement (annulus area + plane orientation accuracy).

## Next Steps

- [ ] Phase 2 narrative updates: §4.5 per-landmark commentary (P1 regression story
      now small; P9 no longer high-uncertainty -> "unique graph position" rationale breaks)
- [ ] Phase 2 figure regen: per_landmark_grouped_bar.pdf from new data
- [ ] Phase 2 Abstract closing: rewrite "underpowered on right ostium" (now false)
- [ ] Phase 2 §5 limitations: drop "RCO underpowered" item
- [ ] Phase 3 downstream: when new clinical measurements arrive, update §4.4 + Table VIII
- [ ] Phase 3 annulus polygons: write .pf parser, add as 6th downstream measurement

---

# Continuation (2026-05-13 late session)

## Additional context

User flagged that the **initial CSV dump was incorrect** -- delivered a corrected
`final_result.csv` with revised per-landmark errors. All Phase 1 + Phase 2.1
work was redone with the corrected numbers. Key consequence: with corrected
data, the previous narrative "P1 is the only landmark not to improve" is
**no longer true** -- all 10 landmarks improve under TRUST. The P1 "unique
adjacency to P7 and P9" mechanism story has been fully removed from the paper.

## Headline number corrections (old wrong -> new correct)

| Metric | Old (wrong CSV) | New (correct CSV) |
|--------|----------------:|------------------:|
| Overall MRE (mm) | 2.16 | **2.19** |
| std | 1.24 | **1.22** |
| Reduction vs Ma (supervised) | 22.6% | **21.5%** |
| Reduction vs SCN | 30.8% | **29.8%** |
| Reduction vs vanilla CPS | 15.3% | **14.1%** |
| SDR@2.0 | 52.00% | 50.67% |
| SDR@4.0 | 92.00% | 92.33% |
| Δ ablation (final row) | -0.30 | **-0.27** |
| Δ std (final row) | -0.34 | **-0.36** |
| Label-eff Δ at 100 labels | -0.63 | **-0.60** |
| P1 vs Sup | TRUST -0.12 mm WORSE | **TRUST -0.14 mm BETTER** |

## Additional changes vs first log

| Location | Old (wrong CSV based) | New (correct CSV) |
|----------|----------------------|-------------------|
| All Phase 1 tables (I, III, IV, V, VI, VII) | Updated with wrong data | **Redone with correct data** |
| §4.3 narrative inline numbers | Updated with wrong data | **Redone** |
| §5 line 735 ablation paragraph | -0.18 (Phase 1 missed this) | **-0.27, "factor of three"** |
| §4.5 "Third" paragraph (P1 regression analysis) | Compressed to 3 sentences | **Rewrote as "all 10 improve, no regressions"** |
| §4.4 line 496 P1 calibration sentence | Softened wording | **Deleted entirely** (P1 no longer underperforms) |
| §5 line 741 P1 limitation paragraph (1 sentence) | Kept as 1 sentence | **Deleted entirely**; line 743 "additional" removed |
| Figure 4 PDF | Had P1 hatch + "TRUST worse" annotation | **Regenerated**: no hatch, no annotation, all bars consistent with "all improve" |
| Figure 4 caption | Mentioned P1 hatching + adjacency | **Rewritten**: "All 10 landmarks improve under TRUST" |
| Figure 5 PDF | Old qualitative results, Apr 28 | **User-supplied new PNG** (Case 8 / 25 / 1), converted to PDF via sips |
| Figure 5 caption | Case A/B/C old numbers (1.69/2.03/6.6) | **Updated**: Case A 1.46/1.5, Case B 2.13/2.5, Case C 2.59/3.8 |

## Plan file (visible in plan mode UI)

Created `/Users/fortisck/.claude/plans/parsed-sniffing-bear.md` as a running
TODO checklist visible in plan mode UI alongside the conversation. Updated
incrementally as items complete.

## New scripts

- `scripts/python/generate_per_landmark_bar.py` -- reproducible Figure 4
  generator, hardcoded values from Table V, restored academic palette
  (steel blue + brick red) at user request after my initial gray+blue draft.

## Verification (Phase 2.4 + 2.5)

| Check | Result |
|-------|--------|
| pdflatex 3-pass + bibtex | 18 pages (was 19; -1 page from §5 P1 paragraph deletion + §4.5 Third shorter) |
| Undefined citations | 0 |
| Undefined references | 0 |
| Overfull \\hbox > 10pt | 0 |
| Figure 4 visual check | All TRUST bars <= matching supervised bars (consistent with "no regression") |
| Figure 5 visual check | New PNG embedded, caption numbers match (1.46 / 2.13 / 2.59) |

## Learnings & Corrections (this round)

- [LEARN:data-trust] First CSV dump was wrong. Verify with user before
  treating CSV numerical dumps as ground truth; specifically check whether
  per-landmark MRE rounds to round numbers (suspect) or stays at full
  precision (more likely correct). The wrong CSV had values that happened
  to make P1 the worst landmark.

- [LEARN:narrative-coupling] When updating Phase 1 numbers, missed that
  §5 line 735 also references the ablation Δ values (-0.18 / "factor of two").
  Cross-section number reuse is common in technical writing -- grep all sections
  for repeated quantitative claims, not just the local table commentary.

- [LEARN:user-instinct] User pushed back on my "drop the P1 explanation
  entirely" recommendation by noting that the topology argument (P1
  adjacent to BOTH P7 and P9) is anatomically invariant and would still
  hold. I conceded and we landed on a softer version. Then the corrected
  data made the whole P1 discussion moot anyway -- still a valid process
  lesson: the user's instinct that an explanation can be saved by
  refactoring its premises (topology, not uncertainty) was correct.

## Outstanding Before 2026-05-15 Addenda

- Phase 2.2 Abstract was still deferred at this point; completed on 2026-05-15.
- Phase 3 §4.4 downstream was waiting for new measurements at this point; later updated using the available phase-5 downstream/discordance outputs.
- Phase 3 Table VII r/u + Fig 3 was waiting for new uncertainty data at this point; the old uncertainty scatter was later removed rather than refreshed.
- Stage 1 retraining status: not confirmed.
- Label-efficiency Table VI |D_L|=20, 50 rows: not confirmed.
- Annulus polygons: parser not yet written.

---

# 2026-05-15 Addendum: Uncertainty Scatter and Fusion Story Cleanup

## Context

The old uncertainty-error scatter plot and Pearson `r = 0.908` were based on stale uncertainty/model outputs and no longer matched the corrected retraining results. The user proposed replacing that figure with a Student 1 / Student 2 / Fuse story using `S1S2&Fuse.csv`, because the G2LCPS dual teacher/student structure does not appear to emphasize inference-time fusion.

After computing the matched 30-case metrics, the fusion result proved practically tied with the stronger student rather than clearly better:

| Output | MRE ± std | SDR@2.0 | SDR@2.5 | SDR@3.0 | SDR@4.0 |
|--------|----------:|--------:|--------:|--------:|--------:|
| Student 1 | 2.20 ± 1.27 | 51.67 | 67.67 | 78.00 | 91.00 |
| Student 2 | 2.44 ± 1.36 | 43.33 | 57.67 | 70.33 | 88.33 |
| Fuse | 2.19 ± 1.22 | 50.67 | 67.00 | 77.33 | 92.33 |

Professional judgment: this does **not** support a standalone Results story or novelty claim. Fuse is useful as the final-output rule in the Methods, but not as an independent accuracy contributor in the main Results.

## Changes Made

| Location | Change |
|----------|--------|
| `paper/main.tex` Results | Removed the old `Uncertainty calibration` paragraph and scatter figure. |
| `paper/main.tex` Results | Temporarily added, then removed, the standalone `Inference-time cross-student fusion` paragraph and Table `tab:fusion` after realizing Fuse is practically tied with Student 1. |
| `paper/main.tex` Contributions | Replaced `inference-time fusion analysis` with `ablation studies`. |
| `paper/main.tex` Methods | Kept the fusion equation and inference definition intact, because TRUST's final reported output remains the fused prediction. |
| `quality_reports/plans/2026-05-14_replace-scatter-with-fusion-analysis.md` | Marked as superseded. |
| `quality_reports/plans/2026-05-15_remove-fusion-results-section.md` | Added final plan/result record for deleting the standalone fusion section. |
| `MEMORY.md` | Added reminders to treat Fuse and Student 1 as practically tied and to provide independent professional judgment before agreeing with manuscript-direction suggestions. |

## Verification

After deleting the standalone fusion section:

| Check | Result |
|-------|--------|
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Stale `tab:fusion` references | 0 |
| Overfull \\hbox > 10pt | 0 |
| Remaining overfull boxes | Two tiny existing warnings: 0.61pt and 0.55pt |

## Updated Outstanding Items

- Abstract final correction completed on 2026-05-15: `2.16 / 22.6% / five measurements / RCO underpowered` was replaced with `2.19 / 21.5% / six measurements` and a balanced closing that frames annular-area under-prediction plus cohort-limited per-ostium sensitivity as validation targets.
- The old uncertainty scatter is no longer an outstanding item; it has been removed.
- The Student1/Student2/Fuse comparison should not be promoted in the main paper unless moved to supplement or requested explicitly.
- Methods still define fusion as the final output rule; Results should not frame it as an independent contribution.

## 2026-05-15 Abstract Finalization

Rewrote the abstract as a single 247-word paragraph focused on unified uncertainty as a training-time reliability signal consumed by CPS, UG-HCO, and Topo-GCN. The abstract now matches the corrected retraining result (`2.19` mm, `21.5%`) and the six-measurement downstream scope. It no longer mentions the old left-ostium/right-ostium asymmetric conclusion.

Verification after the abstract update:

| Check | Result |
|-------|--------|
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Stale old abstract terms in `paper/main.tex` | 0 |
| Overfull \\hbox > 10pt | 0 |

## 2026-05-18 Chinese Mirror Expansion

The user correctly questioned whether `paper/main_zh.md` was truly consistent with the English source, because the 2026-05-16 Chinese file was numerically aligned but intentionally condensed.

Actions:

- Replaced the 354-line condensed Chinese reading draft with a 673-line paragraph-level Chinese mirror of the current `paper/main.tex`.
- Preserved the current final scientific decisions: TRUST `2.19` mm, `21.5%` reduction, six downstream clinical measurements, no standalone Student/Fuse results story, and no stale uncertainty scatter narrative.
- Expanded the Chinese file to include the main formulas, table values, figure-caption summaries, clinical measurement definitions, result interpretations, discussion, limitations, and conclusion.
- Updated `quality_reports/plans/2026-05-18_expand-main-zh-full-mirror.md` with completed checklist and verification notes.

Verification:

| Check | Result |
|-------|--------|
| `paper/main_zh.md` line count | 673 lines |
| Stale terms (`2.28`, `18.3%`, `0.908`, `scatter`, `逐病例均值`, `per-case mean`) | 0 matches |
| Current key terms (`2.19`, `21.5%`, `六项`, `46.7%`, `75.0%`, `n+=1`, `9.4%`, `43.1`, `6.7`, `4.8`) | Present |
| `paper/main.tex` modified in this step | No |

## 2026-05-18 Sentence-Level Chinese Review Draft

The user clarified that the Chinese manuscript should support direct review without repeatedly returning to the English source. A paragraph-level mirror was still too compressed for that workflow.

Actions:

- Rewrote `paper/main_zh.md` again as a sentence-level Chinese review draft of the current `paper/main.tex`.
- Expanded the file from 673 lines to 1466 lines.
- Preserved the English manuscript order, main citation keys, formulas, figure-caption content, table values, method details, ablation interpretations, clinical measurement definitions, downstream results, discussion, limitations, and conclusion.
- Added `quality_reports/plans/2026-05-18_sentence-level-main-zh.md` and marked it complete.

Verification:

| Check | Result |
|-------|--------|
| `paper/main_zh.md` line count | 1466 lines |
| Stale terms (`2.28`, `18.3%`, `22.6`, `2.16`, `0.908`, `scatter`, `逐病例均值`, `per-case mean`) | 0 matches |
| Current key terms (`2.19`, `21.5%`, `六项`, `46.7%`, `75.0%`, `n$_+$=1`, `9.4%`, `43.1`, `6.7`, `4.8`, `-0.27`, `-0.36`) | Present |
| `paper/main.tex` modified in this step | No |

## 2026-05-26 Carlos Review: Per-Landmark / Clinical Analysis Cleanup

Addressed Carlos's annotations around Section 4.5 without changing the reported metrics:

- Compressed the redundant functional enumeration before Table VI into one sentence that points back to Fig. 1.
- Removed arbitrary bold styling from the per-landmark table values for `P6` and overall TRUST.
- Replaced the problematic `\mathbb{1}` indicator in the DLZ calcification equation with `\mathbb{I}\{...\}`.
- Regenerated `figs/per_landmark_grouped_bar.pdf/.png`; group shading now follows Fig. 1 semantics: hinges blue, commissures yellow, center green, MS red, coronary ostia purple.
- Removed redundant `P7` from the MS group label in Fig. 3.
- Synchronized `paper/main_zh.md`.

Verification:

| Check | Result |
|-------|--------|
| `python3 scripts/python/generate_per_landmark_bar.py` | Passed; PDF/PNG regenerated |
| `python3 -m py_compile scripts/python/generate_per_landmark_bar.py` | Passed |
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |

## 2026-06-08 Threshold-Language Cleanup

Addressed the clinician feedback that MSL and coronary-height millimeter cutoffs should not be presented as direct clinical classification rules.

Actions:

- Reframed MSL 2/5 mm bins as analytical MSL strata used for model evaluation, not standalone clinical risk categories.
- Reframed coronary height 12 mm as a reference cutoff for threshold-based evaluation, emphasizing that coronary obstruction risk depends on multiple anatomical factors.
- Replaced overclaiming phrases such as clinical decision discordance, high-risk/low-risk classification, and misclassification with threshold-based discordance, below-cutoff cases, threshold crossings, and extreme-tier shifts.
- Updated the abstract, clinical downstream validation, Table IX, Table X, Discussion, Limitations, and Conclusion in `paper/main.tex`.
- Synchronized the corresponding wording in `paper/main_zh.md`.
- Saved the plan in `quality_reports/plans/2026-06-08_threshold-language-cleanup.md`.

Verification:

| Check | Result |
|-------|--------|
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |

## 2026-06-08 Same-Test-Set Multi-Reader Analysis

Integrated the new n=30 multi-reader/model comparison results into the manuscript as a restrained single-center, same-test-set analysis rather than a multi-center validation claim.

Actions:

- Validated and archived the three n=30 CSV summaries under `quality_reports/multi_reader_analysis/`.
- Added a plan in `quality_reports/plans/2026-06-08_add-multi-reader-analysis.md`.
- Updated `paper/main.tex` dataset and evaluation-metric descriptions to state that the same 30-case test set has independent additional 8-landmark annotations.
- Added the compact multi-reader results table comparing Reader 1 vs Reader 2, TRUST vs Reader 1, and TRUST vs Reader 2.
- Integrated the main interpretation: P7 is the dominant reader-dependent landmark; TRUST is in the same order of magnitude as inter-reader variability but does not fully reach it; DLZ totals remain highly correlated.
- Updated Discussion, Limitations, and Conclusion to distinguish single-center multi-reader context from full multi-center or 10-landmark multi-reader validation.
- Added the 2026 Lemarchand cusp-overlap reference and synchronized the corresponding Chinese mirror text in `paper/main_zh.md`.

Verification:

| Check | Result |
|-------|--------|
| pdflatex after BibTeX update | Passed |
| PDF page count | 19 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |

## 2026-05-28 Unlabeled-Pool Ablation

Added the Carlos-requested ablation over the number of unlabeled training volumes.

Actions:

- Added a compact `Unlabeled-pool size` paragraph and new table to `paper/main.tex`.
- Fixed the labeled set at \(|\mathcal{D}_L|=100\) and varied \(|\mathcal{D}_U| \in \{0,100,300,620\}\).
- Reported MRE, SDR\(_{2.0}\), SDR\(_{4.0}\), and \(\Delta\)MRE relative to the no-unlabeled supervised counterpart.
- Framed the result as monotonic improvement with the strongest gain at the full unlabeled pool, avoiding a linear-scaling claim.
- Synchronized the Chinese review mirror in `paper/main_zh.md` and shifted later Chinese table numbers accordingly.
- Saved the plan in `quality_reports/plans/2026-05-28_unlabeled-pool-ablation.md`.

Verification:

| Check | Result |
|-------|--------|
| pdflatex + bibtex + pdflatex + pdflatex | Passed after rerunning bibtex with `BIBINPUTS=..:$BIBINPUTS` |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |
| Visual page check | New Table VI fits on page 11 |

## 2026-05-28 Carlos Review: Restore Discussion, Shorten Conclusion

The "shorter" comment was clarified to apply to the Conclusion rather than the Discussion.

Actions:

- Restored the detailed Discussion text after the mistaken shortening attempt.
- Preserved the revised Limitations ordering requested by Carlos: single-center/single-annotator and high-risk case scarcity first; inference-time uncertainty saturation second; cascaded ROI failure third; sparse landmarks vs full annular contour last.
- Compressed the Conclusion to a shorter paragraph focused on TRUST, unified uncertainty, the main 2.19 mm / 21.5% result, key downstream findings, and future validation/contour work.
- Synchronized `paper/main_zh.md`.
- Removed the abandoned `2026-05-28_discussion-shortening.md` plan and kept `2026-05-28_restore-discussion-compress-conclusion.md`.

Verification:

| Check | Result |
|-------|--------|
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |

## 2026-05-27 Carlos Review: Discussion Opening Summary

Addressed Carlos's suggestion that the Discussion should begin with a paragraph summarising the main contributions, using the first paragraph of the Conclusion as source material without copying it verbatim.

Actions:

- Added a new opening paragraph to `paper/main.tex` Discussion before the ablation interpretation.
- The paragraph summarizes TRUST as a semi-supervised 10-landmark TAVI planning framework, unified uncertainty as the central methodological finding, the main 2.19 mm / 21.5% result, and the key downstream preservation findings.
- Added a transition sentence explaining that the Discussion first interprets unified uncertainty and then residual landmark-error propagation.
- Synchronized the corresponding paragraph in `paper/main_zh.md`.
- Saved the plan in `quality_reports/plans/2026-05-27_discussion-opening-summary.md`.

Verification:

| Check | Result |
|-------|--------|
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |

## 2026-05-27 Annular Area Definition Correction

The downstream evaluation code confirmed that annular area is not computed by projecting the manual polygon onto the predicted annular plane. The reference area is the 2D shoelace area of the MITK annular polygon, and the prediction-side area is the three-hinge circumcircle approximation, \(A_{\text{circ}}^{\text{pred}} = \pi R_{\triangle,\text{pred}}^2\), using TRUST-predicted \(P_0\)--\(P_2\).

Actions:

- Corrected `paper/main.tex` to describe manual polygon area vs predicted three-hinge circumcircle area.
- Changed the area-analysis sample size from all 30 test cases to the 14 cases with available annular-contour polygons.
- Reframed the -9.4% bias as circular-approximation plus hinge-localization error, not oblique-plane projection.
- Updated the table row label to `Annular area approx.`.
- Synchronized the corresponding explanation, discussion, and conclusion text in `paper/main_zh.md`.
- Added a `[LEARN:annular-area]` note to `MEMORY.md`.

Verification:

| Check | Result |
|-------|--------|
| Stale projection terms in English/Chinese | 0 matches for the targeted old phrases |
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |

## 2026-05-28 Acronym Consistency Pass

Addressed the global abbreviation issue raised after the Carlos review pass.

Actions:

- Treated the abstract and main text as separately self-contained for first-use abbreviation definitions.
- Standardized first-use definitions in `paper/main.tex` for CT, 3D, CPS, MSL, LCO/RCO, ROI, EMA, UG-HCO, Topo-GCN, MSE, IDCT, MLP, GPU, FPS, CI, DLZ, HU, CTA, SCCT, LAO/RAO, CRA/CAU, MAE, ICC, LoA, Sens., and MITK where relevant.
- Removed or rewrote undefined/over-compact table and prose abbreviations including `GT`, `MS Point`, and one-off `TAVR`.
- Kept common method names and product names as abbreviations once defined or self-evident in context.
- Synchronized the corresponding wording and abbreviation explanations in `paper/main_zh.md`.
- Saved the plan in `quality_reports/plans/2026-05-28_acronym-consistency-pass.md`.

Verification:

| Check | Result |
|-------|--------|
| pdflatex + bibtex + pdflatex + pdflatex | Passed |
| PDF page count | 18 pages |
| Undefined citations/references | 0 |
| Remaining overfull boxes | One tiny existing warning: 0.55pt at line 354 |
