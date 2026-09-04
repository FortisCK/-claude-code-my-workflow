# Validation notes for n=30 multi-reader/model comparison CSVs

Date: 2026-06-08

Checked files:

- `all_comparison_downstream_summary_n30_consistent.csv`
- `all_comparison_landmark_summary_by_point_n30_consistent.csv`
- `all_comparison_landmark_summary_n30_consistent.csv`

## Basic integrity

- All three files load correctly as CSV.
- No duplicate rows were found.
- Landmark summaries are internally consistent:
  - each comparison has 8 landmark rows with `n=30` each.
  - the per-point weighted means match the overall summary rows.
  - total landmark observations per comparison: `30 cases x 8 landmarks = 240`.
- Downstream summary has 3 comparisons x 10 measurements = 30 rows.
- Bland-Altman limits match `bias +/- 1.96*sd` where applicable.
- Missing values in the downstream table are expected for non-applicable fields:
  - plane-normal angle has no scalar reference/target mean, bias, Pearson, or ICC.
  - MSL tier discordance applies only to MSL.
  - D_circ bin discordance applies only to D_circ.
  - DLZ any-discordance applies only to DLZ total.

## Important limitation

These files are `n30_consistent` by design: they use the same 30-case paper test set selected from the 50-case re-annotation pool.

They can support a compact same-test-set, single-center multi-reader analysis, but they should not be described as results from all 50 re-annotated cases.

## Key landmark results

Overall 8-landmark summary:

| Comparison | n | MRE (mm) | SDR@2mm | SDR@3mm | SDR@4mm | >10mm outliers |
|---|---:|---:|---:|---:|---:|---:|
| Reader1 vs Reader2 | 240 | 1.87 +/- 1.80 | 70.42% | 87.08% | 89.58% | 2 |
| TRUST vs Reader1 | 240 | 2.20 +/- 1.09 | 46.67% | 78.33% | 94.58% | 0 |
| TRUST vs Reader2 | 240 | 2.47 +/- 1.71 | 43.33% | 75.42% | 88.75% | 4 |

P7/MS point is the dominant difficulty:

| Comparison | P0-P6 weighted mean (mm) | P7 mean (mm) | P7 max (mm) | P7 >10mm outliers |
|---|---:|---:|---:|---:|
| Reader1 vs Reader2 | 1.47 | 4.69 | 13.68 | 2 |
| TRUST vs Reader1 | 2.08 | 3.07 | 6.22 | 0 |
| TRUST vs Reader2 | 2.08 | 5.20 | 12.53 | 4 |

Interpretation:

- The data strongly support the claim that P7/MS is intrinsically difficult and reader-dependent.
- Before manuscript integration, the P7 outlier cases should be inspected at case level to rule out point-order or coordinate issues.

## Key downstream results

- MSL:
  - Reader1 vs Reader2 MAE: 2.26 mm; tier discordance: 13/30; critical discordance: 2/30.
  - TRUST vs Reader1 MAE: 1.51 mm; tier discordance: 15/30; critical discordance: 0/30.
  - TRUST vs Reader2 MAE: 2.25 mm; tier discordance: 15/30; critical discordance: 0/30.
- D_circ:
  - Reader1 vs Reader2 MAE: 0.81 mm; sizing-bin discordance: 4/30.
  - TRUST vs Reader1 MAE: 1.03 mm; sizing-bin discordance: 9/30.
  - TRUST vs Reader2 MAE: 1.25 mm; sizing-bin discordance: 9/30.
- C-arm angles:
  - Reader1 vs Reader2 MAE: 4.17 deg LAO/RAO, 2.85 deg CRA/CAU.
  - TRUST vs readers: approximately 6.5 deg LAO/RAO and 3.7-4.7 deg CRA/CAU.
- DLZ calcification:
  - Very high correlations across comparisons.
  - DLZ total any-discordance is 0/30 in all comparisons.
  - The exact clinical meaning of `dlz_any_discordance` should be confirmed from the server script before writing it in the manuscript.

## Usability judgment

Usable now:

- overall 8-landmark reader-reader variability.
- per-landmark variability, especially P7.
- MSL reader variability and tier discordance.
- D_circ and C-arm reader variability.

Use cautiously:

- DLZ any-discordance, until the exact definition is confirmed.
- wording around `Reader2`, because the folder-level source may combine Jean/Leo re-annotations even though the comparison table treats it as a single independent reader/reference stream.

Still needed:

- case-level CSVs with case IDs and per-case/per-landmark errors.
- outlier list for P7 >10 mm.
- confirmation of the exact case IDs for auditability, although the summary is intended to match the paper's 30-case test set.
- confirmation whether `Reader2` is one additional reader or a combined Jean/Leo re-annotation set before final wording is frozen.
