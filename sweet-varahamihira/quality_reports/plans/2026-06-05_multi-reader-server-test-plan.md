# 50-case multi-reader test plan for TRUST

Date: 2026-06-05

Purpose: use the newly provided 50-case 8-landmark annotations to quantify human annotation variability, then decide how much of that analysis should be added to the TRUST paper.

## 1. What the new data are

Local source paths:

- `/Users/fortisck/Downloads/Jean/`
- `/Users/fortisck/Downloads/Leo-2/`

Observed structure:

- `Jean/`: 30 `.mps` files, covering cases `101-110` and `131-150`.
- `Leo-2/`: 20 `.mps` files, covering cases `111-130`.
- Together they cover `101-150`, exactly 50 cases.
- Each `.mps` contains 8 points, not 10 points.

Interpretation:

- These are suitable for the original 8-landmark protocol, i.e. `P0-P7`.
- They do not include coronary ostia `P8/P9`, so they cannot directly validate coronary height or coronary obstruction screening.
- Before writing the manuscript text, confirm whether `Jean` and `Leo-2` are the same second reader split across folders, or two different additional readers. If two readers, write "additional clinical readers" rather than "a second reader".

Recommended wording if used in the paper:

> To contextualize model performance against human annotation variability, we performed a single-center multi-reader analysis on an additional 50-case subset independently re-annotated using the original 8-landmark protocol.

Avoid:

- "multi-center"
- "external validation"
- "10-landmark multi-reader validation"
- "coronary-ostium reader variability"

## 2. Required server inputs

To run the analysis on the server, locate or copy the following:

1. New second-reader annotations:
   - `Jean/*.mps`
   - `Leo-2/*.mps`

2. Original first-reader annotations for the same 50 case IDs:
   - likely the original Ma/Qixiang 8-landmark `.mps` files, or the original 10-landmark GT converted to the same physical coordinate system.
   - Required case IDs: `101-150`.
   - Required points: at least `P0-P7`.

3. TRUST predictions, if available:
   - best FUSE predictions for cases `101-150`, preferably in physical coordinates or with enough metadata to convert to physical coordinates.
   - If predictions are only available for the official 30-case test set, then only the overlap can be used. Previously identified overlap with the paper's 30-case downstream cohort:
     `103, 105, 107, 108, 111, 112, 115, 117, 119, 120, 123, 124, 126, 127, 129, 130` (`n=16`).

4. CT volumes and preprocessing metadata, only if running DLZ calcification:
   - HU volumes in the same coordinate system as the annotations.
   - voxel spacing / affine / origin metadata.
   - existing downstream calcification code, if available.

## 3. First-pass data QC

Run these checks before calculating any metrics:

1. Count files:
   - total `n=50`.
   - no missing IDs in `101-150`.

2. Count points:
   - every `.mps` should have exactly 8 points.

3. Point order:
   - confirm `id=0..7` corresponds to the paper's `P0-P7` order.
   - Expected protocol:
     - `P0-P2`: hinge points defining the annular plane.
     - `P3-P5`: commissures.
     - `P6`: valve center.
     - `P7`: membranous septum point.

4. Coordinate system:
   - first-reader and second-reader annotations must be in the same physical coordinate system.
   - A quick sanity check is to compute per-case centroid distance between readers. If many cases show tens or hundreds of mm, there is a coordinate transform mismatch.

5. Outlier check:
   - flag any per-landmark reader-reader distance greater than 10 mm.
   - visually inspect those cases before reporting final statistics.

## 4. Core test A: landmark-level reader variability

Question answered:

> How large is human annotation variability for the same 8 landmark points?

Compute, for each case and each point `P0-P7`:

```text
error_k = || reader2(P_k) - reader1(P_k) ||_2
```

Report:

- overall 8-landmark MRE: mean and standard deviation.
- per-landmark MRE for `P0-P7`.
- median and IQR, if the distribution is skewed.
- SDR at `2.0`, `2.5`, `3.0`, and `4.0` mm.
- number of gross outliers, e.g. `error > 10 mm`.

Recommended table:

| Landmark | Reader-reader MRE (mm) | SDR2.0 (%) | SDR2.5 (%) | SDR3.0 (%) | SDR4.0 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | | | | | |
| ... | | | | | |
| P7 | | | | | |
| Mean | | | | | |

How to use in the paper:

- This is the strongest and cleanest use of the 50 cases.
- It directly addresses the limitation that the main test set was annotated by one trained clinician.
- Compare cautiously against TRUST's published 8-landmark result (`2.07 +/- 0.98 mm`) because the case sets may differ.

## 5. Core test B: model error relative to reader variability

Question answered:

> Is TRUST's 8-landmark error comparable to human reader variability?

Best version:

- Run TRUST FUSE inference on all 50 cases.
- Compare model predictions to reader 1 and reader 2 separately:
  - `model vs reader1`
  - `model vs reader2`
  - `reader1 vs reader2`

Fallback version:

- If model predictions exist only for the official test cases, use the overlap subset only (`n=16`).
- Report this as a small sensitivity analysis, not a replacement for the main test result.

Recommended output:

| Comparison | Cases | Landmarks | MRE (mm) | SDR2.0 (%) | SDR2.5 (%) | SDR3.0 (%) | SDR4.0 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reader 1 vs Reader 2 | 50 | P0-P7 | | | | | |
| TRUST vs Reader 1 | 50 or overlap | P0-P7 | | | | | |
| TRUST vs Reader 2 | 50 or overlap | P0-P7 | | | | | |

Interpretation guide:

- If TRUST vs reader is close to reader-reader variability, that is a strong result.
- If TRUST is clearly worse, still useful: it tells us how far model performance is from human variability.
- If only `n=16`, phrase cautiously and avoid strong claims.

## 6. Core test C: MSL reader variability

Question answered:

> How reproducible is membranous septum length (MSL) under independent annotation?

Compute:

- annular plane from `P0, P1, P2`.
- MSL as perpendicular distance from `P7` to that plane.

Report:

- MAE between readers.
- mean bias (`reader2 - reader1`).
- Pearson `r`.
- ICC(2,1), if implemented.
- Bland-Altman limits of agreement.
- risk-tier discordance at:
  - high risk: `MSL <= 2 mm`
  - intermediate: `2 < MSL <= 5 mm`
  - low risk: `MSL > 5 mm`
- critical discordance:
  - high risk classified as low risk, or low risk classified as high risk.

Why this matters:

- The meeting emphasized MSL as clinically important for conduction disturbance / pacemaker risk.
- The new clinical paper also supports the importance of MSL and implantation depth relative to MSL.
- If human readers often disagree near the 2 mm / 5 mm thresholds, it contextualizes TRUST's adjacent-tier MSL shifts.

Recommended table:

| Measurement | Reader-reader MAE | Bias | Pearson r | ICC | 95% LoA | Clinical discordance |
|---|---:|---:|---:|---:|---:|---:|
| MSL | | | | | | |

## 7. Core test D: annular geometry reader variability

Question answered:

> How stable are annular plane and three-hinge sizing quantities under independent annotation?

Compute from `P0-P2`:

1. Annular plane normal difference:
   - angle between reader 1 and reader 2 plane normals.

2. Annular circumdiameter `D_circ`:
   - circumcircle diameter through `P0, P1, P2`.

3. Sizing-bin discordance:
   - use the same bins currently used in the paper: `23 / 26 / 29 mm`.
   - report adjacent-bin and more-than-adjacent-bin shifts.

Recommended output:

| Measurement | Reader-reader MAE | Bias | ICC | Discordance |
|---|---:|---:|---:|---:|
| Plane normal angle | | | | |
| D_circ | | | | |
| D_circ sizing bin | | | | |

Why this matters:

- It directly connects to the doctors' concern that clinical sizing relies on annular area/perimeter, while our landmark-only `D_circ` is a three-hinge approximation.
- If `D_circ` is stable between readers, it supports its use as a geometric proxy.
- It still should not be described as full prosthesis sizing validation.

## 8. Optional test E: C-arm cusp-overlap angle reader variability

Question answered:

> How sensitive is the predicted C-arm projection to reader annotation differences?

Compute:

- annular plane from `P0-P2`.
- cusp-overlap beam direction using the LCC-RCC axis (`P2-P1`) as in the current paper.
- LAO/RAO and CRA/CAU angle differences between readers.

Report:

- MAE for LAO/RAO.
- MAE for CRA/CAU.
- Bland-Altman limits, if space permits.

Use in paper:

- Useful, but lower priority than MSL and D_circ.
- Can be a supplement or one sentence in the multi-reader subsection.

## 9. Optional test F: DLZ calcification reader variability

Question answered:

> How much do per-cusp DLZ calcification volumes change when the annular plane and cusp sectors are defined by a different reader?

Required inputs:

- CT HU volumes.
- reader 1 and reader 2 annotations for `P0-P5`.
- calcification code.

Compute:

- per-cusp DLZ calcification volume using reader 1 landmarks.
- per-cusp DLZ calcification volume using reader 2 landmarks.
- agreement for NCC, LCC, RCC volumes.

Important protocol issue:

- The current TRUST downstream code uses:
  - slab: `5 mm below` the annular plane.
  - HU threshold: `800`.
- The new Lemarchand et al. 2026 paper uses:
  - DLZ cylinder: `3 mm above` to `2 mm below` the annular plane.
  - HU threshold: `850`.
  - systematic expert review to detect artifacts.

Recommendation:

- If the goal is consistency with the current TRUST paper, first use the current TRUST protocol.
- If time permits, run a sensitivity analysis with the Lemarchand 2026 protocol.
- Do not mix the two definitions without explicitly stating the difference.

Why this matters:

- The meeting emphasized DLZ calcification as clinically relevant but sensitive to definition and contrast/artifact.
- A reader-variability analysis would show whether the calcification endpoint is stable under landmark annotation differences.

## 10. What the new 2026 paper contributes

PDF path:

- `/Users/fortisck/项目/arotic-landmarks/.claude/worktrees/sweet-varahamihira/s12928-026-01276-0.pdf`

Paper:

- Lemarchand et al., 2026.
- Title: "Influence of The Cusp-Overlap and three cusps coplanar techniques on new-onset conduction disturbances following transcatheter aortic valve implantation".
- Journal: Cardiovascular Intervention and Therapeutics.
- DOI visible in PDF: `10.1007/s12928-026-01276-0`.

Useful content for TRUST:

1. Clinical rationale for MSL:
   - The paper reinforces that implantation depth relative to membranous septum length is a key determinant of new-onset conduction disturbances after TAVI.
   - This supports our choice of MSL as a downstream clinical measurement.

2. Clinical rationale for DLZ calcification:
   - The paper treats calcification volume and distribution in the aortic valve complex as clinically relevant covariates.
   - It reports per-sector DLZ calcification, including NCC/LCC/RCC sectors.
   - This supports the idea that per-cusp DLZ calcification is a meaningful downstream endpoint.

3. Cusp-overlap context:
   - The paper discusses cusp-overlap vs three-cusp coplanar projections.
   - It frames cusp-overlap as a facilitating imaging technique rather than a direct independent determinant of conduction outcomes.
   - This supports our cautious wording that TRUST-derived C-arm angles can serve as a starting projection refined intraoperatively.

4. Annular measurement context:
   - The paper states that annulus area and diameter are manually obtained during CT postprocessing.
   - This supports our limitation that true prosthesis sizing relies on contour-derived area/perimeter, not only three hinge points.

5. Observer variability:
   - The paper mentions intra-observer variability assessment for MSL on 50 randomly selected cases.
   - This supports the relevance of adding our own single-center multi-reader analysis.
   - If the supplementary material contains exact variability numbers, they may be useful for comparison.

Important caution:

- The paper is not an AI landmark detection validation paper.
- It should not replace our Ma et al. comparison.
- It should be cited mainly for clinical motivation and downstream measurement definitions, not as a model baseline.

## 11. Recommended manuscript placement

Do not merge this analysis into the main benchmark table.

Recommended placement:

- Add a separate subsection after the current downstream clinical analysis:

```text
Single-center Multi-reader Analysis
```

Suggested subsection logic:

1. Briefly describe the 50-case re-annotation subset.
2. State that it follows the original 8-landmark protocol and excludes coronary ostia.
3. Report reader-reader landmark variability.
4. Report MSL and D_circ reader variability.
5. Compare qualitatively with TRUST 8-landmark error.
6. State that this contextualizes, but does not replace, the primary 10-landmark test evaluation.

Recommended main-text output:

- one compact table.
- one short paragraph of interpretation.

Possible supplement:

- per-landmark detailed reader variability.
- Bland-Altman plots for MSL and D_circ.
- optional C-arm angle results.
- optional DLZ calcification reader-variability results.

## 12. Suggested final decision rules

After results are computed:

1. If reader-reader MRE is close to TRUST 8-landmark MRE:
   - use this as a strong supporting analysis.
   - emphasize that TRUST approaches human annotation variability for the original 8-landmark subset.

2. If reader-reader MRE is much lower than TRUST:
   - still include as context, but write conservatively.
   - emphasize that multi-reader analysis quantifies the remaining gap to human-level reproducibility.

3. If P7 reader variability is high:
   - this supports our existing claim that the membranous septum point is intrinsically difficult.
   - connect it to the MSL adjacent-tier discordance discussion.

4. If MSL tier discordance is frequent between readers:
   - use it to explain why intermediate-tier MSL shifts are expected near clinical thresholds.

5. If D_circ bin discordance is low:
   - supports the stability of the hinge-derived sizing proxy.
   - still maintain the limitation that full sizing requires annular contour area/perimeter.

6. If DLZ calcification is robust between readers:
   - strong support for keeping the DLZ downstream endpoint.
   - mention protocol-specific definition and artifact review limitations.

## 13. Minimal deliverables from the server run

Please export these files if possible:

1. `multi_reader_landmark_errors.csv`
   - case ID, landmark ID, reader-reader error.

2. `multi_reader_landmark_summary.csv`
   - per-landmark MRE/SDR summary.

3. `multi_reader_downstream_summary.csv`
   - MSL, D_circ, C-arm, optional DLZ agreement metrics.

4. `multi_reader_model_comparison.csv`
   - only if TRUST predictions are available for the same cases.

5. `multi_reader_outliers.csv`
   - cases/landmarks with large reader-reader disagreement.

6. Optional figures:
   - `ba_msl_reader_variability.pdf`
   - `ba_dcirc_reader_variability.pdf`
   - `per_landmark_reader_variability.pdf`

