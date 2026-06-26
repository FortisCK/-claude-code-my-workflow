# Annular Area Circumcircle Correction

Date: 2026-05-27

Context:
- Carlos highlighted the annular-area comparison because the current manuscript says the predicted area is computed by projecting the manual polygon onto the TRUST-predicted annular plane.
- The downstream evaluation code confirms a different computation:
  - `gt_area = polygon_area_2d(poly)`: 2D shoelace area of the manually annotated MITK annular polygon.
  - `pred_area = np.pi * Rtri * Rtri`: three-hinge circumcircle area from TRUST-predicted `P0`, `P1`, `P2`.

Planned edits:
- Replace the incorrect polygon-projection description with the true manual-contour-vs-predicted-circumcircle definition.
- Clarify that this metric combines circular-approximation error and landmark prediction error.
- Rename the table row to avoid implying full automatic contour prediction.
- Remove "oblique-plane projection" explanations from Results, Discussion, and Conclusion.
- Synchronize `paper/main_zh.md`.
- Compile and verify.
