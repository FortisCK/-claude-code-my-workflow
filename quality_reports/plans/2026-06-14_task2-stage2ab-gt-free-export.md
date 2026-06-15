# Task2 Stage2AB GT-Free Export Plan

Date: 2026-06-14

## Goal

Turn the current Stage2AB public-validation champion into a reusable inference/export path that does not require validation ground truth. The output should keep the frozen domain-aware score policy while preventing GT or diagnostic leakage fields from entering prediction CSVs.

## Steps

1. Inspect the current Stage2AB sweep/export assumptions and identify which files still require GT.
2. Add a frozen-policy exporter that reads only candidate rows and verifier prediction rows.
3. Validate candidate/prediction alignment, domain handling, score-column availability, box geometry, and clean output schema.
4. Add tests for policy selection, missing domain handling, and row alignment failures.
5. Run the exporter on the current public-validation Stage2U YOLO-only run and compare the generated predictions against the existing Stage2AB public-valid export.
6. Record a packaging decision with the command to use once official validation/hidden-test candidate rows are available.

## Acceptance Criteria

- Exporter can generate `*_domain_policy_predictions.csv` files without reading GT columns.
- Output schema equals the existing clean internal schema.
- Forbidden validation fields are absent from output.
- Current public-validation export reproduces Stage2AB scores/rows when passed through the existing verifier and metric script.
- Tests pass.

