# Task2 Stage2AQ Multi-Source Hard-Negative Reranker Plan

Date: 2026-06-14

## Goal

Improve the weakest current Task2 mode: phantom class1 detection.

Stage2AP showed that the current Stage2AE yolo-only pool has both low candidate
coverage and poor hard-negative ranking for phantom class1. Stage2AK also
showed that older multi-source candidate pools can improve phantom class1
candidate recall. Stage2AQ should therefore test whether multi-source class1
candidates can be reintroduced with stricter source-aware scoring, without
collapsing the combined precision-recall curve.

## Constraints

- Do not replace the current Stage2AE/AI/AN champion family unless evidence is
  stronger on public validation.
- Do not tune on hidden test data.
- Keep the champion export/evaluation surface intact.
- Focus on phantom class1 first; do not launch another generic detector run.

## Steps

1. Inspect existing multi-source and source-aware artifacts from Stage2X/Y/AA.
2. Identify the candidate CSV and prediction rows that can be used without
   retraining.
3. Build a focused Stage2AQ fusion/rerank candidate:
   - keep Stage2AE for class0 and animal;
   - use a stricter multi-source or source-aware policy for phantom class1;
   - suppress known high-score hard negatives where possible.
4. Evaluate against Stage2AE/AI/AN on `valid_combined`, `valid_phantom`, and
   class-level phantom class1 AP50.
5. Record whether Stage2AQ promotes or remains diagnostic-only.

## Acceptance Criteria

- Report phantom class1 AP50 relative to Stage2AE's
  `0.045713631775941704`.
- Report valid_combined mAP50 and mAP50-95 relative to Stage2AE and Stage2AI.
- Export clean predictions if Stage2AQ is usable.
- Run py_compile and Task2 regression tests after code changes.
