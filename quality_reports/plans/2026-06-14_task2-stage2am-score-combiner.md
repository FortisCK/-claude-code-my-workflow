# Task2 Stage2AM Score Combiner Plan

Date: 2026-06-14

## Goal

Test a conservative score-level reranking/calibration route after Stage2AL failed.

Stage2AM should not fine-tune the CNN backbone or generate new candidates. It
should use existing Stage2U prediction columns plus candidate metadata to build
a lightweight score combiner that starts from the current Stage2AE/AB score
surface and only promotes if it beats the current champion on public validation.

## Motivation

Stage2AK showed a large same-candidate oracle ranking gap:

- Stage2AE `valid_phantom` mAP50: 0.10462265654206386;
- oracle-score `valid_phantom` mAP50: 0.4003400993223441;
- Stage2AE `valid_phantom` class1 AP50: 0.045713631775941704;
- oracle-score `valid_phantom` class1 AP50: 0.24037906909174153.

Stage2AL showed that full CNN yolo-only fine-tuning hurts ranking:

- best `valid_phantom` class1 AP50 across modes: 0.043619;
- Stage2AE baseline: 0.045713631775941704.

Therefore the next step should be a conservative score combiner using already
available signals, with Stage2AE retained as fallback.

## Steps

1. Inspect existing Stage2T/Stage2S tabular selector code and current prediction
   schema.
2. Implement a lightweight score-combiner diagnostic that:
   - reads candidate rows and Stage2U prediction rows;
   - derives per-class/domain features from existing score columns and candidate
     metadata;
   - trains only on non-leakage train candidate rows;
   - evaluates on `valid_combined`, `valid_phantom`, and `valid_animal`;
   - compares against Stage2AE and Stage2AI.
3. Prefer simple models first: logistic calibration / gradient boosting if
   available in the environment, otherwise deterministic score formulas.
4. Promote only if public-validation evidence improves Stage2AE without a bad
   mAP50 tradeoff.

## Acceptance Criteria

- No hidden-test tuning.
- Machine-readable JSON/CSV outputs.
- Report class/domain metrics, especially `valid_phantom` class1 AP50.
- Keep Stage2AE and Stage2AI unchanged unless Stage2AM clearly beats one.

