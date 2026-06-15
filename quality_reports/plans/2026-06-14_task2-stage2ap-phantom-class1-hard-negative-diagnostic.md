# Task2 Stage2AP Phantom Class1 Hard-Negative Diagnostic Plan

Date: 2026-06-14

## Goal

Diagnose why phantom class1 remains the weakest part of the current Task2
pipeline, despite the candidate pool containing some valid boxes.

Stage2AK showed that phantom class1 is both a candidate and ranking problem:

- any-candidate recall at IoU 0.50 is low;
- top-ranked predictions are much worse than the oracle candidate selection;
- simple score scaling does not repair the combined precision-recall curve.

## Questions

1. For phantom class1 samples, how often does the correct candidate exist but
   rank below false positives?
2. Which sources dominate the true positives, near misses, and hard false
   positives?
3. Are false positives mostly class-confusion, duplicate boxes around the same
   sample, low-IoU localization errors, or score-mode/ranking failures?
4. Which concrete next intervention is justified: new candidate generation,
   verifier hard-negative mining, source filtering, temporal smoothing, or box
   refinement?

## Steps

1. Build a diagnostic script over the Stage2AE public-validation candidate rows
   and exported clean predictions.
2. Focus first on `valid_phantom`, `class_id=1`.
3. Report per-sample oracle-vs-top1 rank gaps, hard-negative score margins,
   source distributions, score-mode distributions, and IoU buckets.
4. Export compact CSVs with the worst hard negatives and missed positives for
   manual inspection.
5. Record the decision and update the current Task2 status report.

## Acceptance Criteria

- Produce a JSON summary with counts/rates for phantom class1 ranking failure.
- Produce CSV samples of high-score false positives and missed/low-rank true
  positives.
- Identify the next optimization direction from evidence rather than model
  preference.
- Run py_compile and relevant tests after adding the diagnostic.
