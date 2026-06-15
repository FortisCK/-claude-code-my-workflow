# Task 1 Dice Diagnostic Package Plan

Date: 2026-06-02

## Goal

Explain why the current Task 1 local score is around `0.65` before spending more
time on `+0.001` style refinements.

The key question is whether the score is low because of:

- foreground localization failure;
- label_1 / label_2 class confusion;
- sub-pixel or 1-3 pixel line displacement;
- animal / phantom domain shift;
- evaluation or pipeline mismatch.

## Diagnostic Metrics

For an existing prediction manifest:

1. Exact multiclass Dice using labels `[1, 2]`, matching the current local
   evaluator.
2. Binary foreground Dice after collapsing `label_1 | label_2` into one tool
   class.
3. Class-swap Dice after swapping predicted labels `1 <-> 2`.
4. Oracle swap Dice, taking per-image max of normal and swapped Dice.
5. Tolerance Dice at radii `1`, `2`, and `3` pixels, separately for multiclass
   and binary foreground.
6. Domain and collection summaries.
7. Worst-case and highest-gap per-sample tables.

## Interpretation

- High binary Dice but low multiclass Dice means label assignment is the main
  bottleneck.
- High tolerance Dice but low exact Dice means thin-line displacement is the
  main bottleneck.
- Low binary Dice means foreground detection/localization is the main
  bottleneck.
- Strong animal/phantom gap means domain handling should take priority.

## First Run

Use the existing full five-model prediction manifest:

`outputs/task1/stage5_submission_smoke/predict_full/predictions.csv`

This is not the final seven-model champion, but it is close enough to diagnose
failure modes quickly without first rerunning full seven-model prediction
writing.

If the diagnostic changes the conclusion materially, generate predictions for
the seven-model champion and rerun the package.

