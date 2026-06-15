# Task2 Stage2AB Domain-Aware Calibration Plan

Date: 2026-06-14

## Goal

Test whether the current Stage2V yolo-only champion can be improved without adding new candidate sources, by selecting score modes separately for each class and domain.

## Rationale

Stage2AA showed that adding `stage2x_class1` candidates and training a tabular selector does not beat Stage2V. The remaining low-risk calibration question is whether class 0/class 1 and phantom/animal prefer different score columns on the same yolo-only candidate source.

## Steps

1. Add `scripts/task2/sweep_stage2v_domain_policy.py`.
2. Search score-mode pairs per class:
   - class 0: phantom score mode + animal score mode;
   - class 1: phantom score mode + animal score mode.
3. Optimize on `valid_combined` mAP50-95.
4. Evaluate the selected policy on `valid_combined`, `valid_phantom`, and `valid_animal`.
5. Promote only if full `valid_combined` beats Stage2V's 0.04924044637514719 by a meaningful margin and the policy is not an obvious public-valid overfit.

