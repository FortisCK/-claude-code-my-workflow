# Task 2 Coarse Detection Literature Mini-Review

Date: 2026-06-09

Status: approved by user request ("这种情况别的论文里有出现过吗，我们可以小调研一下")

## Objective

Check whether the current Task 2 failure mode has appeared in prior work:
coarse detectors produce boxes that are visually noisy, poorly aligned with
the annotated catheter/guidewire tip ROI, and especially fail on collision
cases under domain/procedure shift.

## Scope

1. Prioritize CathAction-specific sources.
2. Search adjacent catheter/guidewire fluoroscopy localization, endpoint,
   tracking, and segmentation papers.
3. Search general small/tiny object detection and medical domain-shift
   literature only where it directly explains our observed failure mode.
4. Extract actionable implications for the next Task 2 stage.

## Non-Goals

- Do not make a full systematic review.
- Do not decide the final Task 2 architecture from literature alone.
- Do not tune on hidden-test feedback.

## Verification

Use source-linked claims from primary papers/pages where possible and flag
claims that remain interpretive.
