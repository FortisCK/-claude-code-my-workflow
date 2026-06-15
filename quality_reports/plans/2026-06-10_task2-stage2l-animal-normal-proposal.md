# Task 2 Stage2L: Animal-Normal-Aware Class-Agnostic Proposal Detector

Date: 2026-06-10

Status: completed; see
`quality_reports/decisions/2026-06-11_task2_stage2l_animal_normal_proposal_result.md`

## Objective

Improve Task 2 proposal coverage by explicitly adding animal normal examples to
the proposal detector training split.

## Motivation

Stage2K showed that the two-stage route is limited by proposal coverage:

- held-out animal collision proposals are often covered;
- held-out animal normal proposals are not covered;
- phantom collision remains weak when using the animal-adapted detector as the
  proposal source.

The current `train_v1_val_v2` split contains only 3 animal normal frames because
`video_1_animal` is mostly collision. `video_0_animal` contains 108 animal
normal frames and should be included in proposal training before investing more
in the ROI verifier.

## Experiment

Create a class-agnostic proposal split:

- Train:
  - original `train_clean` phantom data;
  - `video_0_animal` normal procedure;
  - `video_1_animal` mostly-collision procedure.
- Validate:
  - combined held-out validation: `video_2_animal` plus the phantom
    balanced-small panel, so checkpoint selection sees both domains.
- Additional validation panels:
  - `video_2_animal` alone, to inspect animal normal/collision behavior;
  - phantom balanced-small and phantom-full panels, to monitor whether phantom
    proposal quality collapses.

All generated YOLO lists should point to `datasets/collision_detection_agnostic`
labels, where both original classes are mapped to class `0: tool_roi`.

## Success Criteria

Continue this route if the new proposal detector improves:

- animal normal recall@0.50 relative to Stage2K's Stage2H proposal source;
- animal collision recall@0.50 remains high;
- phantom balanced-small recall@0.50 is not worse than the Stage2H proposal
  source.

If proposal recall improves, retrain the Stage2K background-aware ROI verifier
with this proposal source.
