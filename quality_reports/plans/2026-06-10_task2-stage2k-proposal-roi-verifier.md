# Task 2 Stage2K: Proposal + ROI Verifier Pilot

Date: 2026-06-10

Status: approved by user request ("试试吧")

## Objective

Test whether a two-stage Task 2 design can improve over direct YOLO detection:

1. Stage 1 proposes multiple candidate boxes with high recall.
2. Stage 2 scores candidate crops with an ROI classifier/verifier.
3. Final detections are ranked by the second-stage scores.

## Rationale

Previous experiments show that direct YOLO is unstable across domains:

- phantom-only YOLO fails on animal collision;
- animal-adapted YOLO can recover held-out animal collision but hurts phantom;
- simple balanced-list training fails on both animal and phantom collision.

The existing GT-ROI classifier performs well when given the correct ROI, so the
remaining question is whether proposal boxes can expose the correct local patch
to a second-stage model.

## Pilot Steps

1. Run the existing YOLO + GT-ROI classifier fusion script with `candidate-mode=topk`.
2. Use the best Stage2H animal-adapted detector as the proposal source for a
   small pilot:
   - animal validation: held-out `video_2_animal`;
   - phantom validation: balanced-small phantom panel.
3. Compare:
   - direct Stage2H detector result;
   - top-k fusion result;
   - localization recall from candidate boxes.
4. If GT-only ROI fusion fails, implement a proper 3-class ROI verifier with a
   background class generated from false/near-miss proposal boxes.

## Success Criteria

This route is worth continuing if top-k fusion improves either:

- class-wise mAP50 on animal normal/collision; or
- phantom collision recovery;
- without severely reducing localization recall.

If it fails, the result still identifies the missing component: the second stage
must learn background rejection, not just normal-vs-collision classification on
GT-centered crops.
