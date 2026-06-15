# Task 2 Stage2M-C: Localizer Top-K Proposal Export

Date: 2026-06-11

Status: completed; result recorded in
`quality_reports/decisions/2026-06-11_task2_stage2m_sequence_tip_proposals_result.md`

## Objective

Turn the pretrained ConvNeXt/FPN sequence tip localizer into a first-stage
proposal source and compare it against YOLO Stage2L and Task1 geometry.

## Experiment

- Load the best stable ConvNeXt/FPN localizer checkpoint.
- Predict heatmap peaks on:
  - held-out `video_2_animal`;
  - balanced phantom panel;
  - combined balanced panel.
- Export top-K candidates per frame.
- Convert heatmap centers into proposal boxes using:
  - localizer-predicted width/height;
  - fixed template sizes around each peak.
- Evaluate with the same proposal-recall metrics used for YOLO:
  - top1/top5/top10/top20/top50 recall at IoU 0.25/0.50/0.75;
  - center recall at 5/10/20 px;
  - animal/phantom and class0/class1 breakdown.

## Success Criteria

Continue localizer branch if top-K proposals improve any current first-stage
failure:

- animal class0 coverage above YOLO's near-zero coverage;
- phantom class1 proposal recall above YOLO Stage2L;
- combined proposal recall after union with YOLO/geometry.

## Verification

- Add a script under `scripts/task2/`.
- Compile the script.
- Run the export/evaluation on the stable fp32 ConvNeXt checkpoint.
- Save metrics under `outputs/task2/sequence_tip_proposals/`.

## Result

Implemented `scripts/task2/evaluate_sequence_tip_proposals.py` and
`scripts/task2/evaluate_proposal_csv_union.py`.

The ConvNeXt/FPN localizer is not a standalone replacement for YOLO proposals:
top50 combined R@0.50 is only 0.2352. It is useful as a complementary candidate
source. In oracle union with YOLO and Task1 geometry proposals, combined R@0.50
rises from 0.5661 to 0.5832, and phantom R@0.50 rises from 0.5158 to 0.5352.

Decision: keep the sequence-tip localizer as an optional supplemental proposal
source for the next ROI verifier/ranker experiment; do not train it further as
the primary detector right now.
