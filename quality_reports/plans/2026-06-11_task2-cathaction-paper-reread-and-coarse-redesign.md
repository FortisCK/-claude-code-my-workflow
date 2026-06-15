# Task 2: CathAction Paper Reread and Coarse Detector Redesign

Date: 2026-06-11

Status: completed; decision recorded in `quality_reports/decisions/2026-06-11_task2_cathaction_paper_reread_coarse_redesign.md`

## Objective

Re-read `master_supporting_docs/supporting_papers/Cathaction.pdf` with a Task2
collision-detection lens and use it to redesign the coarse-detection strategy.

## Questions

- What exactly does the paper say about Task2 labels, classes, and metrics?
- How are collision instances represented: frame labels, boxes, points, or local
  regions?
- What baseline models and numbers are reported for Task2?
- Does the paper imply that YOLO-style object detection is the right primitive,
  or should Task2 be reformulated as localization, retrieval, heatmap prediction,
  temporal classification, or tool-interaction reasoning?
- What new experiment should replace the current weak coarse-detector direction?

## Verification

- Extract PDF text with `pdftotext`.
- Inspect Task2-related sections, tables, figures, and evaluation descriptions.
- Save a decision record under `quality_reports/decisions/`.
