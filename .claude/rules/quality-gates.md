---
paths:
  - "Slides/**/*.tex"
  - "Quarto/**/*.qmd"
  - "scripts/**/*.R"
  - "scripts/**/*.py"
  - "src/**/*.py"
  - "tests/**/*.py"
  - "*.tex"
  - "*.md"
  - "Dockerfile"
---

# Quality Review & Scoring Rubrics

> **Framing:** Thresholds are advisory at the harness level. The `/commit` skill runs quality checks and halts on failure until the user fixes or explicitly overrides. A direct `git commit` bypasses this review.

## Thresholds

- **80/100 = Commit** -- good enough to save.
- **90/100 = Submission/PR** -- ready for serious external review.
- **95/100 = Excellence** -- publication-ready / spotlight-ready.

## CATHACTION Challenge Code

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Syntax/import failure on the intended runtime | -100 |
| Critical | Train/validation/test leakage across procedures/cases | -100 |
| Critical | Hidden-test tuning, manual interaction, or private clinical data use | -100 |
| Critical | Metric implementation contradicts challenge ranking metric | -50 |
| Critical | Docker inference path cannot run or omits required outputs | -50 |
| Major | No smoke test for preprocessing/inference/evaluation path | -20 |
| Major | Non-reproducible config/checkpoint/result provenance | -20 |
| Major | Hardcoded local paths or credentials | -20 |
| Major | Domain performance not stratified when labels/domain metadata permit | -10 |
| Minor | Missing docstrings or comments around non-obvious data transforms | -3 |

## Task 1: Segmentation

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | DSC not computed or reported incorrectly | -50 |
| Critical | Catheter/guidewire class handling contradicts labels | -40 |
| Major | Mask resizing/interpolation can corrupt class labels | -20 |
| Major | Thin-tool postprocessing is undocumented or non-reproducible | -10 |
| Minor | Secondary metrics lack definitions | -3 |

## Task 2: Collision Detection

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | mAP not computed or ranking is based on the wrong metric | -50 |
| Critical | Temporal event/frame interpretation contradicts official schema | -40 |
| Major | Threshold selection tuned on non-validation data | -20 |
| Major | False alarm / missed collision tradeoff not described | -10 |
| Minor | Confidence-score calibration not documented | -3 |

## Method Report / Paper

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Method description contradicts code or submission container | -50 |
| Critical | Challenge facts, dataset counts, or ranking metrics are wrong | -40 |
| Major | External data/pretraining policy unclear | -15 |
| Major | No reproducibility information for training/inference | -15 |
| Major | Clinical-safety claims exceed evidence | -10 |
| Minor | Missing limitations or domain-shift discussion | -5 |

## Visuals

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Misleading metric, axis, domain, or sample labeling | -30 |
| Major | Figure is not readable at paper or slide size | -15 |
| Major | Missing caption context or units | -10 |
| Minor | Raster figure used where vector output is feasible | -3 |

## Slides (.tex/.qmd), When Used

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Compilation/render failure | -100 |
| Critical | Broken citation or wrong metric notation | -15 |
| Major | Text or equation overflow | -10 |
| Major | Beamer/Quarto fact or notation drift | -10 |
| Minor | Long lines or minor formatting issues | -1 |

## Enforcement

- **Score < 80:** Halt within `/commit`; list blocking issues. User may override explicitly with a reason.
- **Score < 90:** Allow commit within `/commit`, but warn and list recommended fixes.
- **Submission work:** Aim for at least 90 before challenge upload or method-report external review.

## Quality Reports

Generated only at merge/submission milestones. Use `templates/quality-report.md` and save to `quality_reports/merges/YYYY-MM-DD_[branch-name].md` or an equivalent submission-named report.

## Tolerance Thresholds

| Quantity | Tolerance | Rationale |
|----------|-----------|-----------|
| Case counts / split membership | Exact match | Leakage prevention and reproducibility |
| DSC / IoU / mIoU | Documented rounding only | Challenge metrics should be deterministic |
| AP / mAP | Documented rounding only | Ranking metric must be stable |
| Reported table values | Match generated outputs | Prevent free-floating numbers |
