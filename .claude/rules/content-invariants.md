---
paths:
  - "Slides/**/*.tex"
  - "Quarto/**/*.qmd"
  - "Quarto/**/*.scss"
  - "Preambles/header.tex"
  - "Figures/**/*"
  - "scripts/**/*.R"
  - "scripts/**/*.py"
  - "src/**/*.py"
  - "tests/**/*.py"
  - "*.tex"
  - "*.md"
---

# CATHACTION Content Invariants (INV-1 through INV-18)

Numbered non-negotiable rules for content, code, experiments, and submission artifacts produced in this repository. Reviewers should cite invariants by number when flagging issues.

## Challenge Invariants

- **INV-1: Challenge specification source.** The 2026-04-22 CATHACTION MICCAI PDF is the working source of truth until an official 2026 challenge website or platform page supersedes it. Do not mix the older website counts (about 500k frames / 25k masks) into challenge-facing work without explicitly labeling them as older website information.
- **INV-2: Case-level split integrity.** Training, validation, and test splits must be procedure/case-level. Never frame-randomize across the same video/procedure, and never let frames from one case appear in multiple splits.
- **INV-3: Hidden-test discipline.** Hidden test labels and outputs are evaluation-only. Do not hand-tune, reverse-engineer, manually curate, or otherwise adapt predictions from hidden-test feedback.
- **INV-4: Public-only external data.** External datasets, annotations, pretrained models, and prompts must be publicly available at challenge launch. Private or proprietary clinical data is prohibited.
- **INV-5: No patient-identifying data.** Do not commit patient identifiers, hospital identifiers, unredacted clinical metadata, access tokens, dataset credentials, or private download links.
- **INV-6: No user interaction at inference.** Challenge inference must be fully automated and suitable for Docker execution with no manual curation or interactive steps.

## Metric And Reporting Invariants

- **INV-7: Task 1 metric fidelity.** Catheter/guidewire segmentation is ranked by Dice Similarity Coefficient (DSC). IoU/Jaccard, mIoU, and pixel-wise accuracy are secondary unless the official platform changes the ranking.
- **INV-8: Task 2 metric fidelity.** Collision detection is ranked by mAP on the private test set. AP is complementary. Reports must not imply AP alone is the final ranking metric.
- **INV-9: Domain-stratified reporting.** Whenever data access allows it, report performance by domain (phantom, animal, human) in addition to aggregate metrics.
- **INV-10: Submission provenance.** Every reported result must trace to code version, config, checkpoint, split file, and evaluation command. Do not paste free-floating numbers into papers or slides.

## Visual And Manuscript Invariants

- **INV-11: Publication-ready visuals.** Figures must have clear titles/captions, axis labels, units, metric definitions, sample/domain scope, and readable typography. Prefer vector outputs for diagrams and paper figures.
- **INV-12: Clinical-safety claim discipline.** Claims about collision avoidance, real-time safety, or clinical deployment must distinguish demonstrated benchmark performance from future clinical utility.
- **INV-13: Single bibliography.** `Bibliography_base.bib` is the canonical bibliography when present. Avoid scattered per-artifact `.bib` files unless a venue requires them.
- **INV-14: Method report coherence.** The method report/paper must match the actual code path, training data policy, metrics, and submission packaging.

## Slide Invariants (When Slides Are Used)

- **INV-15: Palette sync.** Color names in `Preambles/header.tex` must match SCSS variables in `Quarto/theme-template.scss`. Verify with `./scripts/check-palette-sync.sh`.
- **INV-16: Beamer/Quarto notation parity.** If a Beamer deck has a Quarto mirror, math notation and stated metrics must remain identical across both.
- **INV-17: TikZ as SVG in HTML.** Browsers cannot render PDF images inline. TikZ diagrams used in Quarto/HTML must be SVG, produced via `/extract-tikz` or an equivalent verified path.

## Code Invariants

- **INV-18: Repository-relative paths.** Code must use repository-relative paths or configurable environment variables. No committed hardcoded `/Users/...`, `~`, machine-specific data roots, or local credentials.
