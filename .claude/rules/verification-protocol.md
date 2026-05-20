---
paths:
  - "Slides/**/*.tex"
  - "Quarto/**/*.qmd"
  - "docs/**"
  - "scripts/**/*"
  - "src/**/*"
  - "tests/**/*"
  - "Dockerfile"
  - "*.tex"
  - "*.md"
---

# Task Completion Verification Protocol

At the end of every task, verify the output with the strongest available check. If the ideal check cannot run because the project is not yet scaffolded, document the missing dependency or file.

## For CATHACTION Code

1. Run syntax/import checks for touched Python/R files.
2. Run targeted unit tests or smoke tests when `tests/` exists.
3. Confirm data paths are repository-relative or configurable.
4. Check split files or loaders preserve procedure/case-level separation.
5. For metric code, run a toy example with known expected DSC/AP/mAP behavior.
6. For inference code, verify outputs match the expected schema and are generated without interaction.
7. For Docker packaging, build the image and run a minimal inference smoke test when `Dockerfile` exists.

## For Experiment Results

1. Record code version, config, split, checkpoint, command, and output path.
2. Recompute reported metrics from saved predictions where possible.
3. Check aggregate and domain-stratified metrics when metadata permits.
4. Verify that tables/figures in the method report are generated from outputs, not manually edited.

## For Method Reports / Papers

1. Compile or render the manuscript when a build path exists.
2. Verify every challenge fact against the 2026 PDF or documented later official source.
3. Verify metric names and ranking claims: Task 1 DSC, Task 2 mAP.
4. Check external-data/pretraining disclosure against challenge policy.
5. Run `/verify-claims` or an equivalent claim check before external submission.

## For Publication Visuals

1. Confirm figure files exist and are non-empty.
2. Prefer vector formats for diagrams and paper plots.
3. Check labels, units, metric definitions, domain/sample scope, and caption context.
4. Open or render outputs when possible; do not rely only on file existence.

## For Quarto/HTML Slides

1. Run `./scripts/sync_to_docs.sh` or `./scripts/sync_to_docs.sh LectureN`.
2. Open the HTML in a browser when available.
3. Verify images display and paths resolve from `docs/`.
4. Check dense slides for overflow.
5. Verify environment parity when a Beamer source exists.

## For LaTeX/Beamer Slides

1. Compile with XeLaTeX and check for errors.
2. Check for overfull boxes.
3. Open the PDF or inspect rendered pages when possible.

## Verification Checklist

```
[ ] Output file or artifact created successfully
[ ] No syntax/build/render errors
[ ] Challenge metrics and split logic are correct for the touched scope
[ ] Figures/tables are generated from reproducible outputs
[ ] Docker or inference path is checked when relevant
[ ] Results reported to the user include evidence and residual risks
```
