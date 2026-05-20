# Requirements Specification: CATHACTION Workflow Configuration

**Date:** 2026-05-20
**Status:** APPROVED

---

## Objective

Adapt the academic workflow scaffold into a rigorous CATHACTION project workflow for a MICCAI challenge on endovascular intervention tool segmentation and collision detection.

---

## Source Material Read

- `CLAUDE.md`, `AGENTS.md`, `MEMORY.md`
- `.claude/rules/*`, `.claude/agents/domain-reviewer.md`, `.codex/agents/domain-reviewer.toml`
- `.claude/WORKFLOW_QUICK_REF.md`, `.claude/settings.json`, `.codex/hooks.json`
- `templates/*`, `scripts/*`, `Preambles/header.tex`, `Quarto/theme-template.scss`
- User-provided PDF: `343-CATHACTION_Endovascular_Intervention_Tool_Segmentation_and_Collision_Detection_2026-04-22T16-37-17.pdf`
- Official dataset website checked on 2026-05-20: https://airvlab.github.io/cathaction/

---

## Challenge Facts To Encode

- Project: CATHACTION: Endovascular Intervention Tool Segmentation and Collision Detection.
- Venue/year: MICCAI challenge, 2026.
- Core tasks:
  - Task 1: Catheter and guidewire segmentation in X-ray fluoroscopy.
  - Task 2: Collision detection in endovascular intervention.
- Domains:
  - Silicon vascular phantom systems.
  - Preclinical animal X-ray fluoroscopy.
  - Real human X-ray procedures.
- Dataset and annotations from the PDF:
  - 650 videos/cases total.
  - Task 1: approximately 40,000 annotated frames with pixel-level masks distinguishing catheter and guidewire.
  - Task 2: approximately 600,000 annotated frames with frame-level collision labels.
  - Split: 70% train, 15% validation, 15% hidden test, split at procedure/case level.
- Metrics:
  - Task 1: DSC, IoU/Jaccard, mIoU, pixel-wise accuracy; primary ranking metric DSC.
  - Task 2: AP and mAP; primary ranking metric mAP on the private test set.
- Submission:
  - Docker container, predefined result file, short method description/methodology report.
  - Platform expected to be Grand Challenge or Synapse.
  - No user interaction during algorithm assessment.
- Data policy:
  - Challenge data plus publicly available datasets and pretrained models are allowed.
  - Private or proprietary clinical datasets are not allowed.
  - Additional annotations are allowed only on publicly available data and not from private sources.
- Schedule from the PDF:
  - Website and registration open: 2026-04-01.
  - Training data release: 2026-04-10.
  - Registration period: 2026-07-10 to 2026-08-15.
  - Validation data release: 2026-07-10.
  - Submission deadline: 2026-08-23.
  - Joint challenge paper preparation/submission: March 2027.
- Website discrepancy:
  - The current public website still says approximately 500,000 annotated frames and 25,000 masks and was last updated 2024-11-28.
  - For this repo, the 2026 challenge PDF should be treated as the authoritative challenge specification unless the official challenge website is later updated with newer 2026 instructions.

---

## Requirements

### MUST Have (Non-Negotiable)

- [ ] Replace template placeholders in `CLAUDE.md` and `AGENTS.md` with CATHACTION-specific project identity, folder map, commands, quality thresholds, and current project state.
- [ ] Preserve the plan-first workflow: non-trivial work requires a saved plan in `quality_reports/plans/` and user approval before implementation.
- [ ] Encode challenge-specific invariants: no case-level leakage, metric fidelity, Docker/reproducibility, public-only external data, hidden-test discipline, and de-identification/PII safety.
- [ ] Customize the domain reviewer away from generic/econometrics examples toward endovascular surgical AI, fluoroscopy, segmentation, collision detection, domain shift, and clinical-safety claims.
- [ ] Update the workflow quick reference with the user's preferences: rigorous collaboration, polished publication-ready visuals, smart memory, and extra check-ins for early sessions.
- [ ] Keep Beamer/Quarto rules available for slides and reports, but clarify that challenge implementation and method artifacts are also first-class project outputs.
- [ ] Add or adjust verification guidance for ML challenge work: smoke tests, metric checks, split checks, Docker checks when available, and reproducible output provenance.
- [ ] Verify after edits by scanning for unreplaced placeholders and running relevant lightweight checks.

### SHOULD Have (Preferred)

- [ ] Add a CATHACTION knowledge base or update `.claude/rules/knowledge-base-template.md` so future slide/report/review work sees challenge facts automatically.
- [ ] Add challenge-specific quality rubrics for segmentation, collision detection, methodology reports, visuals, and experiment logs.
- [ ] Update README/project-facing docs enough that the repo no longer reads primarily as a generic workflow template.
- [ ] Keep Claude and Codex surfaces in sync: `CLAUDE.md`/`.claude/*` and `AGENTS.md`/`.codex/*` should not contradict each other.
- [ ] Add notes for the official website/PDF discrepancy so future work does not silently mix old dataset numbers with 2026 challenge numbers.

### MAY Have (Optional, If Time)

- [ ] Add a dedicated `quality_reports/decisions/` record for the source-of-truth decision: PDF as authoritative until the official website updates.
- [ ] Propose later creation of ML-specific skills such as `/run-baseline`, `/evaluate-cathaction`, `/package-submission`, or `/review-experiment`.
- [ ] Propose later Python/PyTorch conventions if/when the implementation stack is known.

---

## Clarity Status

| Aspect | Status | Notes |
|--------|--------|-------|
| Project name | CLEAR | CATHACTION. |
| Project purpose | CLEAR | MICCAI challenge for endovascular intervention tool segmentation and collision detection. |
| Source of challenge facts | ASSUMED | Use the 2026 PDF as authoritative; website is older and currently discrepant. |
| Institution line | CLEAR | User specified `Universite de Rennes` / `Université de Rennes`; use the accented form in user-facing docs. |
| Repo role | CLEAR | Participant submission repo containing all challenge code plus the final paper/method description. |
| Coding stack | ASSUMED | Do not hard-code PyTorch/nnU-Net/MONAI yet; add workflow rules that are ML-stack-agnostic. |
| Slides vs code emphasis | ASSUMED | Keep Beamer/Quarto infrastructure, but make challenge implementation/reports equally prominent. |

---

## Success Criteria

- No remaining project-identity placeholders in `CLAUDE.md`, `AGENTS.md`, or `.claude/WORKFLOW_QUICK_REF.md`.
- Domain reviewer no longer contains the auto-detect template marker and uses CATHACTION-specific review lenses.
- Core rules mention CATHACTION-specific quality, verification, and data-policy invariants.
- `rg` confirms only intentional generic-template placeholders remain in reusable templates.
- Lightweight checks pass or any failures are documented with next steps.

---

## Approval

- [x] User approved: 2026-05-20

## Approval Notes

- Institution: Université de Rennes.
- Repo role: participant submission repository containing all code and final method paper/report.
- Scope: update all surfaces, including README, guide, and docs.
