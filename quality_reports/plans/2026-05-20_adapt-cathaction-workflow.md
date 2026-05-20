# Plan: Adapt Workflow Configuration for CATHACTION

**Date:** 2026-05-20
**Status:** COMPLETED
**Spec:** `quality_reports/specs/2026-05-20_cathaction_workflow_configuration.md`

---

## Goal

Turn the generic academic workflow scaffold into a CATHACTION-specific working environment for a rigorous MICCAI challenge project, without losing the existing plan-first, verify-after, memory, visual-quality, and review-agent discipline.

---

## Key Decisions Proposed

1. **Challenge source of truth:** Use the 2026 PDF as the repo's challenge source of truth for now. The public website is useful background but currently shows older dataset counts.
2. **Artifact model:** Treat the repo as a hybrid challenge workspace: ML code/experiments/submission package, methodology report, visuals, and optional Beamer/Quarto decks.
3. **Workflow style:** Keep plan approval before non-trivial work. After approval, operate in contractor mode with early-session check-ins.
4. **Visual standard:** All figures and diagrams should be publication-ready: vector where possible, metric labels explicit, no unlabeled axes, no hidden sample/domain assumptions.
5. **Data discipline:** No case leakage; no private/proprietary clinical data; external data must be public at challenge launch; hidden-test outputs are not reverse-engineered or hand-tuned.

---

## Phase 1: Project Identity And Top-Level Configuration

**Files**
- `CLAUDE.md`
- `AGENTS.md`
- `.claude/WORKFLOW_QUICK_REF.md`

**Edits**
- Fill project name as `CATHACTION: Endovascular Intervention Tool Segmentation and Collision Detection`.
- Fill institution once the user provides the exact string.
- Replace sample lecture rows with current CATHACTION project state:
  - Challenge specification source.
  - Task 1 segmentation.
  - Task 2 collision detection.
  - Methodology report/submission package.
- Update core principles to include challenge data policy, reproducibility, and polished visuals.
- Replace placeholder non-negotiables in the quick reference with CATHACTION-specific conventions.

**Verification**
- `rg -n "\\[YOUR|\\[your|YOUR-PROJECT|example — delete|HelloWorld \\*\\(sample" CLAUDE.md AGENTS.md .claude/WORKFLOW_QUICK_REF.md`

---

## Phase 2: Challenge-Specific Rules

**Files**
- `.claude/rules/content-invariants.md`
- `.claude/rules/single-source-of-truth.md`
- `.claude/rules/quality-gates.md`
- `.claude/rules/verification-protocol.md`
- `.claude/rules/beamer-quarto-sync.md`
- `.claude/rules/knowledge-base-template.md`

**Edits**
- Add CATHACTION invariants:
  - split by case/procedure, never frame-random split across procedures;
  - Task 1 metric fidelity: DSC primary, IoU/mIoU/pixel accuracy secondary;
  - Task 2 metric fidelity: mAP primary, AP secondary;
  - no hidden-test tuning or manual interaction during assessment;
  - Docker inference path must be reproducible;
  - public-only external data and pretrained weights;
  - no patient-identifying information in repo artifacts.
- Keep slide-specific Beamer/Quarto rules, but clarify they apply when creating decks.
- Add ML verification checklist: smoke test, metric sanity test, split audit, output-schema check, Docker build/run when packaging.
- Replace generic knowledge-base title with CATHACTION knowledge base and seed it with task/domain/metric facts from the PDF.
- Remove fake lecture mappings from `beamer-quarto-sync.md`; leave an empty/project-ready mapping.

**Verification**
- Scan for contradictory task metrics and old dataset counts.
- Check Markdown structure and links.

---

## Phase 3: Review Agents And Codex/Claude Parity

**Files**
- `.claude/agents/domain-reviewer.md`
- `.codex/agents/domain-reviewer.toml`
- Optional: `.claude/agents/r-reviewer.md`, `.codex/agents/r-reviewer.toml` if we add only light ML-aware notes.

**Edits**
- Replace the generic field reviewer with an endovascular surgical AI reviewer.
- Use five CATHACTION review lenses:
  - clinical and procedural plausibility;
  - fluoroscopy/tool segmentation correctness;
  - collision detection and temporal reasoning;
  - domain generalization and leakage risk;
  - metric/submission/reproducibility fidelity.
- Preserve read-only review behavior.
- Keep `.claude` and `.codex` agent wording aligned.

**Verification**
- Confirm the domain-reviewer template marker is removed.
- Confirm no econometrics-specific examples remain in the active reviewer instructions.

---

## Phase 4: Scripts And Project-Facing Documentation

**Files**
- `README.md`
- `scripts/validate-setup.sh`
- Possibly `scripts/check-surface-sync.py` if README/guide surface assumptions become project-inappropriate.

**Edits**
- Replace the template README opening with a CATHACTION project overview and workflow entry points.
- Keep attribution/community material only if the user wants the repo to remain a forkable workflow template; otherwise move it below project content or trim it.
- Adjust setup validation messaging to this repo:
  - Python 3 and git required.
  - LaTeX/Quarto optional unless building slides.
  - Docker recommended/required for challenge packaging.
  - R optional unless using R scripts.
- Be conservative with generated `docs/`/`guide/`: update only if we decide the repo's public site should represent this project now.

**Verification**
- `bash -n scripts/validate-setup.sh`
- `python3 -m py_compile scripts/check-surface-sync.py scripts/check-palette-sync.py scripts/quality_score.py`

---

## Phase 5: Final Verification And Report

**Checks**
- `rg -n "\\[YOUR|\\[your|YOUR-PROJECT|example — delete|HelloWorld \\*\\(sample" CLAUDE.md AGENTS.md .claude/WORKFLOW_QUICK_REF.md .claude/rules .claude/agents .codex/agents README.md scripts`
- `./scripts/check-palette-sync.sh`
- `./scripts/check-surface-sync.sh` if README/guide surfaces remain in scope; otherwise document why not.
- `git diff --check`

**Deliverable**
- Concise summary of changed files.
- Verification results.
- Proposed next customizations:
  - decide Python/ML stack conventions;
  - add experiment registry;
  - add submission packaging/evaluation skills once official platform format is known.

**Completed verification (2026-05-20):**
- `git diff --check` passed.
- Placeholder/template-marker scan passed.
- Old Emory/economics/legacy-template scan passed on configured surfaces.
- `python3 -m py_compile scripts/check-surface-sync.py scripts/check-palette-sync.py scripts/quality_score.py` passed.
- `bash -n scripts/validate-setup.sh scripts/check-palette-sync.sh scripts/check-surface-sync.sh` passed.
- `./scripts/check-palette-sync.sh` passed.
- `./scripts/check-surface-sync.sh` passed, including skill-integrity checks.
- `./scripts/validate-setup.sh` passed with optional warnings for missing Quarto, R, and GitHub CLI.

---

## User Approval Notes

- Institution: Université de Rennes.
- Repo framing: participant submission repo containing code plus final paper/method description.
- Documentation scope: update all public and operational surfaces, including `README.md`, `guide/`, and `docs/`.

## Open Questions Before Implementation

No blocking questions remain for this configuration pass. Later implementation work still needs stack choices such as PyTorch/MONAI/nnU-Net, experiment tracking, and official platform I/O once the challenge releases final submission instructions.
