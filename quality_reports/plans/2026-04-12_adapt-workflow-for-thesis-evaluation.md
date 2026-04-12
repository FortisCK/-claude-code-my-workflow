# Plan: Adapt Workflow Configuration for Master's Thesis Evaluation

**Status:** COMPLETED
**Date:** 2026-04-12
**Goal:** Reconfigure the academic workflow (forked from pedrohcgs/claude-code-my-workflow) from a lecture-slide-development tool to a master's thesis evaluation tool for a CS/AI thesis.

---

## Context

- **User role:** Thesis advisor
- **Thesis field:** Computer Science / Artificial Intelligence
- **Thesis format:** LaTeX (.tex)
- **Objective:** Provide a detailed, structured evaluation to help the thesis pass blind review
- **Current state:** All config files are template defaults designed for Beamer-to-Quarto lecture slide workflows

---

## What Needs to Change

### Phase 1: Core Configuration (CLAUDE.md + MEMORY.md)

**File: `CLAUDE.md`**
1. Replace `[YOUR PROJECT NAME]` → "Master's Thesis Evaluation — CS/AI"
2. Replace `[YOUR INSTITUTION]` → leave as TBD (ask user) or remove
3. Update **Core Principles** — change from slide-centric to thesis-evaluation-centric:
   - "Single source of truth" → thesis .tex is authoritative (not Beamer slides)
   - Remove Beamer/Quarto sync references
   - Add: "Read-only by default — produce evaluation reports, never edit the student's source files without explicit permission"
4. Update **Folder Structure** to reflect thesis evaluation:
   - `Thesis/` — student's LaTeX source files (the material under review)
   - `master_supporting_docs/` — reference papers, guidelines, rubrics
   - Remove/de-emphasize `Slides/`, `Quarto/`, `Preambles/` (not needed)
5. Update **Commands** section — remove Beamer/Quarto commands, add:
   - LaTeX compilation for the thesis (if needed for verification)
   - Quality scoring for thesis chapters
6. Update **Skills Quick Reference** — highlight thesis-relevant skills, de-emphasize slide skills
7. Remove **Beamer Custom Environments** and **Quarto CSS Classes** sections
8. Replace **Current Project State** with thesis chapter tracking table

**File: `MEMORY.md`** — no changes needed (empty, ready for use)

### Phase 2: Rules Adaptation

**File: `.claude/rules/single-source-of-truth.md`**
- Rewrite: The student's thesis .tex files are READ-ONLY source of truth
- We produce evaluation reports, not derived artifacts
- Remove TikZ freshness protocol and Beamer→Quarto sync (irrelevant)

**File: `.claude/rules/quality-gates.md`**
- Replace slide/script rubrics with thesis evaluation rubrics:
  - Argument structure & logical flow
  - Technical correctness (proofs, algorithms, experiments)
  - Literature review completeness
  - Writing quality & clarity
  - Methodology rigor
  - Results presentation & interpretation
- Keep scoring thresholds (80/90/95) but apply to evaluation report quality

**File: `.claude/rules/verification-protocol.md`**
- Adapt: verification = ensuring our evaluation reports are thorough, not compilation checks
- Add: verify thesis compiles (as a check), verify citations resolve
- Remove: Quarto deployment, TikZ SVG, sync_to_docs references

**File: `.claude/rules/beamer-quarto-sync.md`**
- Mark as inactive or delete — not relevant for thesis evaluation

**File: `.claude/rules/proofreading-protocol.md`**
- Adapt: proofread the thesis itself (grammar, typos, notation consistency)
- Output = evaluation report section, not direct edits

**Files unchanged (still useful as-is):**
- `plan-first-workflow.md` — keep as-is
- `orchestrator-protocol.md` — keep the loop structure
- `orchestrator-research.md` — keep for any R/analysis work
- `session-logging.md` — keep as-is
- `exploration-fast-track.md` — keep for exploratory analysis
- `r-code-conventions.md` — keep if thesis has R code
- `pdf-processing.md` — keep (useful for reading thesis PDFs)
- `knowledge-base-template.md` — keep

### Phase 3: Agent Customization

**File: `.claude/agents/domain-reviewer.md`** (CRITICAL — most important adaptation)
- Change persona from "Econometrica referee" to "top CS/AI venue reviewer" (e.g., NeurIPS/ICML/AAAI-level)
- Customize 5 lenses for CS/AI thesis:
  - Lens 1: Problem Formulation & Motivation (is the research question well-defined?)
  - Lens 2: Technical Correctness (proofs, algorithm analysis, complexity claims)
  - Lens 3: Experimental Methodology (baselines, metrics, statistical significance, reproducibility)
  - Lens 4: Literature Positioning (complete survey? proper attribution? clear contribution differentiation?)
  - Lens 5: Presentation & Argumentation Quality (flow, notation, figures, tables)
- Add CS/AI-specific pitfalls (cherry-picked results, missing ablations, unfair baselines, etc.)

**File: `.claude/agents/proofreader.md`** — keep largely as-is, works for thesis too

**Files to de-emphasize (not delete):**
- `beamer-translator.md`, `quarto-critic.md`, `quarto-fixer.md` — not needed for thesis eval
- `slide-auditor.md`, `tikz-reviewer.md` — not primary use

### Phase 4: Skill Adaptation

**File: `.claude/skills/review-paper/SKILL.md`**
- Adapt from economics focus to CS/AI:
  - Replace "Identification Strategy" with "Technical Approach & Novelty"
  - Replace "Econometric Specification" with "Experimental Design & Evaluation"
  - Add CS/AI-specific dimensions: reproducibility, code availability, ethical considerations
- This becomes the primary skill for thesis evaluation

**New skill: `.claude/skills/review-thesis/SKILL.md`** (optional — or extend review-paper)
- Chapter-by-chapter evaluation mode
- Blind-review readiness checklist
- Generates structured evaluation report matching typical thesis defense rubrics

### Phase 5: Settings & Hooks

**File: `.claude/settings.json`**
- Add thesis .tex files to protected patterns (read-only by default)
- Keep existing git/LaTeX permissions

**File: `.claude/hooks/protect-files.sh`**
- Add `Thesis/*.tex` pattern to protected files
- Student's thesis source should not be accidentally modified

### Phase 6: Folder Structure & Thesis Import

- Create `Thesis/` directory for the student's LaTeX source
- User will push thesis files from local: `master_thesis/` → `Thesis/`
- Create `quality_reports/thesis_evaluation/` for evaluation outputs
- Update `.gitignore` if needed

### Phase 7: WORKFLOW_QUICK_REF.md

- Update non-negotiables for thesis evaluation context
- Update preferences section
- Adjust "I Ask You When" / "I Just Execute When" for evaluation workflow

---

## Verification Steps

1. All placeholders in CLAUDE.md are filled or intentionally marked TBD
2. Domain reviewer agent is customized for CS/AI
3. Review-paper skill is adapted for CS/AI thesis
4. Protected files hook includes thesis source
5. Folder structure supports thesis import
6. No broken references to removed/changed files
7. Git commit with all changes

---

## Files to Modify (ordered)

1. `CLAUDE.md`
2. `.claude/rules/single-source-of-truth.md`
3. `.claude/rules/quality-gates.md`
4. `.claude/rules/verification-protocol.md`
5. `.claude/rules/beamer-quarto-sync.md`
6. `.claude/rules/proofreading-protocol.md`
7. `.claude/agents/domain-reviewer.md`
8. `.claude/skills/review-paper/SKILL.md`
9. `.claude/hooks/protect-files.sh`
10. `.claude/settings.json`
11. `.claude/WORKFLOW_QUICK_REF.md`
12. `.gitignore`
13. Create: `Thesis/` directory
14. Create: `quality_reports/thesis_evaluation/` directory

---

## Out of Scope (for this plan)

- Actually reading/evaluating the thesis (that's the next task, after thesis files are imported)
- Creating new skills from scratch (adapt existing ones first)
- Modifying templates (session-log and quality-report templates work as-is)
