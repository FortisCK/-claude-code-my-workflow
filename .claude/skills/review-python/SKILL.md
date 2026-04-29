---
name: review-python
description: Read-only Python code review for `.py` scripts (PyTorch + MONAI flavour). Checks code quality, reproducibility, GPU-memory hygiene, MONAI/PyTorch idioms, run-card discipline, and professional standards; produces a report without editing. Use when user says "review this Python script", "check the training code", "audit the model", "code review on the .py", or when a Python file is touched as part of a paper submission. NOT for running the code — pair with `/audit-reproducibility` for numeric verification when applicable.
argument-hint: "[filename or 'all' or 'training' or 'evaluation']"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Task"]
---

# Review Python Scripts

Run the comprehensive Python code-review protocol for the cardiac-CT
diffusion project. This is the Python analogue of `/review-r` (now archived).

## Steps

1. **Identify scripts to review:**
   - If `$ARGUMENTS` is a specific `.py` filename: review that file only.
   - If `$ARGUMENTS` is `training`: review all scripts under `code/training/`.
   - If `$ARGUMENTS` is `evaluation`: review all scripts under `code/evaluation/`.
   - If `$ARGUMENTS` is `inference`: review all scripts under `code/inference/`.
   - If `$ARGUMENTS` is `all`: review every `.py` under `code/` and `scripts/python/`.
   - Excluded by default: `code/notebooks/` (exploratory), `code/tests/`
     (pytest discovery), and any file matching `**/_*.py`.

2. **For each script, launch the `python-reviewer` agent** with instructions to:
   - Follow the full protocol in the agent's system prompt.
   - Read [`.claude/rules/python-code-conventions.md`](../../rules/python-code-conventions.md) for current standards.
   - Read [`.claude/rules/experiments-protocol.md`](../../rules/experiments-protocol.md) when reviewing a training or evaluation script.
   - Save the report to `quality_reports/<script_name>_python_review.md`.

3. **After all reviews complete**, present a summary:
   - Total issues found per script.
   - Breakdown by severity (Critical / High / Medium / Low).
   - Top 3 most critical issues.
   - For training scripts: an explicit PASS / FAIL on the run-card-link check.

4. **IMPORTANT: Do NOT edit any Python source files.**
   Reports only. Fixes are applied by the user (or by a subsequent
   targeted edit) after review.

## Pairings

- After this skill, run `/audit-reproducibility` if numeric claims are at stake.
- Before this skill, ensure the script(s) under review are committed (or
  staged) — diff hygiene helps the agent point at line numbers that won't
  shift mid-review.
- If reviewing as part of `/review-paper` (cross-artifact), this skill is
  invoked automatically per the `cross-artifact-review.md` rule.
