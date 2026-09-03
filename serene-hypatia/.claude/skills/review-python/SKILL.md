---
name: review-python
description: Run the Python code reviewer on visualization and figure-generation scripts. Produces a quality report.
disable-model-invocation: true
argument-hint: "[script filename or 'all']"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Task"]
---

# Review Python Scripts

Run the python-reviewer agent on visualization and figure-generation scripts.

## Steps

1. **Identify files to review:**
   - If `$ARGUMENTS` is a specific filename: review that file only
   - If `$ARGUMENTS` is "all": review all `.py` files in `scripts/python/`

2. **For each file, launch the python-reviewer agent** that checks for:
   - Script structure and imports
   - Reproducibility (seeds, paths)
   - Function design (naming, type hints, docstrings)
   - Domain correctness
   - Figure quality (DPI, format, colors)
   - Comment quality
   - Error handling

3. **Save each report** to `quality_reports/[SCRIPT_NAME]_python_review.md`

4. **Present summary** to the user:
   - Total issues found per file
   - Breakdown by category
   - Most critical issues highlighted
