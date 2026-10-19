---
name: compile-paper
description: Compile the IEEE TMI paper with pdflatex (3 passes + bibtex). Use when compiling the paper.
disable-model-invocation: true
argument-hint: "[filename without .tex extension, default: main]"
allowed-tools: ["Read", "Bash", "Glob"]
---

# Compile IEEE TMI Paper

Compile the paper using pdflatex with full citation resolution.

## Steps

1. **Navigate to paper/ directory** and compile with 3-pass sequence:

```bash
cd paper
pdflatex -interaction=nonstopmode ${ARGUMENTS:-main}.tex
BIBINPUTS=..:$BIBINPUTS bibtex ${ARGUMENTS:-main}
pdflatex -interaction=nonstopmode ${ARGUMENTS:-main}.tex
pdflatex -interaction=nonstopmode ${ARGUMENTS:-main}.tex
```

2. **Check for warnings:**
   - Grep output for `Overfull \\hbox` warnings
   - Grep for `undefined citations` or `Label(s) may have changed`
   - Report any issues found

3. **Open the PDF** for visual verification:
   ```bash
   open paper/${ARGUMENTS:-main}.pdf
   ```

4. **Report results:**
   - Compilation success/failure
   - Number of overfull hbox warnings
   - Any undefined citations
   - PDF page count

## Why 3 passes?
1. First pdflatex: Creates `.aux` file with citation keys
2. bibtex: Reads `.aux`, generates `.bbl` with formatted references
3. Second pdflatex: Incorporates bibliography
4. Third pdflatex: Resolves all cross-references with final page numbers

## Important
- **Always use pdflatex** (ieeecolor/IEEEtran is not compatible with xelatex by default)
- **BIBINPUTS** is required: your `.bib` file lives in the repo root
- Check page count against IEEE TMI limits
