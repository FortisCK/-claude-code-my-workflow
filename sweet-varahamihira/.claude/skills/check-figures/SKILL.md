---
name: check-figures
description: Verify all figure references in the paper resolve to existing files and meet quality standards.
disable-model-invocation: true
allowed-tools: ["Read", "Grep", "Glob", "Bash"]
---

# Check Figures

Verify all figure references in the paper resolve to existing files and meet publication quality standards.

## Steps

1. **Scan paper for figure references:**
   ```
   Grep all .tex files in paper/ for \includegraphics
   Extract file paths from each \includegraphics command
   ```

2. **For each referenced figure:**
   - Verify the file exists in `figs/`
   - Check file size > 0
   - Determine format (PDF/PNG/EPS)
   - Check if format is appropriate (PDF for vector, PNG for raster)

3. **Check for orphan figures:**
   - List all files in `figs/`
   - Cross-reference against paper figure references
   - Report any figures not referenced in the paper

4. **Quality checks** (using `pdfinfo` or `file` command):
   - For PNG files: check resolution (should be >= 300 DPI)
   - For PDF files: check page count (should be 1)
   - Check file sizes are reasonable (not suspiciously small or large)

5. **Report findings:**

```markdown
## Figure Check Report

### Referenced Figures
| Figure | Path | Exists | Format | Size | Status |
|--------|------|--------|--------|------|--------|
| fig:arch | figs/architecture.pdf | Yes | PDF | 150KB | OK |
| fig:results | figs/results.png | Yes | PNG | 2.1MB | OK |

### Issues
- [List any broken references, missing files, wrong formats]

### Orphan Figures (not referenced)
- [List any files in figs/ not referenced in paper]

### Summary
- Total references: N
- Resolved: N
- Broken: N
- Orphan figures: N
```
