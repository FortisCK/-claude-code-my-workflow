---
paths:
  - "figs/**/*"
  - "scripts/python/**/*.py"
---

# Publication Figure Quality Standards

**Every figure must be publication-ready before it is considered complete.**

## Resolution & Format

- **Vector graphics** (PDF/EPS): mandatory for line art, diagrams, architecture figures
- **Raster graphics** (PNG): acceptable only for CT images, heatmaps, qualitative results
- **Minimum DPI:** 300 for raster images
- **Font embedding:** all fonts must be embedded in PDF figures

## IEEE TMI Figure Sizing

| Type | Width | Use |
|------|-------|-----|
| Single-column | 3.5 in (88 mm) | Most figures |
| Double-column | 7.16 in (181 mm) | Architecture diagrams, comparison grids |

## Typography

- Font family: match paper body (Times New Roman or similar serif)
- Minimum font size: 7pt in final print size
- Axis labels, tick labels, and legends must be readable at print size
- Use consistent font sizes across all figures in the paper

## Color Standards

- Use a colorblind-safe palette (e.g., `seaborn` colorblind, Okabe-Ito)
- Figures must be interpretable in grayscale (for print)
- Use color consistently: same color = same meaning across all figures
- CT images: appropriate window/level settings, include window/level in caption if relevant

## Medical Imaging Specifics

- CT slices: include orientation markers (A/P, L/R, S/I) where relevant
- Landmark annotations: consistent marker style (color, size, shape)
- Overlay transparency: semi-transparent overlays on anatomical images
- Scale bars: include when showing physical measurements

## Python Figure Generation

```python
# Standard figure saving
plt.savefig('figs/figure_name.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('figs/figure_name.png', dpi=300, bbox_inches='tight', format='png')
```

## Checklist

```
[ ] Correct format (PDF for vector, PNG for raster)
[ ] Resolution >= 300 DPI (raster only)
[ ] Fonts embedded and readable at print size
[ ] Colorblind-safe palette
[ ] Interpretable in grayscale
[ ] Correct IEEE column width
[ ] All text in figure matches paper notation
[ ] Referenced in paper with \includegraphics
```
