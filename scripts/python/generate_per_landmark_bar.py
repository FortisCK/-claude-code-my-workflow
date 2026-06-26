"""Regenerate figs/per_landmark_grouped_bar.png/.pdf.

Per-landmark MRE comparison: supervised baseline vs TRUST.

Data source: Table VI of paper/main.tex.
- Supervised baseline numbers are static (Ma et al. on our 10-landmark set, not retrained).
- TRUST numbers come from final_result.csv (30 test cases x 10 landmarks,
  retrained 2026-05-13 with Leo's expanded annotations).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIGS_DIR: Path = REPO_ROOT / "figs"

landmarks = [f"$P_{{{i}}}$" for i in range(10)]

# Supervised baseline (static -- Ma et al. on 10-landmark task)
sup_mean = [2.95, 2.61, 2.20, 2.74, 2.62, 2.91, 1.91, 3.64, 2.73, 3.57]
sup_std  = [1.61, 1.45, 1.15, 1.43, 1.84, 1.52, 0.81, 1.66, 1.71, 2.37]

# TRUST (retrained 2026-05-13, corrected data dump)
trust_mean = [2.41, 2.47, 1.87, 2.02, 2.19, 1.84, 1.58, 3.29, 2.17, 2.10]
trust_std  = [1.24, 1.14, 0.86, 1.10, 0.94, 0.89, 0.64, 1.88, 1.29, 0.94]

SUP_OVERALL = 2.79
TRUST_OVERALL = 2.19

# Ensure PDF embeds fonts (TrueType, type 42)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Desaturated academic palette (steel blue / muted brick red)
SUP   = "#7B9BB5"
SUP_D = "#4A6A82"
TRU   = "#B85C47"
TRU_D = "#7A3325"

x = np.arange(len(landmarks))
w = 0.35

fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="white")

ax.bar(x - w/2, sup_mean, w, yerr=sup_std, label="Supervised",
       color=SUP, alpha=0.88, capsize=4,
       error_kw=dict(elinewidth=1.2, ecolor=SUP_D))
ax.bar(x + w/2, trust_mean, w, yerr=trust_std, label="TRUST (Ours)",
       color=TRU, alpha=0.88, capsize=4,
       error_kw=dict(elinewidth=1.2, ecolor=TRU_D))

# Clinical-function grouping with shaded background.
# Colors match the semantic code in Fig. 1: hinges blue, commissures yellow,
# center green, membranous septum red, and coronary ostia purple.
shades = [
    (0, 2, "#d8edf7", "Hinge"),
    (3, 5, "#fff1b8", "Commissure"),
    (6, 6, "#dff1dc", "Center"),
    (7, 7, "#f8d4ce", "MS"),
    (8, 9, "#eadcf4", "Coronary"),
]
for lo, hi, color, name in shades:
    ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.5, color=color, zorder=0)
    ax.text((lo + hi)/2, 0.18, name, ha="center", va="bottom",
            fontsize=11, color="#333333", fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"))

# Overall MRE reference lines
ax.axhline(SUP_OVERALL, color=SUP, linestyle="--", linewidth=1.3, alpha=0.8,
           label=f"Supervised overall ({SUP_OVERALL:.2f} mm)")
ax.axhline(TRUST_OVERALL, color=TRU, linestyle="--", linewidth=1.3, alpha=0.8,
           label=f"TRUST overall ({TRUST_OVERALL:.2f} mm)")

ax.set_xticks(x)
ax.set_xticklabels(landmarks, fontsize=14)
ax.set_ylabel("Localization Error (mm)", fontsize=14)
ax.set_ylim(0, 6.2)
ax.set_xlim(-0.6, 9.6)
ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_title("Per-Landmark Localization Error: Supervised vs. TRUST",
             fontsize=15, fontweight="bold", pad=12)

plt.tight_layout()

FIGS_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = FIGS_DIR / "per_landmark_grouped_bar.pdf"
png_path = FIGS_DIR / "per_landmark_grouped_bar.png"
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
