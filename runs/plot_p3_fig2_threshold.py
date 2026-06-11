#!/usr/bin/env python3
"""P3 Fig 2: Percolation threshold n_c by orbit family.

Horizontal bar chart comparing Moon and Mars thresholds.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import BODY_COLORS, apply_style, figsize_single, save_fig

apply_style("ieee")

RUNS = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(RUNS, "epyc_results", "campaign_2026_03_11", "campaign_summary.json")) as f:
    summary = json.load(f)

nc_raw = summary["per_bucket"]["1"]["analysis"]["n_c_per_family"]

# Parse into Moon/Mars pairs
families_order = [
    "polar",
    "sunsync",
    "frozen_elliptical",
    "nrho",
    "lissajous_l2",
    "dro",
    "areostationary",
    "molniya",
]
family_labels = {
    "polar": "Polar",
    "sunsync": "Sun-sync",
    "frozen_elliptical": "Frozen ellip.",
    "nrho": "NRHO",
    "lissajous_l2": "Lissajous L2",
    "dro": "DRO",
    "areostationary": "Areostationary",
    "molniya": "Molniya",
}

moon_nc = []
mars_nc = []
labels = []

for fam in families_order:
    moon_key = f"Moon_{fam}"
    mars_key = f"Mars_{fam}"
    moon_val = nc_raw.get(moon_key)
    mars_val = nc_raw.get(mars_key)
    # Include if at least one body has this family
    if moon_key in nc_raw or mars_key in nc_raw:
        labels.append(family_labels[fam])
        moon_nc.append(moon_val if moon_val is not None else -1)
        mars_nc.append(mars_val if mars_val is not None else -1)

y = np.arange(len(labels))
bar_h = 0.35

fig, ax = plt.subplots(figsize=figsize_single("ieee", height_ratio=0.91))

# Moon bars
moon_mask = [v >= 0 for v in moon_nc]
moon_vals = [v if v >= 0 else 0 for v in moon_nc]
bars_moon = ax.barh(
    y + bar_h / 2,
    moon_vals,
    bar_h,
    color=BODY_COLORS["Moon"],
    edgecolor="white",
    linewidth=0.5,
    label="Moon",
)
# Mars bars
mars_mask = [v >= 0 for v in mars_nc]
mars_vals = [v if v >= 0 else 0 for v in mars_nc]
bars_mars = ax.barh(
    y - bar_h / 2,
    mars_vals,
    bar_h,
    color=BODY_COLORS["Mars"],
    edgecolor="white",
    linewidth=0.5,
    label="Mars",
)

# Mark "does not percolate" families
for i, (mv, mav) in enumerate(zip(moon_nc, mars_nc)):
    if mv < 0:
        ax.text(0.15, i + bar_h / 2, "—", fontsize=7, va="center", color="#888888")
    else:
        ax.text(
            mv + 0.15, i + bar_h / 2, str(mv), fontsize=7, va="center", color=BODY_COLORS["Moon"]
        )
    if mav < 0:
        ax.text(0.15, i - bar_h / 2, "—", fontsize=7, va="center", color="#888888")
    else:
        ax.text(
            mav + 0.15, i - bar_h / 2, str(mav), fontsize=7, va="center", color=BODY_COLORS["Mars"]
        )

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Percolation threshold $n_c$")
ax.set_xlim(0, 7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=8, framealpha=0.88)
ax.grid(True, axis="x", alpha=0.25)

save_fig(fig, "p3_percolation_threshold")
plt.close(fig)
