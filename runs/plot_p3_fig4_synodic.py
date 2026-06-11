#!/usr/bin/env python3
"""P3 Fig 4: Synodic variation — DR vs SEP angle by band.

Shows conjunction survival: Ka-band maintains marginal connectivity,
UHF/X fail at SEP < 3°.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import BAND_STYLES, apply_style, figsize_single, save_fig

apply_style("ieee")

RUNS = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(RUNS, "epyc_results", "campaign_2026_03_11", "campaign_bucket_6_results.json")
) as f:
    records = json.load(f)

# Aggregate: mean DR by (band, sep_override_deg) for polar n=4
agg = defaultdict(list)
for r in records:
    if r["family"] != "polar" or r["n_sats"] != 4:
        continue
    dr = r["DR"]
    if dr is None or (isinstance(dr, float) and np.isnan(dr)):
        dr = 0.0  # failed links → DR=0
    agg[(r["band"], r["sep_override_deg"])].append(dr)

band_styles = {
    "UHF": {**BAND_STYLES["UHF"], "label": "UHF"},
    "X": {**BAND_STYLES["X"], "label": "X-band"},
    "Ka": {**BAND_STYLES["Ka"], "label": "Ka-band"},
}

fig, ax = plt.subplots(figsize=figsize_single("ieee", height_ratio=0.86))

for band, style in band_styles.items():
    seps, drs = [], []
    for (b, sep), vals in sorted(agg.items()):
        if b == band:
            seps.append(sep)
            drs.append(np.mean(vals))
    ax.plot(
        seps,
        drs,
        marker=style["marker"],
        color=style["color"],
        label=style["label"],
        markersize=4,
        lw=1.2,
        alpha=0.85,
    )

# Conjunction zone shading
ax.axvspan(0, 5, alpha=0.06, color="#c0392b")
ax.text(
    3,
    0.85,
    "conjunction\nzone",
    fontsize=6,
    color="#c0392b",
    ha="center",
    style="italic",
    alpha=0.7,
)

# Opposition label
ax.text(170, 0.85, "opposition", fontsize=6, color="#888888", ha="center", style="italic")

ax.set_xlabel("Sun-Earth-Probe angle (deg)")
ax.set_ylabel("Delivery ratio DR")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.2)
ax.legend(loc="center right", fontsize=7, framealpha=0.88)
ax.set_xlim(0, 185)
ax.set_ylim(-0.02, 1.0)
ax.invert_xaxis()  # opposition (180°) on left, conjunction (0°) on right

save_fig(fig, "p3_synodic_variation")
plt.close(fig)
