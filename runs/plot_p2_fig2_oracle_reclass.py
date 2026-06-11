#!/usr/bin/env python3
"""P2 Fig 2: Oracle reclassification across 8 bodies.

Arrows show gamma_greedy → gamma_oracle.  Mercury and Europa cross
the gamma=0 boundary (routing traps).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import apply_style, figsize_single, save_fig

apply_style("pre")

RUNS = os.path.dirname(os.path.abspath(__file__))

RECLASSIFY_COLOR = "#4477AA"  # blue — reclassifies
REMAINS_COLOR = "#CC3311"  # red — remains trap

with open(os.path.join(RUNS, "gamma_oracle_canonical_results.json")) as f:
    data = json.load(f)

# Sort by gamma_greedy (gamma_normal) for visual clarity
bodies = sorted(data["bodies"], key=lambda b: b["gamma_normal"])

names = [b["body"].capitalize() for b in bodies]
gamma_greedy = [b["gamma_normal"] for b in bodies]
gamma_oracle = [b["gamma_retry"] for b in bodies]

fig, ax = plt.subplots(figsize=figsize_single("pre", height_ratio=1.13))

y = np.arange(len(names))

# Draw arrows from greedy to oracle
for i, (gg, go, name) in enumerate(zip(gamma_greedy, gamma_oracle, names)):
    reclassifies = (gg < 0 and go > 0) or (gg > 0 and go < 0)
    color = RECLASSIFY_COLOR if reclassifies else REMAINS_COLOR
    lw = 1.8 if reclassifies else 1.0
    alpha = 1.0 if reclassifies else 0.7

    # Arrow
    ax.annotate(
        "",
        xy=(go, i),
        xytext=(gg, i),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
    )
    # Greedy dot
    ax.plot(gg, i, "o", color=color, markersize=5, alpha=alpha)
    # Oracle dot
    ax.plot(go, i, "s", color=color, markersize=5, alpha=alpha)

# Zero line
ax.axvline(0, color="#555555", ls="--", lw=0.8, zorder=0)

# Shade trap region
ax.axvspan(ax.get_xlim()[0], 0, alpha=0.04, color=REMAINS_COLOR, zorder=0)

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("$\\gamma$")
ax.spines[["top", "right"]].set_visible(False)

# Legend
ax.plot(
    [],
    [],
    "o-",
    color=RECLASSIFY_COLOR,
    lw=1.8,
    markersize=5,
    label="Reclassifies ($\\gamma$ crosses 0)",
)
ax.plot([], [], "o-", color=REMAINS_COLOR, lw=1.0, markersize=5, alpha=0.7, label="Remains trap")
ax.legend(loc="lower right", fontsize=7, framealpha=0.88)

# Annotations: trap/cluster labels in data coords
ax.text(0.05, len(names) - 0.3, "cluster", fontsize=7, color=RECLASSIFY_COLOR)
ax.text(-0.05, len(names) - 0.3, "trap", fontsize=7, color=REMAINS_COLOR, ha="right")

save_fig(fig, "p2_oracle_reclassification")
plt.close(fig)
