#!/usr/bin/env python3
"""Historical greedy/retry gamma comparison across eight bodies.

Arrows preserve an archived geometric-budget comparison from gamma_greedy to
gamma_oracle. The gamma=0 line is a descriptive sign reference, not a current
classifier, reclassification boundary, or routing-trap test.
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

CROSSING_COLOR = "#4477AA"  # blue — crosses the historical zero reference
NO_CROSSING_COLOR = "#CC3311"  # red — does not cross the reference

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
    crosses_zero = (gg < 0 and go > 0) or (gg > 0 and go < 0)
    color = CROSSING_COLOR if crosses_zero else NO_CROSSING_COLOR
    lw = 1.8 if crosses_zero else 1.0
    alpha = 1.0 if crosses_zero else 0.7

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

# Historical zero-sign reference; classifier interpretation is retired.
ax.axvline(0, color="#555555", ls="--", lw=0.8, zorder=0)

# Shade the negative-slope side for visual orientation only.
ax.axvspan(ax.get_xlim()[0], 0, alpha=0.04, color=NO_CROSSING_COLOR, zorder=0)

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("$\\gamma$")
ax.spines[["top", "right"]].set_visible(False)

# Legend
ax.plot(
    [],
    [],
    "o-",
    color=CROSSING_COLOR,
    lw=1.8,
    markersize=5,
    label="Crosses historical $\\gamma=0$ reference",
)
ax.plot(
    [],
    [],
    "o-",
    color=NO_CROSSING_COLOR,
    lw=1.0,
    markersize=5,
    alpha=0.7,
    label="Does not cross historical reference",
)
ax.legend(loc="lower right", fontsize=7, framealpha=0.88)

# Descriptive sign labels in data coordinates.
ax.text(0.05, len(names) - 0.3, "$\\gamma>0$", fontsize=7, color=CROSSING_COLOR)
ax.text(-0.05, len(names) - 0.3, "$\\gamma<0$", fontsize=7, color=NO_CROSSING_COLOR, ha="right")
ax.set_title("Historical slope comparison — classifier retired", fontsize=9)

save_fig(fig, "p2_oracle_reclassification")
plt.close(fig)
