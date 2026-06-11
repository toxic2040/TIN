#!/usr/bin/env python3
"""P2 Fig 4: Layered severity hierarchy as diagnostic flowchart.

Layers -1 through 2.  Each checked in order; Layer 2 never reached
in 344 experiments.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tin_figure_style import LAYER_COLORS, apply_style, figsize_single, save_fig

apply_style("pre")

RUNS = os.path.dirname(os.path.abspath(__file__))

layers = [
    {
        "layer": "$-1$",
        "name": "Percolation\nthreshold",
        "diag": "$S_T = 0$",
        "observed": "Yes (54%)",
        "color": LAYER_COLORS["-1"],
    },
    {
        "layer": "0",
        "name": "Static geometric\nsupport",
        "diag": "Link-type counts",
        "observed": "Yes",
        "color": LAYER_COLORS["0"],
    },
    {
        "layer": "0.5",
        "name": "Temporal\ncoverage",
        "diag": "Per-cycle surface\ncontacts",
        "observed": "Yes (all 33)",
        "color": LAYER_COLORS["0.5"],
    },
    {
        "layer": "1",
        "name": "Phase-class\nstructure",
        "diag": "Binary RAAN\ncoloring",
        "observed": "Yes",
        "color": LAYER_COLORS["1"],
    },
    {
        "layer": "2",
        "name": "Temporal\ncomposition",
        "diag": "Per-cycle chain\nanalysis",
        "observed": "Not observed",
        "color": LAYER_COLORS["2"],
    },
]

fig, ax = plt.subplots(figsize=figsize_single("pre", height_ratio=1.33))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, len(layers) * 1.6 + 0.5)
ax.axis("off")

box_w = 3.0
box_h = 0.9
x_center = 1.5

for i, layer in enumerate(reversed(layers)):
    y = i * 1.6
    alpha = 0.3 if layer["observed"].startswith("Not") else 0.85

    # Box
    rect = FancyBboxPatch(
        (x_center - box_w / 2, y),
        box_w,
        box_h,
        boxstyle="round,pad=0.1",
        facecolor=layer["color"],
        edgecolor="white",
        alpha=alpha,
        linewidth=1.5,
    )
    ax.add_patch(rect)

    # Layer number
    ax.text(
        x_center - box_w / 2 + 0.15,
        y + box_h / 2,
        f"L{layer['layer']}",
        fontsize=8,
        fontweight="bold",
        va="center",
        color="white" if alpha > 0.5 else "#555555",
    )

    # Name
    ax.text(
        x_center,
        y + box_h / 2 + 0.12,
        layer["name"],
        fontsize=7.5,
        va="center",
        ha="center",
        color="white" if alpha > 0.5 else "#555555",
        fontweight="bold",
    )

    # Diagnostic (below name, smaller)
    ax.text(
        x_center,
        y + 0.12,
        layer["diag"],
        fontsize=6,
        va="center",
        ha="center",
        color="white" if alpha > 0.5 else "#888888",
        style="italic",
    )

    # Observed tag on right
    obs_color = "#27ae60" if not layer["observed"].startswith("Not") else "#c0392b"
    ax.text(
        x_center + box_w / 2 + 0.05,
        y + box_h / 2,
        layer["observed"],
        fontsize=5.5,
        va="center",
        ha="left",
        color=obs_color,
        fontweight="bold",
    )

    # Arrow to next layer
    if i < len(layers) - 1:
        ax.annotate(
            "",
            xy=(x_center, y + box_h + 0.05),
            xytext=(x_center, y + box_h + 0.65),
            arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=1.2),
        )
        ax.text(x_center + 0.15, y + box_h + 0.35, "pass", fontsize=6, color="#bbbbbb", va="center")

# Title
ax.text(
    x_center,
    len(layers) * 1.6 + 0.2,
    "Diagnostic hierarchy",
    fontsize=10,
    ha="center",
    va="bottom",
    fontweight="bold",
)
ax.text(
    x_center,
    len(layers) * 1.6,
    "Check upstream before claiming downstream",
    fontsize=7,
    ha="center",
    va="bottom",
    color="#888888",
    style="italic",
)

save_fig(fig, "p2_hierarchy_diagram")
plt.close(fig)
