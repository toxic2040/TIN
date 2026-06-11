#!/usr/bin/env python3
"""P4 Fig 1: TASEP phase diagram — eta vs lambda for all 4 configs.

Log-log scale showing LD, HD phases and power-law fits.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import CONFIG_STYLES, apply_style, figsize_single, save_fig

apply_style("pre")

RUNS = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(RUNS, "load_sweep_v2_results.json")) as f:
    data = json.load(f)

styles = CONFIG_STYLES

fig, ax = plt.subplots(figsize=figsize_single("pre", height_ratio=0.89))

for cfg_key, style in styles.items():
    loads = data[cfg_key]["load"]
    lambdas = sorted([int(k) for k in loads.keys()])
    etas = [loads[str(lam)]["eta_mean"] for lam in lambdas]

    ax.plot(
        lambdas,
        etas,
        marker=style["marker"],
        color=style["color"],
        label=style["label"],
        markersize=4,
        lw=1.2,
        alpha=0.9,
    )

# Reference lines for power-law scaling
lam_ref = np.array([5, 50])
# HD prediction: eta ~ 1/lambda
ax.plot(lam_ref, 0.95 * 5 / lam_ref, "--", color="#bbbbbb", lw=0.7)
ax.text(30, 0.12, "$\\eta \\sim \\lambda^{-1}$", fontsize=7, color="#888888", rotation=-35)

# Phase labels
ax.text(
    1.3,
    0.85,
    "LD",
    fontsize=9,
    color=CONFIG_STYLES["moon_K1"]["color"],
    fontweight="bold",
    alpha=0.5,
)
ax.text(
    30,
    0.55,
    "HD",
    fontsize=9,
    color=CONFIG_STYLES["mars_K1"]["color"],
    fontweight="bold",
    alpha=0.5,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Injection rate $\\lambda$ (bundles/sol)")
ax.set_ylabel("Routing efficiency $\\eta$")
ax.set_xlim(0.8, 70)
ax.set_ylim(0.05, 1.2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.2, which="both")
ax.legend(loc="lower left", framealpha=0.88)

save_fig(fig, "p4_tasep_phase_diagram")
plt.close(fig)
