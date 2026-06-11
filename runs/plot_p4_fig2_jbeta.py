#!/usr/bin/env python3
"""P4 Fig 2: J_beta saturation invariance across 4 configurations.

Shows J_beta ≈ 0.242 is constant despite K_eff spanning 6.6× range.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import CONFIG_STYLES, apply_style, figsize_double, save_fig

apply_style("pre")

RUNS = os.path.dirname(os.path.abspath(__file__))

# Data from Table 1 (confirmed exact by provenance audit)
cfg_keys = ["moon_K1", "moon_K2", "mars_K1", "mars_K2"]
configs = [CONFIG_STYLES[k]["label"] for k in cfg_keys]
jbeta = [0.242, 0.241, 0.243, 0.242]
keff = [48.2, 126.2, 19.0, 19.0]
rho_inf = [0.591, 0.403, 0.418, 0.408]

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=figsize_double("pre", height_ratio=0.40),
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45},
)

# Left panel: J_beta (near-constant)
x = np.arange(len(configs))
colors = [CONFIG_STYLES[k]["color"] for k in cfg_keys]
ax1.bar(x, jbeta, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax1.axhline(0.25, color="#bbbbbb", ls="--", lw=0.8, label="TASEP $J_{\\max}=1/4$")
ax1.axhline(0.242, color="#555555", ls=":", lw=0.8, label="Mean $J_\\beta=0.242$")
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=25, ha="right", fontsize=8)
ax1.set_ylabel("$J_\\beta = \\beta_{\\mathrm{eff}}(1-\\beta_{\\mathrm{eff}})$")
ax1.set_ylim(0.22, 0.26)
ax1.legend(loc="lower left", fontsize=7, framealpha=0.88)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title("Per-lane saturation current", fontsize=10)

# Right panel: K_eff (varies 6.6×)
ax2.bar(x, keff, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, rotation=25, ha="right", fontsize=8)
ax2.set_ylabel("$K_{\\mathrm{eff}}$ (effective lanes)")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title("Effective lane count", fontsize=10)

# Annotate the 6.6× range
ax2.annotate(
    "", xy=(1, 126.2), xytext=(2, 19.0), arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2)
)
ax2.text(1.5, 72, "$6.6\\times$", ha="center", fontsize=9, color="#555555")

save_fig(fig, "p4_jbeta_invariance")
plt.close(fig)
