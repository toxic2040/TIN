#!/usr/bin/env python3
"""Historical gamma vs Var[ln p] scatter from archived production panels.

This reproduces a pooled in-sample association from the 2026-03-11 corpus. It
does not establish an order parameter, mechanism, classifier, or current law.
"""

import json
import math

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import BODY_COLORS, apply_style, figsize_single, save_fig

apply_style("pre")

RUNS = Path(__file__).parent

PROD_DIR = RUNS / "epyc_results" / "production_2026_03_11"
PANELS = [
    "production_P1_results.json",
    "production_P3_results.json",
    "production_P4_results.json",
    "production_P5P6_results.json",
    "production_P7_results.json",
    "production_P8_results.json",
    "production_P12_results.json",
    "production_P14a_results.json",
    "production_P14b_results.json",
    "production_P14c_results.json",
    "production_P14d_results.json",
]

# Body colors from unified palette (colorblind-safe, Paul Tol derived)

# Load all production panels
print("Loading historical production panels...")
by_body = defaultdict(lambda: {"vlp": [], "gamma": []})
n_total = 0
n_valid = 0

for panel_name in PANELS:
    panel_path = PROD_DIR / panel_name
    if not panel_path.exists():
        print(f"  SKIP {panel_name}")
        continue
    with open(panel_path) as f:
        records = json.load(f)
    for r in records:
        n_total += 1
        vlp = r.get("var_log_p")
        lyap = r.get("lyapunov")
        body = r.get("body", "Unknown")
        n_paths = r.get("n_paths", 0)
        if vlp is None or lyap is None:
            continue
        if isinstance(vlp, float) and math.isnan(vlp):
            continue
        if isinstance(lyap, float) and math.isnan(lyap):
            continue
        if n_paths == 0:
            continue
        gamma = -lyap  # gamma = -lyapunov
        by_body[body]["vlp"].append(vlp)
        by_body[body]["gamma"].append(gamma)
        n_valid += 1

print(f"  Total: {n_total}, Valid: {n_valid}")

# Compute overall regression
all_vlp = []
all_gamma = []
for body, data in by_body.items():
    all_vlp.extend(data["vlp"])
    all_gamma.extend(data["gamma"])

all_vlp = np.array(all_vlp)
all_gamma = np.array(all_gamma)

# OLS
slope, intercept = np.polyfit(all_vlp, all_gamma, 1)
ss_res = np.sum((all_gamma - (slope * all_vlp + intercept)) ** 2)
ss_tot = np.sum((all_gamma - np.mean(all_gamma)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"  Historical pooled OLS: slope={slope:.2f}, intercept={intercept:.4f}, R²={r2:.4f}")

# Plot
fig, ax = plt.subplots(figsize=figsize_single("pre", height_ratio=0.95))

# Scatter each body
for body in sorted(by_body.keys()):
    data = by_body[body]
    vlp = np.array(data["vlp"])
    gamma = np.array(data["gamma"])
    color = BODY_COLORS.get(body, "#888888")
    n = len(vlp)
    # Use small alpha for large bodies, larger for small
    alpha = max(0.08, min(0.5, 500 / n))
    ax.scatter(
        vlp,
        gamma,
        s=2,
        color=color,
        alpha=alpha,
        label=f"{body} ({n:,})",
        rasterized=True,
        zorder=2,
    )

# Regression line
x_fit = np.linspace(0, all_vlp.max() * 1.05, 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, "-", color="#333333", lw=1.2, zorder=3)

# R² annotation
ax.text(
    0.97,
    0.03,
    "Historical pooled fit\n"
    f"$\\gamma = {slope:.1f} \\times \\mathrm{{Var}}[\\ln p] + \\varepsilon$\n"
    f"$R^2 = {r2:.3f}$,  $n = {n_valid:,}$",
    transform=ax.transAxes,
    fontsize=7,
    ha="right",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.92),
)

ax.set_xlabel("$\\mathrm{Var}[\\ln p]$")
ax.set_ylabel("Archived $\\gamma$ diagnostic")
ax.set_title("Historical Production-Corpus Association — Classifier Retired")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.2)

# Legend — compact, sorted by n
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper left",
    fontsize=5.5,
    framealpha=0.88,
    markerscale=3,
    ncol=2,
    handletextpad=0.3,
    columnspacing=0.8,
)

save_fig(fig, "p2_varlogp_gamma_scatter")
plt.close(fig)
