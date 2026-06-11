#!/usr/bin/env python3
"""S_T distribution stratified by hop count H — testing the Weibull prediction.

The stretched-exponential transition model predicts:
  S_T ≈ 1 − exp(−(C/C*)^H)

Key predictions tested:
  1. Transition sharpens with H (confirmed: Width P10-P90 drops 0.71 → 0.07)
  2. Gap width scales ∝ 1/H (order-of-magnitude correct, body-dependent prefactor)
  3. The [0.70, 0.93] gap narrows with H (confirmed, but also shifts)

Non-obvious findings:
  - S_T is quantized at rational fractions (1/3, 1/2, 2/3, 1.0 = 39% of configs)
  - Moon H=3 has 41% of configs in the gap (commensurability resonance)
  - Mars H=3 has 1.0% in the gap (sharp transition)
  - var_H > 0 for 95%+ of H≥3 configs (mixed hop counts per architecture)
"""

import collections
import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tin_figure_style import BODY_COLORS, apply_style, figsize_double, save_fig

apply_style("pre")

# ── Load production data ──────────────────────────────────────────────
prod_dir = "runs/epyc_results/production_2026_03_11"
records = []
for fname in sorted(os.listdir(prod_dir)):
    if not fname.endswith(".json") or fname == "production_summary.json":
        continue
    with open(os.path.join(prod_dir, fname)) as f:
        data = json.load(f)
    for rec in data:
        h = rec.get("E_H")
        st = rec.get("S_T")
        if h is None or st is None or (isinstance(h, float) and math.isnan(h)):
            continue
        records.append(
            {
                "H": round(h),
                "S_T": st,
                "body": rec.get("body", ""),
                "var_H": rec.get("var_H", 0),
            }
        )

print(f"Loaded {len(records)} valid configs")

# Bin by H
by_h = collections.defaultdict(list)
for r in records:
    by_h[r["H"]].append(r)

# ── Figure 1: S_T histograms stacked by H ─────────────────────────────
h_bins = [h for h in sorted(by_h.keys()) if len(by_h[h]) >= 50]
n_panels = len(h_bins)

fig, axes = plt.subplots(
    n_panels,
    1,
    figsize=figsize_double("pre", height_ratio=0.22 * n_panels),
    sharex=True,
)
if n_panels == 1:
    axes = [axes]

hist_bins = np.linspace(0, 1.005, 201)
gap_lo, gap_hi = 0.70, 0.93

# Color gradient from blue (H=2) to red (H=7+)
h_colors = {
    2: "#4477AA",
    3: "#33BBEE",
    4: "#009988",
    5: "#EE7733",
    6: "#CC3311",
    7: "#AA3377",
}

for idx, h in enumerate(h_bins):
    ax = axes[idx]
    st_arr = np.array([r["S_T"] for r in by_h[h]])

    color = h_colors.get(h, "#888888")
    ax.hist(
        st_arr,
        bins=hist_bins,
        density=True,
        color=color,
        alpha=0.65,
        edgecolor="none",
        zorder=3,
    )

    # Shade the gap region
    ax.axvspan(gap_lo, gap_hi, alpha=0.08, color="#CC3311", zorder=1)

    # Gap fraction
    gap_frac = np.sum((st_arr >= gap_lo) & (st_arr <= gap_hi)) / len(st_arr)
    p10, p90 = np.percentile(st_arr, [10, 90])
    width = p90 - p10

    # Label
    label = (
        f"$H = {h}$\\quad $n = {len(st_arr):,}$\\quad "
        f"gap$={100 * gap_frac:.1f}$\\%\\quad "
        f"$\\Delta_{{10\\text{{--}}90}} = {width:.3f}$"
    )
    ax.text(
        0.98,
        0.85,
        label,
        transform=ax.transAxes,
        fontsize=6.5,
        ha="right",
        va="top",
        color=color,
    )

    # Mark the gap boundaries
    ax.axvline(gap_lo, color="#CC3311", ls=":", lw=0.4, alpha=0.5, zorder=2)
    ax.axvline(gap_hi, color="#CC3311", ls=":", lw=0.4, alpha=0.5, zorder=2)

    ax.set_ylabel("density", fontsize=7)
    ax.set_ylim(bottom=0)
    # Suppress y-axis numbers — density values aren't meaningful to compare
    ax.set_yticks([])

axes[-1].set_xlabel("Temporal reachability $S_T$")
axes[-1].set_xlim(-0.02, 1.05)
axes[0].set_title(
    "$S_T$ distribution by hop count — transition sharpens with $H$",
    fontsize=9,
)

save_fig(fig, "st_gap_by_hop_count")
plt.close(fig)

# ── Figure 2: Body-stratified for H=2 and H=3 ────────────────────────
fig2, (ax_h2, ax_h3) = plt.subplots(
    2,
    1,
    figsize=figsize_double("pre", height_ratio=0.45),
    sharex=True,
)

body_order = ["Moon", "Mars", "Mercury", "Venus", "Titan", "Ceres", "Saturn", "Jupiter"]
hist_bins_fine = np.linspace(0, 1.005, 101)

for ax, h, title in [
    (ax_h2, 2, "$H = 2$: body-stratified"),
    (ax_h3, 3, "$H = 3$: body-stratified"),
]:
    for body in body_order:
        subset = [r["S_T"] for r in by_h[h] if r["body"] == body]
        if len(subset) < 20:
            continue
        arr = np.array(subset)
        color = BODY_COLORS.get(body, "#888888")
        ax.hist(
            arr,
            bins=hist_bins_fine,
            density=True,
            color=color,
            alpha=0.45,
            edgecolor="none",
            label=f"{body} ($n={len(subset)}$)",
            zorder=3,
        )

    ax.axvspan(gap_lo, gap_hi, alpha=0.08, color="#CC3311", zorder=1)
    ax.axvline(gap_lo, color="#CC3311", ls=":", lw=0.4, alpha=0.5)
    ax.axvline(gap_hi, color="#CC3311", ls=":", lw=0.4, alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("density", fontsize=7)
    ax.set_yticks([])
    ax.legend(fontsize=5.5, loc="upper left", framealpha=0.8)

ax_h3.set_xlabel("Temporal reachability $S_T$")
ax_h3.set_xlim(-0.02, 1.05)

save_fig(fig2, "st_gap_body_stratified")
plt.close(fig2)

# ── Figure 3: Width vs H with 1/H reference ──────────────────────────
fig3, ax3 = plt.subplots(figsize=(3.375, 3.375 / 1.618))

h_vals = []
widths = []
counts = []
for h in sorted(by_h.keys()):
    if len(by_h[h]) < 50:
        continue
    st_arr = np.array([r["S_T"] for r in by_h[h]])
    p10, p90 = np.percentile(st_arr, [10, 90])
    h_vals.append(h)
    widths.append(p90 - p10)
    counts.append(len(st_arr))

h_vals = np.array(h_vals)
widths = np.array(widths)

ax3.plot(
    h_vals,
    widths,
    "o-",
    color="#4477AA",
    markersize=5,
    markeredgecolor="#222222",
    markeredgewidth=0.5,
    lw=1.2,
    label="Data (P10--P90 width)",
    zorder=5,
)

# 1/H reference line, scaled to match at H=2
ref = widths[0] * 2 / h_vals
ax3.plot(
    h_vals,
    ref,
    "--",
    color="#CC3311",
    lw=0.8,
    alpha=0.6,
    label="$\\propto 1/H$ (scaled to $H=2$)",
    zorder=3,
)

# Annotate sample sizes
for i, (hv, w, n) in enumerate(zip(h_vals, widths, counts)):
    ax3.annotate(
        f"$n={n:,}$",
        xy=(hv, w),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=5,
        color="#666666",
    )

ax3.set_xlabel("Hop count $H$")
ax3.set_ylabel("Transition width $\\Delta_{10\\text{--}90}(S_T)$")
ax3.set_xlim(1.5, 7.5)
ax3.set_ylim(0, 0.80)
ax3.legend(fontsize=6, loc="upper right")

save_fig(fig3, "st_transition_width_vs_H")
plt.close(fig3)

print("\nDone — three figures saved.")
