#!/usr/bin/env python3
"""P3 Fig 3: Braess paradox — DR(n) curves showing non-monotonic scaling.

Highlights worst case: Mars frozen elliptical n=2→3, ΔDR = −0.330.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import BODY_COLORS, FAMILY_STYLES, apply_style, figsize_single, save_fig

apply_style("ieee")

RUNS = os.path.dirname(os.path.abspath(__file__))

# Load bucket 4 raw data for DR(n) curves
bucket4_path = os.path.join(
    RUNS, "epyc_results", "campaign_2026_03_11", "campaign_bucket_4_results.json"
)
with open(bucket4_path) as f:
    records = json.load(f)

# Load Braess cases for annotation
summary_path = os.path.join(RUNS, "epyc_results", "campaign_2026_03_11", "campaign_summary.json")
with open(summary_path) as f:
    summary = json.load(f)
braess_cases = summary["per_bucket"]["4"]["analysis"]["braess_cases"]

# Aggregate DR by (body, family, band, n_sats)
agg = defaultdict(list)
for r in records:
    key = (r["body"], r["family"], r["band"], r["n_sats"])
    agg[key].append(r["DR"])

# Select families with Braess for illustration
# Focus on the worst: Mars frozen_elliptical, and a Moon case (NRHO)
focus = [
    (
        "Mars",
        "frozen_elliptical",
        "Ka",
        FAMILY_STYLES["frozen_elliptical"]["color"],
        FAMILY_STYLES["frozen_elliptical"]["marker"],
        "Mars frozen ellip.",
    ),
    ("Mars", "polar", "Ka", BODY_COLORS["Mars"], "o", "Mars polar"),
    (
        "Moon",
        "nrho",
        "X",
        FAMILY_STYLES["nrho"]["color"],
        FAMILY_STYLES["nrho"]["marker"],
        "Moon NRHO",
    ),
    ("Moon", "polar", "X", BODY_COLORS["Moon"], "D", "Moon polar"),
]

fig, ax = plt.subplots(figsize=figsize_single("ieee", height_ratio=0.91))

for body, fam, band, color, marker, label in focus:
    ns, drs = [], []
    for (b, f, ba, n), vals in sorted(agg.items()):
        if b == body and f == fam and ba == band:
            ns.append(n)
            drs.append(np.mean(vals))
    if ns:
        ax.plot(ns, drs, marker=marker, color=color, label=label, markersize=4, lw=1.2, alpha=0.85)

# Highlight worst Braess drop
ax.annotate(
    "$\\Delta\\mathrm{DR} = -0.330$",
    xy=(3, 0.660),
    xytext=(5, 0.55),
    fontsize=7,
    color=FAMILY_STYLES["frozen_elliptical"]["color"],
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=FAMILY_STYLES["frozen_elliptical"]["color"], lw=1.0),
)

# Shade Braess region for the worst case
ax.fill_between(
    [2, 3], [0.990, 0.660], alpha=0.08, color=FAMILY_STYLES["frozen_elliptical"]["color"]
)

ax.set_xlabel("Constellation size $n$")
ax.set_ylabel("Delivery ratio DR")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.2)
ax.legend(loc="upper right", fontsize=6.5, framealpha=0.88)
ax.set_xticks([1, 2, 3, 4, 6, 8, 12])

save_fig(fig, "p3_braess_paradox")
plt.close(fig)
