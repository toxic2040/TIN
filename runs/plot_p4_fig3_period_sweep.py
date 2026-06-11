#!/usr/bin/env python3
"""P4 Fig 3: Period sweep — K_eff vs altitude.

Distinguishes irrational (DQ > 0) vs rational (DQ = 0) periods.
Highlights golden mean, ELFO actual, and baseline.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tin_figure_style import apply_style, figsize_single, save_fig

apply_style("pre")

RUNS = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(RUNS, "period_sweep_results.json")) as f:
    data = json.load(f)

# Separate into categories
rational_alt, rational_keff = [], []
irrational_alt, irrational_keff, irrational_labels = [], [], []
special = {}  # golden mean, ELFO, circular-at-ELFO-period

for key, rec in data.items():
    if key == "baseline":
        special["baseline"] = rec["K_eff"]
        continue
    t = rec.get("torus", {})
    alt = t.get("altitude_km")
    dq = t.get("diophantine_quality", 0)
    keff = rec["K_eff"]
    if alt is None:
        continue

    if key == "ELFO_actual":
        special["ELFO"] = (alt, keff)
    elif key == "ratio_r4.40_ELFO_period":
        special["circular_ELFO"] = (alt, keff)
    elif key == "ratio_phi":
        special["phi"] = (alt, keff)
        irrational_alt.append(alt)
        irrational_keff.append(keff)
        irrational_labels.append("$\\varphi$")
    elif key == "ratio_sqrt2":
        irrational_alt.append(alt)
        irrational_keff.append(keff)
        irrational_labels.append("$\\sqrt{2}$")
    elif key == "ratio_e":
        irrational_alt.append(alt)
        irrational_keff.append(keff)
        irrational_labels.append("$e$")
    elif key == "ratio_pi":
        irrational_alt.append(alt)
        irrational_keff.append(keff)
        irrational_labels.append("$\\pi$")
    elif dq > 0:
        irrational_alt.append(alt)
        irrational_keff.append(keff)
        irrational_labels.append("")
    else:
        rational_alt.append(alt)
        rational_keff.append(keff)

fig, ax = plt.subplots(figsize=figsize_single("pre", height_ratio=0.89))

# Rational periods (DQ = 0)
ax.scatter(
    rational_alt,
    rational_keff,
    marker="o",
    s=25,
    color="#4477AA",
    alpha=0.7,
    label="Rational period",
    zorder=3,
)

# Irrational periods (DQ > 0)
ax.scatter(
    irrational_alt,
    irrational_keff,
    marker="D",
    s=30,
    color="#CC3311",
    edgecolor="#CC3311",
    alpha=0.9,
    label="Irrational period",
    zorder=4,
)

# Label irrational points
for alt, keff, lbl in zip(irrational_alt, irrational_keff, irrational_labels):
    if lbl:
        ax.annotate(
            lbl,
            (alt, keff),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=7,
            color="#CC3311",
        )

# ELFO actual (star)
ea, ek = special["ELFO"]
ax.scatter(
    [ea],
    [ek],
    marker="*",
    s=120,
    color="#EE7733",
    edgecolor="#EE7733",
    zorder=5,
    label="ELFO ($e=0.58$)",
)
ax.annotate(
    "ELFO",
    (ea, ek),
    textcoords="offset points",
    xytext=(6, 4),
    fontsize=7,
    color="#EE7733",
    fontweight="bold",
)

# Circular at ELFO period
ca, ck = special["circular_ELFO"]
ax.scatter([ca], [ck], marker="o", s=40, color="white", edgecolor="#555555", lw=1.2, zorder=5)
ax.annotate(
    "circ. at\nELFO $T$",
    (ca, ck),
    textcoords="offset points",
    xytext=(6, -12),
    fontsize=6,
    color="#555555",
)

# Eccentricity bonus annotation
ax.annotate(
    "", xy=(ea, ek), xytext=(ca, ck), arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.8)
)
ax.text(
    (ea + ca) / 2 + 200,
    (ek + ck) / 2,
    "+50%\n$e$-bonus",
    fontsize=6,
    color="#888888",
    ha="left",
    va="center",
)

# Baseline
ax.axhline(special["baseline"], color="#bbbbbb", ls=":", lw=0.8)
ax.text(500, special["baseline"] + 2, "baseline (no relay)", fontsize=6, color="#bbbbbb")

ax.set_xlabel("Relay altitude (km)")
ax.set_ylabel("$K_{\\mathrm{eff}}$ (effective lanes)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.2)
ax.legend(loc="upper left", fontsize=6.5, framealpha=0.88)

save_fig(fig, "p4_period_sweep")
plt.close(fig)
