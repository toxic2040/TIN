#!/usr/bin/env python3
"""Commodity Phase Diagram: feasibility boundaries in (transit time, τ_half) space.

The first commodity-resolved feasibility map of the solar system.
Combines the one-tau rule (ITN §8.2) with Hohmann transfer times to derive
which commodities can traverse which routes via self-sustaining pipeline.

Two boundaries on the (T_transit, τ_half) plane:

  1. Exponential decay (cryogenics): τ_half > T × ln(2)
     Below this, no self-sustaining resupply pipeline exists.
     Derived from the steady-state reserve recurrence (§8.2).

  2. Step-function decay (perishables/crew): τ_shelf > T
     Below this, cargo arrives dead — no pipeline helps.

Body-pair transit times are Hohmann minimums (minimum-energy transfer).
Faster transfers are possible at higher Δv cost.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tin_figure_style import apply_style, figsize_double, save_fig

apply_style("pre")

# ── Physical constants ────────────────────────────────────────────────
AU_KM = 149_597_870.7
MU_SUN = 132_712_440_018.0  # km³/s²
MU_EARTH = 398_600.44  # km³/s²
LN2 = np.log(2)
DAY = 86_400.0

# ── Heliocentric semi-major axes (AU, from bodies.py) ─────────────────
BODIES_AU = {
    "Mercury": 0.387,
    "Venus": 0.723,
    "Earth": 1.000,
    "Mars": 1.524,
    "Ceres": 2.769,
    "Jupiter": 5.203,
    "Saturn": 9.537,
}


def hohmann_time_days(r1_km, r2_km, mu):
    """Hohmann transfer time between two circular orbits (days)."""
    a = (r1_km + r2_km) / 2.0
    return np.pi * np.sqrt(a**3 / mu) / DAY


# ── Compute all transfer times ────────────────────────────────────────
earth_km = BODIES_AU["Earth"] * AU_KM

transfers = {}
for name, a_au in BODIES_AU.items():
    if name == "Earth":
        continue
    t = hohmann_time_days(earth_km, a_au * AU_KM, MU_SUN)
    transfers[f"Earth\u2013{name}"] = t

# Earth-Moon (within Earth SOI, not heliocentric)
R_LEO = 6_371.0 + 400.0  # 400 km LEO
R_MOON = 384_400.0
transfers["Earth\u2013Moon"] = hohmann_time_days(R_LEO, R_MOON, MU_EARTH)

# Mars-Jupiter relay segment
mars_km = BODIES_AU["Mars"] * AU_KM
jupiter_km = BODIES_AU["Jupiter"] * AU_KM
transfers["Mars\u2013Jupiter"] = hohmann_time_days(mars_km, jupiter_km, MU_SUN)

# EMJ relay minimum (Earth-Mars + Mars-Jupiter, zero dwell floor)
transfers["EMJ relay (floor)"] = transfers["Earth\u2013Mars"] + transfers["Mars\u2013Jupiter"]

# Sort by transit time
transfers = dict(sorted(transfers.items(), key=lambda x: x[1]))

# Print table
print("Hohmann transfer times:")
print(f"  {'Pair':<25s} {'T (days)':>10s}  {'T (years)':>10s}")
print("  " + "-" * 48)
for pair, t in transfers.items():
    print(f"  {pair:<25s} {t:10.1f}  {t / 365.25:10.2f}")

# ── Commodity half-lives ──────────────────────────────────────────────
# (label, τ_half in days, color, decay type)
COMMODITIES = [
    ("Biological samples", 1, "#AA3377", "step"),
    ("mRNA vaccines", 3, "#EE3377", "step"),
    ("Standard vaccines", 30, "#EE7733", "step"),
    ("LH$_2$ propellant", 180, "#CC3311", "exp"),
    ("LOX", 500, "#33BBEE", "exp"),
    ("Food supplies", 730, "#009988", "step"),
]

# ── Body-pair styling ────────────────────────────────────────────────
# (key from transfers dict, color, show_label)
BODY_PAIRS = [
    ("Earth\u2013Moon", "#4477AA", True),
    ("Earth\u2013Venus", "#EE3377", True),
    ("Earth\u2013Mars", "#CC3311", True),
    ("Earth\u2013Ceres", "#AA3377", False),
    ("Earth\u2013Jupiter", "#BBBBBB", True),
    ("Mars\u2013Jupiter", "#BBBBBB", False),
    ("Earth\u2013Saturn", "#CCBB44", True),
    ("EMJ relay (floor)", "#BBBBBB", True),
]

# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=figsize_double("pre", height_ratio=0.55))

T_range = np.logspace(np.log10(1), np.log10(5000), 500)

# ── Feasibility regions ──────────────────────────────────────────────
# Below exponential boundary: infeasible for all
# Between exp and step boundaries: exponential can sustain, step cannot
# Above step boundary: feasible for all

tau_exp = T_range * LN2  # exponential pipeline boundary
tau_step = T_range  # step-function survival boundary

# Infeasible region (below exponential line)
ax.fill_between(
    T_range,
    0.3,
    tau_exp,
    alpha=0.10,
    color="#CC3311",
    zorder=0,
    label=None,
)

# Marginal region (between exp and step lines)
ax.fill_between(
    T_range,
    tau_exp,
    tau_step,
    alpha=0.06,
    color="#EE7733",
    zorder=0,
    label=None,
)

# Boundary lines
ax.plot(
    T_range,
    tau_exp,
    "-",
    color="#CC3311",
    lw=1.4,
    zorder=5,
    label=r"Pipeline: $\tau_{1/2} = T \ln 2$",
)
ax.plot(
    T_range,
    tau_step,
    "--",
    color="#888888",
    lw=1.0,
    zorder=5,
    label=r"Survival: $\tau_\mathrm{shelf} = T$",
)

# ── Region labels ─────────────────────────────────────────────────────
ax.text(
    3.0,
    1500,
    "\\textit{feasible}",
    fontsize=7,
    color="#009988",
    ha="left",
    va="center",
    alpha=0.7,
)
ax.text(
    30,
    1.2,
    "\\textit{infeasible}",
    fontsize=7,
    color="#CC3311",
    ha="center",
    va="center",
    alpha=0.7,
)
ax.text(
    3.5,
    2.0,
    "\\textit{exp.~OK,}",
    fontsize=5.5,
    color="#EE7733",
    ha="left",
    va="center",
    alpha=0.7,
)
ax.text(
    3.5,
    1.3,
    "\\textit{step~fails}",
    fontsize=5.5,
    color="#EE7733",
    ha="left",
    va="center",
    alpha=0.7,
)

# ── Body-pair vertical lines ─────────────────────────────────────────
for pair_key, color, show_label in BODY_PAIRS:
    t = transfers[pair_key]
    ls = "-" if "relay" not in pair_key else ":"
    lw = 0.7 if "relay" not in pair_key else 0.9
    ax.axvline(
        t,
        color=color,
        ls=ls,
        lw=lw,
        alpha=0.5,
        zorder=2,
    )
    if show_label:
        # Place label at top of plot
        label = pair_key.replace("Earth\u2013", "").replace(" (floor)", "")
        if pair_key == "EMJ relay (floor)":
            label = "EMJ relay"
        rot = 90
        ax.text(
            t * 1.06,
            3500,
            label,
            fontsize=5.5,
            color=color,
            rotation=rot,
            ha="left",
            va="top",
        )

# ── Commodity horizontal lines ───────────────────────────────────────
for name, tau, color, dtype in COMMODITIES:
    ls = "-" if dtype == "exp" else "--"
    ax.axhline(tau, color=color, ls=ls, lw=0.6, alpha=0.45, zorder=2)
    ax.text(
        4500,
        tau * 1.10,
        name,
        fontsize=5.5,
        color=color,
        ha="right",
        va="bottom",
    )

# ── Key data points from the theory ──────────────────────────────────
# EMJ worked example (§7.3)
t_ej = transfers["Earth\u2013Jupiter"]

# Jupiter + hardware: feasible (DR = 0.418, η = 0.914)
ax.plot(
    t_ej,
    5000,
    "s",
    color="#4477AA",
    markersize=5,
    markeredgecolor="#222222",
    markeredgewidth=0.5,
    zorder=8,
    clip_on=False,
)
ax.annotate(
    r"HW: DR\,=\,0.42",
    xy=(t_ej, 5000),
    xytext=(t_ej * 0.55, 4200),
    fontsize=5,
    color="#4477AA",
    ha="center",
    arrowprops=dict(arrowstyle="-", color="#4477AA", lw=0.4),
)

# Jupiter + LH2: infeasible pipeline (DR = 0.027)
ax.plot(
    t_ej,
    180,
    "o",
    color="#CC3311",
    markersize=5.5,
    markeredgecolor="#222222",
    markeredgewidth=0.5,
    zorder=8,
)
ax.annotate(
    r"LH$_2$: DR\,=\,0.027",
    xy=(t_ej, 180),
    xytext=(t_ej * 1.55, 100),
    fontsize=5,
    color="#CC3311",
    ha="center",
    arrowprops=dict(arrowstyle="-", color="#CC3311", lw=0.4),
)

# Mars + LH2: marginal (τ_half/ln2 ≈ 260d, T_Hohmann ≈ 259d — right at boundary)
t_em = transfers["Earth\u2013Mars"]
ax.plot(
    t_em,
    180,
    "D",
    color="#EE7733",
    markersize=5,
    markeredgecolor="#222222",
    markeredgewidth=0.5,
    zorder=8,
)
ax.annotate(
    r"LH$_2$ to Mars: $\xi \approx 1.0$",
    xy=(t_em, 180),
    xytext=(t_em * 0.30, 220),
    fontsize=5,
    color="#EE7733",
    ha="center",
    arrowprops=dict(arrowstyle="-", color="#EE7733", lw=0.4),
)

# ── One-tau marker on the EMJ line ────────────────────────────────────
# The one-tau crossing: where τ_half = T × ln(2) on the Jupiter line
tau_crit_ej = t_ej * LN2
ax.plot(
    t_ej,
    tau_crit_ej,
    "x",
    color="#CC3311",
    markersize=6,
    markeredgewidth=1.2,
    zorder=9,
)
ax.annotate(
    r"$\xi = 1$ threshold",
    xy=(t_ej, tau_crit_ej),
    xytext=(t_ej * 0.50, 850),
    fontsize=5,
    color="#CC3311",
    ha="center",
    arrowprops=dict(arrowstyle="-", color="#CC3311", lw=0.4),
)

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 5000)
ax.set_ylim(0.5, 6000)
ax.set_xlabel("Minimum transit time $T_\\mathrm{Hohmann}$ (days)")
ax.set_ylabel("Commodity half-life $\\tau_{1/2}$ (days)")

# Secondary x-axis in years
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax.get_xlim())
year_ticks = [1, 7, 30, 365.25, 365.25 * 3, 365.25 * 10]
year_labels = ["1\\,d", "1\\,wk", "1\\,mo", "1\\,yr", "3\\,yr", "10\\,yr"]
ax2.set_xticks(year_ticks)
ax2.set_xticklabels(year_labels, fontsize=6)
ax2.tick_params(axis="x", which="minor", bottom=False, top=False)
ax2.minorticks_off()

# Legend — centered, side by side
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    fontsize=6,
    framealpha=0.85,
)

save_fig(fig, "commodity_phase_diagram")
plt.close(fig)
print("\nDone.")
