#!/usr/bin/env python3
"""Historical synthetic one-tau reserve-recurrence figure.

Replays the illustrative hub recurrence used in an archived whitepaper draft:

    R_{n+1} = R_n * D_store  +  delivery  -  consumption

where delivery = dispatch * 2^{-T_transit / tau_half}.  The displayed
zero-crossing follows from the fixed dispatch, consumption, and storage-decay
constants below.

The former universal one-tau, sustainability, and mission-design readings are
retired.  This is a synthetic model illustration, not a feasibility result,
reserve requirement, fleet rule, or operational recommendation.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tin_figure_style import apply_style, figsize_double, save_fig

apply_style("pre")

# ── Recurrence parameters ──────────────────────────────────────────
D_STORE = 0.92  # per-epoch decay of stored reserve
DISPATCH = 1.0  # normalised dispatch per epoch
CONSUME = 0.50  # consumption per epoch (dispatch/consume = 2 → one-tau)
R0 = 3.0  # initial reserve (epochs of consumption)
N_EPOCHS = 40  # simulation length

# xi = T_transit / tau_half.  The one-tau threshold is at xi = 1.0
# when dispatch/consume = 2 (exactly half the cargo survives transit).
SCENARIOS = [
    (0.2, "Cislunar\n(54\\,d)", "#4477AA"),
    (0.5, "Earth--Mars\nw/ ZBO (90\\,d)", "#33BBEE"),
    (0.8, "Earth--Mars\nstandard (144\\,d)", "#009988"),
    (1.0, "Archived one-tau\nreference", "#333333"),
    (1.5, "Long coast\n(270\\,d)", "#EE7733"),
    (2.5, "Earth--Jupiter\nrelay (450\\,d)", "#CC3311"),
    (4.0, "Mars--Jupiter\n(730\\,d)", "#881100"),
]


def run_recurrence(xi, n_epochs):
    """Run the reserve recurrence for a given xi = T/tau_half."""
    delivery = DISPATCH * 2.0 ** (-xi)
    R = np.empty(n_epochs + 1)
    R[0] = R0
    for n in range(n_epochs):
        R[n + 1] = max(0.0, R[n] * D_STORE + delivery - CONSUME)
    return R


# ── Figure: two panels ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double("pre", height_ratio=0.48))
fig.subplots_adjust(wspace=0.32, top=0.82)
fig.suptitle(
    "Historical Synthetic Reserve Recurrence — One-Tau Design Claim Retired",
    fontsize=9,
    y=0.98,
)

epochs = np.arange(N_EPOCHS + 1)

# ═══════════════════════════════════════════════════════════════════
# LEFT PANEL: reserve trajectories R_n vs epoch
# ═══════════════════════════════════════════════════════════════════
for xi, label, color in SCENARIOS:
    R = run_recurrence(xi, N_EPOCHS)
    ls = "--" if xi == 1.0 else "-"
    lw = 1.5 if xi == 1.0 else 1.1
    ax1.plot(epochs, R, ls, color=color, lw=lw, alpha=0.85, label="$\\xi=%.1f$" % xi)

    # Label at right edge — stagger depleted curves to avoid overlap
    y_end = R[-1]
    if y_end < 0.15 and xi > 1.0:
        y_end = 0.15 + (xi - 1.0) * 0.35
    ax1.text(N_EPOCHS + 0.8, y_end, label, fontsize=4.3, color=color, va="center", linespacing=1.0)

# Descriptive trajectory annotations under the fixed recurrence inputs.
ax1.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
ax1.text(
    20,
    R0 * 0.92,
    "modeled reserve remains positive",
    fontsize=5.5,
    color="#009988",
    ha="center",
    alpha=0.5,
)
ax1.text(
    20,
    R0 * 0.05,
    "modeled reserve clips at zero",
    fontsize=5.5,
    color="#CC3311",
    ha="center",
    alpha=0.5,
)

ax1.set_xlabel("Resupply epoch $n$")
ax1.set_ylabel("Hub reserve $R_n$ (normalised)")
ax1.set_xlim(0, N_EPOCHS + 14)
ax1.set_ylim(-0.3, R0 * 1.65)

# ═══════════════════════════════════════════════════════════════════
# RIGHT PANEL: steady-state reserve vs xi
# ═══════════════════════════════════════════════════════════════════
xi_range = np.linspace(0.01, 5.0, 500)
delivery = DISPATCH * 2.0 ** (-xi_range)
# Steady state: R_ss = (delivery - consume) / (1 - D_store)
R_ss = (delivery - CONSUME) / (1.0 - D_STORE)
R_ss_clipped = np.maximum(R_ss, 0.0)

# Archived zero-crossing reference for dispatch/consume = 2.
xi_crit = 1.0
ax2.fill_between(
    xi_range[xi_range <= xi_crit], 0, R_ss_clipped[xi_range <= xi_crit], alpha=0.08, color="#009988"
)
ax2.fill_between(
    xi_range[xi_range >= xi_crit],
    0,
    np.zeros(np.sum(xi_range >= xi_crit)),
    alpha=0.04,
    color="#CC3311",
)

ax2.plot(xi_range, R_ss_clipped, "-", color="#333333", lw=1.5)

# Historical reference line under the stated constants.
ax2.axvline(xi_crit, color="#CC3311", ls="--", lw=0.8, alpha=0.6)
ax2.text(
    xi_crit + 0.08,
    R_ss_clipped.max() * 0.75,
    "$\\xi = 1$ archived reference\n$T = \\tau_{1/2}$",
    fontsize=6.5,
    color="#CC3311",
    va="top",
)

# EMJ scenario markers
emj_scenarios = [
    (0.2, "Cislunar", "#4477AA"),
    (0.5, "E--M (ZBO)", "#33BBEE"),
    (2.5, "E--J relay", "#CC3311"),
    (4.0, "M--J coast", "#881100"),
]
for xi_s, lab, col in emj_scenarios:
    y_s = max(R_ss_clipped[np.argmin(np.abs(xi_range - xi_s))], 0)
    ax2.plot(xi_s, y_s, "o", color=col, markersize=4, zorder=8)
    dy = 0.35 if xi_s < xi_crit else 0.20
    ax2.text(xi_s, y_s + dy, lab, fontsize=5, color=col, ha="center", va="bottom")

# Descriptive model-region labels; no operational verdict is implied.
ax2.text(
    0.5,
    -0.35,
    "positive modeled equilibrium",
    fontsize=6.5,
    color="#009988",
    ha="center",
    alpha=0.5,
    fontstyle="italic",
)
ax2.text(
    3.0,
    -0.35,
    "zero-clipped modeled equilibrium",
    fontsize=6.5,
    color="#CC3311",
    ha="center",
    alpha=0.5,
    fontstyle="italic",
)

ax2.set_xlabel("Exposure ratio $\\xi = T_{\\mathrm{transit}} \\,/\\, \\tau_{1/2}$")
ax2.set_ylabel("Steady-state reserve $R_{\\mathrm{ss}}$")
ax2.set_xlim(0, 5.0)
y_top = R_ss_clipped.max() * 1.15
ax2.set_ylim(-0.65, y_top)

# ── Save ───────────────────────────────────────────────────────────
save_fig(fig, "itn_one_tau_rule")
plt.close(fig)
print("Wrote historical synthetic recurrence figure; design and feasibility claims retired.")
