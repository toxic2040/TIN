"""Historical transit-time/half-life model diagram with archived schedule rows.

Reproduces the original diagram and adds:
- Arrows from T_Hohmann (open symbols) to T_actual (filled symbols)
  for EMJ relay chain and Mars direct
- Annotations showing the archived scheduling-interval percentage

The model-reference lines and plotted rows do not establish mission
feasibility, preservation requirements, fleet design, or operational risk.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Figure setup ──────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 9))

# Log-log axes
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 8000)
ax.set_ylim(1, 6000)

# ── Boundary lines ───────────────────────────────────────────
T = np.logspace(0, 4.2, 500)

# Pipeline threshold: τ_half = T * ln(2)
# At this line, half the commodity survives one transit leg
# ξ = τ_half / T = ln(2) ≈ 0.693
tau_pipeline = T * np.log(2)

# Survival threshold: τ_shelf = T  (ξ = 1)
# Below this, the commodity doesn't survive a single transit
tau_survival = T * 1.0

ax.plot(
    T,
    tau_pipeline,
    "k-",
    linewidth=2.0,
    zorder=3,
    label=r"Archived exponential-decay reference: $\tau_{1/2} = T \ln 2$",
)
ax.plot(
    T,
    tau_survival,
    "k--",
    linewidth=1.5,
    zorder=3,
    label=r"Archived step-decay reference: $\tau_{\mathrm{shelf}} = T$",
)

# ── Historical model-reference regions ──────────────────────
# Above the exponential-decay reference line
ax.fill_between(T, tau_pipeline, 5000, alpha=0.08, color="green", zorder=1)
ax.text(
    4,
    2500,
    "above both\nmodel lines",
    fontsize=16,
    color="#2d7d2d",
    fontweight="bold",
    ha="left",
    va="center",
    alpha=0.7,
)

# Below the step-decay reference line
ax.fill_between(T, 1, tau_survival, alpha=0.08, color="red", zorder=1)
ax.text(
    2000,
    8,
    "below both\nmodel lines",
    fontsize=16,
    color="#8b2020",
    fontweight="bold",
    ha="center",
    va="center",
    alpha=0.7,
)

# Wedge (between lines)
ax.fill_between(T, tau_survival, tau_pipeline, alpha=0.06, color="orange", zorder=1)
ax.text(
    30,
    12,
    "between\nmodel lines",
    fontsize=11,
    color="#b8700a",
    fontweight="bold",
    ha="center",
    va="center",
    alpha=0.8,
    fontstyle="italic",
)

# ── Destination markers (Hohmann floor) ──────────────────────
destinations = {
    "Moon": {"T": 5.0, "color": "#666666"},
    "Venus": {"T": 146, "color": "#cc8800"},
    "Mars": {"T": 259, "color": "#cc3300"},
    "Jupiter": {"T": 997, "color": "#886644"},
    "Saturn": {"T": 2190, "color": "#aa9966"},
}

# Draw vertical destination lines (faint)
for name, d in destinations.items():
    ax.axvline(d["T"], color=d["color"], alpha=0.15, linewidth=1, zorder=1)
    # Label at top
    y_label = 3800 if name != "Saturn" else 3800
    ax.text(
        d["T"],
        y_label,
        name,
        fontsize=10,
        color=d["color"],
        ha="center",
        va="top",
        fontweight="bold",
        alpha=0.6,
        rotation=0,
    )

# ── Commodity bands (horizontal) ─────────────────────────────
commodities = {
    "Biological\nsamples": {"tau": 1, "color": "#cc0066"},
    "mRNA\nvaccines": {"tau": 3, "color": "#9900cc"},
    "Standard\nvaccines": {"tau": 30, "color": "#3366cc"},
    "LH₂": {"tau": 180, "color": "#0088cc"},
    "LOX": {"tau": 500, "color": "#555555"},
    "Food": {"tau": 730, "color": "#338833"},
    "CH₄": {"tau": 3600, "color": "#dd6600"},
}

# Custom y-offsets for labels to avoid overlap
label_offsets = {
    "LOX": 1.25,  # push up, away from Food
    "Food": 0.75,  # push down, away from LOX
    "Biological\nsamples": 1.8,  # push up, clear both lines
    "mRNA\nvaccines": 1.5,  # push up slightly
    "CH₄": 1.15,
}

for name, c in commodities.items():
    ax.axhline(c["tau"], color=c["color"], alpha=0.2, linewidth=1, linestyle=":", zorder=1)
    y_mult = label_offsets.get(name, 1.12)
    ax.text(
        1.3,
        c["tau"] * y_mult,
        name,
        fontsize=8.5,
        color=c["color"],
        ha="left",
        va="bottom",
        fontweight="bold",
        alpha=0.7,
    )

# ── Line labels (horizontal, just below Standard vaccines height) ──
# Survival threshold — left of dashed line
ax.text(
    10,
    22,
    r"$\xi = 1$ model reference",
    fontsize=9,
    rotation=0,
    color="black",
    alpha=0.5,
    ha="right",
    va="center",
)
# Pipeline — sits on the solid line, offset right
ax.text(
    55,
    22,
    r"Archived model line: $\tau_{1/2} = T \ln 2$",
    fontsize=9,
    rotation=0,
    color="black",
    alpha=0.5,
    ha="left",
    va="center",
)

# ── Key intersection points ──────────────────────────────────
# Mars × LH2 at Hohmann (the ξ ≈ 1.0 point)
ax.plot(
    259,
    180,
    "o",
    markersize=11,
    markerfacecolor="none",
    markeredgecolor="#0066aa",
    markeredgewidth=2.5,
    zorder=5,
)

# ── Mars archived scheduling-model row (22 March 2026) ───────
# From run_mars_scheduling_overhead.py: 4 scenarios, 10yr, 5 synodic cycles
mars_T_hohmann = 259
mars_T_scheduled = 611  # measured mean, 30d departure window, +136% overhead

# Measured scheduled transit (filled diamond)
ax.plot(
    mars_T_scheduled,
    180,
    "D",
    markersize=10,
    markerfacecolor="#cc5500",
    markeredgecolor="#993300",
    markeredgewidth=1.5,
    zorder=6,
)
ax.annotate(
    "",
    xy=(mars_T_scheduled, 180),
    xytext=(mars_T_hohmann, 180),
    arrowprops=dict(arrowstyle="->", color="#cc5500", lw=2.5),
)

# Mars annotations
ax.annotate(
    "LH₂ model row\n(Hohmann floor, ξ ≈ 1.0)",
    xy=(259, 180),
    xytext=(40, 350),
    fontsize=9,
    color="#0066aa",
    arrowprops=dict(arrowstyle="->", color="#0066aa", lw=1.2, connectionstyle="arc3,rad=0.2"),
    fontweight="bold",
    ha="left",
)

ax.annotate(
    "+136% archived schedule interval\nξ = 2.35, modeled DR = 9.5%\nno mission verdict",
    xy=(mars_T_scheduled, 200),
    xytext=(80, 900),
    fontsize=9,
    color="#cc5500",
    arrowprops=dict(arrowstyle="->", color="#cc5500", lw=1.2, connectionstyle="arc3,rad=0.15"),
    fontweight="bold",
    ha="center",
)

# ── LOX archived Mars scheduling-model row ──────────────────
# LOX Hohmann (open circle)
ax.plot(
    mars_T_hohmann,
    500,
    "o",
    markersize=9,
    markerfacecolor="none",
    markeredgecolor="#555555",
    markeredgewidth=2,
    zorder=5,
)
# LOX scheduled (filled diamond)
ax.plot(
    mars_T_scheduled,
    500,
    "D",
    markersize=9,
    markerfacecolor="#555555",
    markeredgecolor="#333333",
    markeredgewidth=1.5,
    zorder=6,
)
ax.annotate(
    "",
    xy=(mars_T_scheduled, 500),
    xytext=(mars_T_hohmann, 500),
    arrowprops=dict(arrowstyle="->", color="#555555", lw=2.0),
)
ax.annotate(
    "LOX model row\nξ = 0.85, modeled DR = 43%\nno ZBO requirement inferred",
    xy=(mars_T_scheduled, 500),
    xytext=(2500, 500),
    fontsize=9,
    color="#555555",
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0, connectionstyle="arc3,rad=-0.1"),
    fontweight="bold",
    ha="left",
)

# ── CH4 archived Mars scheduling-model row ──────────────────
ax.plot(
    mars_T_scheduled,
    3600,
    "D",
    markersize=8,
    markerfacecolor="#dd6600",
    markeredgecolor="#aa4400",
    markeredgewidth=1.5,
    zorder=6,
)
ax.annotate(
    "CH₄ model row: ξ = 0.12, modeled DR = 89%",
    xy=(mars_T_scheduled, 3600),
    xytext=(30, 4500),
    fontsize=9,
    color="#dd6600",
    arrowprops=dict(arrowstyle="->", color="#dd6600", lw=1.0, connectionstyle="arc3,rad=0.2"),
    fontweight="bold",
    ha="left",
)

# Archived interval difference between Hohmann and scheduled inputs
ax.annotate(
    "",
    xy=(mars_T_scheduled, 115),
    xytext=(mars_T_hohmann, 115),
    arrowprops=dict(arrowstyle="<->", color="#884400", lw=1.5),
)
ax.text(
    (mars_T_hohmann * mars_T_scheduled) ** 0.5,
    82,
    "archived schedule-interval difference",
    fontsize=8,
    color="#884400",
    ha="center",
    va="top",
    fontstyle="italic",
    fontweight="bold",
)

# ── EMJ relay chain: Hohmann floor vs actual ─────────────────
emj_T_hohmann = 1385
emj_T_actual = 1611  # from hull §7.7

# LH2 at EMJ
ax.plot(
    emj_T_hohmann,
    180,
    "o",
    markersize=9,
    markerfacecolor="none",
    markeredgecolor="#cc3300",
    markeredgewidth=2,
    zorder=5,
)
ax.plot(
    emj_T_actual,
    180,
    "o",
    markersize=9,
    markerfacecolor="#cc3300",
    markeredgecolor="#cc3300",
    markeredgewidth=2,
    zorder=5,
)
ax.annotate(
    "",
    xy=(emj_T_actual, 180),
    xytext=(emj_T_hohmann, 180),
    arrowprops=dict(arrowstyle="->", color="#cc3300", lw=2.0),
)

# HW (durable cargo) archived EMJ model row
hw_tau = 2000
ax.plot(
    emj_T_hohmann,
    hw_tau,
    "s",
    markersize=8,
    markerfacecolor="none",
    markeredgecolor="#338833",
    markeredgewidth=2,
    zorder=5,
)
ax.plot(
    emj_T_actual,
    hw_tau,
    "s",
    markersize=8,
    markerfacecolor="#338833",
    markeredgecolor="#338833",
    markeredgewidth=2,
    zorder=5,
)
ax.annotate(
    "",
    xy=(emj_T_actual, hw_tau),
    xytext=(emj_T_hohmann, hw_tau),
    arrowprops=dict(arrowstyle="->", color="#338833", lw=2.0),
)

# EMJ labels — long leader lines to far corners
ax.annotate(
    "EMJ model row: HW\nmodeled DR = 0.42",
    xy=(emj_T_actual, hw_tau),
    xytext=(4500, 1500),
    fontsize=9,
    color="#338833",
    arrowprops=dict(arrowstyle="->", color="#338833", lw=1, connectionstyle="arc3,rad=-0.1"),
    fontweight="bold",
    ha="center",
)

ax.annotate(
    "EMJ model row: LH₂\nmodeled DR = 0.027",
    xy=(emj_T_actual, 180),
    xytext=(5500, 30),
    fontsize=9,
    color="#cc3300",
    arrowprops=dict(arrowstyle="->", color="#cc3300", lw=1, connectionstyle="arc3,rad=0.15"),
    fontweight="bold",
    ha="center",
)

# ── Secondary time axis (top) ────────────────────────────────
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax.get_xlim())
time_labels = {1: "1 d", 7: "1 wk", 30: "1 mo", 365: "1 yr", 1095: "3 yr", 3650: "10 yr"}
ax2.set_xticks(list(time_labels.keys()))
ax2.set_xticklabels(list(time_labels.values()), fontsize=9)
ax2.set_xlabel("")

# ── Legend for arrow meaning ─────────────────────────────────
ax.text(
    60,
    2800,
    "○ → ●  Hohmann floor → archived $T_{actual}$\n◇  archived scheduling-model interval",
    fontsize=8.5,
    color="#555555",
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
)

# ── Axis labels ──────────────────────────────────────────────
ax.set_xlabel(
    r"Two-body Hohmann transfer-time floor $T_{\mathrm{Hohmann}}$ (days)",
    fontsize=13,
    labelpad=8,
)
ax.set_ylabel(r"Commodity half-life $\tau_{1/2}$ (days)", fontsize=13, labelpad=8)

# Title
ax.set_title(
    "Historical transit-time/half-life model diagnostic\n"
    r"Archived Hohmann-floor and $T_{\mathrm{actual}}$ inputs"
    "\nNot mission feasibility, preservation, or design guidance",
    fontsize=12,
    pad=15,
    linespacing=1.4,
)

# Grid
ax.grid(True, which="both", alpha=0.15, linewidth=0.5)
ax.tick_params(labelsize=10)

plt.tight_layout()
_FIG_DIR = Path(__file__).parent.parent / "figures"
_FIG_DIR.mkdir(exist_ok=True)
plt.savefig(str(_FIG_DIR / "commodity_phase_v2.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(str(_FIG_DIR / "commodity_phase_v2.png"), dpi=200, bbox_inches="tight")

# Also save to site directory for percolate.space
_SITE_DIR = Path(__file__).parent.parent / "site"
if _SITE_DIR.exists():
    plt.savefig(str(_SITE_DIR / "commodity_phase_diagram.png"), dpi=200, bbox_inches="tight")
    print(f"Site:   {_SITE_DIR / 'commodity_phase_diagram.png'}")

print(f"Done — saved to {_FIG_DIR / 'commodity_phase_v2.pdf'}")
print(f"       and {_FIG_DIR / 'commodity_phase_v2.png'}")
