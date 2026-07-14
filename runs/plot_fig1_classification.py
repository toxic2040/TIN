"""Historical Figure 1 reproducer for classification_theorem.tex.

This script preserves the published composite table and figure. It is not a
current classifier: four orbital values below are hardcoded with unavailable
source rows, and the published cross-domain gap is retired.

Two-panel historical figure:
  Left:  raw slope γ = d[ln(Phi)]/d[E[H]] vs p_eff for cluster-class traces
  Right: historical composite raw-slope bar chart for all 8 orbital bodies

The left panel uses trace values at explicit p_eff, including p_eff=0.1. The
right panel reproduces the historical orbital composite; a common probability
convention for those eight values is not established. Its values are from
runs/helio_gamma_missing_bodies_results.json (4 new bodies) and from
gen_classification_figures.py HISTORICAL_ORBITAL_GAMMA dict (4 existing bodies).

Data sources:
  runs/crawdad_cross_trace_analysis.json        — CRAWDAD gamma_normal by p_eff
  runs/helio_gamma_missing_bodies_results.json  — Venus, Europa, Jupiter, Saturn raw slopes
  TRAP_GAMMA below                              — historical composite table values
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(RUNS), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Historical heliocentric composite (raw slopes) ───────────────────────────
# Sources:
#   Mercury, Mars, Ceres, Titan: HISTORICAL_ORBITAL_GAMMA dict in
#   gen_classification_figures.py
#   Venus, Europa, Jupiter, Saturn: runs/helio_gamma_missing_bodies_results.json
# Historical table values; this composite is not a validated common-scale set.
# Sorted most-negative → least-negative (top to bottom in bar chart).
TRAP_GAMMA = {
    "Ceres": -1.20,
    "Jupiter": -1.14,
    "Mercury": -1.01,
    "Saturn": -0.67,
    "Europa": -0.54,
    "Mars": -0.40,
    "Venus": -0.21,
    "Titan": -0.10,
}

# ── Colour scheme ─────────────────────────────────────────────────────────────
CLUSTER_BLUES = ["#08306b", "#2171b5", "#4292c6", "#9ecae1"]  # dark → light
TRAP_RED = "#c0392b"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load_crawdad():
    path = os.path.join(RUNS, "crawdad_cross_trace_analysis.json")
    with open(path) as f:
        return json.load(f)


def _crawdad_raw_slopes(gc):
    """Convert gamma_normal to raw slope: raw = gamma_normal * (-ln(p_eff))."""
    import math

    raw = {}
    for trace in ["Exp1", "Exp2", "Exp3", "Exp6"]:
        raw[trace] = {}
        for p_str, gn in gc[trace]["gamma_by_p"].items():
            p = float(p_str)
            raw[trace][p] = gn * (-math.log(p))
    return raw


def plot():
    craw = _load_crawdad()
    gc = craw["gamma_classification"]["gamma_normal"]
    # Convert to raw slopes for consistent scale with trap panel
    raw_slopes = _crawdad_raw_slopes(gc)

    # ── Layout: 2 panels, left wider than right ───────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(10, 4.8),
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    fig.suptitle(
        "Historical composite figure — global classifier retired",
        fontsize=12,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.86, wspace=0.35)

    # ══ LEFT panel — Cluster class (raw slopes) ═══════════════════════════
    traces = ["Exp1", "Exp2", "Exp3", "Exp6"]
    markers = ["o", "s", "D", "^"]
    labels = [
        r"Exp1 ($n{=}9$)",
        r"Exp2 ($n{=}12$)",
        r"Exp3 ($n{=}41$)",
        r"Exp6 ($n{=}98$)",
    ]

    # Published arithmetic: the supported Exp2 p_eff=0.1 value was paired with
    # the historical composite Titan value. No common p_eff is claimed.
    historical_trap_max = max(TRAP_GAMMA.values())  # −0.10 (Titan)
    cluster_min_p01 = min(raw_slopes[t][0.1] for t in traces)  # +1.85 (Exp2)

    # γ=0 class boundary
    ax_l.axhline(0, color="#bbbbbb", linewidth=0.85, linestyle="--", zorder=1)

    for i, (trace, marker, label) in enumerate(zip(traces, markers, labels)):
        rdata = raw_slopes[trace]
        ps = sorted(rdata.keys())
        ys = [rdata[p] for p in ps]
        color = CLUSTER_BLUES[i]
        ax_l.plot(
            ps,
            ys,
            marker=marker,
            linestyle="-",
            color=color,
            markersize=5.5,
            linewidth=1.4,
            label=label,
            zorder=3,
        )

    ax_l.set_xscale("log")
    ax_l.set_xlim(0.015, 0.65)
    ax_l.set_ylim(-0.5, 4.2)
    ax_l.set_xlabel(r"$p_{\mathrm{eff}}$", fontsize=11)
    ax_l.set_ylabel(r"$\gamma$ (raw slope)", fontsize=11)
    ax_l.set_title("Tested Bluetooth traces", fontsize=11, fontweight="semibold")
    ax_l.legend(
        loc="upper right",
        framealpha=0.88,
        handlelength=1.8,
        borderpad=0.5,
    )
    ax_l.grid(True, alpha=0.25)
    # Preserve the published arithmetic while labeling its retired status.
    ax_l.annotate(
        r"published $\Delta\gamma = 1.95$" + "\n(retired)",
        xy=(0.1, cluster_min_p01),
        xytext=(0.18, (cluster_min_p01 + historical_trap_max) / 2),
        fontsize=8,
        color="#888888",
        style="italic",
        ha="left",
        va="center",
    )

    # ══ RIGHT panel — Trap class ══════════════════════════════════════════
    bodies = list(TRAP_GAMMA.keys())  # already sorted most → least negative
    gammas = [TRAP_GAMMA[b] for b in bodies]
    y_pos = np.arange(len(bodies))

    ax_r.barh(
        y_pos,
        gammas,
        color=TRAP_RED,
        alpha=0.80,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_r.axvline(0, color="#555555", linewidth=0.8, zorder=3)

    # Value labels just outside the right edge of the zero axis (white space)
    for i, (body, g) in enumerate(zip(bodies, gammas)):
        ax_r.text(
            0.03,
            i,
            f"{g:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
        )

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(bodies, fontsize=9.5)
    ax_r.invert_yaxis()
    ax_r.set_xlabel(r"$\gamma$", fontsize=11)
    ax_r.set_title("Historical composite orbital table", fontsize=11, fontweight="semibold")
    ax_r.set_xlim(-1.45, 0.35)
    ax_r.grid(True, axis="x", alpha=0.25)
    ax_r.tick_params(left=False)
    ax_r.spines[["top", "right", "left"]].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"fig1_gamma_vs_p.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
