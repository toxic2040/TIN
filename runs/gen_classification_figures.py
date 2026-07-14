"""Regenerate historical working-paper figures.

The global classifier is retired. The four orbital series below are hardcoded
historical values with unavailable source rows; they are retained only to
reproduce the released figure surface.

Four figures:
1. Historical γ vs p_eff composite (orbital + social source groups)
2. Historical four-trace γ_normal vs ρ_pair association
3. Historical Mars 4-tier time series (DR, η, S_T)
4. Historical η vs d_AU fit for relay tiers (T3/T4)

Reads:
  runs/crawdad_cross_trace_analysis.json
  runs/mars_architecture_results.json
  (orbital γ hardcoded in the historical figure generator; source rows unavailable)

Writes to figures/ directory.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(RUNS), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_crawdad():
    with open(os.path.join(RUNS, "crawdad_cross_trace_analysis.json")) as f:
        return json.load(f)


def load_mars():
    with open(os.path.join(RUNS, "mars_architecture_results.json")) as f:
        return json.load(f)


# Historical orbital values hardcoded by the original figure generator. Their
# underlying sweep rows and exact probability conventions were not recovered.
HISTORICAL_ORBITAL_GAMMA = {
    "Mercury": {0.02: -0.20, 0.05: -0.60, 0.10: -1.01, 0.20: -1.50, 0.50: -1.71},
    "Mars": {0.02: -0.08, 0.05: -0.25, 0.10: -0.40, 0.20: -0.47, 0.50: -0.52},
    "Ceres": {0.02: -0.30, 0.05: -0.80, 0.10: -1.20, 0.20: -1.59, 0.50: -1.83},
    "Titan": {0.02: -0.01, 0.05: -0.06, 0.10: -0.10, 0.20: -0.15, 0.50: -0.19},
}


def fig1_gamma_vs_p():
    """Reproduce the historical composite gamma panel."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Historical hardcoded orbital group — dashed, red tones
    red_shades = ["#8B0000", "#CC3333", "#E06666", "#FF9999"]
    for i, (name, data) in enumerate(HISTORICAL_ORBITAL_GAMMA.items()):
        ps = sorted(data.keys())
        gs = [data[p] for p in ps]
        ax.plot(ps, gs, "s--", color=red_shades[i], label=name, markersize=5)

    # Tested Bluetooth source group — solid, blue tones
    craw = load_crawdad()
    gc = craw["gamma_classification"]["gamma_normal"]
    blue_shades = ["#000080", "#3366CC", "#4488DD", "#66AAEE"]
    labels = ["Exp1 (n=9)", "Exp2 (n=12)", "Exp3 (n=41)", "Exp6 (n=98)"]
    for i, trace in enumerate(["Exp1", "Exp2", "Exp3", "Exp6"]):
        gdata = gc[trace]["gamma_by_p"]
        ps = [float(p) for p in sorted(gdata.keys(), key=float)]
        gs = [gdata[str(p) if str(p) in gdata else f"{p}"] for p in ps]
        # Handle key format
        ps_clean, gs_clean = [], []
        for k in sorted(gdata.keys(), key=float):
            ps_clean.append(float(k))
            gs_clean.append(gdata[k])
        ax.plot(ps_clean, gs_clean, "o-", color=blue_shades[i], label=labels[i], markersize=5)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Link reliability $p_{\mathrm{eff}}$", fontsize=12)
    ax.set_ylabel(r"$\gamma$", fontsize=14)
    ax.set_title(r"Historical $\gamma$ panel — global classifier retired", fontsize=12)
    ax.set_xscale("log")

    # Custom legend: two groups
    handles_trap = [
        Line2D(
            [0],
            [0],
            color=red_shades[0],
            linestyle="--",
            marker="s",
            markersize=5,
            label="Historical hardcoded orbital values",
        )
    ]
    handles_cluster = [
        Line2D(
            [0],
            [0],
            color=blue_shades[0],
            linestyle="-",
            marker="o",
            markersize=5,
            label="Tested Bluetooth traces",
        )
    ]
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.set_ylim(-2.1, 1.15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_gamma_vs_peff.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def fig2_gamma_vs_density():
    """Reproduce the historical four-trace gamma/density association."""
    craw = load_crawdad()
    traces = ["Exp1", "Exp2", "Exp3", "Exp6"]
    ns = [craw["traces"][t]["n_nodes"] for t in traces]
    rhos = [craw["traces"][t]["rho_pair"] for t in traces]

    gc = craw["gamma_classification"]["gamma_normal"]
    gamma_means = []
    for t in traces:
        vals = list(gc[t]["gamma_by_p"].values())
        gamma_means.append(sum(vals) / len(vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: γ vs n (non-monotonic)
    ax1.plot(ns, gamma_means, "o-", color="#3366CC", markersize=8, linewidth=2)
    for i, t in enumerate(traces):
        ax1.annotate(
            f"{t}\n(n={ns[i]})",
            (ns[i], gamma_means[i]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )
    ax1.set_xlabel("Network size $n$", fontsize=12)
    ax1.set_ylabel(r"$\bar{\gamma}_{\mathrm{normal}}$", fontsize=13)
    ax1.set_title("Observed ordering by $n$", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.75, 1.02)

    # Right: γ vs ρ_pair
    # Sort by rho for line
    order = sorted(range(4), key=lambda i: rhos[i])
    rhos_s = [rhos[i] for i in order]
    gs_s = [gamma_means[i] for i in order]
    ax2.plot(rhos_s, gs_s, "s-", color="#CC3333", markersize=8, linewidth=2)
    for i, t in enumerate(traces):
        ax2.annotate(
            f"{t}\n(n={ns[i]})",
            (rhos[i], gamma_means[i]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )
    ax2.set_xlabel(
        r"Per-pair contact density $\rho_{\mathrm{pair}}$ (contacts/pair/hr)", fontsize=11
    )
    ax2.set_ylabel(r"$\bar{\gamma}_{\mathrm{normal}}$", fontsize=13)
    ax2.set_title(r"Observed association with $\rho_{\mathrm{pair}}$", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.75, 1.02)

    fig.suptitle(
        "Historical four-trace association — descriptive, not causal",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_gamma_saturation.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig3_mars_timeseries():
    """Reproduce the historical Mars 4-tier model time series."""
    mars = load_mars()
    results = mars["results"]
    from collections import defaultdict

    tiers = defaultdict(list)
    for r in results:
        tiers[r["tier"]].append(r)

    tier_labels = {1: "T1 (3 polar)", 2: "T2 (+6 orbiters)", 3: "T3 (+L4 relays)", 4: "T4 (full)"}
    colors = {1: "#CC3333", 2: "#E08833", 3: "#3366CC", 4: "#228B22"}

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for metric_idx, (metric, label) in enumerate(
        [("DR", r"$\mathrm{DR}$"), ("eta", r"$\eta$"), ("S_T", r"$S_T$")]
    ):
        ax = axes[metric_idx]
        for tn in [1, 2, 3, 4]:
            rows = sorted(tiers[tn], key=lambda r: r["epoch_day"])
            days = [r["epoch_day"] for r in rows]
            vals = [r[metric] for r in rows]
            ax.plot(
                days,
                vals,
                "-o",
                color=colors[tn],
                label=tier_labels[tn],
                markersize=3,
                linewidth=1.5,
            )
        ax.set_ylabel(label, fontsize=13)
        ax.grid(True, alpha=0.3)
        if metric_idx == 0:
            ax.legend(fontsize=9, ncol=2, loc="upper right")
        # Mark conjunction zone
        ax.axvspan(100, 180, alpha=0.08, color="red")
        if metric_idx == 0:
            ax.annotate("conjunction", xy=(140, 0.02), fontsize=8, color="red", ha="center")
        # Mark opposition
        ax.axvline(480, color="green", linewidth=0.8, linestyle=":", alpha=0.5)

    axes[2].set_xlabel("Epoch day (780-day synodic cycle)", fontsize=12)
    fig.suptitle(
        "Historical Mars 4-Tier Model Output — Not Design Guidance",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_mars_architecture.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def fig4_eta_vs_distance():
    """Reproduce the historical η vs d_AU fit for T3/T4."""
    mars = load_mars()
    results = mars["results"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = {3: "#3366CC", 4: "#228B22"}
    labels = {3: "T3 (L4 relays)", 4: "T4 (full)"}

    all_d, all_ln_eta = [], []
    for r in results:
        if r["tier"] in [3, 4] and r["eta"] > 0:
            d = r["dist_au"]
            ln_eta = np.log(r["eta"])
            ax.scatter(d, ln_eta, color=colors[r["tier"]], s=25, alpha=0.6, edgecolors="none")
            all_d.append(d)
            all_ln_eta.append(ln_eta)

    # Fit line
    all_d = np.array(all_d)
    all_ln_eta = np.array(all_ln_eta)
    coeffs = np.polyfit(all_d, all_ln_eta, 1)
    d_fit = np.linspace(all_d.min(), all_d.max(), 100)
    ax.plot(
        d_fit,
        np.polyval(coeffs, d_fit),
        "k--",
        linewidth=2,
        label=rf"$\ln\eta = {coeffs[1]:.2f} {coeffs[0]:+.2f}\,d_{{\mathrm{{AU}}}}$"
        f"  ($R^2 = {1 - np.sum((all_ln_eta - np.polyval(coeffs, all_d)) ** 2) / np.sum((all_ln_eta - all_ln_eta.mean()) ** 2):.2f}$)",
    )

    # Legend patches
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=colors[3], label=labels[3]),
        Patch(facecolor=colors[4], label=labels[4]),
        ax.get_lines()[-1],
    ]
    ax.legend(handles=handles, fontsize=10)
    ax.set_xlabel(r"Earth--Mars distance $d_{\mathrm{AU}}$ (AU)", fontsize=12)
    ax.set_ylabel(r"$\ln\eta$", fontsize=13)
    ax.set_title("Historical Routing-Efficiency Fit — Archived Model Output", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_eta_vs_distance.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


if __name__ == "__main__":
    print("Generating historical working-paper figures...")
    fig1_gamma_vs_p()
    fig2_gamma_vs_density()
    fig3_mars_timeseries()
    fig4_eta_vs_distance()
    print("Done.")
