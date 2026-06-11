"""plot_fig4_phase_space.py — Phase space of the classification theorem.

Single-panel scatter: E[H] vs ln(Phi) for all source-dest pairs at
reference p_eff ~ 0.1.  Orbital (trap) points slope downward; social
(cluster) points slope upward.  The sign of the slope IS gamma.

Data sources:
  runs/phi_decompose_results.json       — 8 orbital targets (230k configs)
  runs/crawdad_contacts.Exp{1,2,3,6}_results.json — 4 CRAWDAD traces

Writes:
  figures/fig4_phase_space.{pdf,png}
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

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ── Colour scheme ────────────────────────────────────────────────────
TRAP_RED = "#c0392b"
TRAP_FILL = "#e74c3c"
CLUSTER_BLUE = "#2c3e50"
CLUSTER_FILL = "#3498db"


def _load_orbital(p_ref=0.1184):
    """Load orbital data at reference p_ref, return (E_H, ln_Phi) arrays."""
    path = os.path.join(RUNS, "phi_decompose_results.json")
    d = json.load(open(path))
    ehs, lnphis, targets = [], [], []
    for r in d["results"]:
        if r["p_ref"] != p_ref:
            continue
        phi = r.get("phi_normal", 0.0)
        eh = r.get("E_H", 0.0)
        if not (phi > 0 and eh > 0 and np.isfinite(phi)):
            continue
        ehs.append(eh)
        lnphis.append(np.log(phi))
        targets.append(r["target"])
    return np.array(ehs), np.array(lnphis), targets


def _load_social(p_eff=0.1):
    """Load CRAWDAD data at reference p_eff, return (E_H, ln_Phi) arrays."""
    traces = ["Exp1", "Exp2", "Exp3", "Exp6"]
    ehs, lnphis, names = [], [], []
    for trace in traces:
        path = os.path.join(RUNS, f"crawdad_contacts.{trace}_results.json")
        d = json.load(open(path))
        for r in d["results"]:
            if r["p_eff"] != p_eff:
                continue
            phi = r.get("phi_normal", 0.0)
            eh = r.get("E_H", 0.0)
            if not (phi > 0 and eh > 0 and np.isfinite(phi)):
                continue
            ehs.append(eh)
            lnphis.append(np.log(phi))
            names.append(trace)
    return np.array(ehs), np.array(lnphis), names


def plot():
    eh_trap, lnphi_trap, _ = _load_orbital()
    eh_clust, lnphi_clust, _ = _load_social()

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))

    # ── Individual points (high transparency) ────────────────────────
    ax.scatter(
        eh_trap,
        lnphi_trap,
        c=TRAP_FILL,
        s=8,
        alpha=0.06,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        eh_clust,
        lnphi_clust,
        c=CLUSTER_FILL,
        s=8,
        alpha=0.08,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # ── OLS regression lines (restricted to data range) ──────────────
    # Trap
    mask_t = np.isfinite(eh_trap) & np.isfinite(lnphi_trap)
    c_trap = np.polyfit(eh_trap[mask_t], lnphi_trap[mask_t], 1)
    x_trap = np.linspace(eh_trap.min() - 0.1, eh_trap.max() + 0.1, 100)
    ax.plot(
        x_trap,
        np.polyval(c_trap, x_trap),
        color=TRAP_RED,
        linewidth=2.8,
        linestyle="-",
        label=rf"Orbital — trap ($\gamma = {c_trap[0]:+.2f}$)",
        zorder=4,
    )

    # Cluster
    mask_c = np.isfinite(eh_clust) & np.isfinite(lnphi_clust)
    c_clust = np.polyfit(eh_clust[mask_c], lnphi_clust[mask_c], 1)
    x_clust = np.linspace(eh_clust.min() - 0.1, eh_clust.max() + 0.1, 100)
    ax.plot(
        x_clust,
        np.polyval(c_clust, x_clust),
        color=CLUSTER_BLUE,
        linewidth=2.8,
        linestyle="-",
        label=rf"Social — cluster ($\gamma = {c_clust[0]:+.2f}$)",
        zorder=4,
    )

    # ── Annotations ──────────────────────────────────────────────────
    # Trap label — positioned near middle of trap data range
    trap_xmid = np.median(eh_trap)
    ax.annotate(
        r"$\gamma < 0$",
        xy=(trap_xmid + 0.6, np.polyval(c_trap, trap_xmid + 0.6)),
        xytext=(trap_xmid + 1.2, np.polyval(c_trap, trap_xmid) + 2.0),
        fontsize=11,
        color=TRAP_RED,
        fontweight="semibold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=TRAP_RED, lw=1.5),
        zorder=5,
    )

    # Cluster label — positioned near middle of cluster data range
    clust_xmid = np.median(eh_clust)
    ax.annotate(
        r"$\gamma > 0$",
        xy=(clust_xmid + 0.8, np.polyval(c_clust, clust_xmid + 0.8)),
        xytext=(clust_xmid + 1.4, np.polyval(c_clust, clust_xmid) - 2.5),
        fontsize=11,
        color=CLUSTER_BLUE,
        fontweight="semibold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=CLUSTER_BLUE, lw=1.5),
        zorder=5,
    )

    # ── Count annotation ─────────────────────────────────────────────
    n_total = len(eh_trap) + len(eh_clust)
    ax.text(
        0.98,
        0.02,
        f"$n = {n_total:,}$ pairs at $p_{{\\mathrm{{eff}}}} \\approx 0.1$",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#777777",
        ha="right",
        va="bottom",
    )

    # ── Axes and labels ──────────────────────────────────────────────
    ax.set_xlabel(r"Expected hop count $E[H]$", fontsize=12)
    ax.set_ylabel(r"$\ln\,\Phi$", fontsize=13)
    ax.set_xlim(0.8, 7.2)
    ax.set_ylim(-4.5, 15.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # ── Save ─────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"fig4_phase_space.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  Wrote {path}")
    plt.close(fig)

    # Print stats
    print(f"\n  Trap:    {len(eh_trap):,} points, slope = {c_trap[0]:+.3f}")
    print(f"  Cluster: {len(eh_clust):,} points, slope = {c_clust[0]:+.3f}")


if __name__ == "__main__":
    plot()
