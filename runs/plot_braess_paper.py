"""plot_braess_paper.py — Figure for the Braess Paradox paper.

1. DR vs synodic epoch for n=2,4,8,12: non-monotonic architecture dependence

Reads:  runs/helio_multi_arch_results.json
Writes: figures/fig_braess_dr_vs_epoch.pdf
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).parent
_FIG = _HERE.parent / "figures"
_FIG.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)

COLORS = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c", 12: "#d62728"}
MARKERS = {2: "o", 4: "s", 8: "^", 12: "D"}


def plot_braess_dr_vs_epoch():
    with open(_HERE / "helio_multi_arch_results.json") as f:
        data = json.load(f)
    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[2, 1], sharex=True)

    for n in [2, 4, 8, 12]:
        subset = sorted(
            [r for r in results if r["n_orb"] == n],
            key=lambda r: r["epoch_day"],
        )
        epochs = np.array([r["epoch_day"] for r in subset])
        dr = np.array([r["DR"] for r in subset])
        dist = np.array([r["dist_au"] for r in subset])

        ax1.plot(
            epochs,
            dr,
            "-",
            marker=MARKERS[n],
            color=COLORS[n],
            markersize=3,
            linewidth=1.2,
            label=f"n = {n}",
            markevery=5,
            alpha=0.85,
        )

    ax1.set_ylabel("DR")
    ax1.set_title("Braess paradox: DR over Mars synodic cycle by architecture")
    ax1.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 0.85)

    # Lower panel: distance
    sample = sorted(
        [r for r in results if r["n_orb"] == 2],
        key=lambda r: r["epoch_day"],
    )
    epochs = np.array([r["epoch_day"] for r in sample])
    dist = np.array([r["dist_au"] for r in sample])

    ax2.fill_between(epochs, dist, alpha=0.3, color="#9467bd")
    ax2.plot(epochs, dist, "-", color="#9467bd", lw=1.5)
    ax2.set_ylabel("Distance (AU)")
    ax2.set_xlabel("Epoch (days since J2000)")
    ax2.grid(True, alpha=0.3)

    # Annotate opposition
    min_idx = np.argmin(dist)
    ax2.annotate(
        "Opposition",
        xy=(epochs[min_idx], dist[min_idx]),
        xytext=(epochs[min_idx] + 60, dist[min_idx] + 0.3),
        arrowprops=dict(arrowstyle="->", color="0.4"),
        fontsize=9,
        color="0.3",
    )

    fig.tight_layout()

    out = _FIG / "fig_braess_dr_vs_epoch.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


def main():
    print()
    print("Generating Braess paper figure...")
    plot_braess_dr_vs_epoch()
    print("Done.")


if __name__ == "__main__":
    main()
