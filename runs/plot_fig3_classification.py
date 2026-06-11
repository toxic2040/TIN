"""plot_fig3_classification.py — Figure 3 for classification_theorem.tex.

η vs d_AU for Mars relay Tier 3 and Tier 4 over a 780-day synodic cycle.
Log-linear (distance law) fit per tier.

Improvements over original:
  - T3/T4 points nudged ±0.008 AU so overlapping pairs become legible
  - Fit equation annotations moved above the data cloud (upper area), clear of points
  - Fit lines differentiated by both colour and style (solid / dashed)

Data source: runs/mars_architecture_results.json
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

TIER_STYLE = {
    3: dict(color="#2171b5", marker="o", label="Tier 3", linestyle="-", nudge=-0.008),
    4: dict(color="#c0392b", marker="s", label="Tier 4", linestyle="--", nudge=+0.008),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load():
    with open(os.path.join(RUNS, "mars_architecture_results.json")) as f:
        return json.load(f)["results"]


def plot():
    results = _load()
    fig, ax = plt.subplots(figsize=(7, 5))

    fits = {}
    for tier, sty in TIER_STYLE.items():
        rows = [r for r in results if r["tier"] == tier and r.get("eta", 0) > 0]
        dists = np.array([r["dist_au"] for r in rows])
        lneta = np.array([np.log(r["eta"]) for r in rows])

        # Scatter with small x-nudge to separate overlapping T3/T4 pairs
        ax.scatter(
            dists + sty["nudge"],
            lneta,
            color=sty["color"],
            marker=sty["marker"],
            s=28,
            alpha=0.65,
            edgecolors="none",
            zorder=3,
        )

        # Fit
        c = np.polyfit(dists, lneta, 1)
        r2 = 1 - (np.sum((lneta - np.polyval(c, dists)) ** 2) / np.sum((lneta - lneta.mean()) ** 2))
        fits[tier] = (c, r2)

        d_fit = np.linspace(dists.min(), dists.max(), 120)
        ax.plot(
            d_fit,
            np.polyval(c, d_fit),
            color=sty["color"],
            linestyle=sty["linestyle"],
            linewidth=1.8,
            zorder=4,
        )

    # ── Equation annotations — upper area, above the data cloud ──────────
    # Data is densest near ln(η)≈−0.1 at short range and ≈−1.6 at far end.
    # Upper left is clear of points (short distance, near ln(η)=0).
    eq_y_base = 0.13  # axes fraction from top
    for i, (tier, sty) in enumerate(TIER_STYLE.items()):
        c, r2 = fits[tier]
        sign = "+" if c[1] >= 0 else ""
        eq = (
            rf"$\ln(\eta) = {sign}{c[1]:.2f} {c[0]:+.2f}\,d_{{\mathrm{{AU}}}}$"
            rf"  ($R^2={r2:.2f}$)"
        )
        ax.text(
            0.97,
            0.97 - i * 0.095,
            eq,
            transform=ax.transAxes,
            fontsize=9,
            color=sty["color"],
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=sty["color"],
                alpha=0.88,
                linewidth=0.8,
            ),
            zorder=6,
        )

    # ── Legend ───────────────────────────────────────────────────────────
    handles = [
        Line2D(
            [0],
            [0],
            color=sty["color"],
            linestyle=sty["linestyle"],
            marker=sty["marker"],
            markersize=6,
            linewidth=1.6,
            label=sty["label"],
        )
        for sty in TIER_STYLE.values()
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        framealpha=0.88,
        fontsize=9,
        handlelength=2.2,
        borderpad=0.5,
    )

    ax.set_xlabel(r"$d_{\mathrm{AU}}$  (Earth–Mars distance)", fontsize=11)
    ax.set_ylabel(r"$\ln(\eta)$", fontsize=12)
    ax.set_title(
        "Routing efficiency vs distance: the chain-law ceiling", fontsize=11, fontweight="semibold"
    )
    ax.grid(True, alpha=0.22)

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"fig3_mars_distance_law.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
