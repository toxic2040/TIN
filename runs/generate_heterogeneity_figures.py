"""generate_heterogeneity_figures.py — Paper-ready figures for heterogeneity findings.

Generates four key figures:
  1. s vs K across bodies (regime collapse)
  2. Moon ISL ablation (with/without ISL comparison)
  3. Dead-end filter depth effect (K=1 vs K=2)
  4. Universal L=0 vs L=1 comparison across all configs
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent

# Publication style
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load(name):
    p = _HERE / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def fig1_regime_collapse():
    """Figure 1: s vs K across bodies — the regime collapse."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Data points
    configs = [
        # (label, K, s, body, marker)
        ("Mars K=1\n(6 polar)", 1, 293.7, "Mars", "D"),
        ("Mars K=2\n(6P+ecc)", 2, 156.9, "Mars", "D"),
        ("Moon K=1\n(8 polar)", 1, 50.4, "Moon", "o"),
        ("Moon K=2\n(8P+ELFO)", 2, 6.1, "Moon", "o"),
        ("Moon K=3\n(8P+E+H)", 3, 8.9, "Moon", "o"),
    ]

    # Also add no-ISL points
    no_isl = [
        ("Mars no-ISL", 1, 339.0, "Mars-noISL", "^"),
        ("Moon no-ISL", 1, 159.0, "Moon-noISL", "^"),
    ]

    colors = {"Mars": "C1", "Moon": "C0", "Mars-noISL": "C1", "Moon-noISL": "C0"}

    for label, k, s, body, marker in configs:
        ax.scatter(
            k,
            s,
            s=150,
            marker=marker,
            color=colors[body],
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.annotate(
            label, (k, s), textcoords="offset points", xytext=(12, 5), fontsize=8, ha="left"
        )

    for label, k, s, body, marker in no_isl:
        ax.scatter(
            k + 0.1,
            s,
            s=100,
            marker=marker,
            color=colors[body],
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.6,
        )
        ax.annotate(
            label,
            (k + 0.1, s),
            textcoords="offset points",
            xytext=(12, -5),
            fontsize=8,
            ha="left",
            alpha=0.7,
        )

    # Reference lines
    ax.axhline(1.0, color="red", ls="--", alpha=0.3, lw=1, label="s=1 (η=0.5)")
    ax.axhline(10, color="orange", ls=":", alpha=0.3, lw=1, label="s=10 (η=0.91)")

    ax.set_xlabel("Constellation Heterogeneity K")
    ax.set_ylabel("Relay SNR  s = η/(1−η)")
    ax.set_title("Routing Efficiency vs Constellation Heterogeneity")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 3.5)
    ax.set_xticks([1, 2, 3])
    ax.legend(loc="lower left")

    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_regime_collapse.{ext}")
    print("  fig_regime_collapse saved")
    plt.close()


def fig2_isl_ablation():
    """Figure 2: Moon ISL ablation bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    configs = ["With ISL", "Without ISL"]
    moon_s = [50.4, 159.0]
    moon_eta = [0.9806, 0.9938]
    moon_nr = [28, 9]
    moon_hops = [8.9, 2.0]
    colors = ["C0", "C2"]

    # Panel A: s
    ax = axes[0]
    bars = ax.bar(configs, moon_s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("s = η/(1−η)")
    ax.set_title("A: Relay SNR")
    for bar, val in zip(bars, moon_s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val:.0f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Panel B: η
    ax = axes[1]
    bars = ax.bar(configs, moon_eta, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("η")
    ax.set_title("B: Routing Efficiency")
    ax.set_ylim(0.95, 1.005)
    for bar, val in zip(bars, moon_eta):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            fontsize=10,
        )

    # Panel C: hops and no_route
    ax = axes[2]
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(
        x - w / 2, moon_hops, w, label="Mean hops", color="C4", edgecolor="black", linewidth=0.5
    )
    ax2 = ax.twinx()
    b2 = ax2.bar(
        x + w / 2, moon_nr, w, label="no_route", color="C3", edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Mean Hops")
    ax2.set_ylabel("no_route Failures")
    ax.set_title("C: Path Depth & Failures")
    lines = [b1, b2]
    ax.legend(lines, ["Mean hops", "no_route"], loc="upper center")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_moon_isl_ablation.{ext}")
    print("  fig_moon_isl_ablation saved")
    plt.close()


def fig3_filter_depth():
    """Figure 3: Filter depth sweep K=1 vs K=2."""
    d = _load("filter_depth_sweep_results.json")
    if not d:
        print("  SKIP fig3: no filter_depth_sweep_results.json")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for label, results_list in d.items():
        depths = [r["depth"] for r in results_list]
        ss = [r["s"] for r in results_list]
        etas = [r["eta"] for r in results_list]
        stds = [r.get("eta_std", 0) for r in results_list]
        color = "C0" if "K=1" in label else "C1"
        marker = "o" if "K=1" in label else "s"

        ax = axes[0]
        ax.errorbar(
            depths,
            etas,
            yerr=stds,
            fmt=f"{marker}-",
            color=color,
            capsize=4,
            label=label,
            markersize=8,
        )

        ax = axes[1]
        ax.plot(depths, ss, f"{marker}-", color=color, label=label, markersize=8)

    axes[0].set_xlabel("Lookahead Depth L")
    axes[0].set_ylabel("η")
    axes[0].set_title("A: Efficiency vs Filter Depth")
    axes[0].legend()
    axes[0].set_xticks([0, 1, 2, 3, 4, 6])

    axes[1].set_xlabel("Lookahead Depth L")
    axes[1].set_ylabel("s = η/(1−η)")
    axes[1].set_title("B: Relay SNR vs Filter Depth")
    axes[1].legend()
    axes[1].set_xticks([0, 1, 2, 3, 4, 6])
    axes[1].set_yscale("log")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_filter_depth_paper.{ext}")
    print("  fig_filter_depth_paper saved")
    plt.close()


def fig4_universal_filter():
    """Figure 4: Universal L=0 vs L=1 comparison."""
    d = _load("filter_universal_results.json")
    if not d:
        print("  SKIP fig4: no filter_universal_results.json")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parse paired L=0/L=1 results
    labels = []
    s_l0 = []
    s_l1 = []
    nr_l0 = []
    nr_l1 = []

    for i in range(0, len(d), 2):
        r0 = d[i]
        r1 = d[i + 1]
        labels.append(r0["label"].replace(" (", "\n("))
        s_l0.append(r0["s"])
        s_l1.append(r1["s"])
        nr_l0.append(r0["no_route"])
        nr_l1.append(r1["no_route"])

    x = np.arange(len(labels))
    w = 0.35

    # Panel A: s comparison (log scale)
    ax = axes[0]
    ax.bar(
        x - w / 2,
        s_l0,
        w,
        label="L=0 (no filter)",
        color="C3",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        s_l1,
        w,
        label="L=1 (1-hop filter)",
        color="C0",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("s = η/(1−η)")
    ax.set_title("A: Relay SNR with and without 1-Hop Filter")
    ax.legend()
    ax.set_yscale("log")

    # Panel B: no_route comparison
    ax = axes[1]
    ax.bar(
        x - w / 2,
        nr_l0,
        w,
        label="L=0 (no filter)",
        color="C3",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        nr_l1,
        w,
        label="L=1 (1-hop filter)",
        color="C0",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("no_route Failures (5 seeds)")
    ax.set_title("B: Dead-End Failures with and without Filter")
    ax.legend()

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_universal_filter_paper.{ext}")
    print("  fig_universal_filter_paper saved")
    plt.close()


def fig5_cross_body():
    """Figure 5: Cross-body η comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Mars configs from universality sweep
    mars_labels = [
        "n=2",
        "n=4",
        "n=6",
        "n=6\nDSN8",
        "n=6\nDSN16",
        "n=6\nDSN20",
        "lat=0°",
        "lat=−45°",
    ]
    mars_eta = [0.9948, 0.9715, 0.9966, 0.9955, 0.9962, 0.9968, 0.9975, 0.9979]

    # Cislunar configs
    cis_labels = ["K=1\n(8P)", "K=2\n(8P+E)", "K=3\n(8P+E+H)"]
    cis_eta = [0.9806, 0.8583, 0.8990]

    x_mars = np.arange(len(mars_labels))
    x_cis = np.arange(len(cis_labels)) + len(mars_labels) + 1

    ax.bar(
        x_mars, mars_eta, color="C1", alpha=0.8, edgecolor="black", linewidth=0.5, label="Mars K=1"
    )
    ax.bar(
        x_cis,
        cis_eta,
        color="C0",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="Moon K=1/2/3",
    )

    all_x = list(x_mars) + list(x_cis)
    all_labels = mars_labels + cis_labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=8, ha="center")

    ax.axhline(0.681, color="gray", ls="--", alpha=0.5, label="cislunar η₀ (paper)")
    ax.set_ylabel("η (routing efficiency)")
    ax.set_title("Cross-Body Routing Efficiency Comparison")
    ax.set_ylim(0.7, 1.02)
    ax.legend(loc="lower left")

    # Separator
    sep_x = len(mars_labels) + 0.5
    ax.axvline(sep_x, color="gray", ls="-", alpha=0.3)
    ax.text(
        sep_x - 0.5, 0.72, "Mars", ha="right", fontsize=10, fontweight="bold", color="C1", alpha=0.7
    )
    ax.text(
        sep_x + 0.5, 0.72, "Moon", ha="left", fontsize=10, fontweight="bold", color="C0", alpha=0.7
    )

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_cross_body_eta.{ext}")
    print("  fig_cross_body_eta saved")
    plt.close()


def main():
    print()
    print("=" * 60)
    print("  Generating Heterogeneity Paper Figures")
    print("=" * 60)

    fig1_regime_collapse()
    fig2_isl_ablation()
    fig3_filter_depth()
    fig4_universal_filter()
    fig5_cross_body()

    print("\nAll figures generated.")
    print("Done.")


if __name__ == "__main__":
    main()
