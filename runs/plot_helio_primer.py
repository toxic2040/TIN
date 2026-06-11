"""plot_helio_primer.py — Four figures for the Heliocentric Primer paper.

1. Synodic DR vs distance: 780-epoch Mars scatter, colored by SEP regime
2. Phase collapse: DR vs p_eff for 20 epochs × 12 p_ref values
3. Multi-body: 3-panel DR vs epoch for Venus, Mars, Jupiter
4. CGR recovery: paired bars showing greedy vs CGR η at 5 epochs

Reads:  runs/synodic_sweep_results.json
        runs/helio_phase_diagram_results.json
        runs/multi_body_sweep_results.json
        runs/helio_cgr_results.json
Writes: figures/fig_synodic_dr_vs_d.pdf
        figures/fig_phase_collapse.pdf
        figures/fig_multi_body.pdf
        figures/fig_cgr_recovery.pdf
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


# -----------------------------------------------------------------------
# Plot 1: Synodic DR vs distance — colored by SEP regime
# -----------------------------------------------------------------------
def plot_synodic_dr_vs_d():
    with open(_HERE / "synodic_sweep_results.json") as f:
        data = json.load(f)
    ts = data["time_series"]

    # Filter active epochs only (DR > 0)
    active = [r for r in ts if r["DR"] > 0]

    dist = np.array([r["dist_au"] for r in active])
    dr = np.array([r["DR"] for r in active])
    sep = np.array([r["sep_deg"] for r in active])

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Color by SEP regime: opposition (>120), quadrature (60-120), conjunction (<60)
    colors = []
    for s in sep:
        if s >= 120:
            colors.append("#2ca02c")  # opposition
        elif s >= 60:
            colors.append("#1f77b4")  # quadrature
        else:
            colors.append("#d62728")  # conjunction approach

    sc = ax.scatter(dist, dr, c=colors, s=12, alpha=0.6, edgecolors="none")

    # Legend via proxy artists
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ca02c",
            markersize=8,
            label="Opposition (SEP \u2265 120\u00b0)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markersize=8,
            label="Quadrature (60\u00b0\u201320\u00b0)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markersize=8,
            label="Conjunction (SEP < 60\u00b0)",
        ),
    ]
    ax.legend(
        handles=legend_elements, frameon=True, fancybox=False, edgecolor="0.7", loc="upper right"
    )

    ax.set_xlabel("Earth-Mars distance (AU)")
    ax.set_ylabel("DR")
    ax.set_title("Mars synodic sweep: 780 daily epochs")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 2.7)
    ax.set_ylim(-0.02, 1.0)

    out = _FIG / "fig_synodic_dr_vs_d.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 2: Phase collapse — DR vs p_eff
# -----------------------------------------------------------------------
def plot_phase_collapse():
    with open(_HERE / "helio_phase_diagram_results.json") as f:
        data = json.load(f)
    results = data["results"]

    p_eff = np.array([r["p_eff"] for r in results])
    dr = np.array([r["DR"] for r in results])
    dist = np.array([r["dist_au"] for r in results])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    sc = ax.scatter(p_eff, dr, c=dist, cmap="viridis", s=18, alpha=0.7, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Distance (AU)")

    # Fit sigmoid-like curve through the data
    valid = (p_eff > 0) & (dr > 0)
    if valid.sum() > 5:
        pe = p_eff[valid]
        d = dr[valid]
        order = np.argsort(pe)
        pe_s = pe[order]
        d_s = d[order]
        # Running mean for trend line
        window = max(1, len(pe_s) // 15)
        if window > 1:
            smooth_p = np.convolve(pe_s, np.ones(window) / window, mode="valid")
            smooth_d = np.convolve(d_s, np.ones(window) / window, mode="valid")
            ax.plot(smooth_p, smooth_d, "-", color="red", lw=2, alpha=0.7, label="Running mean")

    ax.set_xscale("log")
    ax.set_xlabel(r"$p_{\mathrm{eff}}$ (effective link probability)")
    ax.set_ylabel("DR")
    ax.set_title(r"Phase diagram collapse: DR vs $p_{\mathrm{eff}}$")
    if valid.sum() > 5:
        ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.85)

    out = _FIG / "fig_phase_collapse.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 3: Multi-body — 3-panel DR vs epoch
# -----------------------------------------------------------------------
def plot_multi_body():
    with open(_HERE / "multi_body_sweep_results.json") as f:
        data = json.load(f)
    ts = data["time_series"]

    targets = ["venus", "mars", "jupiter"]
    titles = {"venus": "Venus (584 d)", "mars": "Mars (780 d)", "jupiter": "Jupiter (399 d)"}
    colors = {"venus": "#ff7f0e", "mars": "#d62728", "jupiter": "#9467bd"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, tgt in zip(axes, targets):
        subset = [r for r in ts if r["target"] == tgt]
        subset.sort(key=lambda r: r["epoch_day"])

        epochs = np.array([r["epoch_day"] for r in subset])
        dr_arr = np.array([r["DR"] for r in subset])
        dist_arr = np.array([r["dist_au"] for r in subset])

        ax.plot(epochs, dr_arr, "-", color=colors[tgt], lw=1.2, alpha=0.8)

        # Mark blackout epochs (DR = 0)
        blackout = dr_arr == 0
        if blackout.any():
            ax.scatter(
                epochs[blackout],
                dr_arr[blackout],
                color="black",
                marker="x",
                s=20,
                zorder=3,
                label="Blackout",
            )

        ax.set_xlabel("Epoch (days)")
        ax.set_title(titles[tgt])
        ax.grid(True, alpha=0.3)

        # Secondary axis for distance
        ax2 = ax.twinx()
        ax2.plot(epochs, dist_arr, "--", color="0.6", lw=0.8, alpha=0.6)
        ax2.set_ylabel("Distance (AU)", color="0.5", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="0.5", labelsize=7)

    axes[0].set_ylabel("DR")
    fig.suptitle("Multi-body synodic sweeps: full orbital cycles", fontsize=12)
    fig.tight_layout()

    out = _FIG / "fig_multi_body.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 4: CGR recovery — paired bars
# -----------------------------------------------------------------------
def plot_cgr_recovery():
    with open(_HERE / "helio_cgr_results.json") as f:
        results = json.load(f)

    # Handle potential "inf" strings
    for r in results:
        for sec in ("greedy", "cgr"):
            for k, v in r[sec].items():
                if v == "inf":
                    r[sec][k] = float("inf")

    results = sorted(results, key=lambda r: r["dist_au"])
    labels = [f"d={r['dist_au']:.2f}" for r in results]
    greedy_eta = np.array([r["greedy"]["eta"] for r in results])
    cgr_eta = np.array([r["cgr"]["eta"] for r in results])

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x - w / 2, greedy_eta, w, label="Greedy", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, cgr_eta, w, label="CGR", color="#2ca02c", alpha=0.85)

    # Add recovery % annotations
    for i, r in enumerate(results):
        rec = r["recovery_pct"]
        ax.text(
            x[i],
            max(greedy_eta[i], cgr_eta[i]) + 0.02,
            f"{rec:.0f}%",
            ha="center",
            fontsize=8,
            color="0.3",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\eta$ (routing efficiency)")
    ax.set_title("CGR recovery of greedy efficiency gap")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis="y")

    out = _FIG / "fig_cgr_recovery.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
def main():
    print()
    print("Generating helio primer figures...")
    plot_synodic_dr_vs_d()
    plot_phase_collapse()
    plot_multi_body()
    plot_cgr_recovery()
    print("Done.")


if __name__ == "__main__":
    main()
