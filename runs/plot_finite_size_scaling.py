"""plot_finite_size_scaling.py — Four-panel figure suite for the routing-limited
percolation experiment.

1. DR vs p_ref for n=3,6,12,24 at d~1.24 AU (non-monotonicity)
2. p_crit(n) flat + Δp(n) rising (dual-axis)
3. Master collapse: DR vs rescaled p_tilde = p_ref / d^2
4. Braess sweet-spot heatmap: best n* vs distance

Reads: runs/finite_size_scaling_results.json
Writes: figures/fig_fss_*.pdf
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

_HERE = Path(__file__).parent
_FIG = _HERE.parent / "figures"
_FIG.mkdir(exist_ok=True)

# -- Style --
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

COLORS = {
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#2ca02c",
    6: "#d62728",
    8: "#9467bd",
    12: "#8c564b",
    16: "#e377c2",
    24: "#7f7f7f",
}
MARKERS = {
    2: "o",
    3: "s",
    4: "^",
    6: "D",
    8: "v",
    12: "p",
    16: "h",
    24: "*",
}


def load():
    with open(_HERE / "finite_size_scaling_results.json") as f:
        data = json.load(f)
    results = data["results"]
    # Convert "inf"/"nan" strings back
    for r in results:
        for k, v in r.items():
            if v == "inf":
                r[k] = float("inf")
            elif v == "-inf":
                r[k] = float("-inf")
            elif v == "nan":
                r[k] = float("nan")
    return data["config"], results


# -----------------------------------------------------------------------
# Plot 1: DR vs p_ref — non-monotonicity at d ~ 1.24 AU
# -----------------------------------------------------------------------
def plot_dr_vs_p(config, results):
    epoch = 390  # d ~ 1.24 AU
    show_n = [3, 6, 12, 24]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for n in show_n:
        subset = sorted(
            [r for r in results if r["n_orb"] == n and r["epoch_day"] == epoch],
            key=lambda r: r["p_ref"],
        )
        p = [r["p_ref"] for r in subset]
        dr = [r["DR"] for r in subset]
        ax.plot(
            p,
            dr,
            "-",
            marker=MARKERS[n],
            color=COLORS[n],
            markersize=5,
            label=f"n = {n}",
            linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$p_{\mathrm{ref}}$")
    ax.set_ylabel("DR")
    ax.set_title("DR vs link reliability (d = 1.24 AU)")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xlim(0.008, 1.1)
    ax.set_ylim(-0.02, 0.85)
    ax.axhline(0.1, ls=":", color="0.5", lw=0.8, label=None)
    ax.text(0.012, 0.115, r"$p_{\mathrm{crit}}$ threshold", fontsize=8, color="0.5")
    ax.grid(True, alpha=0.3)

    out = _FIG / "fig_fss_dr_vs_p.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 2: p_crit(n) flat + Δp(n) rising — dual axis
# -----------------------------------------------------------------------
def plot_pcrit_and_width(config, results):
    epoch = 390  # mid-distance
    ns = config["n_orbiters"]

    # p_crit: raw threshold crossing at DR > 0.1
    p_crits = []
    widths = []
    for n in ns:
        subset = sorted(
            [r for r in results if r["n_orb"] == n and r["epoch_day"] == epoch],
            key=lambda r: r["p_ref"],
        )
        p_arr = np.array([r["p_ref"] for r in subset])
        dr_arr = np.array([r["DR"] for r in subset])

        above = dr_arr > 0.1
        pc = float(p_arr[above][0]) if above.any() else np.nan
        p_crits.append(pc)

        # Width: p(DR=0.2) - p(DR=0.02)
        above_hi = dr_arr >= 0.2
        above_lo = dr_arr >= 0.02
        p_hi = float(p_arr[above_hi][0]) if above_hi.any() else np.nan
        p_lo = float(p_arr[above_lo][0]) if above_lo.any() else np.nan
        if np.isfinite(p_hi) and np.isfinite(p_lo):
            widths.append(p_hi - p_lo)
        else:
            widths.append(np.nan)

    p_crits = np.array(p_crits)
    widths = np.array(widths)

    fig, ax1 = plt.subplots(figsize=(5.5, 4))

    # p_crit on left axis
    valid_pc = np.isfinite(p_crits)
    ax1.plot(
        np.array(ns)[valid_pc],
        p_crits[valid_pc],
        "s-",
        color="#d62728",
        markersize=7,
        linewidth=1.5,
        label=r"$p_{\mathrm{crit}}$ (DR > 0.1)",
    )
    ax1.set_xlabel("Constellation size n")
    ax1.set_ylabel(r"$p_{\mathrm{crit}}$", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0, 1.2)

    # Δp on right axis
    ax2 = ax1.twinx()
    valid_w = np.isfinite(widths)
    ax2.plot(
        np.array(ns)[valid_w],
        widths[valid_w],
        "o--",
        color="#1f77b4",
        markersize=7,
        linewidth=1.5,
        label=r"$\Delta p$ (transition width)",
    )
    ax2.set_ylabel(r"$\Delta p = p(\mathrm{DR}=0.2) - p(\mathrm{DR}=0.02)$", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 1.2)

    # Reference line: classical expectation Δp ~ n^{-1/ν}
    ax1.axhline(np.nanmean(p_crits[valid_pc]), ls=":", color="#d62728", lw=0.8, alpha=0.5)

    ax1.set_title(r"Routing-limited percolation: $\gamma \approx 0$, reverse FSS")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, fancybox=False, edgecolor="0.7")
    ax1.set_xscale("log")
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.grid(True, alpha=0.3)

    out = _FIG / "fig_fss_pcrit_width.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 3: Master collapse — DR vs p_tilde = p_ref / d^2
# -----------------------------------------------------------------------
def plot_collapse(config, results):
    ns_show = [3, 6, 12, 24]
    epochs = config["epoch_days"]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for n in ns_show:
        p_tilde_all = []
        dr_all = []
        for epoch in epochs:
            subset = [r for r in results if r["n_orb"] == n and r["epoch_day"] == epoch]
            for r in subset:
                d = r["dist_au"]
                pt = r["p_ref"] / (d**2)
                p_tilde_all.append(pt)
                dr_all.append(r["DR"])

        order = np.argsort(p_tilde_all)
        pt_sorted = np.array(p_tilde_all)[order]
        dr_sorted = np.array(dr_all)[order]

        ax.plot(pt_sorted, dr_sorted, ".", color=COLORS[n], markersize=4, alpha=0.6)
        # Bin and smooth for clarity
        n_bins = 20
        if len(pt_sorted) > n_bins:
            bins = np.logspace(
                np.log10(max(pt_sorted.min(), 1e-4)), np.log10(pt_sorted.max()), n_bins + 1
            )
            bin_centers = []
            bin_means = []
            for i in range(n_bins):
                mask = (pt_sorted >= bins[i]) & (pt_sorted < bins[i + 1])
                if mask.sum() > 0:
                    bin_centers.append(np.sqrt(bins[i] * bins[i + 1]))
                    bin_means.append(dr_sorted[mask].mean())
            ax.plot(bin_centers, bin_means, "-", color=COLORS[n], linewidth=1.8, label=f"n = {n}")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tilde{p} = p_{\mathrm{ref}} / d^2$  (effective link probability)")
    ax.set_ylabel("DR")
    ax.set_title("Master collapse: all distances onto single curve")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.85)

    out = _FIG / "fig_fss_collapse.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 4: Braess sweet-spot heatmap — best n* vs (d, p_ref)
# -----------------------------------------------------------------------
def plot_braess_heatmap(config, results):
    epochs = config["epoch_days"]
    p_vals = sorted(set(r["p_ref"] for r in results))
    ns = config["n_orbiters"]

    # For each (epoch, p_ref), find n* that maximizes DR
    # Build grid: x = distance, y = p_ref
    dist_map = {}
    for epoch in epochs:
        sample = [r for r in results if r["epoch_day"] == epoch]
        if sample:
            dist_map[epoch] = sample[0]["dist_au"]

    dists = [dist_map[e] for e in epochs]

    # Build the n* grid and DR* grid
    n_star = np.zeros((len(p_vals), len(epochs)))
    dr_star = np.zeros((len(p_vals), len(epochs)))

    for j, epoch in enumerate(epochs):
        for i, p in enumerate(p_vals):
            best_dr = -1
            best_n = 2
            for n in ns:
                match = [
                    r
                    for r in results
                    if r["n_orb"] == n and r["epoch_day"] == epoch and abs(r["p_ref"] - p) < 0.001
                ]
                if match and match[0]["DR"] > best_dr:
                    best_dr = match[0]["DR"]
                    best_n = n
            n_star[i, j] = best_n
            dr_star[i, j] = best_dr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1]})

    # Left panel: optimal n* heatmap — sequential palette (n* has natural order)
    bounds = [1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 12.5, 16.5, 24.5]
    cmap = matplotlib.colormaps.get_cmap("YlOrRd").resampled(len(ns))
    norm = BoundaryNorm(bounds, cmap.N)

    im1 = ax1.pcolormesh(
        range(len(epochs)),
        range(len(p_vals)),
        n_star,
        cmap=cmap,
        norm=norm,
        shading="nearest",
    )
    ax1.set_xticks(range(len(epochs)))
    ax1.set_xticklabels([f"{d:.2f}" for d in dists])
    # Show subset of p_ref ticks
    ytick_idx = list(range(0, len(p_vals), 4)) + [len(p_vals) - 1]
    ax1.set_yticks(ytick_idx)
    ax1.set_yticklabels([f"{p_vals[i]:.3f}" for i in ytick_idx])
    ax1.set_xlabel("Distance (AU)")
    ax1.set_ylabel(r"$p_{\mathrm{ref}}$")
    ax1.set_title(r"Optimal $n^*$ (Braess sweet spot)")
    cbar1 = fig.colorbar(im1, ax=ax1, ticks=ns)
    cbar1.set_label("n*")

    # Right panel: DR at optimal n*
    im2 = ax2.pcolormesh(
        range(len(epochs)),
        range(len(p_vals)),
        dr_star,
        cmap="viridis",
        shading="nearest",
        vmin=0,
        vmax=0.8,
    )
    ax2.set_xticks(range(len(epochs)))
    ax2.set_xticklabels([f"{d:.2f}" for d in dists])
    ax2.set_yticks(ytick_idx)
    ax2.set_yticklabels([f"{p_vals[i]:.3f}" for i in ytick_idx])
    ax2.set_xlabel("Distance (AU)")
    ax2.set_ylabel(r"$p_{\mathrm{ref}}$")
    ax2.set_title(r"$\mathrm{DR}^*$ at optimal $n^*$")
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("DR*")

    fig.suptitle("Braess paradox: more orbiters can hurt greedy DR", fontsize=12, y=1.02)
    fig.tight_layout()

    out = _FIG / "fig_fss_braess_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
def main():
    print()
    print("Generating finite-size scaling figures...")
    config, results = load()
    plot_dr_vs_p(config, results)
    plot_pcrit_and_width(config, results)
    plot_collapse(config, results)
    plot_braess_heatmap(config, results)
    print("Done.")


if __name__ == "__main__":
    main()
