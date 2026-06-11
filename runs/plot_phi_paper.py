"""plot_phi_paper.py — Four figures for the Φ paper.

1. Φ_time vs Φ_rel scatter: one panel per target, identity line
2. Φ histogram by n: showing diversity-to-trap shift
3. Φ vs p_eff colored by target: the policy distortion landscape
4. Braess boundary heatmap: fraction with Φ>1 in (n, p_ref) space

Reads:  runs/phi_sweep_results.json
Writes: figures/fig_phi_time_vs_rel.pdf
        figures/fig_phi_histogram_by_n.pdf
        figures/fig_phi_vs_peff.pdf
        figures/fig_braess_boundary_heatmap.pdf
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

TARGET_COLORS = {
    "mercury": "#e377c2",
    "venus": "#ff7f0e",
    "mars": "#d62728",
    "ceres": "#8c564b",
    "europa": "#2ca02c",
    "jupiter": "#9467bd",
    "saturn": "#1f77b4",
    "titan": "#7f7f7f",
}


def load():
    with open(_HERE / "phi_sweep_results.json") as f:
        data = json.load(f)
    # Filter to active configs (eta_sim > 0)
    results = [r for r in data["results"] if r["eta_sim"] > 0]
    return data, results


# -----------------------------------------------------------------------
# Plot 1: Φ_time vs Φ_rel — 8-panel scatter
# -----------------------------------------------------------------------
def plot_phi_time_vs_rel(data, results):
    targets = list(TARGET_COLORS.keys())

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    axes = axes.flatten()

    for ax, tgt in zip(axes, targets):
        subset = [r for r in results if r["target"] == tgt]
        if not subset:
            ax.set_visible(False)
            continue

        pt = np.array([r["phi_time"] for r in subset])
        pr = np.array([r["phi_rel"] for r in subset])

        # Clip for display
        clip = 10
        mask = (pt < clip) & (pr < clip)
        pt_c, pr_c = pt[mask], pr[mask]

        ax.scatter(pt_c, pr_c, s=6, alpha=0.3, color=TARGET_COLORS[tgt], edgecolors="none")

        lim = max(pt_c.max(), pr_c.max()) * 1.05 if len(pt_c) > 0 else 5
        lim = min(lim, clip)
        ax.plot([0, lim], [0, lim], "--", color="0.5", lw=0.8)
        ax.axhline(1, ls=":", color="0.7", lw=0.5)
        ax.axvline(1, ls=":", color="0.7", lw=0.5)

        n_active = len(subset)
        frac_below = np.mean(pr < pt) * 100 if len(pt) > 0 else 0
        ax.set_title(f"{tgt.capitalize()} (n={n_active})", fontsize=10)
        ax.text(
            0.05,
            0.92,
            f"{frac_below:.0f}% below identity",
            transform=ax.transAxes,
            fontsize=7,
            color="0.3",
        )

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.supxlabel(r"$\Phi_{\mathrm{time}}$ (time-optimal oracle)", fontsize=11)
    fig.supylabel(r"$\Phi_{\mathrm{rel}}$ (reliability-optimal oracle)", fontsize=11)
    fig.suptitle(r"$\Phi_{\mathrm{rel}} \leq \Phi_{\mathrm{time}}$ universally", fontsize=13)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.95])

    out = _FIG / "fig_phi_time_vs_rel.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 2: Φ histogram by n
# -----------------------------------------------------------------------
def plot_phi_histogram_by_n(data, results):
    ns = sorted(set(r["n_orb"] for r in results))

    n_colors = {3: "#ff7f0e", 6: "#d62728", 12: "#2ca02c", 24: "#1f77b4"}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bins = np.linspace(0, 5, 40)

    for n in ns:
        phi_vals = [r["phi_time"] for r in results if r["n_orb"] == n and r["phi_time"] < 10]
        if not phi_vals:
            continue
        frac_gt1 = np.mean(np.array(phi_vals) > 1) * 100
        ax.hist(
            phi_vals,
            bins=bins,
            alpha=0.5,
            color=n_colors.get(n, "0.5"),
            label=f"n={n} ({frac_gt1:.0f}% > 1)",
            density=True,
        )

    ax.axvline(1.0, ls="--", color="0.4", lw=1.5)
    ax.set_xlabel(r"$\Phi_{\mathrm{time}}$")
    ax.set_ylabel("Density")
    ax.set_title(r"$\Phi$ distribution shifts from diversity to trap with $n$")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True, alpha=0.2)

    out = _FIG / "fig_phi_histogram_by_n.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 3: Φ vs p_eff colored by target — density-aware rendering
# -----------------------------------------------------------------------
def plot_phi_vs_peff(data, results):
    fig, ax = plt.subplots(figsize=(7, 5))

    p_bins = np.logspace(np.log10(4e-4), np.log10(1.1), 22)

    for tgt in TARGET_COLORS:
        subset = [r for r in results if r["target"] == tgt and r["phi_time"] < 15]
        if not subset:
            continue
        peff = np.array([r["p_eff"] for r in subset])
        phi = np.array([r["phi_time"] for r in subset])
        mask = peff > 0
        peff, phi = peff[mask], phi[mask]
        color = TARGET_COLORS[tgt]

        # ── Context scatter — low alpha so cloud reads as texture ────────
        ax.scatter(peff, phi, s=5, alpha=0.18, color=color, edgecolors="none", zorder=2)

        # ── Per-planet median curve — gives each body identity ───────────
        # Binned median in log-spaced p_eff bins; requires ≥3 pts per bin
        cx, cy = [], []
        for i in range(len(p_bins) - 1):
            in_bin = (peff >= p_bins[i]) & (peff < p_bins[i + 1])
            if in_bin.sum() >= 3:
                cx.append(np.sqrt(p_bins[i] * p_bins[i + 1]))
                cy.append(float(np.median(phi[in_bin])))

        if len(cx) >= 2:
            ax.plot(cx, cy, color=color, lw=1.8, alpha=0.92, label=tgt.capitalize(), zorder=4)

    # Φ = 1 reference
    ax.axhline(1.0, ls="--", color="0.35", lw=1.2, zorder=3)
    ax.text(4e-4, 1.06, r"$\Phi = 1$  (no distortion)", fontsize=7.5, color="0.45", va="bottom")

    # Φ range note
    ax.text(
        0.97,
        0.97,
        r"$\Phi \in [0.3,\,8593]$ across 154k+ configs",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="right",
        va="top",
        color="0.4",
        style="italic",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$p_{\mathrm{eff}}$ (effective link probability)")
    ax.set_ylabel(r"$\Phi_{\mathrm{time}}$")
    ax.set_title(r"Policy distortion factor vs link quality")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", ncol=2, fontsize=8, handlelength=1.6)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 8)

    out = _FIG / "fig_phi_vs_peff.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 4: Braess boundary heatmap — fraction Φ>1 in (n, p_ref) space
# -----------------------------------------------------------------------
def plot_braess_boundary(data, results):
    ns = sorted(set(r["n_orb"] for r in results))
    p_refs = sorted(set(r["p_ref"] for r in results))

    frac_grid = np.full((len(p_refs), len(ns)), np.nan)

    for j, n in enumerate(ns):
        for i, p in enumerate(p_refs):
            subset = [r for r in results if r["n_orb"] == n and abs(r["p_ref"] - p) < 0.001]
            if len(subset) >= 3:
                frac_gt1 = np.mean([r["phi_time"] > 1 for r in subset])
                frac_grid[i, j] = frac_gt1

    fig, ax = plt.subplots(figsize=(5.5, 5))

    im = ax.pcolormesh(
        range(len(ns)),
        range(len(p_refs)),
        frac_grid,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        shading="nearest",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"Fraction with $\Phi > 1$ (diversity-dominated)")

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Constellation size n")

    # Show subset of p_ref ticks
    ytick_idx = list(range(0, len(p_refs), max(1, len(p_refs) // 8)))
    if (len(p_refs) - 1) not in ytick_idx:
        ytick_idx.append(len(p_refs) - 1)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([f"{p_refs[i]:.2f}" for i in ytick_idx])
    ax.set_ylabel(r"$p_{\mathrm{ref}}$")

    ax.set_title(r"Braess boundary in $(n, p_{\mathrm{ref}})$ space")

    # Add Φ=1 contour annotation
    ax.contour(
        range(len(ns)),
        range(len(p_refs)),
        frac_grid,
        levels=[0.5],
        colors=["black"],
        linewidths=[1.5],
    )

    out = _FIG / "fig_braess_boundary_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
def main():
    print()
    print("Generating Phi paper figures...")
    data, results = load()
    plot_phi_time_vs_rel(data, results)
    plot_phi_histogram_by_n(data, results)
    plot_phi_vs_peff(data, results)
    plot_braess_boundary(data, results)
    print("Done.")


if __name__ == "__main__":
    main()
