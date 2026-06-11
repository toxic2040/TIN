"""plot_math_primer.py — Four figures for the Math Primer paper.

1. Three-factor decomposition: grouped bars for S_T, exp(E[H]·λ), Φ per config
2. Lyapunov validation: η_sim vs η_lyap scatter with identity line
3. Φ range: dot plot by config with Φ=1 reference
4. Wald validity: Wald error vs σ² showing divergence above 0.5

Reads:  runs/lyapunov_validation_results.json
Writes: figures/fig_three_factor_decomposition.pdf
        figures/fig_lyapunov_validation.pdf
        figures/fig_phi_range.pdf
        figures/fig_wald_validity.pdf
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


def load():
    with open(_HERE / "lyapunov_validation_results.json") as f:
        data = json.load(f)
    return data["results"]


# -- Short labels for configs --
def short_label(r):
    lab = r["label"]
    if lab.startswith("Helio"):
        return f"d={r['dist_au']:.2f}"
    return lab.replace("Moon ", "M").replace("Mars ", "Ma")


# -----------------------------------------------------------------------
# Plot 1: Three-factor decomposition — horizontal grouped bars, sorted by DR
# -----------------------------------------------------------------------
def plot_three_factor(results):
    labels = [short_label(r) for r in results]
    s_t = np.array([r["S_T"] for r in results])
    eta_lyap = np.array([r["eta_lyapunov"] for r in results])
    phi = np.array([r["eta_sim"] / r["eta_lyapunov"] for r in results])
    dr = np.array([r["eta_sim"] * r["S_T"] for r in results])

    # Sort by DR descending — highest delivery at top
    order = np.argsort(dr)[::-1]
    labels = [labels[i] for i in order]
    s_t = s_t[order]
    eta_lyap = eta_lyap[order]
    phi = phi[order]
    dr = dr[order]

    n = len(labels)
    y = np.arange(n)
    h = 0.17  # bar height per factor

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.barh(y + 1.5 * h, s_t, h, label=r"$S_T$", color="#2ca02c", alpha=0.85)
    ax.barh(y + 0.5 * h, eta_lyap, h, label=r"$\exp(E[H]\lambda)$", color="#1f77b4", alpha=0.85)
    ax.barh(y - 0.5 * h, phi, h, label=r"$\Phi$", color="#c0392b", alpha=0.85)
    ax.barh(
        y - 1.5 * h,
        dr,
        h,
        label=r"DR (product)",
        color="#555555",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.35,
    )

    # Φ = 1 reference — vertical, clean
    ax.axvline(1.0, ls="--", color="0.45", lw=1.0, zorder=1)
    ax.text(1.02, n - 0.55, r"$\Phi{=}1$", fontsize=7.5, color="0.5", va="top")

    # DR value label at right end of each DR bar
    for i, d in enumerate(dr):
        ax.text(
            d + 0.03,
            y[i] - 1.5 * h,
            f"{d:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # highest DR at top
    ax.set_xlabel("Factor value", fontsize=10)
    ax.set_title(
        r"Three-factor decomposition:  $\mathrm{DR} = S_T \cdot \exp(E[H]\lambda) \cdot \Phi$",
        fontsize=11,
    )
    ax.legend(
        frameon=True, fancybox=False, edgecolor="0.7", ncol=2, fontsize=8.5, loc="lower right"
    )
    ax.set_xlim(0, 2.65)
    ax.grid(True, alpha=0.2, axis="x")

    out = _FIG / "fig_three_factor_decomposition.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 2: Lyapunov validation — η_sim vs η_lyap scatter
# -----------------------------------------------------------------------
def plot_lyapunov_validation(results):
    fig, ax = plt.subplots(figsize=(5, 4.5))

    body = [r for r in results if r["type"] == "body"]
    helio = [r for r in results if r["type"] == "helio"]

    for group, color, marker, glabel in [
        (body, "#1f77b4", "o", "Body-centric"),
        (helio, "#d62728", "s", "Heliocentric"),
    ]:
        eta_lyap = [r["eta_lyapunov"] for r in group]
        eta_sim = [r["eta_sim"] for r in group]
        ax.scatter(
            eta_lyap,
            eta_sim,
            c=color,
            marker=marker,
            s=60,
            zorder=3,
            label=glabel,
            edgecolors="white",
            linewidth=0.5,
        )

        for r in group:
            ax.annotate(
                short_label(r),
                (r["eta_lyapunov"], r["eta_sim"]),
                fontsize=7,
                textcoords="offset points",
                xytext=(5, 5),
            )

    # Identity line
    lims = [0, 1.05]
    ax.plot(lims, lims, "--", color="0.5", lw=1, zorder=1, label="Identity")

    ax.set_xlabel(r"$\eta_{\mathrm{Lyap}} = \exp(E[H]\lambda)$")
    ax.set_ylabel(r"$\eta_{\mathrm{sim}}$")
    ax.set_title(r"Lyapunov validation: $\Phi$ separates sim from oracle chain")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out = _FIG / "fig_lyapunov_validation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 3: Φ range dot plot
# -----------------------------------------------------------------------
def plot_phi_range(results):
    labels = [short_label(r) for r in results]
    phi = np.array([r["eta_sim"] / r["eta_lyapunov"] for r in results])

    # Sort by Φ for visual clarity
    order = np.argsort(phi)
    labels = [labels[i] for i in order]
    phi = phi[order]

    colors = ["#2ca02c" if p > 1 else "#d62728" for p in phi]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    y = np.arange(len(labels))
    ax.barh(y, phi, color=colors, alpha=0.8, height=0.6, edgecolor="white")

    ax.axvline(1.0, ls="--", color="0.4", lw=1.2, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Phi = \eta_{\mathrm{sim}} / \eta_{\mathrm{Lyap}}$")
    ax.set_title(r"Policy distortion factor $\Phi$ by configuration")

    # Annotate regions
    ax.text(
        0.75,
        len(labels) - 0.5,
        r"$\Phi < 1$: dead-end trap",
        fontsize=8,
        color="#d62728",
        ha="center",
    )
    ax.text(
        1.8,
        len(labels) - 0.5,
        r"$\Phi > 1$: diversity gain",
        fontsize=8,
        color="#2ca02c",
        ha="center",
    )
    ax.set_xlim(0.5, 2.5)
    ax.grid(True, alpha=0.2, axis="x")

    out = _FIG / "fig_phi_range.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
# Plot 4: Wald validity — error vs σ²
# -----------------------------------------------------------------------
def plot_wald_validity(results):
    sigma2 = np.array([r["var_log_p"] for r in results])
    # Relative error of Wald vs simulation
    eta_sim = np.array([r["eta_sim"] for r in results])
    eta_wald = np.array([r["eta_wald"] for r in results])
    rel_error = (eta_wald - eta_sim) / eta_sim * 100  # percentage

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for i, r in enumerate(results):
        color = "#1f77b4" if r["type"] == "body" else "#d62728"
        marker = "o" if r["type"] == "body" else "s"
        ax.scatter(
            sigma2[i],
            rel_error[i],
            c=color,
            marker=marker,
            s=60,
            zorder=3,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            short_label(r),
            (sigma2[i], rel_error[i]),
            fontsize=7,
            textcoords="offset points",
            xytext=(5, 3),
        )

    # Threshold
    ax.axvline(0.5, ls="--", color="#ff7f0e", lw=1.5, label=r"$\sigma^2 = 0.5$ boundary")
    ax.axhspan(-10, 10, color="#2ca02c", alpha=0.08)
    ax.axhline(0, ls=":", color="0.5", lw=0.8)

    ax.set_xlabel(r"$\sigma^2 = \mathrm{Var}[\log p_h]$")
    ax.set_ylabel("Wald relative error (%)")
    ax.set_title("Wald CLT validity boundary")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xlim(-0.05, 0.9)
    ax.grid(True, alpha=0.3)

    out = _FIG / "fig_wald_validity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")


# -----------------------------------------------------------------------
def main():
    print()
    print("Generating math primer figures...")
    results = load()
    plot_three_factor(results)
    plot_lyapunov_validation(results)
    plot_phi_range(results)
    plot_wald_validity(results)
    print("Done.")


if __name__ == "__main__":
    main()
