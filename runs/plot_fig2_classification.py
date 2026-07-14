"""Historical four-trace gamma/rho_pair comparison.

γ̄ (mean across p_eff) vs ρ_pair for four CRAWDAD traces.

This archived figure shows a descriptive association in four selected traces.
It does not establish that rho_pair causally controls gamma, support a current
classifier, or generalize beyond the displayed observations. Marker area
encodes network size (n).

Data source: runs/crawdad_cross_trace_analysis.json
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
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

BLUE = "#2171b5"


def _load():
    with open(os.path.join(RUNS, "crawdad_cross_trace_analysis.json")) as f:
        return json.load(f)


def plot():
    d = _load()
    gc = d["gamma_classification"]["gamma_normal"]
    traces = ["Exp1", "Exp2", "Exp3", "Exp6"]

    ns, rhos, gmeans = [], [], []
    for t in traces:
        ns.append(d["traces"][t]["n_nodes"])
        rhos.append(d["traces"][t]["rho_pair"])
        gmeans.append(float(np.mean(list(gc[t]["gamma_by_p"].values()))))

    # ── Marker size proportional to n ────────────────────────────────────
    # Visual encoding: larger network = larger dot.
    # The ordering by dot area (9→98) differs from the observed gamma ordering.
    # This is descriptive for the four displayed traces only.
    size_scale = [12 + (n**0.72) * 4.5 for n in ns]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Historical gamma=1 reference line; no ceiling claim is made here.
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.9, linestyle="--", zorder=1)
    ax.text(
        0.635,
        1.006,
        r"historical $\gamma = 1$ reference",
        fontsize=8.5,
        color="#999999",
        va="bottom",
        ha="right",
    )

    # Data points
    ax.scatter(rhos, gmeans, s=size_scale, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4)

    # Labels — hand-placed to avoid overlap
    offsets = {
        "Exp1": (0.010, -0.010),
        "Exp2": (-0.012, -0.004),
        "Exp3": (-0.010, 0.008),
        "Exp6": (0.012, -0.025),
    }
    haligns = {"Exp1": "left", "Exp2": "right", "Exp3": "right", "Exp6": "left"}
    for t, rho, gm, n in zip(traces, rhos, gmeans, ns):
        dx, dy = offsets[t]
        ax.annotate(
            f"{t}  ($n={n}$)",
            xy=(rho, gm),
            xytext=(rho + dx, gm + dy),
            fontsize=9,
            color="#333333",
            ha=haligns[t],
            va="center",
            arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.7),
            zorder=5,
        )

    # ── Inset key — clean white box replacing old tan/beige ──────────────
    # Preserve the observed n and gamma orderings without assigning a cause.
    n_order = sorted(range(4), key=lambda i: ns[i])
    g_order = sorted(range(4), key=lambda i: gmeans[i])
    n_seq = " < ".join(traces[i] for i in n_order)
    g_seq = " < ".join(traces[i] for i in g_order)
    key_text = (
        r"Marker area $\propto\,n$"
        "\n"
        r"By $n$:  " + n_seq + "\n"
        r"By $\bar{\gamma}$:  " + g_seq
    )
    ax.text(
        0.04,
        0.06,
        key_text,
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
        zorder=6,
    )

    ax.set_xlabel(r"$\rho_{\mathrm{pair}}$  (mean pairwise contact density)", fontsize=11)
    ax.set_ylabel(r"archived $\bar{\gamma}$  (mean across $p_{\mathrm{eff}}$)", fontsize=11)
    ax.set_title(
        "Historical four-trace association: "
        r"$\bar{\gamma}$ and $\rho_{\mathrm{pair}}$"
        "\n(descriptive; classifier retired)",
        fontsize=11,
        fontweight="semibold",
    )

    ax.set_xlim(0.28, 0.68)
    ax.set_ylim(0.74, 1.05)
    ax.grid(True, alpha=0.22)

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"fig2_gamma_saturation.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
