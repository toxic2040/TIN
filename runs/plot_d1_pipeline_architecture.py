"""plot_d1_pipeline_architecture.py — TIN pipeline architecture diagram.

Five-stage flow: Config → Contacts → Oracle → Efficiency → Classification.
Each stage shows input/output and the engine module responsible.

Writes:
  figures/d1_pipeline_architecture.{pdf,png}
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RUNS = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(RUNS), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def plot():
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 3.0)
    ax.set_axis_off()

    # Stage definitions: (x, label, module, input, output, color)
    stages = [
        (0.5, "Configure", "schema.py\nregistry.py", "YAML file", "PercConfig", "#3498db"),
        (
            2.7,
            "Generate\nContacts",
            "contact_gen.py\nhelio_contact_gen.py",
            "PercConfig",
            "Contact plan",
            "#2ecc71",
        ),
        (4.9, "Oracle\nSweep", "oracle.py\nsweep.py", "Contact plan", "S_T, paths", "#f39c12"),
        (
            7.1,
            "Efficiency\nEstimation",
            "efficiency.py\nanalytic_s.py",
            "Plan + paths",
            "η, Φ, E[H]",
            "#e74c3c",
        ),
        (9.3, "Classify", "analytic_s.py\ncompute_gamma()", "η, Φ, E[H]", "γ, class", "#9b59b6"),
    ]

    box_w = 1.6
    box_h = 1.8

    for x, label, module, inp, out, color in stages:
        # Main box
        box = FancyBboxPatch(
            (x - box_w / 2, -0.2),
            box_w,
            box_h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            alpha=0.15,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(box)

        # Stage label (bold, centered)
        ax.text(
            x, 1.1, label, ha="center", va="center", fontsize=10, fontweight="bold", color=color
        )

        # Module name (smaller, below label)
        ax.text(
            x,
            0.45,
            module,
            ha="center",
            va="center",
            fontsize=7,
            color="#555555",
            fontstyle="italic",
        )

        # Input (above box)
        ax.text(
            x,
            2.0,
            inp,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#777777",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#f8f8f8", edgecolor="#cccccc", linewidth=0.5
            ),
        )
        ax.annotate(
            "",
            xy=(x, 1.65),
            xytext=(x, 1.95),
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0),
        )

        # Output (below box)
        ax.text(
            x,
            -0.7,
            out,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#777777",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#f8f8f8", edgecolor="#cccccc", linewidth=0.5
            ),
        )
        ax.annotate(
            "",
            xy=(x, -0.55),
            xytext=(x, -0.15),
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0),
        )

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + box_w / 2 + 0.05
        x2 = stages[i + 1][0] - box_w / 2 - 0.05
        ax.annotate(
            "",
            xy=(x2, 0.7),
            xytext=(x1, 0.7),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#333333",
                lw=2.0,
                connectionstyle="arc3,rad=0",
            ),
        )

    # Title
    ax.text(
        5.0,
        2.7,
        "TIN Pipeline Architecture",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#2c3e50",
    )

    # Subtitle
    ax.text(
        5.0,
        2.35,
        "perc classify mars_6polar.yaml",
        ha="center",
        va="center",
        fontsize=9,
        color="#888888",
        fontstyle="italic",
        fontfamily="monospace",
    )

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"d1_pipeline_architecture.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
