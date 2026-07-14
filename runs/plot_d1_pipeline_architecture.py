"""Render the historical TIN research-analysis pipeline.

This diagram records a retired internal workflow, not the public ``tin``
package API. Private analysis-module names are omitted, and the former gamma
classifier is retained only as a visibly retired branch.

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

    # Stage definitions: (x, label, scope, input, output, color)
    stages = [
        (0.5, "Configure", "historical internal stage", "scenario file", "model config", "#3498db"),
        (
            2.7,
            "Generate\nContacts",
            "historical internal stage",
            "model config",
            "Contact plan",
            "#2ecc71",
        ),
        (
            4.9,
            "Path\nReference",
            "historical analysis stage",
            "Contact plan",
            "S_T, paths",
            "#f39c12",
        ),
        (
            7.1,
            "Delivery\nDiagnostics",
            "historical analysis stage",
            "Plan + paths",
            "η, Φ, E[H]",
            "#e74c3c",
        ),
        (
            9.3,
            "Retired\nClassifier",
            "historical branch",
            "η, Φ, E[H]",
            "archived γ label",
            "#9b59b6",
        ),
    ]

    box_w = 1.6
    box_h = 1.8

    for x, label, scope, inp, out, color in stages:
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

        # Historical scope (smaller, below label)
        ax.text(
            x,
            0.45,
            scope,
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
        "Historical TIN Research Pipeline — Classifier Retired",
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
        "archival analysis flow — not the public tin package API",
        ha="center",
        va="center",
        fontsize=9,
        color="#888888",
        fontstyle="italic",
    )

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"d1_pipeline_architecture.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
