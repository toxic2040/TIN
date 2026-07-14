"""Historical configuration-to-output concept flowchart.

Shows an archived concept path from a generic configuration through analysis
stages to recorded outputs. It does not document a public CLI, package API, or
current classifier/predictor interface.

Writes:
  figures/d4_yaml_to_result.{pdf,png}
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


def _box(ax, x, y, w, h, text, subtext, color, style="round,pad=0.1"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=style,
        facecolor=color,
        alpha=0.15,
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.12, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    if subtext:
        ax.text(
            x,
            y - 0.18,
            subtext,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#666666",
            fontstyle="italic",
        )


def _arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=6.5, color="#888888")


def plot():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-2.5, 3.5)
    ax.set_axis_off()

    # ── Row 1: Historical configuration concept ────────────────
    _box(
        ax,
        1.0,
        2.5,
        2.0,
        0.9,
        "Example\nConfiguration",
        "scenario and link assumptions",
        "#3498db",
    )

    _arrow(ax, 2.1, 2.5, 3.4, 2.5, "parse")

    _box(ax, 4.5, 2.5, 2.0, 0.9, "Historical\nRunner", "configuration to output", "#2c3e50")

    # ── Row 2: Archived analysis stages ─────────────────────────
    _arrow(ax, 4.5, 2.0, 4.5, 1.2)

    stages = [
        (1.5, "Contact-Plan\nSummary", "archived stage", "#27ae60"),
        (4.0, "Reachability\nSummary", "archived stage", "#f39c12"),
        (6.5, "Efficiency\nSummary", "archived stage", "#e67e22"),
        (9.0, "Slope\nDiagnostic", "historical only", "#e74c3c"),
        (11.5, "Retired Claim\nBranch", "not a current output", "#9b59b6"),
    ]

    for x, label, out, color in stages:
        _box(ax, x, 0.5, 2.0, 1.0, label, out, color)

    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 1.1
        x2 = stages[i + 1][0] - 1.1
        _arrow(ax, x1, 0.5, x2, 0.5)

    # Connect the historical runner concept to the archived stages.
    _arrow(ax, 4.5, 2.0, 1.5, 1.1)

    # ── Row 3: Historical outputs ────────────────────────────────
    outputs = [
        (3.5, "Console\nSummary", "historical record", "#2c3e50"),
        (7.0, "Result\nRecord", "archived output", "#16a085"),
        (10.5, "Retired Claim\nBranch", "not a current interface", "#8e44ad"),
    ]

    for x, label, sub, color in outputs:
        _box(ax, x, -1.5, 2.2, 0.9, label, sub, color)

    _arrow(ax, 11.5, -0.05, 3.5, -1.0)
    _arrow(ax, 11.5, -0.05, 7.0, -1.0)
    _arrow(ax, 11.5, -0.05, 10.5, -1.0)

    # ── Title ────────────────────────────────────────────────────
    ax.text(
        6.0,
        3.3,
        "Historical Configuration-to-Output Concept — Not a Public API",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#2c3e50",
    )

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"d4_yaml_to_result.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
