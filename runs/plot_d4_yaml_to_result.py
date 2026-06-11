"""plot_d4_yaml_to_result.py — YAML-to-result flowchart.

Shows the user journey: YAML config file → perc CLI → engine pipeline →
output (classification, prediction, figures).

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

    # ── Row 1: User input ────────────────────────────────────────
    _box(ax, 1.0, 2.5, 2.0, 0.9, "mars_6polar.yaml", "body, orbiters, DSN, p_eff", "#3498db")

    _arrow(ax, 2.1, 2.5, 3.4, 2.5, "load_config()")

    _box(ax, 4.5, 2.5, 2.0, 0.9, "perc CLI", "perc classify config.yaml", "#2c3e50")

    # ── Row 2: Engine pipeline (horizontal) ──────────────────────
    _arrow(ax, 4.5, 2.0, 4.5, 1.2)

    stages = [
        (1.5, "Contact\nGeneration", "951 contacts", "#27ae60"),
        (4.0, "Oracle\nSweep", "S_T = 0.990", "#f39c12"),
        (6.5, "Efficiency\nEstimation", "η per pair", "#e67e22"),
        (9.0, "Gamma\nRegression", "γ = −3.67", "#e74c3c"),
        (11.5, "Classification", "TRAP", "#9b59b6"),
    ]

    for x, label, out, color in stages:
        _box(ax, x, 0.5, 2.0, 1.0, label, out, color)

    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 1.1
        x2 = stages[i + 1][0] - 1.1
        _arrow(ax, x1, 0.5, x2, 0.5)

    # Connect CLI to pipeline
    _arrow(ax, 4.5, 2.0, 1.5, 1.1)

    # ── Row 3: Outputs (fan out from classification) ─────────────
    outputs = [
        (3.5, "Terminal\nOutput", "γ=−3.67, class=TRAP", "#2c3e50"),
        (7.0, "JSON\nExport", "results.json", "#16a085"),
        (10.5, "Sparse Law\nPredictor", "DR(d, p_ref)", "#8e44ad"),
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
        "From YAML to Classification in One Command",
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
