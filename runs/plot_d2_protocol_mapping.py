"""Render a historical research-layer-to-DTN concept map.

The right-hand stack is an archival research abstraction, not the public
``tin`` package API. Private analysis modules are omitted, and the former
gamma-classification branch is explicitly retired. Inspired by Burleigh's
tutorial (p. 22).

Writes:
  figures/d2_protocol_mapping.{pdf,png}
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


def _draw_stack(ax, x, layers, title, title_color):
    """Draw a vertical protocol stack at position x."""
    box_w = 2.8
    box_h = 0.7
    gap = 0.08
    y_base = 0.0

    ax.text(
        x,
        y_base + len(layers) * (box_h + gap) + 0.3,
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=title_color,
    )

    for i, (label, sublabel, color) in enumerate(layers):
        y = y_base + i * (box_h + gap)
        box = FancyBboxPatch(
            (x - box_w / 2, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            alpha=0.2,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y + box_h / 2 + 0.08,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
        )
        if sublabel:
            ax.text(
                x,
                y + box_h / 2 - 0.15,
                sublabel,
                ha="center",
                va="center",
                fontsize=6.5,
                color="#666666",
                fontstyle="italic",
            )


def plot():
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.8, 5.8)
    ax.set_axis_off()

    # DTN Protocol Stack (left) — bottom to top
    dtn_layers = [
        ("Physical / Link", "RF, optical, Bluetooth, etc.", "#7f8c8d"),
        ("Convergence Layer", "TCP, LTP, UDP", "#7f8c8d"),
        ("Bundle Protocol", "BP7 / BPv6", "#2980b9"),
        ("Bundle Service", "Custody, priority, TTL", "#2980b9"),
        ("Application", "User data", "#8e44ad"),
    ]

    # Historical research stack (right) — bottom to top. These are conceptual
    # roles, not private module names or a public-package API map.
    tin_layers = [
        ("Contact Modeling", "historical research stage", "#27ae60"),
        ("Path Reference", "earliest-arrival diagnostic", "#27ae60"),
        ("Forwarding Model", "modeled routing choices", "#e67e22"),
        ("Delivery Accounting", "DR = S_T × η identity", "#e74c3c"),
        ("Retired Gamma Branch", "historical diagnostic; not public API", "#9b59b6"),
    ]

    _draw_stack(ax, 2.0, dtn_layers, "DTN Protocol Stack", "#2c3e50")
    _draw_stack(ax, 8.0, tin_layers, "Historical TIN Research Layers", "#2c3e50")

    # Mapping arrows between stacks
    mappings = [
        (0, 0, "Contacts represent\nlink opportunities"),
        (1, 1, "Path reference\ncompares routes"),
        (2, 2, "Forwarding model\nsimulates choices"),
        (3, 3, "Bookkeeping records\nservice outcomes"),
        (4, 4, "Historical gamma branch\nretired"),
    ]

    for dtn_idx, tin_idx, label in mappings:
        y_dtn = dtn_idx * 0.78 + 0.35
        y_tin = tin_idx * 0.78 + 0.35
        ax.annotate(
            "",
            xy=(6.3, y_tin),
            xytext=(3.5, y_dtn),
            arrowprops=dict(
                arrowstyle="<->",
                color="#bbbbbb",
                lw=1.0,
                connectionstyle="arc3,rad=0.05",
            ),
        )
        mid_x = 5.0
        mid_y = (y_dtn + y_tin) / 2
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#999999",
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="white", edgecolor="#dddddd", linewidth=0.5
            ),
        )

    # Title
    ax.text(
        5.0,
        5.5,
        "Historical Research Mapping — Not the Public Package API",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#2c3e50",
    )

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"d2_protocol_mapping.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
