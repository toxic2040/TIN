#!/usr/bin/env python3
"""Historical serial-chain and diamond failure schematic.

Left panel:  a serial chain with path multiplicity 1 and a manually failed hub.
Right panel: a diamond with path multiplicity 2 and a pre-specified alternate path.

The former TRAP/CLUSTER classification and causal resilience interpretation
are retired. This construction is an illustrative topology comparison, not
measured cascade evidence, a classifier, or design guidance.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── Style (minimal — schematic, not data plot) ─────────────────────
try:
    from tin_figure_style import FIG_DIR, apply_style, figsize_double, save_fig

    apply_style("pre")
    USE_STYLE = True
except Exception:
    USE_STYLE = False
    FIG_DIR = Path(__file__).parent.parent / "figures"

# ── Palette ────────────────────────────────────────────────────────
C = {
    "active": "#4477AA",
    "failed": "#CC3311",
    "cascaded": "#EE7733",
    "rerouted": "#009988",
    "edge_ok": "#4477AA",
    "edge_brk": "#CC3311",
    "edge_re": "#009988",
    "text": "#333333",
    "bg": "#F8F8F8",
}

NODE_RADIUS = 0.055


def _draw_node(ax, x, y, label, status, sublabel=None):
    """Draw a node as a filled circle with centered label."""
    fc = C[status]
    ec = {"active": "#2255AA", "failed": "#881100", "cascaded": "#BB5500", "rerouted": "#007766"}[
        status
    ]
    circle = plt.Circle((x, y), NODE_RADIUS, fc=fc, ec=ec, lw=1.3, zorder=10, clip_on=False)
    ax.add_patch(circle)

    # Node label (inside circle)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6,
        color="white",
        fontweight="bold",
        zorder=12,
    )

    # Sub-label (below circle)
    if sublabel:
        ax.text(
            x,
            y - NODE_RADIUS - 0.045,
            sublabel,
            ha="center",
            va="top",
            fontsize=5,
            color=C["text"],
            zorder=12,
        )

    # Failed X overlay
    if status == "failed":
        s = NODE_RADIUS * 0.6
        ax.plot(
            [x - s, x + s],
            [y - s, y + s],
            "-",
            color="white",
            lw=2.5,
            zorder=13,
            solid_capstyle="round",
        )
        ax.plot(
            [x - s, x + s],
            [y + s, y - s],
            "-",
            color="white",
            lw=2.5,
            zorder=13,
            solid_capstyle="round",
        )


def _draw_edge(ax, x0, y0, x1, y1, status, label=None):
    """Draw a directed edge as an arrow between two node centres."""
    styles = {
        "active": dict(color=C["edge_ok"], lw=1.2, ls="-"),
        "broken": dict(color=C["edge_brk"], lw=1.0, ls="--"),
        "rerouted": dict(color=C["edge_re"], lw=2.0, ls="-"),
    }
    s = styles[status]
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        shrinkA=NODE_RADIUS * 72 + 2,  # points (72 pt/in, data≈inches here)
        shrinkB=NODE_RADIUS * 72 + 2,
        zorder=5,
        **s,
    )
    ax.add_patch(arrow)

    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(
            mx,
            my + 0.04,
            label,
            fontsize=5,
            color=s["color"],
            ha="center",
            va="bottom",
            fontstyle="italic",
            zorder=8,
        )


def _setup_ax(ax, title):
    """Configure axes for schematic drawing."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8, pad=8)


# ── Figure ─────────────────────────────────────────────────────────
if USE_STYLE:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double("pre", height_ratio=0.40))
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
fig.subplots_adjust(wspace=0.30, top=0.80)
fig.suptitle(
    "Historical topology schematic — not classifier or resilience validation",
    fontsize=9,
    y=0.96,
)

# ═══════════════════════════════════════════════════════════════════
# LEFT PANEL — serial-chain construction
# ═══════════════════════════════════════════════════════════════════
_setup_ax(ax1, "Historical serial-chain example ($m_v = 1$)")

# Nodes
_draw_node(ax1, 0.08, 0.50, "S", "active", "Source")
_draw_node(ax1, 0.35, 0.50, "A", "active", "Hub A")
_draw_node(ax1, 0.62, 0.50, "B", "failed", "Hub B")
_draw_node(ax1, 0.92, 0.50, "D", "cascaded", "Dest")

# Edges
_draw_edge(ax1, 0.08, 0.50, 0.35, 0.50, "active")
_draw_edge(ax1, 0.35, 0.50, 0.62, 0.50, "active")
_draw_edge(ax1, 0.62, 0.50, 0.92, 0.50, "broken")

# Illustrative downstream-loss annotation
ax1.annotate(
    "modeled loss",
    xy=(0.82, 0.50),
    xytext=(0.77, 0.78),
    fontsize=6,
    color=C["cascaded"],
    ha="center",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["cascaded"], lw=1.0, shrinkB=5),
)

# Bottom text
ax1.text(
    0.50,
    -0.08,
    "Illustrative outcome with no alternate path in this construction",
    ha="center",
    va="top",
    fontsize=5.5,
    color="#888888",
)

# ═══════════════════════════════════════════════════════════════════
# RIGHT PANEL — diamond construction
# ═══════════════════════════════════════════════════════════════════
_setup_ax(ax2, "Historical diamond example ($m_v = 2$)")

# Nodes
_draw_node(ax2, 0.08, 0.50, "S", "active", "Source")
_draw_node(ax2, 0.42, 0.78, "A", "failed", "Hub A")
_draw_node(ax2, 0.42, 0.22, "B", "active", "Hub B")
_draw_node(ax2, 0.92, 0.50, "D", "active", "Dest")

# Edges — primary path broken
_draw_edge(ax2, 0.08, 0.50, 0.42, 0.78, "broken")
_draw_edge(ax2, 0.42, 0.78, 0.92, 0.50, "broken")

# Edges — alternate path carries traffic
_draw_edge(ax2, 0.08, 0.50, 0.42, 0.22, "rerouted")
_draw_edge(ax2, 0.42, 0.22, 0.92, 0.50, "rerouted", "rerouted")

# Illustrative alternate-route annotation
ax2.annotate(
    "modeled reroute",
    xy=(0.92, 0.58),
    xytext=(0.80, 0.88),
    fontsize=6,
    color=C["rerouted"],
    ha="center",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["rerouted"], lw=1.0, shrinkB=5),
)

# Bottom text
ax2.text(
    0.50,
    -0.08,
    "Illustrative outcome with the alternate path pre-specified",
    ha="center",
    va="top",
    fontsize=5.5,
    color="#888888",
)

# ── Save ───────────────────────────────────────────────────────────
if USE_STYLE:
    save_fig(fig, "itn_cascade_propagation")
else:
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(str(FIG_DIR / "itn_cascade_propagation.png"), dpi=200, bbox_inches="tight")
    fig.savefig(str(FIG_DIR / "itn_cascade_propagation.pdf"), bbox_inches="tight")
    print(f"  Wrote {FIG_DIR / 'itn_cascade_propagation.png'}")
    print(f"  Wrote {FIG_DIR / 'itn_cascade_propagation.pdf'}")
plt.close(fig)
print("Done.")
