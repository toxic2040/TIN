#!/usr/bin/env python3
"""Diagram 2: Layer Stack Architecture.

Tree-connector style matching the theory DAG and paper pipeline.
Each layer is a trunk with its controls, paper coverage, and
honest negatives as branches.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).parent.parent / "figures"

# ── Palette ────────────────────────────────────────────────────────
C = {
    "L-1": "#CC3311",
    "L0": "#EE7733",
    "L0.5": "#CCBB44",
    "L1": "#4477AA",
    "L2": "#33BBEE",
    "L3": "#009988",
    "L4": "#AA3377",
    "L4.5": "#EE3377",
    "L5": "#4477AA",
    "L5.5": "#BBBBBB",
    "controls": "#333333",
    "paper": "#888888",
    "negative": "#AA3377",
    "connector": "#AAAAAA",
}

# ── Layer tree ─────────────────────────────────────────────────────
# (indent, label, type, extra)
# type: "title", "controls", "paper", "negative"

LAYERS = [
    # ── Layer -1 ───────────────────────────────────────────────────
    (0, "Layer -1: Percolation Foundation", "title", "L-1"),
    (1, "Controls:", "heading", "controls"),
    (2, "Temporal reachability S_T", "controls", ""),
    (2, "Phase transitions in network connectivity", "controls", ""),
    (2, "Percolation threshold detection", "controls", ""),
    (1, "Papers: P1", "paper", ""),
    (-1, "", "", ""),
    # ── Layer 0 ────────────────────────────────────────────────────
    (0, "Layer 0: Commodity Physics", "title", "L0"),
    (1, "Controls:", "heading", "controls"),
    (2, "Decay models  (exponential, Weibull, Arrhenius)", "controls", ""),
    (2, "One-tau sustainability threshold  (Ax1+Ax3+Ax5 only)", "controls", ""),
    (2, "Effective graph pruning  (commodity-dependent topology)", "controls", ""),
    (1, "Papers: P1, P3", "paper", ""),
    (-1, "", "", ""),
    # ── Layer 0.5 ──────────────────────────────────────────────────
    (0, "Layer 0.5: Coverage & Geometric Support", "title", "L0.5"),
    (1, "Controls:", "heading", "controls"),
    (2, "N_eff_src universal predictor  (AUC = 1.0)", "controls", ""),
    (2, "Catastrophe thresholds  (n_src >= 2 at every node)", "controls", ""),
    (2, "ALL S_T collapses trace here  (dominant layer)", "controls", ""),
    (1, "Papers: P2, P3", "paper", ""),
    (1, "Negative:", "heading", "negative"),
    (2, "Pure temporal composition failure NOT OBSERVED  (344 experiments)", "negative", ""),
    (-1, "", "", ""),
    # ── Layer 1 ────────────────────────────────────────────────────
    (0, "Layer 1: Temporal Composition", "title", "L1"),
    (1, "Controls:", "heading", "controls"),
    (2, "Time-respecting path existence", "controls", ""),
    (2, "Phase-class structure  (RAAN coloring)", "controls", ""),
    (2, "Contact graph reachability", "controls", ""),
    (1, "Papers: P2", "paper", ""),
    (-1, "", "", ""),
    # ── Layer 2 ────────────────────────────────────────────────────
    (0, "Layer 2: Routing Policy & Myopia", "title", "L2"),
    (1, "Controls:", "heading", "controls"),
    (2, "TRAP vs CLUSTER classification  (gamma order parameter)", "controls", ""),
    (2, "var_log_p mechanism  (R^2 = 0.903, structural law)", "controls", ""),
    (2, "Greedy vs oracle behaviour  (foresight dominance)", "controls", ""),
    (2, "Dead-end routing failures", "controls", ""),
    (1, "Papers: P2", "paper", ""),
    (1, "Negative:", "heading", "negative"),
    (2, "Hub fragility H does NOT classify  (range 2.1-2.8, no separation)", "negative", ""),
    (-1, "", "", ""),
    # ── Layer 3 ────────────────────────────────────────────────────
    (0, "Layer 3: Within-Path Covariance", "title", "L3"),
    (1, "Controls:", "heading", "controls"),
    (2, "Hop count <-> link quality interaction", "controls", ""),
    (2, "Foresight dominance  (DP beats greedy 9x)", "controls", ""),
    (2, "Oracle annihilation  (gamma_retry -> 0 on 7/8 bodies)", "controls", ""),
    (1, "Papers: P2", "paper", ""),
    (1, "Negative:", "heading", "negative"),
    (2, "Venus anomaly: oracle WORSENS trap  (dur_tail_ratio mechanism)", "negative", ""),
    (-1, "", "", ""),
    # ── Layer 4 ────────────────────────────────────────────────────
    (0, "Layer 4: Multicommodity & Binding Constraint", "title", "L4"),
    (1, "Controls:", "heading", "controls"),
    (2, "Convex hull  (affine parametric shortest paths)", "controls", ""),
    (2, "Commodity-dependent effective graphs  (hull facets)", "controls", ""),
    (2, "Softmax temperature from margin  (beta ~ 1/sigma)", "controls", ""),
    (2, "Binding commodity identification  (propellant, always)", "controls", ""),
    (1, "Papers: P3", "paper", ""),
    (-1, "", "", ""),
    # ── Layer 4.5 ──────────────────────────────────────────────────
    (0, "Layer 4.5: Self-Consistency & Feedback", "title", "L4.5"),
    (1, "Controls:", "heading", "controls"),
    (2, "Pipeline stability conditions C1-C4", "controls", ""),
    (2, "Cascade percolation threshold  (m_v >= 2 required)", "controls", ""),
    (2, "Finite-campaign sustainability  (Bernoulli reserve model)", "controls", ""),
    (1, "Papers: P3", "paper", ""),
    (1, "Negative:", "heading", "negative"),
    (2, "C2/C4 FAIL on EMJ  (Jupiter unsustainable at all tau_half)", "negative", ""),
    (-1, "", "", ""),
    # ── Layer 5 ────────────────────────────────────────────────────
    (0, "Layer 5: Extended First-Principles", "title", "L5"),
    (1, "Controls:", "heading", "controls"),
    (2, "Irreversibility premium  (regret scales 330x at Jupiter)", "controls", ""),
    (2, "Bidirectional DR  (crew safety, return path binding)", "controls", ""),
    (2, "Cascade-adjusted resilience  (buffer sizing)", "controls", ""),
    (1, "Papers: P3", "paper", ""),
    (-1, "", "", ""),
    # ── Layer 5.5 ──────────────────────────────────────────────────
    (0, "Layer 5.5: Statistical Mechanics Lens", "title", "L5.5"),
    (1, "Controls:", "heading", "controls"),
    (2, "Partition function correspondence  (structural, not physical)", "controls", ""),
    (2, "Free energy landscape  (V_beta = (1/beta) log Z)", "controls", ""),
    (2, "Fluctuation-dissipation test  (UNTESTED, make-or-break)", "controls", ""),
    (1, "Papers: --  (theory notes only)", "paper", ""),
    (1, "Negative:", "heading", "negative"),
    (
        2,
        "Correspondence is structural, not thermodynamic  (no thermal fluctuations)",
        "negative",
        "",
    ),
]


def draw():
    x_indent = 0.35
    x_base = 0.7
    y_step = 0.34
    connector_x_off = 0.18

    n_rows = len(LAYERS)
    fig_h = max(n_rows * y_step + 1.5, 6)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, n_rows * y_step + 1.0)
    ax.invert_yaxis()
    ax.axis("off")

    # Title
    ax.text(
        5.5,
        0.15,
        "TIN Layer Architecture",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        5.5,
        0.42,
        "increasing abstraction top to bottom",
        ha="center",
        va="center",
        fontsize=7,
        color="#888888",
        fontstyle="italic",
    )

    y = 0.7
    trunk_lines = {}

    for i, (indent, label, ntype, extra) in enumerate(LAYERS):
        if indent == -1:
            y += y_step * 0.5
            trunk_lines.clear()
            continue

        x = x_base + indent * x_indent

        # ── Connector lines ────────────────────────────────────────
        if indent > 0 and ntype not in ("heading",):
            cx = x - connector_x_off
            ax.plot(
                [cx, cx + connector_x_off * 0.7],
                [y, y],
                color=C["connector"],
                lw=0.8,
                solid_capstyle="round",
            )
            parent_indent = indent - 1
            if parent_indent in trunk_lines:
                px, py = trunk_lines[parent_indent]
                ax.plot([cx, cx], [py, y], color=C["connector"], lw=0.8, solid_capstyle="round")

        # Track for children
        if ntype not in ("heading",):
            trunk_lines[indent] = (x - connector_x_off + x_indent, y + y_step * 0.35)
            for k in list(trunk_lines):
                if k > indent:
                    del trunk_lines[k]

        # ── Render by type ─────────────────────────────────────────
        if ntype == "title":
            color = C.get(extra, "#333333")
            rect = FancyBboxPatch(
                (0.2, y - 0.14),
                10.5,
                0.28,
                boxstyle="round,pad=0.03",
                facecolor=color,
                alpha=0.10,
                edgecolor=color,
                linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.plot(x - 0.12, y, "s", color=color, markersize=6, zorder=5)
            ax.text(x + 0.05, y, label, fontsize=8.5, fontweight="bold", color=color, va="center")

        elif ntype == "heading":
            color = C.get(extra, "#333333")
            ax.text(x + 0.05, y, label, fontsize=6.5, fontweight="bold", color=color, va="center")
            trunk_lines[indent] = (x - connector_x_off + x_indent, y + y_step * 0.35)

        elif ntype == "controls":
            ax.plot(x - 0.08, y, "o", color=C["controls"], markersize=3, zorder=5)
            ax.text(x + 0.05, y, label, fontsize=6, color=C["controls"], va="center")

        elif ntype == "paper":
            ax.text(
                x + 0.05,
                y,
                label,
                fontsize=6,
                color=C["paper"],
                va="center",
                fontstyle="italic",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="#F0F0F0",
                    edgecolor="#DDDDDD",
                    linewidth=0.3,
                ),
            )

        elif ntype == "negative":
            ax.plot(
                x - 0.08, y, "X", color=C["negative"], markersize=4, zorder=5, markeredgewidth=1.2
            )
            ax.text(
                x + 0.05, y, label, fontsize=6, color=C["negative"], va="center", fontstyle="italic"
            )

        y += y_step

    # ── Legend ──────────────────────────────────────────────────────
    ly = y + 0.15
    items = [
        ("o", C["controls"], "Controls"),
        ("X", C["negative"], "Negative result"),
    ]
    lx = 1.0
    for marker, color, label in items:
        ax.plot(
            lx,
            ly,
            marker,
            color=color,
            markersize=4,
            zorder=5,
            markeredgewidth=1.2 if marker == "X" else 0.5,
        )
        ax.text(lx + 0.15, ly, label, fontsize=6, color=color, va="center")
        lx += 2.0

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(str(FIG_DIR / "layer_stack.png"), dpi=200, bbox_inches="tight")
    fig.savefig(str(FIG_DIR / "layer_stack.pdf"), bbox_inches="tight")
    print(f"Wrote {FIG_DIR / 'layer_stack.png'}")
    print(f"Wrote {FIG_DIR / 'layer_stack.pdf'}")
