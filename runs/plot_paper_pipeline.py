#!/usr/bin/env python3
"""Diagram 3: Paper Pipeline — four trees showing what each paper
produces and what it consumes from prior papers.

Same tree-connector style as the theory DAG. No arrows, no clutter.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).parent.parent / "figures"

# ── Palette ────────────────────────────────────────────────────────
C = {
    "P1": "#4477AA",
    "P2": "#CC3311",
    "P3": "#EE7733",
    "P4": "#009988",
    "produces": "#333333",
    "consumes": "#AA3377",
    "connector": "#AAAAAA",
    "status_pub": "#009988",
    "status_dft": "#CCBB44",
}

# ── Paper definitions ──────────────────────────────────────────────
# (indent, label, type)
# type: "title", "status", "produces", "consumes", "key_number"

PAPERS = [
    # ── Paper 1 ────────────────────────────────────────────────────
    (0, "Paper 1: Classification Theorem", "title", "P1"),
    (0, "Published  —  IEEE Aerospace  —  arXiv v7.3", "status", "pub"),
    (1, "Produces:", "heading", "produces"),
    (2, "DR = S_T x eta  (conditional-probability identity)", "produces", ""),
    (2, "Three-factor sparse law  DR = S_T exp(E[H] lambda) Phi", "produces", ""),
    (2, "TRAP / CLUSTER classification  (gamma order parameter)", "produces", ""),
    (2, "Self-averaging  (eta_OPSP matches MC, CV < 0.4%)", "produces", ""),
    (2, "Braess paradox detection  (90% of Mars epochs degraded)", "produces", ""),
    (2, "155K+ configs validated  (8 orbital + 4 CRAWDAD + SF Cab)", "produces", ""),
    (1, "Consumes:", "heading", "consumes"),
    (2, "(foundational — no prior papers)", "consumes", ""),
    (-1, "", "", ""),
    # ── Paper 2 ────────────────────────────────────────────────────
    (0, "Paper 2: Mechanism Taxonomy", "title", "P2"),
    (0, "Draft  —  target Physical Review E", "status", "dft"),
    (1, "Produces:", "heading", "produces"),
    (2, "var_log_p -> gamma  (R^2 = 0.903, slope = 21.17)", "produces", ""),
    (2, "Oracle annihilation  (gamma_retry -> 0 on 7/8 bodies)", "produces", ""),
    (2, "Four-layer diagnostic hierarchy  (L0 - L0.5 - L1 - L2)", "produces", ""),
    (2, "Two-level heterogeneity mechanism  (greedy + oracle traps)", "produces", ""),
    (2, "Venus anomaly  (dur_tail_ratio defeats oracle)", "produces", ""),
    (2, "Layer 0.5 dominance  (344 searches, pure L1 never observed)", "produces", ""),
    (2, "Decomposition discipline  (rule out upstream before downstream)", "produces", ""),
    (1, "Consumes from P1:", "heading", "consumes"),
    (2, "gamma classification", "consumes", ""),
    (2, "Three-factor decomposition  (Phi -> gamma extraction)", "consumes", ""),
    (-1, "", "", ""),
    # ── Paper 3 ────────────────────────────────────────────────────
    (0, "Paper 3: Relay Architectures", "title", "P3"),
    (0, "Draft  —  target IEEE Aerospace", "status", "dft"),
    (1, "Produces:", "heading", "produces"),
    (2, "Percolation thresholds per orbit family  (NRHO n_c=5, polar n_c=1)", "produces", ""),
    (2, "Braess selection rule  (90/1386 configs show paradox)", "produces", ""),
    (2, "NRHO parity resonance  (even constellations 1.30x better)", "produces", ""),
    (2, "Heritage fleet 35% suboptimality  (0.52 vs 0.71 optimal)", "produces", ""),
    (2, "Topology dominates band  (S_T invariant UHF through optical)", "produces", ""),
    (2, "Mixed architectures eliminate Braess  (Phi = 1.00)", "produces", ""),
    (2, "Pipeline conditions C1-C4  (simulation-free screening)", "produces", ""),
    (1, "Consumes from P1:", "heading", "consumes"),
    (2, "S_T monotonicity theorem", "consumes", ""),
    (2, "Braess detection framework", "consumes", ""),
    (1, "Consumes from P2:", "heading", "consumes"),
    (2, "N_eff catastrophe predictor", "consumes", ""),
    (2, "var_log_p trap screening", "consumes", ""),
    (-1, "", "", ""),
    # ── Paper 4 ────────────────────────────────────────────────────
    (0, "Paper 4: TASEP Phase Structure", "title", "P4"),
    (0, "Draft  —  target J. Comp. Phys.", "status", "dft"),
    (1, "Produces:", "heading", "produces"),
    (2, "J_beta = 0.242 +/- 0.001  (saturation invariance, 1.1% spread)", "produces", ""),
    (2, "Three-phase structure  (LD, HD, MC observed in DTN)", "produces", ""),
    (2, "Golden-mean hypothesis FALSIFIED  (rho = -0.50, p = 0.39)", "produces", ""),
    (2, "Altitude primacy  (rho = +0.66, p = 0.003)", "produces", ""),
    (2, "Mars pre-jamming  (no LD phase, boots into HD)", "produces", ""),
    (2, "Eccentricity as design lever  (ELFO +50% K_eff bonus)", "produces", ""),
    (1, "Consumes from P1:", "heading", "consumes"),
    (2, "Three-factor decomposition", "consumes", ""),
    (2, "eta + routing efficiency definition", "consumes", ""),
    (1, "Consumes from P2:", "heading", "consumes"),
    (2, "Trap mechanism  (var_log_p explains per-lane ceiling)", "consumes", ""),
]


def draw():
    x_indent = 0.35
    x_base = 0.7
    y_step = 0.34
    connector_x_off = 0.18

    n_rows = len(PAPERS)
    fig_h = max(n_rows * y_step + 1.5, 6)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, n_rows * y_step + 1.0)
    ax.invert_yaxis()
    ax.axis("off")

    # Title
    ax.text(
        5.5, 0.15, "TIN Paper Pipeline", ha="center", va="center", fontsize=13, fontweight="bold"
    )

    y = 0.65
    trunk_lines = {}

    for i, (indent, label, ntype, extra) in enumerate(PAPERS):
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
            # Background bar
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

        elif ntype == "status":
            color = C["status_pub"] if extra == "pub" else C["status_dft"]
            ax.text(
                x + 0.2,
                y,
                label,
                fontsize=6,
                fontweight="bold",
                color=color,
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor=color,
                    alpha=0.12,
                    edgecolor=color,
                    linewidth=0.4,
                ),
            )

        elif ntype == "heading":
            color = C["produces"] if extra == "produces" else C["consumes"]
            ax.text(x + 0.05, y, label, fontsize=6.5, fontweight="bold", color=color, va="center")
            # Reset trunk tracking for this heading's children
            trunk_lines[indent] = (x - connector_x_off + x_indent, y + y_step * 0.35)

        elif ntype == "produces":
            ax.plot(x - 0.08, y, "o", color=C["produces"], markersize=3, zorder=5)
            ax.text(x + 0.05, y, label, fontsize=6, color=C["produces"], va="center")

        elif ntype == "consumes":
            ax.plot(x - 0.08, y, "o", color=C["consumes"], markersize=3, zorder=5)
            ax.text(
                x + 0.05, y, label, fontsize=6, color=C["consumes"], va="center", fontstyle="italic"
            )

        y += y_step

    # ── Legend ──────────────────────────────────────────────────────
    ly = y + 0.15
    items = [
        ("s", C["P1"], "Paper 1"),
        ("s", C["P2"], "Paper 2"),
        ("s", C["P3"], "Paper 3"),
        ("s", C["P4"], "Paper 4"),
        ("o", C["produces"], "Produces"),
        ("o", C["consumes"], "Consumes"),
    ]
    lx = 1.0
    for marker, color, label in items:
        ax.plot(lx, ly, marker, color=color, markersize=4 if marker == "o" else 5, zorder=5)
        ax.text(lx + 0.15, ly, label, fontsize=6, color=color, va="center")
        lx += 1.6

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(str(FIG_DIR / "paper_pipeline.png"), dpi=200, bbox_inches="tight")
    fig.savefig(str(FIG_DIR / "paper_pipeline.pdf"), bbox_inches="tight")
    print(f"Wrote {FIG_DIR / 'paper_pipeline.png'}")
    print(f"Wrote {FIG_DIR / 'paper_pipeline.pdf'}")
