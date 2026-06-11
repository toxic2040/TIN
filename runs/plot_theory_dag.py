#!/usr/bin/env python3
"""Diagram 1: Theory Dependency Tree.

Structured top-to-bottom like a call graph — axiom subsets produce
results, results branch into theorems/discoveries/design rules.
No diagonal arrows. Reads like code.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent.parent / "figures"

# ── Palette (Paul Tol, colorblind-safe) ────────────────────────────
C = {
    "axiom": "#CC3311",
    "identity": "#EE7733",
    "theorem": "#CCBB44",
    "discovery": "#4477AA",
    "design": "#009988",
    "negative": "#AA3377",
    "connector": "#AAAAAA",
    "bg": "#FAFAFA",
}

# ── Tree structure ─────────────────────────────────────────────────
# Each trunk starts from an axiom subset and branches downward.
# Format: list of (indent, label, type, paper_tag)
# indent=0 is a trunk root, indent=1+ are children.

TREE = [
    # ── Trunk 1: Temporal structure ────────────────────────────────
    (0, "Ax1 (Discrete Contacts) + Ax2 (Directed Time)", "axiom", ""),
    (1, "S_T  — temporal reachability", "identity", "P1"),
    (2, "S_T monotone  (Braess cannot occur in S_T)", "theorem", "P1, GT 34"),
    (2, "Percolation threshold  (phase transition at critical fleet)", "theorem", "P1"),
    (2, "N_eff catastrophe predictor  (AUC = 1.0)", "discovery", "P2"),
    (3, "C1: Support check  (n_src >= 2 at every node)", "design", "P3"),
    (2, "Layer 0.5 dominance  (all S_T collapses trace here)", "discovery", "P2"),
    (3, "Pure temporal composition failure NOT OBSERVED  (344 expts)", "negative", "P2"),
    # spacer
    (-1, "", "", ""),
    # ── Trunk 2: Transport physics ─────────────────────────────────
    (0, "Ax3 (Custody) + Ax4 (Path Composition) + Ax5 (Decay)", "axiom", ""),
    (1, "eta  — transport efficiency", "identity", "P1"),
    (2, "Self-averaging  (eta_OPSP matches MC, CV < 0.4%)", "theorem", "P1"),
    (2, "var_log_p -> gamma  (R^2 = 0.903, structural law)", "discovery", "P2"),
    (3, "Oracle annihilation  (gamma_retry -> 0 on 7/8 bodies)", "discovery", "P2"),
    (4, "Venus anomaly  (oracle WORSENS trap, dur_tail_ratio)", "negative", "P2"),
    (3, "C3: Trap severity screening  (rank by sigma^2_t)", "design", "P3"),
    (2, "Convex hull  (affine parametric shortest paths)", "theorem", "P3"),
    (3, "Commodity-dependent effective graphs  (hull facets)", "discovery", "P3"),
    (4, "C2: Effective graph per commodity", "design", "P3"),
    (2, "TRAP / CLUSTER classification  (gamma, zero overlap)", "discovery", "P1"),
    (3, "Achievability gap  (Moon 35.5x, DP vs greedy)", "discovery", "P1"),
    (3, "Forwarding ratio boundary  (f_fwd = 1.00 exact)", "discovery", "P1"),
    # spacer
    (-1, "", "", ""),
    # ── Trunk 3: The factorization ─────────────────────────────────
    (0, "S_T  +  eta  (from trunks above)", "identity", ""),
    (1, "DR = S_T x eta  — conditional-probability identity", "identity", "P1"),
    (2, "Braess localization  (Braess lives entirely in eta)", "theorem", "P1"),
    (2, "DR monotone despite eta dip  (91,540 configs)", "discovery", "P2"),
    (2, "J_beta = 0.242 +/- 0.001  (TASEP saturation invariance)", "discovery", "P4"),
    (3, "Three-phase structure  (LD, HD, MC)", "discovery", "P4"),
    (3, "Golden-mean hypothesis FALSIFIED", "negative", "P4"),
    (2, "C4: Self-consistency  (finite-campaign sustainability)", "design", "P3"),
    (3, "C2/C4 FAIL on EMJ  (Jupiter unsustainable at all tau)", "negative", "P3"),
    # spacer
    (-1, "", "", ""),
    # ── Trunk 4: One-tau (fewer axioms needed) ─────────────────────
    (0, "Ax1 + Ax3 + Ax5  only  (Ax2, Ax4 NOT required)", "axiom", ""),
    (1, "One-tau sustainability threshold", "theorem", "Theory"),
    (2, "Universality  (DTN, kinetics, cold chain, biology, neural)", "discovery", "Theory"),
    (2, "EMJ: Mars-Jupiter coast at 4.34 tau  (non-perturbative)", "discovery", "P3"),
]


def draw():
    # Layout constants
    x_indent = 0.45  # horizontal indent per level
    x_base = 0.8  # leftmost text position
    y_step = 0.38  # vertical spacing
    tag_x = 11.5  # right column for paper tags
    connector_x_off = 0.2  # connector line offset from text

    n_rows = len(TREE)
    fig_h = max(n_rows * y_step + 1.5, 6)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, n_rows * y_step + 1.0)
    ax.invert_yaxis()
    ax.axis("off")

    # Title
    ax.text(
        6.5,
        0.15,
        "TIN Theory Dependency Tree",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    # Column headers
    header_y = 0.55
    ax.text(x_base, header_y, "Structure", fontsize=8, fontweight="bold", color="#666666")
    ax.text(tag_x, header_y, "Paper", fontsize=8, fontweight="bold", color="#666666", ha="center")
    ax.axhline(y=header_y + 0.15, xmin=0.04, xmax=0.96, color="#DDDDDD", lw=0.5)

    y = 0.85
    # Track the x position of each indent level for connector lines
    # We draw vertical connector lines for parent-child relationships
    trunk_lines = {}  # indent -> (x, y_start)

    for i, (indent, label, ntype, tag) in enumerate(TREE):
        if indent == -1:
            # Spacer
            y += y_step * 0.4
            trunk_lines.clear()
            continue

        x = x_base + indent * x_indent
        color = C.get(ntype, "#333333")

        # ── Connector lines ────────────────────────────────────────
        cx = x - connector_x_off
        if indent > 0:
            # Horizontal tick from connector to text
            ax.plot(
                [cx, cx + connector_x_off * 0.7],
                [y, y],
                color=C["connector"],
                lw=0.8,
                solid_capstyle="round",
            )

            # Vertical line from parent
            parent_indent = indent - 1
            if parent_indent in trunk_lines:
                px, py = trunk_lines[parent_indent]
                # Extend vertical line down to this row
                ax.plot([cx, cx], [py, y], color=C["connector"], lw=0.8, solid_capstyle="round")

        # Update trunk tracking: this node's children will connect here.
        # Offset y_start downward so the vertical line doesn't touch
        # the parent's text/marker.
        trunk_lines[indent] = (cx + x_indent, y + y_step * 0.35)
        # Clear deeper levels (they belong to a previous branch)
        for k in list(trunk_lines):
            if k > indent:
                del trunk_lines[k]

        # ── Type marker (colored dot) ──────────────────────────────
        marker_x = x - 0.08
        if ntype == "axiom":
            ax.plot(marker_x, y, "s", color=color, markersize=5, zorder=5)
        elif ntype == "negative":
            ax.plot(marker_x, y, "X", color=color, markersize=5, zorder=5, markeredgewidth=1.5)
        else:
            ax.plot(marker_x, y, "o", color=color, markersize=4, zorder=5)

        # ── Label ──────────────────────────────────────────────────
        fs = 7.5 if indent == 0 else 7 if indent == 1 else 6.5
        fw = "bold" if indent <= 1 else "normal"
        style = "italic" if ntype == "negative" else "normal"
        ax.text(
            x + 0.05,
            y,
            label,
            fontsize=fs,
            fontweight=fw,
            fontstyle=style,
            color=color,
            va="center",
        )

        # ── Paper tag (right column) ───────────────────────────────
        if tag:
            ax.text(
                tag_x,
                y,
                tag,
                fontsize=5.5,
                color="#888888",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="#F0F0F0",
                    edgecolor="#DDDDDD",
                    linewidth=0.3,
                ),
            )

        y += y_step

    # ── Legend ──────────────────────────────────────────────────────
    legend_items = [
        ("square", "axiom", "Axiom subset"),
        ("o", "identity", "Identity"),
        ("o", "theorem", "Theorem"),
        ("o", "discovery", "Discovery"),
        ("o", "design", "Design rule"),
        ("X", "negative", "Negative result"),
    ]
    ly = y + 0.2
    lx = 1.5
    for marker, ckey, label in legend_items:
        if marker == "square":
            ax.plot(lx, ly, "s", color=C[ckey], markersize=5)
        elif marker == "X":
            ax.plot(lx, ly, "X", color=C[ckey], markersize=5, markeredgewidth=1.5)
        else:
            ax.plot(lx, ly, "o", color=C[ckey], markersize=4)
        ax.text(lx + 0.15, ly, label, fontsize=6, color=C[ckey], va="center")
        lx += 1.8

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(str(FIG_DIR / "theory_dag.png"), dpi=200, bbox_inches="tight")
    fig.savefig(str(FIG_DIR / "theory_dag.pdf"), bbox_inches="tight")
    print(f"Wrote {FIG_DIR / 'theory_dag.png'}")
    print(f"Wrote {FIG_DIR / 'theory_dag.pdf'}")
