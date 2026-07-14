#!/usr/bin/env python3
"""Diagram 1: historical theory dependency tree.

Structured top-to-bottom like a call graph. Historical premise sets lead to
archived claim branches and recorded negative results.
No diagonal arrows. Reads like code.

This is a map of the historical program, including retired claim branches. No
branch is presented as a current theorem, mechanism, predictor, universality
result, or design rule.
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
    "archived": "#4477AA",
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
    (0, "Historical premises: discrete contacts + directed time", "axiom", ""),
    (1, "S_T  — temporal reachability", "identity", "P1"),
    (2, "Archived S_T monotonicity claim  (not reasserted)", "archived", "P1, GT 34"),
    (2, "Retired critical-fleet phase-transition claim", "negative", "P1"),
    (2, "Archived N_eff AUC = 1.0 diagnostic  (predictor claim retired)", "negative", "P2"),
    (3, "Archived support check: n_src >= 2  (not design guidance)", "archived", "P3"),
    (2, "Archived Layer 0.5 association  (dominance claim retired)", "negative", "P2"),
    (3, "No pure temporal-composition failure in 344 archived tests", "archived", "P2"),
    # spacer
    (-1, "", "", ""),
    # ── Trunk 2: Transport physics ─────────────────────────────────
    (0, "Historical premises: custody + path composition + decay", "axiom", ""),
    (1, "eta  — transport efficiency", "identity", "P1"),
    (2, "Archived self-averaging comparison  (scoped tested regime)", "archived", "P1"),
    (2, "Archived var_log_p/gamma association  (R^2 = 0.903; law claim retired)", "negative", "P2"),
    (3, "Archived retry-slope near-zero count  (7/8 bodies; no causal claim)", "archived", "P2"),
    (4, "Archived Venus retry-slope comparison", "archived", "P2"),
    (3, "Archived sigma^2_t ranking proposal  (not design guidance)", "archived", "P3"),
    (2, "Archived affine shortest-path construction", "archived", "P3"),
    (3, "Archived commodity-dependent graph construction", "archived", "P3"),
    (4, "Archived per-commodity graph proposal  (not design guidance)", "archived", "P3"),
    (2, "TRAP / CLUSTER global threshold  (retired)", "negative", "P1"),
    (3, "Archived Moon DP/greedy ratio: 35.5x", "archived", "P1"),
    (3, "Archived forwarding-ratio reference: f_fwd = 1.00", "archived", "P1"),
    # spacer
    (-1, "", "", ""),
    # ── Trunk 3: The factorization ─────────────────────────────────
    (0, "S_T  +  eta  (from trunks above)", "identity", ""),
    (1, "DR = S_T x eta  — conditional-probability identity", "identity", "P1"),
    (2, "Archived Braess decomposition  (causal localization retired)", "negative", "P1"),
    (2, "Archived DR/eta sample summary  (91,540 rows)", "archived", "P2"),
    (2, "Archived J_beta fit: 0.242 +/- 0.001  (invariance claim retired)", "negative", "P4"),
    (3, "Archived LD / HD / MC labels  (descriptive only)", "archived", "P4"),
    (3, "Archived golden-mean test: hypothesis not supported in tested rows", "negative", "P4"),
    (2, "Archived C4 finite-campaign checklist  (not operational guidance)", "archived", "P3"),
    (3, "Archived EMJ model-band comparison  (no mission verdict)", "negative", "P3"),
    # spacer
    (-1, "", "", ""),
    # ── Trunk 4: One-tau (fewer axioms needed) ─────────────────────
    (0, "Historical reduced premise set: Ax1 + Ax3 + Ax5", "axiom", ""),
    (1, "Archived one-tau reference line  (threshold claim retired)", "negative", "Theory"),
    (2, "Retired cross-domain universality analogy", "negative", "Theory"),
    (2, "Archived EMJ modeled ratio: 4.34 tau  (no feasibility inference)", "archived", "P3"),
]


def draw():
    # Layout constants
    x_indent = 0.45  # horizontal indent per level
    x_base = 0.8  # leftmost text position
    y_step = 0.38  # vertical spacing
    tag_x = 11.5  # right column for paper tags
    connector_x_off = 0.2  # connector line offset from text

    n_rows = len(TREE)
    fig_h = max(n_rows * y_step + 2.0, 6)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, n_rows * y_step + 1.4)
    ax.invert_yaxis()
    ax.axis("off")

    # Title
    ax.text(
        6.5,
        0.15,
        "TIN Historical Theory Dependency Tree",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        6.5,
        0.43,
        "Archived claim labels only — no theorem, predictor, mechanism, universality, or design rule is current",
        ha="center",
        va="center",
        fontsize=7,
        color="#666666",
        fontstyle="italic",
    )

    # Column headers
    header_y = 0.72
    ax.text(x_base, header_y, "Structure", fontsize=8, fontweight="bold", color="#666666")
    ax.text(tag_x, header_y, "Paper", fontsize=8, fontweight="bold", color="#666666", ha="center")
    ax.axhline(y=header_y + 0.15, xmin=0.04, xmax=0.96, color="#DDDDDD", lw=0.5)

    y = 1.02
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
        ("square", "axiom", "Historical premise set"),
        ("o", "identity", "Identity / bookkeeping"),
        ("o", "archived", "Archived claim branch"),
        ("X", "negative", "Retired / negative branch"),
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
