#!/usr/bin/env python3
"""Historical paper pipeline for the closed TIN claim program.

Four trees show the labels each draft used for its outputs and inputs. Every
branch is archival: no mechanism, classifier, predictor, optimality, theorem,
or design claim is current.

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
    (0, "Paper 1: Historical classification working paper", "title", "P1"),
    (0, "Zenodo working-paper deposit  —  historical record", "status", "pub"),
    (1, "Archived outputs and claims (not current):", "heading", "produces"),
    (2, "DR = S_T x eta  (conditional-probability identity)", "produces", ""),
    (2, "Three-factor bookkeeping decomposition  (law framing retired)", "produces", ""),
    (2, "TRAP / CLUSTER global classification  (retired)", "produces", ""),
    (2, "Self-averaging result  (scoped tested regime only)", "produces", ""),
    (2, "Braess observation  (35/39 tested Mars epochs)", "produces", ""),
    (2, "Historical run-count arithmetic  (not a unique-config census)", "produces", ""),
    (1, "Historical inputs:", "heading", "consumes"),
    (2, "(foundational — no prior papers)", "consumes", ""),
    (-1, "", "", ""),
    # ── Paper 2 ────────────────────────────────────────────────────
    (0, "Paper 2: Historical mechanism-taxonomy draft", "title", "P2"),
    (0, "Historical draft record  —  former target Physical Review E", "status", "dft"),
    (1, "Archived outputs and claims (not current):", "heading", "produces"),
    (2, "var_log_p / gamma pooled association  (R^2 = 0.903; law claim retired)", "produces", ""),
    (2, "Retry-slope near-zero count  (7/8 bodies; no causal annihilation)", "produces", ""),
    (2, "Four-layer diagnostic outline  (historical proposal)", "produces", ""),
    (2, "Two-level heterogeneity interpretation  (mechanism claim retired)", "produces", ""),
    (2, "Venus retry-slope comparison  (descriptive only)", "produces", ""),
    (2, "Layer 0.5 search summary  (dominance claim retired)", "produces", ""),
    (2, "Historical decomposition checklist", "produces", ""),
    (1, "Historical inputs from P1:", "heading", "consumes"),
    (2, "retired gamma classification (historical input)", "consumes", ""),
    (2, "Three-factor decomposition  (archived Phi/gamma extraction)", "consumes", ""),
    (-1, "", "", ""),
    # ── Paper 3 ────────────────────────────────────────────────────
    (0, "Paper 3: Historical relay-architecture draft", "title", "P3"),
    (0, "Historical draft record  —  former target IEEE Aerospace", "status", "dft"),
    (1, "Archived outputs and claims (not current):", "heading", "produces"),
    (2, "Critical-fleet reference counts  (threshold claim retired)", "produces", ""),
    (2, "Braess-labeled rows: 90/1386  (selection-rule claim retired)", "produces", ""),
    (2, "Even/odd NRHO comparison: 1.30x  (fleet rule retired)", "produces", ""),
    (2, "Heritage-fleet model comparison: 0.52 vs 0.71  (no optimality claim)", "produces", ""),
    (2, "S_T band comparison  (dominance/invariance claims retired)", "produces", ""),
    (2, "Mixed-architecture tested rows: Phi = 1.00  (no elimination claim)", "produces", ""),
    (2, "C1-C4 historical checklist  (not operational screening)", "produces", ""),
    (1, "Historical inputs from P1:", "heading", "consumes"),
    (2, "Archived S_T monotonicity claim  (not a current theorem)", "consumes", ""),
    (2, "Archived Braess diagnostic", "consumes", ""),
    (1, "Historical inputs from P2:", "heading", "consumes"),
    (2, "Retired N_eff catastrophe-predictor claim", "consumes", ""),
    (2, "Retired var_log_p trap-screening claim", "consumes", ""),
    (-1, "", "", ""),
    # ── Paper 4 ────────────────────────────────────────────────────
    (0, "Paper 4: Historical TASEP-phase draft", "title", "P4"),
    (0, "Historical draft record  —  former target J. Comp. Phys.", "status", "dft"),
    (1, "Archived outputs and claims (not current):", "heading", "produces"),
    (2, "J_beta fit: 0.242 +/- 0.001  (invariance claim retired)", "produces", ""),
    (2, "LD / HD / MC labels in archived rows  (descriptive only)", "produces", ""),
    (2, "Golden-mean tested negative: rho = -0.50, p = 0.39", "produces", ""),
    (2, "Altitude correlation: rho = +0.66, p = 0.003  (no causal primacy)", "produces", ""),
    (2, "No LD-labeled Mars rows in the archived sample", "produces", ""),
    (2, "ELFO K_eff comparison: +50%  (design-lever claim retired)", "produces", ""),
    (1, "Historical inputs from P1:", "heading", "consumes"),
    (2, "Three-factor bookkeeping decomposition", "consumes", ""),
    (2, "eta + routing efficiency definition", "consumes", ""),
    (1, "Historical inputs from P2:", "heading", "consumes"),
    (2, "var_log_p association  (mechanism and ceiling claims retired)", "consumes", ""),
]


def draw():
    x_indent = 0.35
    x_base = 0.7
    y_step = 0.34
    connector_x_off = 0.18

    n_rows = len(PAPERS)
    fig_h = max(n_rows * y_step + 2.0, 6)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, n_rows * y_step + 1.4)
    ax.invert_yaxis()
    ax.axis("off")

    # Title
    ax.text(
        5.5,
        0.15,
        "TIN Historical Paper Pipeline — Claim Program Closed",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        5.5,
        0.42,
        "Archived draft labels only — no mechanism, classifier, predictor, optimality, or design claim is current",
        ha="center",
        va="center",
        fontsize=7,
        color="#666666",
        fontstyle="italic",
    )

    y = 0.82
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
        ("o", C["produces"], "Archived output / claim"),
        ("o", C["consumes"], "Historical dependency"),
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
