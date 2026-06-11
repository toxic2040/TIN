#!/usr/bin/env python3
"""gen_emj_network.py — EMJ Interplanetary Transport Network diagram.

Top-down heliocentric view showing the 8-node Earth-Mars-Jupiter relay chain
with commodity-aware route switching: hardware takes the bypass, cryogenic
propellant takes the 7-hop relay.  Same network, different topologies.

Dark-sky aesthetic matching gen_orrery.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Orbital geometry ─────────────────────────────────────────────────
AU = 1.0
R_EARTH = 1.0 * AU
R_MARS = 1.524 * AU
R_JUPITER = 5.203 * AU

# Use log-compressed radial scale so inner system stays visible
# Map: r_visual = a * ln(1 + b * r_physical)
# Tuned so Earth ~1.5, Mars ~2.2, Jupiter ~3.8 on the canvas
SCALE_A = 2.1
SCALE_B = 1.0


def r_visual(r_phys):
    return SCALE_A * np.log(1 + SCALE_B * r_phys)


# Angular positions — maximally spread for readability
EARTH_ANGLE = np.radians(210)
MARS_ANGLE = np.radians(55)
JUPITER_ANGLE = np.radians(310)

# L4 relay: 60° ahead of Mars along its orbit
L4_ANGLE = MARS_ANGLE + np.radians(55)

# Cycler midpoints — push outward from the bodies for clarity
EM_CYCLER_ANGLE = (EARTH_ANGLE + MARS_ANGLE) / 2 + np.radians(5)
MJ_CYCLER_ANGLE = (MARS_ANGLE + JUPITER_ANGLE) / 2 + np.radians(5)

# ── Colors ───────────────────────────────────────────────────────────
BG = "#0a0e1a"
SUN_CORE = "#fff9c4"
SUN_GLOW = "#ffb74d"
EARTH_C = "#4fc3f7"
MARS_C = "#ef5350"
JUPITER_C = "#d4a574"
L2_C = "#80cbc4"
L4_C = "#ce93d8"

# Route colors — the visual headline
HARDWARE_ROUTE = "#4fc3f7"  # cool blue = hardware bypass
CRYO_ROUTE = "#ff7043"  # warm orange = cryo relay chain
SHARED_ROUTE = "#a5d6a7"  # green = shared legs
NODE_RING = "#b0bec5"
LABEL_DIM = "#78909c"
LABEL_BRIGHT = "#eceff1"


def polar_to_xy(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def draw_orbit_ring(ax, r_phys, color, alpha=0.12, lw=0.6, ls="-"):
    rv = r_visual(r_phys)
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(
        rv * np.cos(theta), rv * np.sin(theta), color=color, alpha=alpha, ls=ls, lw=lw, zorder=1
    )


def draw_glow(ax, x, y, color, radius=0.06, alpha=0.15, zorder=2):
    ax.add_patch(plt.Circle((x, y), radius, color=color, alpha=alpha, zorder=zorder))


def bezier_arc(p0, p1, bulge=0.3, n=80):
    """Quadratic Bezier curve between two points, bulging outward from origin."""
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    # Perpendicular direction (away from origin for outward bulge)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = np.sqrt(dx**2 + dy**2)
    # Normal pointing outward from center
    nx, ny = -dy / length, dx / length
    # Check if normal points outward (dot with midpoint vector)
    if mx * nx + my * ny < 0:
        nx, ny = -nx, -ny
    ctrl = (mx + bulge * length * nx, my + bulge * length * ny)
    t = np.linspace(0, 1, n)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t**2 * p1[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t**2 * p1[1]
    return x, y


def draw_route(ax, p0, p1, color, alpha=0.5, lw=1.5, ls="-", bulge=0.25, zorder=3, arrow=True):
    """Draw a curved route between two points with optional arrowhead."""
    bx, by = bezier_arc(p0, p1, bulge=bulge)
    ax.plot(bx, by, color=color, alpha=alpha, lw=lw, ls=ls, zorder=zorder)
    if arrow:
        # Arrowhead at ~85% along the curve
        idx = int(0.85 * len(bx))
        dx = bx[idx] - bx[idx - 2]
        dy = by[idx] - by[idx - 2]
        ax.annotate(
            "",
            xy=(bx[idx], by[idx]),
            xytext=(bx[idx] - dx * 0.5, by[idx] - dy * 0.5),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw * 0.8, mutation_scale=8),
            zorder=zorder + 1,
        )


def draw_straight_route(ax, p0, p1, color, alpha=0.5, lw=1.5, ls="-", zorder=3, arrow=True):
    """Draw a straight route with arrowhead."""
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, alpha=alpha, lw=lw, ls=ls, zorder=zorder)
    if arrow:
        mx = p0[0] + 0.85 * (p1[0] - p0[0])
        my = p0[1] + 0.85 * (p1[1] - p0[1])
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        ax.annotate(
            "",
            xy=(mx, my),
            xytext=(mx - dx * 0.03, my - dy * 0.03),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw * 0.8, mutation_scale=8),
            zorder=zorder + 1,
        )


def main():
    fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")

    # ── Star field ───────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_stars = 1500
    sx = rng.uniform(-6.5, 7.0, n_stars)
    sy = rng.uniform(-5.0, 5.5, n_stars)
    ss = rng.exponential(0.3, n_stars) ** 2
    sa = rng.uniform(0.12, 0.5, n_stars)
    ax.scatter(sx, sy, s=ss, c="white", alpha=sa, zorder=0, edgecolors="none")

    # ── Orbit rings ──────────────────────────────────────────────────
    draw_orbit_ring(ax, R_EARTH, EARTH_C, alpha=0.15, lw=0.7)
    draw_orbit_ring(ax, R_MARS, MARS_C, alpha=0.15, lw=0.7)
    draw_orbit_ring(ax, R_JUPITER, JUPITER_C, alpha=0.12, lw=0.6, ls="--")

    # Orbit labels
    for r_phys, label, color in [
        (R_EARTH, "1.0 AU", EARTH_C),
        (R_MARS, "1.52 AU", MARS_C),
        (R_JUPITER, "5.2 AU", JUPITER_C),
    ]:
        rv = r_visual(r_phys)
        lx = rv * np.cos(np.radians(160)) + 0.02
        ly = rv * np.sin(np.radians(160)) + 0.08
        ax.text(lx, ly, label, color=color, alpha=0.4, fontsize=7, ha="center", zorder=10)

    # ── Sun ──────────────────────────────────────────────────────────
    draw_glow(ax, 0, 0, SUN_GLOW, radius=0.25, alpha=0.06)
    draw_glow(ax, 0, 0, SUN_GLOW, radius=0.16, alpha=0.10)
    draw_glow(ax, 0, 0, "#fff176", radius=0.09, alpha=0.18)
    ax.plot(0, 0, "o", color=SUN_CORE, ms=14, zorder=10)
    ax.text(
        0, -0.18, "Sun", color=SUN_CORE, fontsize=9, ha="center", va="top", zorder=10, alpha=0.7
    )

    # ── Compute node positions ───────────────────────────────────────
    # Major bodies
    earth_xy = polar_to_xy(r_visual(R_EARTH), EARTH_ANGLE)
    mars_xy = polar_to_xy(r_visual(R_MARS), MARS_ANGLE)
    jupiter_xy = polar_to_xy(r_visual(R_JUPITER), JUPITER_ANGLE)

    # L2 hub — near Earth, offset toward Mars chain
    l2_offset = 0.22
    l2_dir = EARTH_ANGLE + np.radians(50)
    l2_xy = (earth_xy[0] + l2_offset * np.cos(l2_dir), earth_xy[1] + l2_offset * np.sin(l2_dir))

    # L4 relay — 60° ahead of Mars on Mars orbit
    l4_xy = polar_to_xy(r_visual(R_MARS), L4_ANGLE)

    # EM cycler positions — push out and spread along transfer arc
    em_r = (r_visual(R_EARTH) + r_visual(R_MARS)) / 2
    em_e_xy = polar_to_xy(em_r - 0.30, EM_CYCLER_ANGLE - np.radians(20))
    em_m_xy = polar_to_xy(em_r + 0.35, EM_CYCLER_ANGLE + np.radians(22))

    # MJ cycler — between Mars and Jupiter at ~40% of gap
    mj_r = r_visual(R_MARS) + 0.40 * (r_visual(R_JUPITER) - r_visual(R_MARS))
    mj_xy = polar_to_xy(mj_r, MJ_CYCLER_ANGLE)

    # ── Draw bodies ──────────────────────────────────────────────────
    # Earth
    draw_glow(ax, *earth_xy, EARTH_C, radius=0.14, alpha=0.10)
    ax.plot(*earth_xy, "o", color=EARTH_C, ms=13, zorder=10)
    ax.text(
        earth_xy[0] - 0.20,
        earth_xy[1] - 0.12,
        "Earth",
        color=EARTH_C,
        fontsize=12,
        ha="center",
        fontweight="bold",
        zorder=10,
    )

    # Mars
    draw_glow(ax, *mars_xy, MARS_C, radius=0.12, alpha=0.10)
    ax.plot(*mars_xy, "o", color=MARS_C, ms=11, zorder=10)
    ax.text(
        mars_xy[0] + 0.18,
        mars_xy[1] - 0.10,
        "Mars",
        color=MARS_C,
        fontsize=12,
        ha="left",
        fontweight="bold",
        zorder=10,
    )

    # Jupiter
    draw_glow(ax, *jupiter_xy, JUPITER_C, radius=0.22, alpha=0.06)
    draw_glow(ax, *jupiter_xy, JUPITER_C, radius=0.12, alpha=0.10)
    ax.plot(*jupiter_xy, "o", color=JUPITER_C, ms=16, zorder=10)
    ax.text(
        jupiter_xy[0],
        jupiter_xy[1] - 0.28,
        "Jupiter",
        color=JUPITER_C,
        fontsize=14,
        ha="center",
        fontweight="bold",
        zorder=10,
    )
    ax.text(
        jupiter_xy[0],
        jupiter_xy[1] - 0.46,
        "(Callisto)",
        color=JUPITER_C,
        fontsize=8,
        ha="center",
        alpha=0.6,
        zorder=10,
    )

    # ── Draw network nodes ───────────────────────────────────────────
    node_style = dict(ms=8, zorder=10, alpha=0.95)

    # L2 Hub
    ax.plot(*l2_xy, "D", color=L2_C, **node_style)
    draw_glow(ax, *l2_xy, L2_C, radius=0.07, alpha=0.12)
    ax.text(
        l2_xy[0] + 0.16,
        l2_xy[1] + 0.06,
        "L2 Hub",
        color=L2_C,
        fontsize=8,
        ha="left",
        va="center",
        zorder=10,
        fontweight="bold",
    )

    # EM cycler — boarding
    ax.plot(*em_e_xy, "s", color=CRYO_ROUTE, **node_style)
    draw_glow(ax, *em_e_xy, CRYO_ROUTE, radius=0.06, alpha=0.08)
    ax.text(
        em_e_xy[0] - 0.04,
        em_e_xy[1] + 0.16,
        "E-M Cycler\n(boarding)",
        color=CRYO_ROUTE,
        fontsize=7,
        ha="center",
        va="bottom",
        linespacing=1.2,
        zorder=10,
        alpha=0.9,
    )

    # EM cycler — arriving
    ax.plot(*em_m_xy, "s", color=CRYO_ROUTE, **node_style)
    draw_glow(ax, *em_m_xy, CRYO_ROUTE, radius=0.06, alpha=0.08)
    ax.text(
        em_m_xy[0] + 0.04,
        em_m_xy[1] + 0.16,
        "E-M Cycler\n(arrival)",
        color=CRYO_ROUTE,
        fontsize=7,
        ha="center",
        va="bottom",
        linespacing=1.2,
        zorder=10,
        alpha=0.9,
    )

    # L4 Relay
    ax.plot(*l4_xy, "D", color=L4_C, ms=9, zorder=10, alpha=0.95)
    draw_glow(ax, *l4_xy, L4_C, radius=0.09, alpha=0.10)
    ax.text(
        l4_xy[0],
        l4_xy[1] + 0.18,
        "L4 Relay",
        color=L4_C,
        fontsize=10,
        ha="center",
        fontweight="bold",
        zorder=10,
    )

    # MJ cycler
    ax.plot(*mj_xy, "s", color=SHARED_ROUTE, ms=9, zorder=10, alpha=0.95)
    draw_glow(ax, *mj_xy, SHARED_ROUTE, radius=0.08, alpha=0.08)
    ax.text(
        mj_xy[0],
        mj_xy[1] + 0.18,
        "M-J Cycler",
        color=SHARED_ROUTE,
        fontsize=9,
        ha="center",
        fontweight="bold",
        zorder=10,
    )

    # Earth surface marker
    es_offset = 0.10
    es_dir = EARTH_ANGLE - np.radians(60)
    es_xy = (earth_xy[0] + es_offset * np.cos(es_dir), earth_xy[1] + es_offset * np.sin(es_dir))
    ax.plot(*es_xy, "^", color="#fff176", ms=6, zorder=11, alpha=0.9)
    ax.text(
        es_xy[0] - 0.10,
        es_xy[1] - 0.08,
        "Surface",
        color="#fff176",
        fontsize=6,
        ha="center",
        alpha=0.7,
        zorder=10,
    )

    # ══════════════════════════════════════════════════════════════════
    # ROUTES — The Visual Headline
    # ══════════════════════════════════════════════════════════════════

    # ── CRYO RELAY CHAIN (warm orange, 7 hops) ───────────────────────
    # Hop 1: earth → L2 hub
    draw_straight_route(ax, earth_xy, l2_xy, CRYO_ROUTE, alpha=0.45, lw=1.8, zorder=4)
    # Hop 2: L2 → EM cycler boarding
    draw_route(ax, l2_xy, em_e_xy, CRYO_ROUTE, alpha=0.45, lw=1.8, bulge=0.15, zorder=4)
    # Hop 3: EM coast (boarding → arrival)
    draw_route(ax, em_e_xy, em_m_xy, CRYO_ROUTE, alpha=0.50, lw=2.0, bulge=0.20, zorder=4)
    # Hop 4: EM arrival → Mars
    draw_route(ax, em_m_xy, mars_xy, CRYO_ROUTE, alpha=0.45, lw=1.8, bulge=0.15, zorder=4)
    # Hop 5: Mars → L4 (shared, but color as cryo here)
    draw_route(ax, mars_xy, l4_xy, CRYO_ROUTE, alpha=0.40, lw=1.5, bulge=0.10, zorder=4)

    # ── HARDWARE BYPASS (cool blue, direct) ──────────────────────────
    # earth_surface → mars_station (1 hop, direct chemical transfer)
    draw_route(ax, earth_xy, mars_xy, HARDWARE_ROUTE, alpha=0.55, lw=2.5, bulge=0.30, zorder=5)

    # ── SHARED LEGS (green, both routes merge here) ──────────────────
    # Mars → L4 (draw again in green, slightly offset)
    draw_route(
        ax, mars_xy, l4_xy, SHARED_ROUTE, alpha=0.35, lw=1.2, bulge=-0.08, zorder=3, arrow=False
    )
    # L4 → MJ cycler
    draw_route(ax, l4_xy, mj_xy, SHARED_ROUTE, alpha=0.45, lw=1.8, bulge=0.20, zorder=4)
    # MJ cycler → Jupiter
    draw_route(ax, mj_xy, jupiter_xy, SHARED_ROUTE, alpha=0.50, lw=2.0, bulge=0.25, zorder=4)

    # ── Hop labels — positioned manually to avoid overlap ─────────────
    def label_midpoint(p0, p1, dx=0.0, dy=0.0):
        return (p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy

    hop_labels = [
        (*label_midpoint(earth_xy, l2_xy, 0.14, 0.10), "p=.98\nweekly", CRYO_ROUTE),
        (*label_midpoint(l2_xy, em_e_xy, -0.20, 0.06), "p=.95\n~26 mo", CRYO_ROUTE),
        (*label_midpoint(em_e_xy, em_m_xy, 0.0, 0.22), "p=.99  |  8 mo coast", CRYO_ROUTE),
        (*label_midpoint(em_m_xy, mars_xy, 0.18, 0.0), "p=.93", CRYO_ROUTE),
        # Bypass — offset well away from relay chain
        (
            *label_midpoint(earth_xy, mars_xy, -0.50, 0.10),
            "BYPASS\np=.82  |  9 mo direct",
            HARDWARE_ROUTE,
        ),
        # Shared legs — offset outward
        (*label_midpoint(mars_xy, l4_xy, 0.16, 0.10), "p=.90\nmonthly", SHARED_ROUTE),
        (*label_midpoint(l4_xy, mj_xy, 0.18, 0.12), "p=.88\n~13 mo", SHARED_ROUTE),
        (*label_midpoint(mj_xy, jupiter_xy, 0.18, 0.18), "p=.85\n2 yr coast", SHARED_ROUTE),
    ]

    for hx, hy, txt, clr in hop_labels:
        ax.text(
            hx,
            hy,
            txt,
            color=clr,
            fontsize=6.5,
            ha="center",
            va="center",
            alpha=0.7,
            zorder=10,
            linespacing=1.3,
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor=BG, edgecolor=clr, alpha=0.20, linewidth=0.4
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # HEADLINE RESULT BOX — commodity-aware oracle switching
    # ══════════════════════════════════════════════════════════════════

    box_x, box_y = 3.5, 4.5
    headline_text = (
        "Commodity-Aware Oracle\n"
        "Same network, different topologies\n"
        "\n"
        "Hardware ($\\tau_{1/2} = \\infty$):\n"
        "  100% bypass  |  H = 1  |  Q = 0.82\n"
        "\n"
        "Cryo ($\\tau_{1/2}$ = 730d):\n"
        "  80% bypass, 20% relay  |  crossover\n"
        "\n"
        "Cryo ($\\tau_{1/2}$ = 180d):\n"
        "  56% bypass, 44% relay\n"
        "  relay mean_p = 5.4$\\times$ bypass\n"
        "  E[H] jumps 1.0 $\\to$ 3.63"
    )

    ax.text(
        box_x,
        box_y,
        headline_text,
        color=LABEL_BRIGHT,
        fontsize=7.5,
        ha="left",
        va="top",
        zorder=10,
        linespacing=1.5,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#0f1428",
            edgecolor="#4fc3f7",
            alpha=0.7,
            linewidth=0.8,
        ),
    )

    # ── Mechanism callout ─────────────────────────────────────────────
    mech_x, mech_y = -5.6, 4.5
    mech_text = (
        "Dwell-Induced Topological Pruning\n"
        "\n"
        "Relay wins for cryo because\n"
        "intermediate dwells (~98d avg)\n"
        "< bypass uninterrupted transit\n"
        "(1826d continuous decay)\n"
        "\n"
        "Section 3.4 + Section 3.7\n"
        "Affine parametric shortest-path"
    )
    ax.text(
        mech_x,
        mech_y,
        mech_text,
        color=CRYO_ROUTE,
        fontsize=7.5,
        ha="left",
        va="top",
        zorder=10,
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#0f1428",
            edgecolor=CRYO_ROUTE,
            alpha=0.5,
            linewidth=0.8,
        ),
    )

    # ── Node count summary ────────────────────────────────────────────
    summary_x, summary_y = -5.6, -3.2
    summary_text = (
        "8 nodes  |  7 relay hops  |  4 bypass hops\n"
        "673 contacts  |  10-year window\n"
        "\n"
        "$\\mathrm{DR} = S_T \\times \\eta$\n"
        "Exact factorization to $10^{-16}$"
    )
    ax.text(
        summary_x,
        summary_y,
        summary_text,
        color=LABEL_DIM,
        fontsize=8,
        ha="left",
        va="top",
        zorder=10,
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#0f1428",
            edgecolor=LABEL_DIM,
            alpha=0.3,
            linewidth=0.5,
        ),
    )

    # ── Legend ────────────────────────────────────────────────────────
    leg_x, leg_y = 3.8, -3.2
    legend_items = [
        ("-", HARDWARE_ROUTE, 2.5, "Hardware bypass (H=1)"),
        ("-", CRYO_ROUTE, 2.0, "Cryo relay chain (H=7)"),
        ("-", SHARED_ROUTE, 1.5, "Shared legs (Mars→Jupiter)"),
        ("o", EARTH_C, None, "Planet"),
        ("D", L4_C, None, "Lagrange relay"),
        ("s", SHARED_ROUTE, None, "Cycler vehicle"),
    ]

    for i, (marker, color, lw, label) in enumerate(legend_items):
        y = leg_y - i * 0.22
        if marker == "-":
            ax.plot([leg_x, leg_x + 0.25], [y, y], color=color, lw=lw, alpha=0.7, zorder=10)
        else:
            ax.plot(leg_x + 0.12, y, marker, color=color, ms=6, zorder=10, alpha=0.9)
        ax.text(leg_x + 0.35, y, label, color=NODE_RING, fontsize=7.5, va="center", zorder=10)

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(
        0,
        5.2,
        "Earth \u2013 Mars \u2013 Jupiter Transport Network",
        color=LABEL_BRIGHT,
        fontsize=22,
        ha="center",
        va="bottom",
        fontweight="bold",
        zorder=10,
    )
    ax.text(
        0,
        4.90,
        "Commodity-Aware Routing  |  Interplanetary Transport Network",
        color=LABEL_DIM,
        fontsize=12,
        ha="center",
        va="bottom",
        zorder=10,
    )

    # ── Conjunction zone (SEP wedge) ─────────────────────────────────
    for body_angle in [MARS_ANGLE, JUPITER_ANGLE]:
        wedge_half = 5
        theta_w = np.linspace(
            body_angle - np.radians(wedge_half), body_angle + np.radians(wedge_half), 50
        )
        r_in, r_out = 0.20, r_visual(R_JUPITER) + 0.5
        wx = np.concatenate([r_in * np.cos(theta_w), r_out * np.cos(theta_w[::-1])])
        wy = np.concatenate([r_in * np.sin(theta_w), r_out * np.sin(theta_w[::-1])])
        ax.fill(wx, wy, color=SUN_GLOW, alpha=0.025, zorder=1)

    # ── Axes cleanup ─────────────────────────────────────────────────
    ax.set_xlim(-6.2, 6.8)
    ax.set_ylim(-4.8, 5.5)
    ax.axis("off")

    # ── Save ─────────────────────────────────────────────────────────
    out_png = FIGURES / "emj_network.png"
    out_pdf = FIGURES / "emj_network.pdf"
    fig.savefig(
        out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.2
    )
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")

    import shutil

    desk = Path.home() / "Desktop" / "emj_network.png"
    shutil.copy2(out_png, desk)
    print(f"  Copied to: {desk}")


if __name__ == "__main__":
    main()
