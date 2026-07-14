#!/usr/bin/env python3
"""Generate a historical, modeled heliocentric DTN concept sketch.

Sun-centered view of the full interplanetary DTN architecture:
Earth + cislunar (Moon, Shackleton, 8 polar, ELFO, EM-L2 Halo),
Mars, L4/L5 relays, polar constellation, DSN links.
Dark-sky aesthetic.  Geometry and archived annotations are illustrative only;
the figure is not a validated architecture, mission-feasibility result, or
network-design recommendation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Orbital parameters ────────────────────────────────────────────────
AU = 1.0
R_EARTH = 1.0 * AU
R_MARS = 1.524 * AU

# Place Earth at 210° and Mars at ~30° (pre-opposition geometry, ~1.8 AU apart)
EARTH_ANGLE = np.radians(210)
MARS_ANGLE = np.radians(30)

# L4/L5 are 60° ahead/behind Mars along its orbit
L4_ANGLE = MARS_ANGLE + np.radians(60)  # leading
L5_ANGLE = MARS_ANGLE - np.radians(60)  # trailing

# Earth-side relay at Earth's L4 (leading Earth by 60°)
EARTH_L4_ANGLE = EARTH_ANGLE + np.radians(60)

# Cislunar geometry (exaggerated scale for visibility)
MOON_OFFSET = 0.20  # exaggerated distance from Earth (true: 0.0026 AU)
MOON_DIR = EARTH_ANGLE + np.radians(25)  # offset angle from Earth


def polar_to_xy(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def draw_orbit(ax, radius, color="#ffffff", alpha=0.12, ls="-", lw=0.6):
    theta = np.linspace(0, 2 * np.pi, 360)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, color=color, alpha=alpha, ls=ls, lw=lw, zorder=1)


def draw_link(ax, p1, p2, color="#4fc3f7", alpha=0.35, lw=0.8, ls="-", zorder=3):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, lw=lw, ls=ls, zorder=zorder)


def draw_glow(ax, x, y, color, radius=0.06, alpha=0.15, zorder=2):
    circle = plt.Circle((x, y), radius, color=color, alpha=alpha, zorder=zorder)
    ax.add_patch(circle)


def draw_cislunar(ax, earth_xy):
    """Draw the cislunar network near Earth at exaggerated scale."""
    ex, ey = earth_xy

    # Moon position
    moon_x = ex + MOON_OFFSET * np.cos(MOON_DIR)
    moon_y = ey + MOON_OFFSET * np.sin(MOON_DIR)

    # Moon orbit ring (exaggerated)
    theta = np.linspace(0, 2 * np.pi, 120)
    ax.plot(
        ex + MOON_OFFSET * np.cos(theta),
        ey + MOON_OFFSET * np.sin(theta),
        color="#b0bec5",
        alpha=0.10,
        lw=0.4,
        zorder=2,
    )

    # Moon body
    draw_glow(ax, moon_x, moon_y, "#b0bec5", radius=0.04, alpha=0.08)
    ax.plot(moon_x, moon_y, "o", color="#cfd8dc", ms=5.5, zorder=10)
    ax.text(
        moon_x + 0.06,
        moon_y + 0.06,
        "Moon",
        color="#cfd8dc",
        fontsize=7.5,
        ha="left",
        zorder=10,
        fontweight="bold",
    )

    # Shackleton station (on Moon's south-pole surface — place at bottom of Moon)
    shack_x = moon_x + 0.015 * np.cos(np.radians(250))
    shack_y = moon_y + 0.015 * np.sin(np.radians(250))
    ax.plot(shack_x, shack_y, "^", color="#fff176", ms=3.5, zorder=11, alpha=0.9)
    ax.text(
        shack_x - 0.01,
        shack_y - 0.05,
        "Shackleton",
        color="#fff176",
        fontsize=5.5,
        ha="center",
        zorder=10,
        alpha=0.8,
    )

    # 8 Polar orbiters (tiny ring around Moon)
    polar_r = 0.040
    for i in range(8):
        th = 2 * np.pi * i / 8 + np.pi / 12
        px = moon_x + polar_r * np.cos(th)
        py = moon_y + polar_r * np.sin(th)
        ax.plot(px, py, "o", color="#81d4fa", ms=1.8, zorder=9, alpha=0.85)
    # Polar orbit ring
    ax.plot(
        moon_x + polar_r * np.cos(theta),
        moon_y + polar_r * np.sin(theta),
        color="#81d4fa",
        alpha=0.18,
        lw=0.4,
        zorder=4,
    )

    # ELFO relay hub (higher elliptical, further ring)
    elfo_r = 0.065
    elfo_angle = MOON_DIR + np.radians(110)  # offset for visibility
    elfo_x = moon_x + elfo_r * np.cos(elfo_angle)
    elfo_y = moon_y + elfo_r * np.sin(elfo_angle)
    ax.plot(elfo_x, elfo_y, "D", color="#ffcc80", ms=3.5, zorder=10, alpha=0.9)
    ax.text(
        elfo_x - 0.01,
        elfo_y + 0.04,
        "ELFO",
        color="#ffcc80",
        fontsize=5.5,
        ha="center",
        zorder=10,
        alpha=0.8,
    )

    # EM-L2 Halo relay (far side of Moon from Earth)
    l2_dir = np.arctan2(moon_y - ey, moon_x - ex)  # direction away from Earth
    l2_dist = 0.10
    l2_x = moon_x + l2_dist * np.cos(l2_dir)
    l2_y = moon_y + l2_dist * np.sin(l2_dir)
    ax.plot(l2_x, l2_y, "D", color="#80cbc4", ms=3.5, zorder=10, alpha=0.9)
    ax.text(
        l2_x - 0.04,
        l2_y + 0.06,
        "EM-L2\nHalo",
        color="#80cbc4",
        fontsize=5,
        ha="center",
        va="bottom",
        zorder=10,
        alpha=0.8,
        linespacing=1.2,
    )

    # ── Cislunar links ────────────────────────────────────────────────
    # Earth → Moon
    draw_link(ax, (ex, ey), (moon_x, moon_y), color="#b0bec5", alpha=0.20, lw=0.6)

    # Shackleton → nearest polar sat (symbolic)
    p0_th = 2 * np.pi * 0 / 8 + np.pi / 12
    p0 = (moon_x + polar_r * np.cos(p0_th), moon_y + polar_r * np.sin(p0_th))
    draw_link(ax, (shack_x, shack_y), p0, color="#fff176", alpha=0.25, lw=0.4)

    # Polar → ELFO
    p4_th = 2 * np.pi * 4 / 8 + np.pi / 12
    p4 = (moon_x + polar_r * np.cos(p4_th), moon_y + polar_r * np.sin(p4_th))
    draw_link(ax, p4, (elfo_x, elfo_y), color="#ffcc80", alpha=0.20, lw=0.4)

    # ELFO → EM-L2
    draw_link(ax, (elfo_x, elfo_y), (l2_x, l2_y), color="#80cbc4", alpha=0.20, lw=0.4)

    # EM-L2 → Earth (far-side relay path)
    draw_link(ax, (l2_x, l2_y), (ex, ey), color="#80cbc4", alpha=0.15, lw=0.5, ls="--")

    # Polar ring → Earth (direct downlinks — a couple representative ones)
    for idx in [2, 6]:
        th = 2 * np.pi * idx / 8 + np.pi / 12
        pi = (moon_x + polar_r * np.cos(th), moon_y + polar_r * np.sin(th))
        draw_link(ax, pi, (ex, ey), color="#81d4fa", alpha=0.12, lw=0.3, ls=":")

    # ── Zoom ring (dashed circle enclosing the cislunar complex) ──────
    zoom_cx = (ex + l2_x) / 2
    zoom_cy = (ey + l2_y) / 2
    zoom_r = 0.23
    ax.plot(
        zoom_cx + zoom_r * np.cos(theta),
        zoom_cy + zoom_r * np.sin(theta),
        color="#546e7a",
        alpha=0.25,
        lw=0.5,
        ls="--",
        zorder=2,
    )


def draw_mars_constellation(ax, center, n_sats=6, orbit_r=0.08, color="#ef5350", ms=3):
    """Draw small satellites orbiting Mars."""
    cx, cy = center
    for i in range(n_sats):
        theta = 2 * np.pi * i / n_sats + np.pi / 7  # offset for aesthetics
        sx = cx + orbit_r * np.cos(theta)
        sy = cy + orbit_r * np.sin(theta)
        ax.plot(sx, sy, "o", color=color, ms=ms, zorder=8, alpha=0.9)
    # Draw the constellation orbit ring
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        cx + orbit_r * np.cos(theta),
        cy + orbit_r * np.sin(theta),
        color=color,
        alpha=0.2,
        lw=0.5,
        zorder=4,
    )


def main():
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    ax.set_aspect("equal")

    # ── Star field ─────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_stars = 800
    sx = rng.uniform(-2.8, 2.8, n_stars)
    sy = rng.uniform(-2.8, 2.8, n_stars)
    ss = rng.exponential(0.3, n_stars) ** 2
    sa = rng.uniform(0.15, 0.6, n_stars)
    ax.scatter(sx, sy, s=ss, c="white", alpha=sa, zorder=0, edgecolors="none")

    # ── Orbits ─────────────────────────────────────────────────────────
    draw_orbit(ax, R_EARTH, color="#4fc3f7", alpha=0.18, lw=0.7)
    draw_orbit(ax, R_MARS, color="#ef5350", alpha=0.18, lw=0.7)

    # Orbit labels
    ax.text(
        R_EARTH * np.cos(np.radians(135)) + 0.02,
        R_EARTH * np.sin(np.radians(135)) + 0.06,
        "1 AU",
        color="#4fc3f7",
        alpha=0.4,
        fontsize=7,
        ha="center",
        zorder=10,
    )
    ax.text(
        R_MARS * np.cos(np.radians(135)) + 0.02,
        R_MARS * np.sin(np.radians(135)) + 0.08,
        "1.52 AU",
        color="#ef5350",
        alpha=0.4,
        fontsize=7,
        ha="center",
        zorder=10,
    )

    # ── Sun ────────────────────────────────────────────────────────────
    draw_glow(ax, 0, 0, "#ffb74d", radius=0.22, alpha=0.08)
    draw_glow(ax, 0, 0, "#ffb74d", radius=0.14, alpha=0.12)
    draw_glow(ax, 0, 0, "#fff176", radius=0.08, alpha=0.2)
    ax.plot(0, 0, "o", color="#fff9c4", ms=14, zorder=10)
    ax.text(
        0, -0.16, "Sun", color="#fff9c4", fontsize=9, ha="center", va="top", zorder=10, alpha=0.8
    )

    # ── Earth ──────────────────────────────────────────────────────────
    ex, ey = polar_to_xy(R_EARTH, EARTH_ANGLE)
    draw_glow(ax, ex, ey, "#4fc3f7", radius=0.10, alpha=0.12)
    ax.plot(ex, ey, "o", color="#4fc3f7", ms=10, zorder=10)
    ax.text(
        ex - 0.14,
        ey + 0.10,
        "Earth",
        color="#4fc3f7",
        fontsize=10,
        ha="center",
        va="bottom",
        zorder=10,
        fontweight="bold",
    )

    # DSN stations (3 around Earth)
    for i, label in enumerate(["Goldstone", "Canberra", "Madrid"]):
        theta = EARTH_ANGLE + np.radians(120 * i + 15)
        dx, dy = polar_to_xy(R_EARTH + 0.06, theta)
        ax.plot(dx, dy, "s", color="#81d4fa", ms=3.5, zorder=9, alpha=0.8)
        draw_link(ax, (ex, ey), (dx, dy), color="#81d4fa", alpha=0.25, lw=0.5)

    # ── Cislunar network (Moon, Shackleton, polars, ELFO, EM-L2) ─────
    draw_cislunar(ax, (ex, ey))

    # ── Mars ───────────────────────────────────────────────────────────
    mx, my = polar_to_xy(R_MARS, MARS_ANGLE)
    draw_glow(ax, mx, my, "#ef5350", radius=0.12, alpha=0.10)
    ax.plot(mx, my, "o", color="#ef5350", ms=8, zorder=10)
    ax.text(
        mx + 0.12,
        my + 0.08,
        "Mars",
        color="#ef5350",
        fontsize=10,
        ha="left",
        zorder=10,
        fontweight="bold",
    )

    # Mars polar constellation (T2: 6 low + 3 high)
    draw_mars_constellation(ax, (mx, my), n_sats=6, orbit_r=0.09, color="#ef9a9a", ms=2.5)
    draw_mars_constellation(ax, (mx, my), n_sats=3, orbit_r=0.15, color="#ffab91", ms=3)

    # ── L4 Relay (Mars leading, 60° ahead) ─────────────────────────────
    l4x, l4y = polar_to_xy(R_MARS, L4_ANGLE)
    draw_glow(ax, l4x, l4y, "#ce93d8", radius=0.08, alpha=0.12)
    ax.plot(l4x, l4y, "D", color="#ce93d8", ms=7, zorder=10)
    ax.text(
        l4x + 0.10,
        l4y + 0.08,
        "L4 Relay",
        color="#ce93d8",
        fontsize=9,
        ha="left",
        zorder=10,
        fontweight="bold",
    )

    # L4 Earth-side relay (at Mars orbit but Earth's leading L4)
    el4x, el4y = polar_to_xy(R_MARS, EARTH_L4_ANGLE)
    draw_glow(ax, el4x, el4y, "#80cbc4", radius=0.07, alpha=0.10)
    ax.plot(el4x, el4y, "D", color="#80cbc4", ms=6, zorder=10)
    ax.text(
        el4x - 0.14, el4y + 0.06, "Earth L4", color="#80cbc4", fontsize=8, ha="center", zorder=10
    )

    # ── L5 Relay (Mars trailing, 60° behind) ───────────────────────────
    l5x, l5y = polar_to_xy(R_MARS, L5_ANGLE)
    draw_glow(ax, l5x, l5y, "#a5d6a7", radius=0.08, alpha=0.12)
    ax.plot(l5x, l5y, "D", color="#a5d6a7", ms=7, zorder=10)
    ax.text(
        l5x + 0.10,
        l5y - 0.10,
        "L5 Relay",
        color="#a5d6a7",
        fontsize=9,
        ha="left",
        zorder=10,
        fontweight="bold",
    )

    # ── Communication Links ────────────────────────────────────────────

    # Illustrative dotted direct Earth-Mars path.
    draw_link(ax, (ex, ey), (mx, my), color="#ffcc80", alpha=0.20, lw=1.2, ls=":", zorder=2)

    # Earth → L4 relay
    draw_link(ax, (ex, ey), (l4x, l4y), color="#ce93d8", alpha=0.30, lw=1.0)
    # L4 → Mars
    draw_link(ax, (l4x, l4y), (mx, my), color="#ce93d8", alpha=0.30, lw=1.0)

    # Earth → L5 relay
    draw_link(ax, (ex, ey), (l5x, l5y), color="#a5d6a7", alpha=0.25, lw=0.9)
    # L5 → Mars
    draw_link(ax, (l5x, l5y), (mx, my), color="#a5d6a7", alpha=0.25, lw=0.9)

    # Earth → Earth L4
    draw_link(ax, (ex, ey), (el4x, el4y), color="#80cbc4", alpha=0.25, lw=0.8)
    # Earth L4 → L4
    draw_link(ax, (el4x, el4y), (l4x, l4y), color="#80cbc4", alpha=0.20, lw=0.7, ls="--")

    # L4 ↔ L5 inter-relay backbone
    draw_link(ax, (l4x, l4y), (l5x, l5y), color="#fff176", alpha=0.15, lw=0.6, ls="--")

    # ── Conjunction zone (SEP wedge from Sun) ──────────────────────────
    # Show the ~5° solar exclusion zone toward Mars
    mars_angle_deg = np.degrees(MARS_ANGLE)
    wedge_half = 5  # degrees
    theta_wedge = np.linspace(
        np.radians(mars_angle_deg - wedge_half),
        np.radians(mars_angle_deg + wedge_half),
        50,
    )
    r_inner = 0.18
    r_outer = 2.0
    wx1 = np.concatenate([r_inner * np.cos(theta_wedge), r_outer * np.cos(theta_wedge[::-1])])
    wy1 = np.concatenate([r_inner * np.sin(theta_wedge), r_outer * np.sin(theta_wedge[::-1])])
    ax.fill(wx1, wy1, color="#ffb74d", alpha=0.04, zorder=1)

    # ── Annotations ────────────────────────────────────────────────────

    # Legend box
    legend_x, legend_y = -2.5, -2.0
    legend_items = [
        ("o", "#4fc3f7", 6, "Planet / Moon"),
        ("D", "#ce93d8", 5, "Lagrange Relay"),
        ("D", "#ffcc80", 4, "ELFO / Halo Relay"),
        ("^", "#fff176", 4, "Surface Station"),
        ("s", "#81d4fa", 4, "DSN Station"),
        ("o", "#ef9a9a", 3, "Mars Orbiter"),
        ("o", "#81d4fa", 2, "Lunar Polar Sat"),
    ]
    for i, (marker, color, ms, label) in enumerate(legend_items):
        y = legend_y - i * 0.16
        ax.plot(legend_x, y, marker, color=color, ms=ms, zorder=10)
        ax.text(legend_x + 0.10, y, label, color="#b0bec5", fontsize=8, va="center", zorder=10)

    # Link legend
    ly = legend_y - len(legend_items) * 0.16 - 0.08
    ax.plot(
        [legend_x - 0.05, legend_x + 0.05], [ly, ly], color="#ffcc80", alpha=0.4, lw=1.2, ls=":"
    )
    ax.text(
        legend_x + 0.10,
        ly,
        "Archived direct-path concept",
        color="#b0bec5",
        fontsize=8,
        va="center",
        zorder=10,
    )
    ly -= 0.16
    ax.plot([legend_x - 0.05, legend_x + 0.05], [ly, ly], color="#ce93d8", alpha=0.5, lw=1.0)
    ax.text(
        legend_x + 0.10,
        ly,
        "Archived relay-path concept",
        color="#b0bec5",
        fontsize=8,
        va="center",
        zorder=10,
    )

    # Title
    ax.text(
        0,
        2.55,
        "Historical Heliocentric DTN Concept Sketch",
        color="#eceff1",
        fontsize=16,
        ha="center",
        va="bottom",
        fontweight="bold",
        zorder=10,
    )
    ax.text(
        0,
        2.38,
        "MODELED GEOMETRY ONLY  |  NOT MISSION FEASIBILITY OR VALIDATED DESIGN",
        color="#78909c",
        fontsize=10,
        ha="center",
        va="bottom",
        zorder=10,
    )

    # Retained bookkeeping identity; the former universal predictor is retired.
    ax.text(
        0,
        -2.55,
        r"$\mathrm{DR} = S_T \cdot \eta$  (bookkeeping identity)",
        color="#b0bec5",
        fontsize=11,
        ha="center",
        va="top",
        zorder=10,
    )
    ax.text(
        0,
        -2.75,
        "Historical concept sketch  |  universal predictor retired",
        color="#546e7a",
        fontsize=8,
        ha="center",
        va="top",
        zorder=10,
    )

    # ── Tier annotations ───────────────────────────────────────────────
    # Small tier labels near Mars
    tier_info = [
        (mx + 0.12, my - 0.12, "T1: 3 polar sats", "#ef9a9a", 7),
        (mx + 0.12, my - 0.20, "T2: +6 low polar", "#ffab91", 7),
        (l4x + 0.10, l4y - 0.08, "T3: +L4/L5 relays", "#ce93d8", 7),
        (l5x + 0.10, l5y - 0.20, "T4: +L5 + DSN 24h", "#a5d6a7", 7),
    ]
    for tx, ty, label, color, fs in tier_info:
        ax.text(tx, ty, label, color=color, fontsize=fs, alpha=0.7, zorder=10)

    # ── Archived model annotations ─────────────────────────────────────
    ax.text(
        2.0,
        2.0,
        "Archived model row:\nT1 DR > T2 DR\nat 90% of tested epochs",
        color="#ffab91",
        fontsize=8,
        ha="center",
        va="center",
        zorder=10,
        alpha=0.6,
        linespacing=1.4,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#0a0e1a",
            edgecolor="#ffab91",
            alpha=0.3,
            linewidth=0.5,
        ),
    )

    ax.text(
        -2.0,
        2.0,
        "Archived model row:\nT3 S_T = 0.993 at\nSEP = 1.4\u00b0",
        color="#ce93d8",
        fontsize=8,
        ha="center",
        va="center",
        zorder=10,
        alpha=0.6,
        linespacing=1.4,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#0a0e1a",
            edgecolor="#ce93d8",
            alpha=0.3,
            linewidth=0.5,
        ),
    )

    # ── Axes cleanup ───────────────────────────────────────────────────
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.9, 2.7)
    ax.axis("off")

    # ── Save ───────────────────────────────────────────────────────────
    out_png = FIGURES / "orrery_heliocentric.png"
    out_pdf = FIGURES / "orrery_heliocentric.pdf"
    fig.savefig(
        out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.2
    )
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")

    # Also save to Desktop
    import shutil

    desk = Path.home() / "Desktop" / "orrery_heliocentric.png"
    shutil.copy2(out_png, desk)
    print(f"  Copied to: {desk}")


if __name__ == "__main__":
    main()
