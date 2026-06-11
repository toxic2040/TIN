#!/usr/bin/env python3
"""
Animated Orrery — The TIN Solar System in Motion
=================================================

Real ephemeris (SPICE DE440s) for Mercury through Jupiter, animated
over a 10-year campaign window. Shows:
  - Orbital tracks with fading trail
  - Body positions at true scale
  - Earth-Mars and Mars-Jupiter transfer windows highlighted
  - Synodic alignment markers
  - EMJ relay chain geometry

Output: figures/orrery.gif (animated)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# SPICE setup
import spiceypy as spice
from matplotlib.animation import FuncAnimation, PillowWriter

KERNEL_DIR = Path(__file__).parent.parent / "data" / "kernels"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── SPICE kernels ────────────────────────────────────────────────────

spice.furnsh(str(KERNEL_DIR / "naif0012.tls"))
spice.furnsh(str(KERNEL_DIR / "de440s.bsp"))

AU_KM = 1.496e8  # 1 AU in km

# ── Bodies ───────────────────────────────────────────────────────────

BODIES = {
    "Mercury": {"id": "MERCURY BARYCENTER", "color": "#EE7733", "size": 4},
    "Venus": {"id": "VENUS BARYCENTER", "color": "#EE3377", "size": 5},
    "Earth": {"id": "EARTH BARYCENTER", "color": "#4477AA", "size": 6},
    "Mars": {"id": "MARS BARYCENTER", "color": "#CC3311", "size": 5},
    "Jupiter": {"id": "JUPITER BARYCENTER", "color": "#BBBBBB", "size": 10},
}

# Orbital periods (days) for trail length
PERIODS = {
    "Mercury": 88,
    "Venus": 225,
    "Earth": 365,
    "Mars": 687,
    "Jupiter": 4333,
}


def get_positions(t0_str, duration_days, dt_days=2.0):
    """Get ecliptic positions for all bodies over time range."""
    et0 = spice.str2et(t0_str)
    n_steps = int(duration_days / dt_days)
    ets = et0 + np.arange(n_steps) * dt_days * 86400.0

    positions = {}
    for name, body in BODIES.items():
        xs, ys = [], []
        for et in ets:
            pos, _ = spice.spkpos(body["id"], et, "ECLIPJ2000", "NONE", "SUN")
            xs.append(pos[0] / AU_KM)
            ys.append(pos[1] / AU_KM)
        positions[name] = (np.array(xs), np.array(ys))

    # Calendar dates for display
    dates = [spice.et2utc(et, "C", 0)[:11] for et in ets]

    return positions, dates, ets


def earth_mars_distance(positions, i):
    """Distance between Earth and Mars at frame i (AU)."""
    ex, ey = positions["Earth"][0][i], positions["Earth"][1][i]
    mx, my = positions["Mars"][0][i], positions["Mars"][1][i]
    return np.sqrt((ex - mx) ** 2 + (ey - my) ** 2)


def mars_jupiter_distance(positions, i):
    """Distance between Mars and Jupiter at frame i (AU)."""
    mx, my = positions["Mars"][0][i], positions["Mars"][1][i]
    jx, jy = positions["Jupiter"][0][i], positions["Jupiter"][1][i]
    return np.sqrt((mx - jx) ** 2 + (my - jy) ** 2)


def main():
    print("Computing ephemeris...")
    t0 = "2035-01-01"
    duration = 3652  # 10 years
    dt = 5.0  # 5-day steps
    positions, dates, ets = get_positions(t0, duration, dt)
    n_frames = len(dates)

    # Precompute full orbit tracks (one full period before start)
    orbit_tracks = {}
    for name in BODIES:
        period_days = PERIODS[name]
        et0 = ets[0] - period_days * 86400.0
        n_track = int(period_days / 2.0)
        track_ets = et0 + np.arange(n_track) * 2.0 * 86400.0
        txs, tys = [], []
        for et in track_ets:
            pos, _ = spice.spkpos(BODIES[name]["id"], et, "ECLIPJ2000", "NONE", "SUN")
            txs.append(pos[0] / AU_KM)
            tys.append(pos[1] / AU_KM)
        orbit_tracks[name] = (np.array(txs), np.array(tys))

    # Precompute distances for transfer window highlighting
    em_dist = np.array([earth_mars_distance(positions, i) for i in range(n_frames)])
    mj_dist = np.array([mars_jupiter_distance(positions, i) for i in range(n_frames)])

    # Transfer window thresholds (approximate)
    em_threshold = 1.5  # AU — favorable for transfer
    mj_threshold = 5.0  # AU — favorable for transfer

    spice.kclear()

    # ── Figure setup ─────────────────────────────────────────────────
    print("Setting up animation...")

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (AU)", color="#888888", fontsize=10)
    ax.set_ylabel("Y (AU)", color="#888888", fontsize=10)
    ax.tick_params(colors="#555555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(True, alpha=0.08, color="#444444")

    # Sun
    ax.plot(0, 0, "o", color="#FFD700", markersize=12, zorder=10)
    ax.plot(0, 0, "o", color="#FFEE88", markersize=6, zorder=11)

    # Orbit tracks (static, faint)
    for name, (tx, ty) in orbit_tracks.items():
        ax.plot(tx, ty, "-", color=BODIES[name]["color"], alpha=0.12, linewidth=0.8, zorder=1)

    # Body dots (will be updated)
    body_dots = {}
    body_labels = {}
    for name, body in BODIES.items():
        (dot,) = ax.plot([], [], "o", color=body["color"], markersize=body["size"], zorder=20)
        body_dots[name] = dot

        label = ax.text(
            0,
            0,
            f"  {name}",
            color=body["color"],
            fontsize=7,
            alpha=0.8,
            zorder=21,
            ha="left",
            va="center",
        )
        body_labels[name] = label

    # Trails (recent history, fading)
    trail_length = 30  # frames of trail
    body_trails = {}
    for name, body in BODIES.items():
        (trail,) = ax.plot([], [], "-", color=body["color"], alpha=0.4, linewidth=1.5, zorder=5)
        body_trails[name] = trail

    # Transfer window lines
    (em_line,) = ax.plot([], [], "--", color="#66AAFF", alpha=0.0, linewidth=1.0, zorder=8)
    (mj_line,) = ax.plot([], [], "--", color="#FFAA66", alpha=0.0, linewidth=1.0, zorder=8)

    # Text overlays
    date_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        color="#CCCCCC",
        fontsize=11,
        fontfamily="monospace",
        va="top",
        zorder=30,
    )
    info_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        color="#888888",
        fontsize=8,
        fontfamily="monospace",
        va="top",
        ha="right",
        zorder=30,
    )

    title_text = ax.text(
        0.5,
        0.02,
        "TIN Orrery — Real Ephemeris (SPICE DE440s)",
        transform=ax.transAxes,
        color="#666666",
        fontsize=9,
        fontfamily="monospace",
        ha="center",
        va="bottom",
        zorder=30,
    )

    # ── Animation function ───────────────────────────────────────────

    def update(frame):
        i = frame

        # Update body positions
        for name in BODIES:
            x = positions[name][0][i]
            y = positions[name][1][i]
            body_dots[name].set_data([x], [y])
            body_labels[name].set_position((x, y))

            # Trail
            i0 = max(0, i - trail_length)
            body_trails[name].set_data(
                positions[name][0][i0 : i + 1],
                positions[name][1][i0 : i + 1],
            )

        # Transfer window highlighting
        em_d = em_dist[i]
        mj_d = mj_dist[i]

        ex, ey = positions["Earth"][0][i], positions["Earth"][1][i]
        mx, my = positions["Mars"][0][i], positions["Mars"][1][i]
        jx, jy = positions["Jupiter"][0][i], positions["Jupiter"][1][i]

        if em_d < em_threshold:
            em_line.set_data([ex, mx], [ey, my])
            em_line.set_alpha(0.6 * (1 - em_d / em_threshold))
        else:
            em_line.set_alpha(0.0)

        if mj_d < mj_threshold:
            mj_line.set_data([mx, jx], [my, jy])
            mj_line.set_alpha(0.4 * (1 - mj_d / mj_threshold))
        else:
            mj_line.set_alpha(0.0)

        # Text
        date_text.set_text(dates[i])

        em_status = "OPEN" if em_d < em_threshold else "     "
        mj_status = "OPEN" if mj_d < mj_threshold else "     "

        info_text.set_text(f"E-M: {em_d:.2f} AU  {em_status}\nM-J: {mj_d:.2f} AU  {mj_status}")

        artists = (
            list(body_dots.values())
            + list(body_labels.values())
            + list(body_trails.values())
            + [em_line, mj_line, date_text, info_text]
        )
        return artists

    # ── Render ────────────────────────────────────────────────────────

    # Subsample for GIF: every 2nd frame → ~365 frames → ~12s at 30fps
    frame_indices = list(range(0, n_frames, 2))
    print(f"Rendering {len(frame_indices)} frames...")

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=33, blit=True)

    out_path = FIG_DIR / "orrery.gif"
    anim.save(
        str(out_path),
        writer=PillowWriter(fps=30),
        savefig_kwargs={"facecolor": fig.get_facecolor()},
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    plt.close(fig)


if __name__ == "__main__":
    main()
