"""tin.core.viz — Mars-grade interactive Plotly visualizations for TIN v0.4.0

Generates:
  1. 3D lunar constellation viewer (orbits, sats, ELFO, bundle trails)
  2. BPv7 custody Gantt timeline (animated, priority color-coded)
  3. Shackleton crater coverage heatmap (PSR glow + rim occlusion)

All outputs are standalone interactive HTML files in results/.

Usage:
    from tin.core.viz import make_all_viz
    make_all_viz(constellation, bundles)

Dependencies: plotly (pip install plotly)
"""

import os
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from tin.core.dtn import PRIORITY_EMERGENCY, PRIORITY_NAMES


def _check_plotly():
    if not HAS_PLOTLY:
        print("  ⚠️  Plotly not installed. Run: pip install plotly")
        return False
    return True


# =========================================================================
# 1. 3D Lunar Constellation Viewer
# =========================================================================

def lunar_3d_constellation(constellation=None, bundles=None,
                            title="TIN Lunar Constellation — Live View"):
    """Interactive 3D Moon globe with neon orbits + emergency bundle trails."""
    if not _check_plotly():
        return None

    fig = go.Figure()
    R = 1737.4  # lunar radius km

    # --- Moon globe ---
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, colorscale="gray", opacity=0.95,
        showscale=False, name="Moon",
    ))

    # --- Shackleton crater marker ---
    # -89.54° lat, 0° lon → near south pole
    s_pos = _latlon_to_xyz(-89.54, 0.0, R)
    fig.add_trace(go.Scatter3d(
        x=[s_pos[0]], y=[s_pos[1]], z=[s_pos[2]],
        mode="markers+text",
        marker=dict(size=18, color="#00ff88", symbol="diamond",
                    line=dict(width=3, color="white")),
        text=["Shackleton"], textposition="top center",
        name="Shackleton Base",
    ))

    # --- Satellite orbits + position markers ---
    if constellation is not None:
        sats = getattr(constellation, "satellites", [])
        if not sats and hasattr(constellation, "config"):
            sats = getattr(constellation.config, "satellites", [])

        for sat in sats:
            a_km = getattr(sat, "a_km", R + 400)
            i_deg = getattr(sat, "i_deg", 89.5)
            raan_deg = getattr(sat, "raan_deg", 0.0)
            sat_id = getattr(sat, "sat_id", "SAT")

            # Generate orbit ring
            theta = np.linspace(0, 2 * np.pi, 120)
            ox, oy, oz = _orbit_points(a_km, i_deg, raan_deg, theta)

            fig.add_trace(go.Scatter3d(
                x=ox, y=oy, z=oz, mode="lines",
                line=dict(color="#00ffff", width=4),
                name=sat_id, showlegend=False,
            ))

            # Sat position marker (at arbitrary true anomaly)
            idx = 30
            fig.add_trace(go.Scatter3d(
                x=[ox[idx]], y=[oy[idx]], z=[oz[idx]],
                mode="markers",
                marker=dict(size=8, color="#00ffff",
                            line=dict(width=1, color="white")),
                name=sat_id,
            ))

        # ELFO hub marker
        relays = getattr(constellation, "relay_hubs", [])
        if not relays and hasattr(constellation, "config"):
            relays = getattr(constellation.config, "relay_hubs", [])

        for relay in relays:
            relay_id = getattr(relay, "sat_id", "ELFO-HUB")
            a_km = getattr(relay, "a_km", 5250)
            # Place at apoapsis over south pole
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[-a_km * 0.8],
                mode="markers+text",
                marker=dict(size=16, color="#ff00ff", symbol="diamond"),
                text=[relay_id], textposition="top center",
                name=relay_id,
            ))

    # --- Emergency bundle trails ---
    if bundles:
        for b in bundles:
            color = "#ff2222" if b.priority == PRIORITY_EMERGENCY else "#00ffcc"
            # Build path from custody chain (approximate 3D positions)
            n_hops = len(b.custody_chain)
            if n_hops > 1:
                path_z = np.linspace(-R, R * 2.5, n_hops)
                path_x = np.linspace(-300, 0, n_hops) + np.random.normal(0, 100, n_hops)
                path_y = np.linspace(-300, 0, n_hops) + np.random.normal(0, 100, n_hops)

                bid = getattr(b, "bundle_id", "bundle")
                is_em = b.priority == PRIORITY_EMERGENCY
                fig.add_trace(go.Scatter3d(
                    x=path_x, y=path_y, z=path_z,
                    mode="lines+markers",
                    line=dict(color=color, width=8,
                              dash="dash" if is_em else "solid"),
                    marker=dict(size=7, color=color),
                    name=f"{'🚨 ' if is_em else ''}Bundle {bid}",
                ))

    # --- Layout ---
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color="white")),
        scene=dict(
            xaxis=dict(title="X (km)", backgroundcolor="#05050f", gridcolor="#222"),
            yaxis=dict(title="Y (km)", backgroundcolor="#05050f", gridcolor="#222"),
            zaxis=dict(title="Z (km)", backgroundcolor="#05050f", gridcolor="#222"),
            bgcolor="#05050f",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        paper_bgcolor="#05050f",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
    )

    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "lunar_3d_constellation.html")
    fig.write_html(path)
    print(f"    📂 {path}")
    return fig


# =========================================================================
# 2. BPv7 Custody Gantt Timeline
# =========================================================================

def custody_gantt(bundles, title="TIN BPv7 Custody Timeline"):
    """Animated Gantt chart showing bundle hop timing with priority colors."""
    if not _check_plotly():
        return None

    fig = go.Figure()

    if not bundles:
        print("    ⚠️  No bundles to visualize.")
        return None

    for i, b in enumerate(bundles):
        color = "#ff2222" if b.priority == PRIORITY_EMERGENCY else (
            "#ffaa00" if b.priority == 1 else "#00ffcc"
        )
        pri_name = PRIORITY_NAMES.get(b.priority, "?")

        for hop in getattr(b, "hop_log", []):
            depart = hop.get("depart_s", 0) / 60
            arrive = hop.get("arrive_s", 0) / 60
            from_n = hop.get("from", "?")
            to_n = hop.get("to", "?")
            link = hop.get("link_type", "")
            dur = hop.get("arrive_s", 0) - hop.get("depart_s", 0)

            fig.add_trace(go.Scatter(
                x=[depart, arrive], y=[i, i],
                mode="lines+markers",
                line=dict(color=color, width=14),
                marker=dict(size=8, color=color),
                hovertemplate=(
                    f"{from_n} → {to_n}<br>"
                    f"Link: {link}<br>"
                    f"Duration: {dur:.3f}s<br>"
                    f"Priority: {pri_name}"
                ),
                name=f"[{pri_name}] {getattr(b, 'bundle_id', f'B-{i}')}",
                showlegend=(hop == b.hop_log[0]),  # only first hop in legend
            ))

    # Y-axis labels
    labels = [getattr(b, "bundle_id", f"B-{i}") for i, b in enumerate(bundles)]
    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color="white")),
        xaxis_title="Time (minutes)",
        yaxis=dict(
            tickvals=list(range(len(labels))),
            ticktext=labels,
            title="Bundle",
        ),
        plot_bgcolor="#05050f",
        paper_bgcolor="#05050f",
        font=dict(color="white"),
        height=max(400, 120 * len(bundles)),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.6)"),
    )

    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "custody_gantt.html")
    fig.write_html(path)
    print(f"    📂 {path}")
    return fig


# =========================================================================
# 3. Shackleton Crater Heatmap
# =========================================================================

def shackleton_heatmap(title="Shackleton Crater — Coverage + PSR Overlay"):
    """Polar coverage heatmap with permanently shadowed region (PSR) glow."""
    if not _check_plotly():
        return None

    lat = np.linspace(-90, -80, 120)
    lon = np.linspace(-25, 25, 120)
    LAT, LON = np.meshgrid(lat, lon)

    # Coverage model: peaks at Shackleton (~89.5°S, ~0°E), drops at edges
    coverage = 100 - 8 * np.exp(-((LAT + 89.2)**2 + (LON + 2)**2) / 8)

    # PSR zones: permanently shadowed regions inside crater (cold-soak zones)
    psr = np.where((LAT < -88.6) & (np.abs(LON) < 6), 35, 0)

    # Effective coverage (reduced in PSR due to cold-soak battery degradation)
    effective = coverage - psr

    fig = go.Figure()

    # Main coverage heatmap
    fig.add_trace(go.Contour(
        z=effective, x=lon, y=lat,
        colorscale="Plasma",
        contours=dict(showlabels=True, coloring="heatmap"),
        colorbar=dict(title="Coverage %", x=1.02),
        name="Coverage",
    ))

    # PSR overlay (blue glow)
    fig.add_trace(go.Contour(
        z=psr, x=lon, y=lat,
        colorscale="Blues_r", opacity=0.6,
        contours=dict(showlabels=True, start=10, end=40, size=10),
        showscale=False,
        name="PSR (cold-soak zone)",
    ))

    # Shackleton center marker
    fig.add_trace(go.Scatter(
        x=[0], y=[-89.54],
        mode="markers+text",
        marker=dict(size=14, color="#00ff88", symbol="star",
                    line=dict(width=2, color="white")),
        text=["Shackleton Base"], textposition="top right",
        textfont=dict(color="#00ff88", size=12),
        name="Shackleton Base",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="white")),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°S)",
        plot_bgcolor="#05050f",
        paper_bgcolor="#05050f",
        font=dict(color="white"),
        height=700,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.6)"),
    )

    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "shackleton_heatmap.html")
    fig.write_html(path)
    print(f"    📂 {path}")
    return fig


# =========================================================================
# Helpers
# =========================================================================

def _latlon_to_xyz(lat_deg, lon_deg, r_km):
    """Convert lat/lon to 3D Cartesian."""
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)
    return np.array([
        r_km * np.cos(phi) * np.cos(lam),
        r_km * np.cos(phi) * np.sin(lam),
        r_km * np.sin(phi),
    ])


def _orbit_points(a_km, i_deg, raan_deg, theta):
    """Generate 3D orbit ring points for given orbital elements."""
    i = np.deg2rad(i_deg)
    raan = np.deg2rad(raan_deg)

    # Position in orbital plane
    x_orb = a_km * np.cos(theta)
    y_orb = a_km * np.sin(theta)

    # Rotate to 3D
    cos_r, sin_r = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)

    x = x_orb * cos_r - y_orb * sin_r * cos_i
    y = x_orb * sin_r + y_orb * cos_r * cos_i
    z = y_orb * sin_i

    return x, y, z


# =========================================================================
# One-click entry point
# =========================================================================

def make_all_viz(constellation=None, bundles=None):
    """Generate all TIN visualizations. Called from CLI with --viz flag."""
    if not _check_plotly():
        return

    print("\n  Generating Mars-grade Plotly visuals...")
    lunar_3d_constellation(constellation, bundles)
    custody_gantt(bundles or [])
    shackleton_heatmap()
    print("\n  ✅ All visuals saved to results/")
    print("  Open any .html file in your browser — fully interactive!")
