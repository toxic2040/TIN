cat > tin/core/viz.py << 'EOT'
import plotly.graph_objects as go
import numpy as np
import os

PRIORITY_EMERGENCY = 0

def lunar_3d_constellation(constellation, bundles=None):
    fig = go.Figure()

    # Moon globe
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = 1737.4 * np.outer(np.cos(u), np.sin(v))
    y = 1737.4 * np.outer(np.sin(u), np.sin(v))
    z = 1737.4 * np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='gray', opacity=0.95, name="Moon"))

    # Shackleton
    fig.add_trace(go.Scatter3d(x=[-280], y=[-320], z=[-1730], mode='markers+text',
                               marker=dict(size=18, color='#00ff88', symbol='diamond'),
                               text=["🟢 Shackleton"], textposition="top center"))

    # Satellites + orbits
    for sat in getattr(constellation, 'satellites', []):
        theta = np.linspace(0, 2*np.pi, 120)
        r = sat.get('a_km', 2137)
        ox = r * np.cos(theta)
        oy = r * np.sin(theta)
        oz = np.linspace(-200, 200, 120)
        fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode='lines',
                                   line=dict(color='#00ffff', width=5), name=sat.get('id', 'SAT')))

    # ELFO hub
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1950], mode='markers+text',
                               marker=dict(size=16, color='#ff00ff', symbol='star'),
                               text=["⭐ ELFO-HUB"]))

    # Emergency bundle trails (red flare)
    if bundles:
        for b in bundles:
            color = '#ff2222' if getattr(b, 'priority', 99) == PRIORITY_EMERGENCY else '#00ffcc'
            path_x = [-380, -280, 80, 0, 3800]
            path_y = [-380, -280, 80, 0, 0]
            path_z = [-1720, -1720, 380, 1950, 4200]
            fig.add_trace(go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines+markers',
                                       line=dict(color=color, width=8, dash='dash'),
                                       marker=dict(size=9, color=color),
                                       name=f"🚨 Bundle {getattr(b, 'bundle_id', 'N/A')}"))

    fig.update_layout(
        title="TIN Lunar Constellation — Live 3D View (Interactive)",
        scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)',
                   bgcolor='#05050f', aspectmode='cube'),
        paper_bgcolor='#05050f',
        font=dict(color='white')
    )
    os.makedirs("results", exist_ok=True)
    fig.write_html("results/lunar_3d_constellation.html")
    return fig

def custody_gantt(bundles):
    fig = go.Figure()
    for i, b in enumerate(bundles):
        color = '#ff2222' if getattr(b, 'priority', 99) == PRIORITY_EMERGENCY else '#00ffcc'
        for hop in getattr(b, 'hop_log', []):
            fig.add_trace(go.Scatter(x=[hop.get('depart_s',0)/60, hop.get('arrive_s',10)/60],
                                     y=[i, i], mode='lines', line=dict(color=color, width=14),
                                     hovertemplate=f"{hop.get('from','?')} → {hop.get('to','?')}",
                                     name=f"Bundle {getattr(b, 'bundle_id', 'N/A')}"))
    fig.update_layout(title="TIN BPv7 Custody Timeline (Animated)", xaxis_title="Time (minutes)",
                      yaxis_title="Bundle ID", plot_bgcolor='#05050f', paper_bgcolor='#05050f',
                      font=dict(color='white'), height=650)
    fig.write_html("results/custody_gantt.html")
    return fig

def shackleton_heatmap():
    lat = np.linspace(-90, -80, 120)
    lon = np.linspace(-25, 25, 120)
    lat, lon = np.meshgrid(lat, lon)
    coverage = 100 - 8 * np.exp(-((lat + 89.2)**2 + (lon + 2)**2) / 8)
    psr = np.where((lat < -88.6) & (np.abs(lon) < 6), 35, 0)

    fig = go.Figure()
    fig.add_trace(go.Contour(z=coverage - psr, x=lon[0], y=lat[:,0], colorscale='Plasma', contours=dict(showlabels=True)))
    fig.add_trace(go.Contour(z=psr, x=lon[0], y=lat[:,0], colorscale='Blues_r', opacity=0.75))
    fig.update_layout(title="Shackleton Crater — Coverage Heatmap + PSR Glow", xaxis_title="Longitude (°E)",
                      yaxis_title="Latitude (°S)", plot_bgcolor='#05050f', paper_bgcolor='#05050f',
                      font=dict(color='white'), height=700)
    fig.write_html("results/shackleton_heatmap.html")
    return fig

def make_all_viz(constellation=None, bundles=None):
    print("🌕 Generating Mars-grade visuals (3D + Gantt + Heatmap)...")
    lunar_3d_constellation(constellation, bundles)
    custody_gantt(bundles or [])
    shackleton_heatmap()
    print("✅ Visuals saved to results/ folder!")
    print("   📂 lunar_3d_constellation.html   ← rotate, zoom, bundle trails")
    print("   📂 custody_gantt.html            ← animated custody timeline")
    print("   📂 shackleton_heatmap.html       ← PSR glow + crater details")
    print("Open any in browser — fully interactive!")
EOT
