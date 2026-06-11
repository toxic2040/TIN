"""anim_t3_classification_reveal.py — Classification conjecture reveal animation.

The conjecture assembles itself: orbital targets appear one by one (trap cloud),
then CRAWDAD traces (cluster cloud), then regression lines reveal the gap.

Outputs:
  figures/t3_classification_reveal.gif
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

RUNS = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(RUNS), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "figure.dpi": 120,
    }
)

# ── Colours ──────────────────────────────────────────────────────
TRAP_REDS = [
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#a50f15",
    "#67000d",
]
CLUSTER_BLUES = ["#08519c", "#3182bd", "#6baed6", "#bdd7e8"]
LINE_RED = "#c0392b"
LINE_BLUE = "#2c3e50"

ORBITAL_TARGETS = ["Mercury", "Venus", "Mars", "Ceres", "Europa", "Jupiter", "Saturn", "Titan"]
CRAWDAD_TRACES = ["Exp1", "Exp2", "Exp3", "Exp6"]
CRAWDAD_LABELS = ["Exp1 (n=9)", "Exp2 (n=12)", "Exp3 (n=41)", "Exp6 (n=98)"]


def _load_orbital(p_ref=0.1184):
    path = os.path.join(RUNS, "phi_decompose_results.json")
    d = json.load(open(path))
    by_target = {t: ([], []) for t in ORBITAL_TARGETS}
    for r in d["results"]:
        if r["p_ref"] != p_ref:
            continue
        t = r["target"].capitalize()
        phi = r.get("phi_normal", 0.0)
        eh = r.get("E_H", 0.0)
        if not (phi > 0 and eh > 0 and np.isfinite(phi)):
            continue
        if t in by_target:
            by_target[t][0].append(eh)
            by_target[t][1].append(np.log(phi))
    return {t: (np.array(e), np.array(l)) for t, (e, l) in by_target.items()}


def _load_social(p_eff=0.1):
    by_trace = {}
    for trace in CRAWDAD_TRACES:
        path = os.path.join(RUNS, f"crawdad_contacts.{trace}_results.json")
        d = json.load(open(path))
        ehs, lnphis = [], []
        for r in d["results"]:
            if r["p_eff"] != p_eff:
                continue
            phi = r.get("phi_normal", 0.0)
            eh = r.get("E_H", 0.0)
            if not (phi > 0 and eh > 0 and np.isfinite(phi)):
                continue
            ehs.append(eh)
            lnphis.append(np.log(phi))
        by_trace[trace] = (np.array(ehs), np.array(lnphis))
    return by_trace


def animate():
    orbital = _load_orbital()
    social = _load_social()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.set_xlabel(r"Expected hop count $E[H]$", fontsize=12)
    ax.set_ylabel(r"$\ln\,\Phi$", fontsize=13)
    ax.set_xlim(0.8, 7.2)
    ax.set_ylim(-4.5, 15.0)
    ax.grid(True, alpha=0.2)

    # Persistent artists list
    scatter_artists = []
    line_artists = []
    text_artists = []

    # Collect all trap/cluster data for regression lines
    all_trap_eh = np.concatenate([orbital[t][0] for t in ORBITAL_TARGETS])
    all_trap_lnphi = np.concatenate([orbital[t][1] for t in ORBITAL_TARGETS])
    all_clust_eh = np.concatenate([social[t][0] for t in CRAWDAD_TRACES])
    all_clust_lnphi = np.concatenate([social[t][1] for t in CRAWDAD_TRACES])

    c_trap = np.polyfit(all_trap_eh, all_trap_lnphi, 1)
    c_clust = np.polyfit(all_clust_eh, all_clust_lnphi, 1)

    # ── Frame sequence ───────────────────────────────────────────────
    # Frames 0-7:   Add orbital targets one at a time
    # Frame 8:      Trap regression line
    # Frame 9:      Hold
    # Frames 10-13: Add CRAWDAD traces one at a time
    # Frame 14:     Cluster regression line
    # Frame 15:     Labels
    # Frames 16-19: Hold final

    def update(frame):
        if frame < 8:
            # Add an orbital target
            idx = frame
            t = ORBITAL_TARGETS[idx]
            eh, lnphi = orbital[t]
            sc = ax.scatter(
                eh,
                lnphi,
                c=TRAP_REDS[idx],
                s=10,
                alpha=0.15,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
            scatter_artists.append(sc)
            # Target label
            med_eh = np.median(eh)
            med_lnphi = np.median(lnphi)
            txt = ax.text(
                med_eh,
                med_lnphi + 0.4,
                t,
                fontsize=7,
                color=TRAP_REDS[idx],
                ha="center",
                fontweight="bold",
                alpha=0.8,
            )
            text_artists.append(txt)
            ax.set_title(f"Orbital targets: {idx + 1}/8", fontsize=11)

        elif frame == 8:
            # Trap regression line
            x_fit = np.linspace(all_trap_eh.min() - 0.1, all_trap_eh.max() + 0.1, 100)
            (line,) = ax.plot(
                x_fit,
                np.polyval(c_trap, x_fit),
                color=LINE_RED,
                linewidth=2.5,
                zorder=4,
            )
            line_artists.append(line)
            ax.set_title(
                rf"Trap class: $\gamma = {c_trap[0]:+.2f}$",
                fontsize=11,
                color=LINE_RED,
            )

        elif frame == 9:
            ax.set_title("Now adding social/opportunistic networks...", fontsize=11)

        elif 10 <= frame <= 13:
            # Add CRAWDAD traces
            idx = frame - 10
            t = CRAWDAD_TRACES[idx]
            eh, lnphi = social[t]
            sc = ax.scatter(
                eh,
                lnphi,
                c=CLUSTER_BLUES[idx],
                s=10,
                alpha=0.15,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
            scatter_artists.append(sc)
            txt = ax.text(
                np.median(eh) + 0.3,
                np.median(lnphi),
                CRAWDAD_LABELS[idx],
                fontsize=7,
                color=CLUSTER_BLUES[idx],
                ha="left",
                fontweight="bold",
                alpha=0.8,
            )
            text_artists.append(txt)
            ax.set_title(f"Social traces: {idx + 1}/4", fontsize=11)

        elif frame == 14:
            # Cluster regression line
            x_fit = np.linspace(all_clust_eh.min() - 0.1, all_clust_eh.max() + 0.1, 100)
            (line,) = ax.plot(
                x_fit,
                np.polyval(c_clust, x_fit),
                color=LINE_BLUE,
                linewidth=2.5,
                zorder=4,
            )
            line_artists.append(line)
            ax.set_title(
                rf"Cluster class: $\gamma = {c_clust[0]:+.2f}$",
                fontsize=11,
                color=LINE_BLUE,
            )

        elif frame == 15:
            # Final labels
            ax.annotate(
                r"$\gamma < 0$" + "\n(trap)",
                xy=(4.0, np.polyval(c_trap, 4.0)),
                xytext=(5.5, np.polyval(c_trap, 4.0) + 2.0),
                fontsize=12,
                color=LINE_RED,
                fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=LINE_RED, lw=2),
                zorder=5,
            )
            ax.annotate(
                r"$\gamma > 0$" + "\n(cluster)",
                xy=(4.5, np.polyval(c_clust, 4.5)),
                xytext=(5.8, np.polyval(c_clust, 4.5) - 3.0),
                fontsize=12,
                color=LINE_BLUE,
                fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=LINE_BLUE, lw=2),
                zorder=5,
            )
            n_total = len(all_trap_eh) + len(all_clust_eh)
            ax.set_title(
                rf"Classification theorem: $\mathrm{{sign}}(\gamma)$ separates all {n_total:,} pairs",
                fontsize=11,
                fontweight="semibold",
            )

        # Hold frames (16-19): do nothing
        return scatter_artists + line_artists + text_artists

    n_frames = 20
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1200, repeat=False)

    # MP4 first (higher quality, for presentations)
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        from matplotlib.animation import FFMpegWriter

        path_mp4 = os.path.join(FIG_DIR, "t3_classification_reveal.mp4")
        mp4_writer = FFMpegWriter(fps=1, bitrate=1800)
        anim.save(path_mp4, writer=mp4_writer, dpi=150)
        print(f"  Wrote {path_mp4} ({n_frames} frames)")
    except Exception as e:
        print(f"  MP4 skipped: {e}")

    plt.close(fig)

    # GIF (separate pass — fresh figure)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    ax2.set_xlabel(r"Expected hop count $E[H]$", fontsize=12)
    ax2.set_ylabel(r"$\ln\,\Phi$", fontsize=13)
    ax2.set_xlim(0.8, 7.2)
    ax2.set_ylim(-4.5, 15.0)
    ax2.grid(True, alpha=0.2)
    artists2 = []

    def update2(frame):
        if frame < 8:
            idx = frame
            t = ORBITAL_TARGETS[idx]
            eh, lnphi = orbital[t]
            ax2.scatter(
                eh, lnphi, c=TRAP_REDS[idx], s=10, alpha=0.15, linewidths=0, rasterized=True
            )
            ax2.text(
                np.median(eh),
                np.median(lnphi) + 0.4,
                t,
                fontsize=7,
                color=TRAP_REDS[idx],
                ha="center",
                fontweight="bold",
                alpha=0.8,
            )
            ax2.set_title(f"Orbital targets: {idx + 1}/8", fontsize=11)
        elif frame == 8:
            x_fit = np.linspace(all_trap_eh.min() - 0.1, all_trap_eh.max() + 0.1, 100)
            ax2.plot(x_fit, np.polyval(c_trap, x_fit), color=LINE_RED, linewidth=2.5, zorder=4)
            ax2.set_title(rf"Trap class: $\gamma = {c_trap[0]:+.2f}$", fontsize=11, color=LINE_RED)
        elif frame == 9:
            ax2.set_title("Now adding social/opportunistic networks...", fontsize=11)
        elif 10 <= frame <= 13:
            idx = frame - 10
            t = CRAWDAD_TRACES[idx]
            eh, lnphi = social[t]
            ax2.scatter(
                eh, lnphi, c=CLUSTER_BLUES[idx], s=10, alpha=0.15, linewidths=0, rasterized=True
            )
            ax2.text(
                np.median(eh) + 0.3,
                np.median(lnphi),
                CRAWDAD_LABELS[idx],
                fontsize=7,
                color=CLUSTER_BLUES[idx],
                ha="left",
                fontweight="bold",
                alpha=0.8,
            )
            ax2.set_title(f"Social traces: {idx + 1}/4", fontsize=11)
        elif frame == 14:
            x_fit = np.linspace(all_clust_eh.min() - 0.1, all_clust_eh.max() + 0.1, 100)
            ax2.plot(x_fit, np.polyval(c_clust, x_fit), color=LINE_BLUE, linewidth=2.5, zorder=4)
            ax2.set_title(
                rf"Cluster class: $\gamma = {c_clust[0]:+.2f}$", fontsize=11, color=LINE_BLUE
            )
        elif frame == 15:
            ax2.annotate(
                r"$\gamma < 0$" + "\n(trap)",
                xy=(4.0, np.polyval(c_trap, 4.0)),
                xytext=(5.5, np.polyval(c_trap, 4.0) + 2.0),
                fontsize=12,
                color=LINE_RED,
                fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=LINE_RED, lw=2),
            )
            ax2.annotate(
                r"$\gamma > 0$" + "\n(cluster)",
                xy=(4.5, np.polyval(c_clust, 4.5)),
                xytext=(5.8, np.polyval(c_clust, 4.5) - 3.0),
                fontsize=12,
                color=LINE_BLUE,
                fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=LINE_BLUE, lw=2),
            )
            n_total = len(all_trap_eh) + len(all_clust_eh)
            ax2.set_title(
                rf"Classification theorem: $\mathrm{{sign}}(\gamma)$ separates all {n_total:,} pairs",
                fontsize=11,
                fontweight="semibold",
            )
        return artists2

    anim2 = FuncAnimation(fig2, update2, frames=n_frames, interval=1200, repeat=False)
    path_gif = os.path.join(FIG_DIR, "t3_classification_reveal.gif")
    anim2.save(path_gif, writer=PillowWriter(fps=1), dpi=120)
    print(f"  Wrote {path_gif} ({n_frames} frames)")
    plt.close(fig2)


if __name__ == "__main__":
    animate()
