#!/usr/bin/env python3
"""Gamma–chi duality plot: Var(chi_DR) vs gamma across all domains.

Shows that vulnerability variance separates by class:
  - Trap configs: hub-concentrated fragility (high or heterogeneous Var(chi))
  - Cluster configs: uniformly distributed vulnerability (tight Var(chi) band)

Each trace traces a trajectory through (gamma, Var(chi)) space as p_eff varies.
Cluster traces: median Var(chi) across (s,d) pairs with IQR whiskers.
Trap traces: single (source, dest) pair per p_eff.

Data sources:
  - runs/chi_completion_results.json  (CRAWDAD per-pair chi detail + gamma)
  - runs/susceptibility_results.json  (Moon, Mars per-node chi)
"""

import json
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


# ── Trace display config ────────────────────────────────────────────
TRACE_STYLE = {
    #                    marker  color           short_label
    "Moon_8polar_ELFO": ("D", "#c0392b", "Moon"),
    "Mars_6polar": ("o", "#e67e22", "Mars"),
    "CRAWDAD_Exp1_n9": ("s", "#2980b9", "Exp1"),
    "CRAWDAD_Exp2_n12": ("^", "#27ae60", "Exp2"),
    "CRAWDAD_Exp3_n41": ("H", "#3498db", "Exp3"),
    "Cambridge_SR10_n36": ("p", "#8e44ad", "Camb."),
}


def _load_trace_trajectories():
    """Build per-trace trajectories: list of (gamma, median_var, lo_iqr, hi_iqr, p_eff)."""
    trajectories = {}  # trace_key -> list of dicts

    # ── 1. Cluster traces (CRAWDAD + Cambridge) ──
    with open("runs/chi_completion_results.json") as f:
        chi_data = json.load(f)

    # Build gamma lookup: trace_key -> {p_eff -> gamma}
    gamma_lookup = {}
    for pt in chi_data["duality_points"]:
        lab = pt["label"]
        parts = lab.rsplit("_p", 1)
        trace_key = parts[0]
        p_eff = float(parts[1]) if len(parts) > 1 else None
        if p_eff is not None:
            gamma_lookup.setdefault(trace_key, {})[p_eff] = pt["gamma"]

    for trace_key, trace_data in chi_data["chi_results"].items():
        by_p = {}
        for entry in trace_data["detail"]:
            p = entry["p_eff"]
            var_chi = entry["std_chi"] ** 2
            by_p.setdefault(p, []).append(var_chi)

        traj = []
        for p_eff in sorted(by_p.keys()):
            vals = np.array(by_p[p_eff])
            gamma = gamma_lookup.get(trace_key, {}).get(p_eff)
            if gamma is None:
                continue
            traj.append(
                {
                    "p_eff": p_eff,
                    "gamma": gamma,
                    "var_chi_median": float(np.median(vals)),
                    "var_chi_q25": float(np.percentile(vals, 25)),
                    "var_chi_q75": float(np.percentile(vals, 75)),
                    "n_pairs": len(vals),
                    "graph_class": "cluster",
                }
            )
        trajectories[trace_key] = traj

    # ── 2. Trap traces (Moon, Mars) ──
    with open("runs/susceptibility_results.json") as f:
        susc = json.load(f)

    trap_gammas = {"Moon_8polar_ELFO": -0.50, "Mars_6polar": -0.30}

    for r in susc["results"]:
        label = r["label"]
        gamma = trap_gammas.get(label)
        if gamma is None:
            continue
        traj = []
        for sweep in r["p_eff_sweeps"]:
            chi_vals = [n["chi_dr"] for n in sweep["nodes"]]
            var_chi = float(np.var(chi_vals))
            traj.append(
                {
                    "p_eff": sweep["p_eff"],
                    "gamma": gamma,
                    "var_chi_median": var_chi,
                    "var_chi_q25": var_chi,  # single measurement, no IQR
                    "var_chi_q75": var_chi,
                    "n_pairs": 1,
                    "graph_class": "trap",
                }
            )
        trajectories[label] = traj

    return trajectories


def _plot(trajectories, outpath="figures/fig_gamma_chi_duality.pdf"):
    """Single-panel trajectory plot."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ── Gap band ──
    ax.axvspan(-0.19, 0.74, color="#e0e0e0", alpha=0.35, zorder=0, label="_nolegend_")
    ax.axvline(0, color="#bbb", ls=":", lw=0.6, zorder=1)
    ax.text(
        0.275,
        1.8,
        r"$\Delta\gamma \geq 0.93$",
        fontsize=8,
        ha="center",
        va="top",
        color="#888",
        style="italic",
    )

    legend_handles = []

    for trace_key, traj in trajectories.items():
        if not traj:
            continue
        marker, color, short = TRACE_STYLE[trace_key]
        cls = traj[0]["graph_class"]

        gammas = [t["gamma"] for t in traj]
        medians = [t["var_chi_median"] for t in traj]
        q25s = [t["var_chi_q25"] for t in traj]
        q75s = [t["var_chi_q75"] for t in traj]
        p_effs = [t["p_eff"] for t in traj]

        # Connect trajectory with thin line
        ax.plot(gammas, medians, color=color, lw=1.2, alpha=0.5, zorder=3)

        # IQR whiskers (vertical error bars)
        yerr_lo = [m - q for m, q in zip(medians, q25s)]
        yerr_hi = [q - m for m, q in zip(medians, q75s)]
        has_iqr = any(lo > 0 or hi > 0 for lo, hi in zip(yerr_lo, yerr_hi))

        if has_iqr:
            ax.errorbar(
                gammas,
                medians,
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                ecolor=color,
                elinewidth=0.8,
                capsize=2,
                capthick=0.7,
                alpha=0.5,
                zorder=3,
            )

        # Markers sized by p_eff (larger = denser network)
        for i, t in enumerate(traj):
            size = {0.1: 50, 0.3: 80, 0.5: 120}.get(t["p_eff"], 70)
            ax.scatter(
                t["gamma"],
                t["var_chi_median"],
                marker=marker,
                s=size,
                c=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )

        # ── Flow arrow: shows direction of increasing p_eff ──────────────
        # Short arrowhead placed at the midpoint of the first segment
        # (p=0.1 → p=0.3), interpolated log-linearly in y.
        if len(traj) >= 2:
            x0 = traj[0]["gamma"]
            y0 = traj[0]["var_chi_median"]
            x1 = traj[1]["gamma"]
            y1 = traj[1]["var_chi_median"]
            flo, fhi = 0.35, 0.65
            xa = x0 + flo * (x1 - x0)
            ya = np.exp(np.log(y0) + flo * (np.log(y1) - np.log(y0)))
            xb = x0 + fhi * (x1 - x0)
            yb = np.exp(np.log(y0) + fhi * (np.log(y1) - np.log(y0)))
            ax.annotate(
                "",
                xy=(xb, yb),
                xytext=(xa, ya),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.3,
                    alpha=0.75,
                    mutation_scale=9,
                ),
                zorder=4,
            )

        # Trace label: (dx_pt, dy_pt, ref_idx, ha, rotation_deg)
        label_cfg = {
            "Moon_8polar_ELFO": (0, -12, 0, "center", 0),
            "Mars_6polar": (-10, 0, 1, "right", 0),
            "CRAWDAD_Exp1_n9": (-10, 0, 2, "right", 0),
            # Exp2: horizontal, left of p=0.1 point — mirrors Camb. on the opposite side
            "CRAWDAD_Exp2_n12": (-16, 7, 0, "right", 0),
            "CRAWDAD_Exp3_n41": (8, 6, 0, "left", 0),
            "Cambridge_SR10_n36": (8, 0, 1, "left", 0),
        }
        dx, dy, idx, ha, rot = label_cfg.get(trace_key, (8, 0, 0, "left", 0))
        idx = min(idx, len(traj) - 1)
        ref = traj[idx]
        ax.annotate(
            short,
            (ref["gamma"], ref["var_chi_median"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold",
            color=color,
            ha=ha,
            va="center",
            rotation=rot,
            zorder=6,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7,
                lw=1.2,
                label=f"{short} ({cls})",
            )
        )

    # ── Size legend for p_eff ──
    size_handles = []
    for p, s in [(0.1, 50), (0.3, 80), (0.5, 120)]:
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#999",
                markeredgecolor="white",
                markersize=np.sqrt(s) * 0.7,
                lw=0,
                label=f"p={p}",
            )
        )

    # Compact legend: traces + sizes side by side
    leg1 = ax.legend(
        handles=legend_handles,
        fontsize=6.5,
        loc="upper right",
        title="Trace (class)",
        title_fontsize=7,
        framealpha=0.85,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=size_handles,
        fontsize=6,
        loc="lower right",
        title=r"$p_{\mathrm{eff}}$",
        title_fontsize=6.5,
        framealpha=0.85,
        handletextpad=0.3,
        borderpad=0.3,
        labelspacing=0.25,
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma$ (classification index)", fontsize=11)
    ax.set_ylabel(r"Var($\chi_{\mathrm{DR}}$)  [node vulnerability spread]", fontsize=11)
    ax.set_xlim(-0.75, 1.85)
    ax.set_ylim(2e-3, 2.0)
    ax.tick_params(labelsize=9)

    # ── Hub amplification annotation (Moon) ──────────────────────────────
    ax.annotate(
        "hub\namplifies",
        xy=(-0.50, 1.07),
        xytext=(-0.50, 0.07),
        fontsize=8,
        color="#c0392b",
        ha="center",
        va="top",
        fontweight="semibold",
        arrowprops={"arrowstyle": "-|>", "color": "#c0392b", "lw": 1.6, "alpha": 0.85},
        zorder=6,
    )
    # Box centred on γ=0 vertical line, mid-height of axes
    # γ=0 in axes fraction: (0 - xlim_lo) / (xlim_hi - xlim_lo) = 0.75/2.60 ≈ 0.288
    ax.text(
        0.288,
        0.50,
        "Trap/cluster median ratio:\n"
        r"$2.7\times$ at $p{=}0.1$"
        "\n"
        r"$137\times$ at $p{=}0.5$",
        transform=ax.transAxes,
        fontsize=7.5,
        va="center",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f5f5f5",
            "edgecolor": "#ccc",
            "alpha": 0.9,
        },
        zorder=7,
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def main():
    trajectories = _load_trace_trajectories()

    # Summary table
    print("\n=== Gamma–Chi Duality: Trace Trajectories ===\n")
    print(
        f"{'Trace':25s}  {'Class':8s}  {'p_eff':>5s}  {'gamma':>7s}  {'Var(chi)':>9s}  {'IQR':>15s}  {'n':>3s}"
    )
    print("-" * 82)
    for trace_key in TRACE_STYLE:
        traj = trajectories.get(trace_key, [])
        for t in traj:
            iqr = f"[{t['var_chi_q25']:.4f}, {t['var_chi_q75']:.4f}]"
            print(
                f"{trace_key:25s}  {t['graph_class']:8s}  {t['p_eff']:5.2f}  "
                f"{t['gamma']:+7.3f}  {t['var_chi_median']:9.4f}  {iqr:>15s}  {t['n_pairs']:3d}"
            )

    # Class separation at each p_eff
    print("\n=== Class Separation by p_eff ===\n")
    for p in [0.1, 0.3, 0.5]:
        trap_vals = []
        cluster_vals = []
        for traj in trajectories.values():
            for t in traj:
                if abs(t["p_eff"] - p) < 0.01:
                    if t["graph_class"] == "trap":
                        trap_vals.append(t["var_chi_median"])
                    else:
                        cluster_vals.append(t["var_chi_median"])
        if trap_vals and cluster_vals:
            ratio = np.median(trap_vals) / np.median(cluster_vals)
            print(
                f"  p={p}: trap median={np.median(trap_vals):.4f}, "
                f"cluster median={np.median(cluster_vals):.4f}, "
                f"ratio={ratio:.1f}×"
            )

    _plot(trajectories)
    print("\nDone.")


if __name__ == "__main__":
    main()
