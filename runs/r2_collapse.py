#!/usr/bin/env python3
"""Historical R-squared collapse diagnostic; universal claim retired.

Computes R² of OLS regression ln(Φ) vs E[H] at each (config, p_eff),
then plots R² against the dimensionless variable x = p_eff × e^γ.

The original analysis asked whether curves collapse across 14 configurations
from three domains. Its cross-domain reference values mix conventions, and four
orbital values are hardcoded with unavailable source rows. This script preserves
the released diagnostic surface; it does not establish a universal boundary.

Two perturbation axes measured:
  - cross_n_orb: orbital bodies, E[H] varies via constellation density
  - cross_pair:  traces, E[H] varies via source-dest pair selection

The two perturbation axes remain separate evidence classes.

c₀ diagnostic: tracks the OLS intercept to check whether the collapse
absorbs a second parameter (p-dependent boundary correction).

Data sources:
  - Phase 1 phi_sweep (8 orbital, phi_time)
  - CRAWDAD Exp1-3, Exp6 (phi_normal)
  - SF Cab vehicular (phi_greedy)
  - Cambridge domain_transfer (phi_normal)
"""

import json
import math
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_LOCAL = _ROOT / "local_results" / "epyc_20260307_0906"
_RUNS = _ROOT / "runs"
_PHASE3 = _RUNS / "epyc_results" / "phase3"
_OUT = _RUNS / "r2_collapse_results.json"

# Reference p_ref for orbital bodies (closest grid point to 0.1)
_ORBITAL_REF_PREF = 0.1184

CLASS_MAP = {
    "ceres": "TRAP",
    "jupiter": "TRAP",
    "mercury": "TRAP",
    "saturn": "TRAP",
    "europa": "TRAP",
    "mars": "TRAP",
    "venus": "TRAP",
    "titan": "TRAP",
    "sfcab": "CLUSTER",
    "exp1": "CLUSTER",
    "exp2": "CLUSTER",
    "exp3": "CLUSTER",
    "exp6": "CLUSTER",
    "cambridge": "CLUSTER",
}

# Historical Table 3 γ values. The table mixes conventions, and the Mercury,
# Mars, Ceres, and Titan source rows were not recovered.
# Orbital: these are gamma_normal = raw_slope / (-mean_lambda), evaluated
#   at p_ref ≈ 0.1184.  (Confirmed: self-computed gamma_normal for Mercury
#   matches -0.976 ≈ -1.01.  Distant bodies have survivorship bias at this
#   p_ref, so self-computed slopes diverge.)
# CRAWDAD: these are raw slopes d[ln Φ]/d[E_H] at p_eff = 0.1.
#   (Self-computed matches to < 0.005 for all 4 traces.)
# SF Cab: Table 3 reports +0.99 = gamma_normal, not raw slope (raw ≈ 2.29).
# Cambridge: not in Table 3; use domain_transfer gamma_by_p at p=0.1.
TABLE3_GAMMA = {
    "ceres": -1.20,
    "jupiter": -1.14,
    "mercury": -1.01,
    "saturn": -0.67,
    "europa": -0.54,
    "mars": -0.40,
    "venus": -0.21,
    "titan": -0.10,
    "sfcab": +0.99,
    "exp1": +1.89,
    "exp2": +1.85,
    "exp3": +2.22,
    "exp6": +2.07,
    "cambridge": +1.65,
}


def ols_fit(ehs, lphis):
    """OLS of ln(Φ) vs E[H]. Returns (R², slope, intercept, n) or Nones."""
    n = len(ehs)
    if n < 5:
        return None, None, None, None
    if np.std(ehs) < 1e-12:
        return None, None, None, None
    slope, intercept = np.polyfit(ehs, lphis, 1)
    pred = slope * ehs + intercept
    ss_res = float(np.sum((lphis - pred) ** 2))
    ss_tot = float(np.sum((lphis - np.mean(lphis)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2, float(slope), float(intercept), n


def load_orbital():
    """Phase 1 phi_sweep: 8 orbital bodies, grouped by (target, p_ref)."""
    path = _LOCAL / "phi_sweep_results.json"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return []

    with open(path) as f:
        data = json.load(f)

    groups = {}
    for r in data["results"]:
        phi = r.get("phi_time", 0)
        eh = r.get("E_H", 0)
        if not (phi and phi > 0 and eh > 0):
            continue
        key = (r["target"], r["p_ref"])
        groups.setdefault(key, []).append((eh, math.log(phi), r["p_eff"]))

    results = []
    for (target, p_ref), pts in sorted(groups.items()):
        ehs = np.array([p[0] for p in pts])
        lphis = np.array([p[1] for p in pts])
        p_effs = np.array([p[2] for p in pts])

        r2, slope, intercept, n = ols_fit(ehs, lphis)
        if r2 is None:
            continue

        results.append(
            {
                "config": target,
                "p_ref": p_ref,
                "p_eff": float(np.mean(p_effs)),
                "r2": r2,
                "slope": slope,
                "intercept": intercept,
                "n_pts": n,
                "graph_class": CLASS_MAP[target],
                "perturbation_axis": "cross_n_orb",
            }
        )

    return results


def _load_trace(path, config, phi_key):
    """Generic loader for trace-based data (CRAWDAD / vehicular / Cambridge)."""
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    # Handle domain_transfer structure (nested under config name)
    if "rows" in data:
        raw_rows = data["rows"]
    elif "results" in data:
        raw_rows = data["results"]
    else:
        return []

    groups = {}
    for r in raw_rows:
        phi = r.get(phi_key, 0)
        eh = r.get("E_H", 0)
        if not (phi and phi > 0 and eh > 0):
            continue
        p = r.get("p_eff")
        if p is None:
            continue
        groups.setdefault(p, []).append((eh, math.log(phi)))

    results = []
    for p_eff, pts in sorted(groups.items()):
        ehs = np.array([p[0] for p in pts])
        lphis = np.array([p[1] for p in pts])

        r2, slope, intercept, n = ols_fit(ehs, lphis)
        if r2 is None:
            continue

        results.append(
            {
                "config": config,
                "p_ref": p_eff,
                "p_eff": float(p_eff),
                "r2": r2,
                "slope": slope,
                "intercept": intercept,
                "n_pts": n,
                "graph_class": CLASS_MAP[config],
                "perturbation_axis": "cross_pair",
            }
        )

    return results


def load_crawdad():
    """CRAWDAD Exp1-3, Exp6 from runs/."""
    results = []
    for trace in ["Exp1", "Exp2", "Exp3", "Exp6"]:
        path = _RUNS / f"crawdad_contacts.{trace}_results.json"
        results.extend(_load_trace(path, trace.lower(), "phi_normal"))
    return results


def load_vehicular():
    """SF Cab from Phase 3."""
    path = _PHASE3 / "vehicular_gamma_results.json"
    return _load_trace(path, "sfcab", "phi_greedy")


def load_cambridge():
    """Cambridge from domain_transfer."""
    path = _RUNS / "domain_transfer_results.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    cam = data.get("cambridge", {})
    if not cam.get("rows"):
        return []
    # Wrap in the format _load_trace expects
    return _load_trace_from_rows(cam["rows"], "cambridge", "phi_normal")


def _load_trace_from_rows(raw_rows, config, phi_key):
    """Load from pre-extracted rows list."""
    groups = {}
    for r in raw_rows:
        phi = r.get(phi_key, 0)
        eh = r.get("E_H", 0)
        if not (phi and phi > 0 and eh > 0):
            continue
        p = r.get("p_eff")
        if p is None:
            continue
        groups.setdefault(p, []).append((eh, math.log(phi)))

    results = []
    for p_eff, pts in sorted(groups.items()):
        ehs = np.array([p[0] for p in pts])
        lphis = np.array([p[1] for p in pts])

        r2, slope, intercept, n = ols_fit(ehs, lphis)
        if r2 is None:
            continue

        results.append(
            {
                "config": config,
                "p_ref": p_eff,
                "p_eff": float(p_eff),
                "r2": r2,
                "slope": slope,
                "intercept": intercept,
                "n_pts": n,
                "graph_class": CLASS_MAP[config],
                "perturbation_axis": "cross_pair",
            }
        )

    return results


def compute_reference_gamma(all_results):
    """Retain the historical reference values for diagnostic reproduction.

    Table 3 uses a mixed convention (gamma_normal for orbital, raw slope
    for CRAWDAD) but these are the published claims.  For the collapse
    plot, we need ONE consistent number per config.  Since the CRAWDAD
    self-computed values match Table 3 closely. The orbital values do not form
    a comparable, fully sourced cross-domain reference set.

    Falls back to self-computed slope at p≈0.1 for any config not in Table 3.
    """
    # Self-computed fallback
    by_config = {}
    for r in all_results:
        by_config.setdefault(r["config"], []).append(r)

    gamma_ref = {}
    gamma_source = {}
    for config, rows in by_config.items():
        if rows[0]["perturbation_axis"] == "cross_n_orb":
            # Orbital: use Table 3 (gamma_normal at p_ref≈0.1184).
            # Self-computed raw slopes have survivorship bias for distant bodies.
            if config in TABLE3_GAMMA:
                gamma_ref[config] = TABLE3_GAMMA[config]
                gamma_source[config] = "table3"
            else:
                best = min(rows, key=lambda r: abs(r["p_ref"] - _ORBITAL_REF_PREF))
                gamma_ref[config] = best["slope"]
                gamma_source[config] = "self-computed"
        else:
            # Cross-pair (traces): use self-computed raw slope at p≈0.1.
            # Gives consistent convention across all trace configs.
            # (CRAWDAD raw slopes match Table 3 to < 0.005.)
            best = min(rows, key=lambda r: abs(r["p_eff"] - 0.1))
            gamma_ref[config] = best["slope"]
            gamma_source[config] = "raw_slope"

    return gamma_ref, gamma_source


def annotate_results(all_results, gamma_ref):
    """Add derived quantities: k, x = p·k, c₀."""
    for r in all_results:
        g = gamma_ref[r["config"]]
        r["gamma_ref"] = g
        r["k"] = math.exp(g)
        r["x_pk"] = r["p_eff"] * math.exp(g)
        r["c0"] = r["intercept"]


def print_table(all_results, gamma_ref):
    """Print the full results table."""
    print("\n" + "=" * 110)
    print(
        f"{'Config':12s} {'Class':7s} {'p_eff':>8s} {'p·k':>8s} "
        f"{'R²':>6s} {'slope':>8s} {'c₀':>8s} {'n':>5s} {'axis':>12s}"
    )
    print("-" * 110)

    for r in sorted(
        all_results,
        key=lambda r: (
            0 if r["graph_class"] == "TRAP" else 1,
            r["config"],
            r["x_pk"],
        ),
    ):
        print(
            f"{r['config']:12s} {r['graph_class']:7s} "
            f"{r['p_eff']:8.4f} {r['x_pk']:8.4f} "
            f"{r['r2']:6.3f} {r['slope']:+8.4f} "
            f"{r['c0']:+8.4f} {r['n_pts']:5d} "
            f"{r['perturbation_axis']:>12s}"
        )

    # Summary statistics
    trap = [r for r in all_results if r["graph_class"] == "TRAP"]
    cluster = [r for r in all_results if r["graph_class"] == "CLUSTER"]

    print("\n" + "-" * 110)
    print("Summary:")
    print(
        f"  TRAP    ({len(trap):3d} points): "
        f"R² = {np.mean([r['r2'] for r in trap]):.3f} ± {np.std([r['r2'] for r in trap]):.3f}, "
        f"p·k range [{min(r['x_pk'] for r in trap):.4f}, {max(r['x_pk'] for r in trap):.4f}]"
    )
    print(
        f"  CLUSTER ({len(cluster):3d} points): "
        f"R² = {np.mean([r['r2'] for r in cluster]):.3f} ± {np.std([r['r2'] for r in cluster]):.3f}, "
        f"p·k range [{min(r['x_pk'] for r in cluster):.4f}, {max(r['x_pk'] for r in cluster):.4f}]"
    )

    # Perturbation axis comparison
    cross_norb = [r for r in all_results]
    ax_norb = [r for r in all_results if r["perturbation_axis"] == "cross_n_orb"]
    ax_pair = [r for r in all_results if r["perturbation_axis"] == "cross_pair"]
    print(
        f"\n  cross_n_orb ({len(ax_norb):3d} pts): all TRAP, "
        f"R² = {np.mean([r['r2'] for r in ax_norb]):.3f} ± {np.std([r['r2'] for r in ax_norb]):.3f}"
    )
    print(
        f"  cross_pair  ({len(ax_pair):3d} pts): mixed, "
        f"R² = {np.mean([r['r2'] for r in ax_pair]):.3f} ± {np.std([r['r2'] for r in ax_pair]):.3f}"
    )

    # c₀ diagnostic preview
    print("\n  c₀ by class (at reference p):")
    for config in sorted(gamma_ref):
        ref_pts = [r for r in all_results if r["config"] == config and abs(r["p_eff"] - 0.1) < 0.2]
        if ref_pts:
            best = min(ref_pts, key=lambda r: abs(r["p_eff"] - 0.1))
            print(
                f"    {config:12s} ({best['graph_class']:7s}): "
                f"c₀ = {best['c0']:+.3f}, R² = {best['r2']:.3f}"
            )


def make_plot(all_results, gamma_ref):
    """Three-panel historical diagnostic figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Historical collapse diagnostic — universal claim retired", fontsize=13)

    trap_configs = sorted(
        set(r["config"] for r in all_results if r["graph_class"] == "TRAP"),
        key=lambda c: gamma_ref[c],
    )
    cluster_configs = sorted(
        set(r["config"] for r in all_results if r["graph_class"] == "CLUSTER"),
        key=lambda c: gamma_ref[c],
    )

    n_trap = len(trap_configs)
    n_cluster = len(cluster_configs)
    trap_colors = plt.cm.Blues(np.linspace(0.35, 0.90, max(n_trap, 1)))
    cluster_colors = plt.cm.Reds(np.linspace(0.30, 0.95, max(n_cluster, 1)))

    cmap = {}
    for i, c in enumerate(trap_configs):
        cmap[c] = trap_colors[i]
    for i, c in enumerate(cluster_configs):
        cmap[c] = cluster_colors[i]

    marker_map = {"cross_n_orb": "o", "cross_pair": "s"}

    # ── Panel A: CLUSTER collapse (cross-pair, the clean test) ──
    ax = axes[0]
    for config in cluster_configs:
        pts = sorted(
            [r for r in all_results if r["config"] == config],
            key=lambda r: r["x_pk"],
        )
        if not pts:
            continue
        xs = [r["x_pk"] for r in pts]
        ys = [r["r2"] for r in pts]
        g = gamma_ref[config]
        lbl = f"{config} (γ={g:+.2f})"
        ax.plot(
            xs, ys, "-s", color=cmap[config], label=lbl, markersize=5, linewidth=1.5, alpha=0.85
        )

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.axvspan(0.5, 2.0, alpha=0.05, color="gray")
    ax.text(0.7, 1.02, "p·k = 1", fontsize=8, color="black", alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("p · k   (p_eff × e^γ)", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("(A) CLUSTER: cross-pair collapse", fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0.08, 15)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # ── Panel B: TRAP (cross-n_orb, perturbation-axis comparison) ──
    ax = axes[1]
    for config in trap_configs:
        pts = sorted(
            [r for r in all_results if r["config"] == config],
            key=lambda r: r["x_pk"],
        )
        if not pts:
            continue
        xs = [r["x_pk"] for r in pts]
        ys = [r["r2"] for r in pts]
        g = gamma_ref[config]
        lbl = f"{config} (γ={g:+.2f})"
        ax.plot(xs, ys, "-o", color=cmap[config], label=lbl, markersize=4, linewidth=1.0, alpha=0.8)

    # CLUSTER envelope for reference
    cluster_pts = [r for r in all_results if r["graph_class"] == "CLUSTER"]
    if cluster_pts:
        xs_c = sorted(set(round(r["x_pk"], 3) for r in cluster_pts))
        # Bin CLUSTER data for envelope
        bins = np.logspace(np.log10(0.1), np.log10(10), 15)
        for i in range(len(bins) - 1):
            in_bin = [r for r in cluster_pts if bins[i] <= r["x_pk"] < bins[i + 1]]
            if len(in_bin) >= 2:
                xc = math.sqrt(bins[i] * bins[i + 1])
                r2_lo = min(r["r2"] for r in in_bin)
                r2_hi = max(r["r2"] for r in in_bin)
                ax.plot(
                    [xc, xc],
                    [r2_lo, r2_hi],
                    color="salmon",
                    alpha=0.15,
                    linewidth=6,
                    solid_capstyle="round",
                )

    ax.set_xscale("log")
    ax.set_xlabel("p · k   (p_eff × e^γ)", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("(B) TRAP: cross-n_orb (structural noise)", fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.03,
        0.03,
        "R² low everywhere:\nvarying n_orb changes\ntopology, not just E[H]",
        transform=ax.transAxes,
        fontsize=7,
        style="italic",
        bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"),
    )

    # ── Panel C: c₀ diagnostic ──
    ax = axes[2]

    # CLUSTER only (cross-pair): R² vs |c₀|, colored by p·k
    c_pts = [r for r in all_results if r["graph_class"] == "CLUSTER"]
    if c_pts:
        c0s = [abs(r["c0"]) for r in c_pts]
        r2s = [r["r2"] for r in c_pts]
        pks = [r["x_pk"] for r in c_pts]
        sc = ax.scatter(
            c0s,
            r2s,
            c=pks,
            cmap="viridis",
            norm=plt.matplotlib.colors.LogNorm(vmin=0.05, vmax=10),
            marker="s",
            s=40,
            alpha=0.8,
            edgecolors="firebrick",
            linewidths=0.5,
        )
        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("p · k", fontsize=9)

    # TRAP
    t_pts = [r for r in all_results if r["graph_class"] == "TRAP"]
    if t_pts:
        c0s_t = [abs(r["c0"]) for r in t_pts]
        r2s_t = [r["r2"] for r in t_pts]
        ax.scatter(
            c0s_t, r2s_t, marker="o", s=15, alpha=0.3, color="steelblue", label="TRAP (cross-n_orb)"
        )

    ax.set_xlabel("|c₀|  (OLS intercept magnitude)", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("(C) c₀ diagnostic", fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="center right")

    ax.text(
        0.03,
        0.97,
        "If |c₀| explains scatter\naround collapse curve:\npoints should stratify\nby |c₀| at fixed p·k",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = _ROOT / "figures" / "r2_collapse.png"
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")


def main():
    print("=" * 60)
    print("HISTORICAL R² COLLAPSE DIAGNOSTIC — UNIVERSAL CLAIM RETIRED")
    print("=" * 60)

    print("\nLoading data...")
    orbital = load_orbital()
    crawdad = load_crawdad()
    vehicular = load_vehicular()
    cambridge = load_cambridge()

    all_results = orbital + crawdad + vehicular + cambridge

    n_configs = len(set(r["config"] for r in all_results))
    print(f"  Orbital:    {len(orbital):3d} groups (8 bodies × ~12 p_refs)")
    print(f"  CRAWDAD:    {len(crawdad):3d} groups (4 traces × 6 p_effs)")
    print(f"  Vehicular:  {len(vehicular):3d} groups (SF Cab × 4 p_effs)")
    print(f"  Cambridge:  {len(cambridge):3d} groups")
    print(f"  Total:      {len(all_results):3d} data points, {n_configs} configs")

    # Reference γ per config (Table 3 where available, self-computed fallback)
    gamma_ref, gamma_source = compute_reference_gamma(all_results)

    print("\nHistorical reference γ (mixed Table 3 values; not a universal scale):")
    print(f"  {'Config':12s} {'γ_ref':>8s} {'source':>14s} {'class':>8s}")
    print("  " + "-" * 50)
    for config in sorted(gamma_ref, key=lambda c: gamma_ref[c]):
        g = gamma_ref[config]
        src = gamma_source[config]
        cls = CLASS_MAP.get(config, "?")
        print(f"  {config:12s} {g:+8.2f} {src:>14s} {cls:>8s}")

    # Annotate all results with k, x = p·k, c₀
    annotate_results(all_results, gamma_ref)

    # Full table
    print_table(all_results, gamma_ref)

    # Save JSON
    output = {
        "description": (
            "Historical R² collapse diagnostic using mixed-convention reference gamma; "
            "the universal-boundary claim is retired"
        ),
        "gamma_ref": {k: round(v, 6) for k, v in gamma_ref.items()},
        "table3_gamma": TABLE3_GAMMA,
        "n_configs": n_configs,
        "n_points": len(all_results),
        "results": [
            {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}
            for r in all_results
        ],
    }
    with open(_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {_OUT}")

    # Plot
    make_plot(all_results, gamma_ref)

    print("\nDone.")


if __name__ == "__main__":
    main()
