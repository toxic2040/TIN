#!/usr/bin/env python3
"""tin_trade_study.py — N-sat polar coverage/latency/cost trade study for TIN v0.4.0

Imports compute_coverage() from tin_coverage_sim.py (must be in same directory).
Runs parameterized sweep across constellation sizes and altitudes, with/without ELFO relay.
Outputs JSON summary + matplotlib trade study plot.

Usage:
    python scripts/tin_trade_study.py --include_elfo
    python scripts/tin_trade_study.py --no_elfo
    python scripts/tin_trade_study.py                  # defaults to --include_elfo

Dependencies: numpy, matplotlib (+ tin_coverage_sim.py in scripts/)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend — works headless and in SSH
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import compute_coverage from the canonical tin_coverage_sim.py
# ---------------------------------------------------------------------------
# Ensure scripts/ directory is on the path so co-located import works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tin_coverage_sim import compute_coverage
except ImportError as e:
    raise ImportError(
        "Cannot import compute_coverage from tin_coverage_sim.py — "
        "ensure the file exists in scripts/ and has no syntax errors."
    ) from e

# ---------------------------------------------------------------------------
# Trade study configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    {"n_sats": 4, "alt_km": 400, "label": "4x400km"},
    {"n_sats": 6, "alt_km": 400, "label": "6x400km"},
    {"n_sats": 8, "alt_km": 400, "label": "8x400km"},
    {"n_sats": 10, "alt_km": 400, "label": "10x400km"},
    {"n_sats": 12, "alt_km": 400, "label": "12x400km"},
    {"n_sats": 8, "alt_km": 300, "label": "8x300km"},
    {"n_sats": 8, "alt_km": 500, "label": "8x500km"},
]

# Cost proxy: Starship-class rideshare estimate
SAT_MASS_KG = 24  # 12U cubesat baseline
COST_PER_KG_M = 1.0  # $M per kg to lunar orbit (rideshare)

# Simulation parameters
ELEV_MIN_DEG = 5.0
SIM_DAYS = 28
LAT_SOUTH_POLE = -89.5


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------
def run_sweep(include_elfo=True):
    """Run all configurations and return list of result dicts."""
    results = []
    mode = "ELFO-augmented" if include_elfo else "pure constellation"
    print(f"\n{'=' * 60}")
    print(f"  TIN v0.4.0 Trade Study — {mode}")
    print(f"  {len(CONFIGS)} configurations | {SIM_DAYS}-day sim | elev >= {ELEV_MIN_DEG} deg")
    print(f"{'=' * 60}\n")

    for cfg in CONFIGS:
        label = cfg["label"]
        print(
            f"  Running: {label} {'+ ELFO' if include_elfo else '(pure)'}...", end=" ", flush=True
        )

        cov_pct, worst_gap_min, avg_gap_min = compute_coverage(
            n_sats=cfg["n_sats"],
            alt_km=cfg["alt_km"],
            elev_min=ELEV_MIN_DEG,
            sim_days=SIM_DAYS,
            lat_deg=LAT_SOUTH_POLE,
            include_elfo=include_elfo,
        )

        cost_est = cfg["n_sats"] * SAT_MASS_KG * COST_PER_KG_M

        result = {
            "config": label,
            "n_sats": cfg["n_sats"],
            "alt_km": cfg["alt_km"],
            "include_elfo": include_elfo,
            "south_pole_coverage_pct": round(float(cov_pct), 2),
            "worst_gap_min": round(float(worst_gap_min), 2),
            "avg_gap_min": round(float(avg_gap_min), 2),
            "cost_proxy_M_usd": round(float(cost_est), 1),
        }
        results.append(result)
        print(
            f"cov={result['south_pole_coverage_pct']:.1f}%  "
            f"worst_gap={result['worst_gap_min']:.1f}min  "
            f"cost=${result['cost_proxy_M_usd']:.0f}M"
        )

    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print formatted console summary table."""
    print(
        f"\n{'Config':<16} {'ELFO':>5} {'Cov%':>7} {'Worst Gap':>11} {'Avg Gap':>10} {'Cost $M':>9}"
    )
    print("-" * 62)
    for r in results:
        elfo_str = "+ELFO" if r["include_elfo"] else "pure"
        print(
            f"{r['config']:<16} {elfo_str:>5} {r['south_pole_coverage_pct']:>6.1f}% "
            f"{r['worst_gap_min']:>10.1f}m {r['avg_gap_min']:>9.1f}m "
            f"{r['cost_proxy_M_usd']:>8.1f}"
        )


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def save_json(results, outdir):
    """Save results to timestamped JSON file."""
    os.makedirs(outdir, exist_ok=True)
    fname = f"trade_study_{datetime.now():%Y%m%d_%H%M}.json"
    outpath = os.path.join(outdir, fname)
    with open(outpath, "w") as f:
        json.dump(
            {
                "generated": datetime.now().isoformat(),
                "tin_version": "0.4.0-dev",
                "sim_days": SIM_DAYS,
                "elev_min_deg": ELEV_MIN_DEG,
                "lat_deg": LAT_SOUTH_POLE,
                "sat_mass_kg": SAT_MASS_KG,
                "cost_per_kg_M": COST_PER_KG_M,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nJSON saved: {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Matplotlib visualization
# ---------------------------------------------------------------------------
def plot_trade_study(results, outdir):
    """Generate 3-axis bar+line trade study chart (coverage, worst gap, cost)."""
    os.makedirs(outdir, exist_ok=True)

    labels = [r["config"] + (" +ELFO" if r["include_elfo"] else " pure") for r in results]
    covs = [r["south_pole_coverage_pct"] for r in results]
    worst_gaps = [r["worst_gap_min"] for r in results]
    costs = [r["cost_proxy_M_usd"] for r in results]

    # --- TIN brand-aligned color scheme ---
    COV_COLOR = "#2E75B6"  # navy blue
    GAP_COLOR = "#E87722"  # orange
    COST_COLOR = "#2EA043"  # green
    BG_COLOR = "#F7F9FC"  # light background

    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax1.set_facecolor(BG_COLOR)

    # --- Coverage bars ---
    x = np.arange(len(labels))
    bar_width = 0.55
    bars = ax1.bar(x, covs, bar_width, color=COV_COLOR, alpha=0.8, label="Coverage %", zorder=3)
    ax1.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax1.set_ylabel("South Pole Coverage (%)", color=COV_COLOR, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COV_COLOR)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Value labels on bars
    for bar, cov in zip(bars, covs):
        ypos = bar.get_height() + 1.0
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{cov:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COV_COLOR,
            fontweight="bold",
        )

    # --- Worst gap line (right axis 1) ---
    ax2 = ax1.twinx()
    line_gap = ax2.plot(
        x,
        worst_gaps,
        color=GAP_COLOR,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Worst Gap (min)",
        zorder=5,
    )
    ax2.set_ylabel("Worst Gap (min)", color=GAP_COLOR, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=GAP_COLOR)
    # Don't let the gap axis go below 0
    max_gap = max(worst_gaps) if worst_gaps else 10
    ax2.set_ylim(0, max_gap * 1.3)

    # Gap value annotations
    for xi, gap in zip(x, worst_gaps):
        ax2.annotate(
            f"{gap:.1f}m",
            (xi, gap),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color=GAP_COLOR,
            fontweight="bold",
        )

    # --- Cost line (right axis 2, offset outward) ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 70))
    line_cost = ax3.plot(
        x,
        costs,
        color=COST_COLOR,
        marker="s",
        linewidth=2,
        markersize=7,
        linestyle="--",
        label="Cost Proxy ($M)",
        zorder=4,
    )
    ax3.set_ylabel("Cost Proxy ($M)", color=COST_COLOR, fontsize=12)
    ax3.tick_params(axis="y", labelcolor=COST_COLOR)

    # --- Title ---
    elfo_mode = "ELFO-Augmented" if results[0]["include_elfo"] else "Pure Constellation"
    fig.suptitle(
        f"TIN v0.4.0 Trade Study — {elfo_mode}\n"
        f"Polar Constellation | {SIM_DAYS}-day South Pole Coverage "
        f"(lat {LAT_SOUTH_POLE}\u00b0, elev \u2265{ELEV_MIN_DEG}\u00b0)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # --- Combined legend ---
    fig.legend(
        [bars, line_gap[0], line_cost[0]],
        ["Coverage %", "Worst Gap (min)", "Cost Proxy ($M)"],
        loc="lower center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fancybox=True,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    fname = f"tin_trade_{datetime.now():%Y%m%d_%H%M}.png"
    plot_path = os.path.join(outdir, fname)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Plot saved: {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TIN v0.4.0 Trade Study — N-sat polar coverage sweep"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--include_elfo",
        action="store_true",
        default=True,
        help="Include Lunar Pathfinder ELFO relay (default)",
    )
    group.add_argument(
        "--no_elfo", action="store_true", help="Pure constellation only, no ELFO relay"
    )
    args = parser.parse_args()

    include_elfo = not args.no_elfo

    # --- Run sweep ---
    results = run_sweep(include_elfo=include_elfo)

    # --- Print summary ---
    print_summary(results)

    # --- Determine output directories (relative to repo root) ---
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(repo_root, "data")
    results_dir = os.path.join(repo_root, "results")

    # --- Save outputs ---
    save_json(results, data_dir)
    plot_trade_study(results, results_dir)

    print(f"\nDone. Trade study complete for {'ELFO-augmented' if include_elfo else 'pure'} mode.")


if __name__ == "__main__":
    main()
