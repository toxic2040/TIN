#!/usr/bin/env python3
"""
C4 Throughput Sensitivity — Does the Jupiter verdict survive varying M?

The C4 reserve formula R_h_min = ceil(log(eps) / log(1 - DR_h)) depends only
on DR and epsilon, not on throughput M.  The mass reserve (R_h_min * M) scales
linearly with M.  This script demonstrates that increasing pipe diameter
cannot rescue Jupiter because the time horizon is set by DR alone.

Grounding audit item: "0.4% delivery ratio doesn't care much about the pipe
diameter — but needs demonstration."
"""

import json
import math
import os

import numpy as np

# ── Constants (from run_pipeline_self_consistency.py) ─────────────────
DAY = 86400.0
YEAR = 365.25 * DAY
SYNODIC_MJ = 398.9 * DAY  # Earth-Jupiter synodic period
EPSILON = 0.01  # Risk tolerance


def r_h_min(dr_h, eps=EPSILON):
    """Minimum reserve in resupply epochs for C4 at confidence 1 - eps."""
    if dr_h <= 0:
        return float("inf")
    if dr_h >= 1.0:
        return 1
    return math.ceil(math.log(eps) / math.log(1 - dr_h))


def main():
    print("=" * 70)
    print("  C4 THROUGHPUT SENSITIVITY ANALYSIS")
    print("  Does increasing pipe diameter rescue Jupiter?")
    print("=" * 70)

    # DR values: Jupiter LH2 is the target; include others for context
    dr_cases = {
        "Jupiter LH2 (baseline)": 0.004,
        "Jupiter LH2 (2x better)": 0.008,
        "Jupiter LH2 (5x better)": 0.020,
        "Jupiter LH2 (10x better)": 0.040,
        "Mars LH2 (reference)": 0.15,
    }

    # Throughput sweep: 5 to 100 t/pass
    m_values = [5, 10, 14.1, 20, 30, 50, 75, 100]

    # ── Part 1: R_h_min depends only on DR ────────────────────────────
    print("\n--- Part 1: R_h_min (epochs) vs DR ---")
    print(f"  epsilon = {EPSILON}")
    print(f"  Synodic period (MJ) = {SYNODIC_MJ / DAY:.1f} days = {SYNODIC_MJ / YEAR:.2f} years\n")

    print(f"  {'Case':30s} | {'DR':>8s} | {'R_h_min':>8s} | {'Years':>10s}")
    print("  " + "-" * 65)

    results = {}
    for label, dr in dr_cases.items():
        r = r_h_min(dr)
        r_years = r * SYNODIC_MJ / YEAR if r < float("inf") else float("inf")
        print(f"  {label:30s} | {dr:8.4f} | {r:8d} | {r_years:10.1f}")
        results[label] = {
            "dr": dr,
            "r_h_min_epochs": r,
            "r_h_min_years": r_years,
        }

    # ── Part 2: Mass reserve scales linearly with M ───────────────────
    print("\n--- Part 2: Mass reserve (tonnes) at Jupiter DR = 0.004 ---")
    dr_jup = 0.004
    r_epochs = r_h_min(dr_jup)
    r_years = r_epochs * SYNODIC_MJ / YEAR

    print(f"  R_h_min = {r_epochs} epochs = {r_years:.1f} years (FIXED)\n")
    print(f"  {'M (t/pass)':>12s} | {'Mass reserve (t)':>18s} | {'Verdict':>10s}")
    print("  " + "-" * 50)

    mass_results = {}
    for m in m_values:
        mass_reserve = r_epochs * m
        verdict = "PASS" if r_years <= 20 else "FAIL"
        print(f"  {m:12.1f} | {mass_reserve:18,.0f} | {verdict:>10s}")
        mass_results[m] = {"mass_reserve_t": mass_reserve, "verdict": verdict}

    # ── Part 3: What DR would Jupiter need to PASS C4? ────────────────
    print("\n--- Part 3: DR required for C4 PASS (R_h < 20 years) ---")
    target_years = 20
    target_epochs = math.floor(target_years * YEAR / SYNODIC_MJ)
    # (1 - DR)^R < eps  =>  DR > 1 - eps^(1/R)
    dr_needed = 1.0 - EPSILON ** (1.0 / target_epochs)
    print(f"  Max epochs in {target_years} years: {target_epochs}")
    print(f"  DR required: {dr_needed:.4f} ({dr_needed / dr_jup:.0f}x current)")
    print(f"  Current Jupiter DR: {dr_jup:.4f}")
    print(f"  Gap: {dr_needed / dr_jup:.0f}x improvement needed in DR, regardless of throughput")

    # ── Part 4: Sensitivity at boundary (Mars) ────────────────────────
    print("\n--- Part 4: Mars boundary sensitivity ---")
    dr_mars_range = np.linspace(0.05, 0.40, 15)
    synodic_em = 779.9 * DAY

    print(f"  {'DR_mars':>8s} | {'R_h_min':>8s} | {'Years':>8s} | {'Verdict':>8s}")
    print("  " + "-" * 42)
    for dr in dr_mars_range:
        r = r_h_min(dr)
        ry = r * synodic_em / YEAR
        v = "PASS" if ry <= 20 else "FAIL"
        print(f"  {dr:8.3f} | {r:8d} | {ry:8.1f} | {v:>8s}")

    # ── Conclusion ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CONCLUSION")
    print("=" * 70)
    print("  R_h_min (time) is independent of throughput M.")
    print(f"  Jupiter at DR = {dr_jup}: {r_epochs} synodic periods = {r_years:.0f} years.")
    print("  Even at 100 t/pass (7x baseline), the TIME requirement")
    print("  is unchanged. Throughput determines tonnes of reserve,")
    print("  not years of reserve. The Jupiter verdict is structural:")
    print(f"  it's set by DR = {dr_jup}, which is a topology × decay")
    print("  product, not a pipe diameter.")
    print(f"\n  To PASS C4, Jupiter needs DR >= {dr_needed:.4f}")
    print(f"  ({dr_needed / dr_jup:.0f}x current). No throughput increase helps.")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "epsilon": EPSILON,
        "synodic_mj_days": SYNODIC_MJ / DAY,
        "dr_cases": results,
        "mass_reserve_by_throughput": {str(m): v for m, v in mass_results.items()},
        "dr_needed_for_pass": dr_needed,
        "improvement_factor": dr_needed / dr_jup,
        "conclusion": "Jupiter verdict insensitive to throughput",
    }

    outpath = os.path.join("runs", "c4_throughput_sensitivity_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
