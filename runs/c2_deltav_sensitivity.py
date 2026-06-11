#!/usr/bin/env python3
"""
C2 Delta-v Sensitivity — How robust is the propulsive cost verdict?

C2 tests whether delta-v differences between relay and bypass paths can flip
the hull ranking.  The test: |mu * (dv_relay - dv_bypass)| < hull_margin.

This script sweeps both delta-v values over +/-30% and maps the ratio of
propulsive perturbation to hull margin, showing where C2 passes and fails.

Grounding audit item: "How sensitive is the 11.4x margin exceedance to the
trajectory assumptions? Needs attention."
"""

import json
import math
import os

import numpy as np

# ── Constants (from run_pipeline_self_consistency.py) ─────────────────
DAY = 86400.0
YEAR = 365.25 * DAY

# Rocket equation parameter
ISP = 450  # seconds
G0 = 9.81  # m/s^2
MU_LOG = 1000 / (G0 * ISP)  # 0.2267 nats per km/s

# LH2 decay
TAU_HALF_LH2 = 180 * DAY
LAMBDA_LH2 = math.log(2) / TAU_HALF_LH2

# Relay path parameters (unique segment: Earth -> Mars via cycler)
# From run_pipeline_self_consistency.py
LINKS_RELAY = [
    ("earth_surface", "l2_hub", 0.98, 3600, 3 * DAY),
    ("l2_hub", "em_cycler_e", 0.95, 4 * 3600, 14 * DAY),
    ("em_cycler_e", "em_cycler_m", 0.99, 243 * DAY, 0),
    ("em_cycler_m", "mars_station", 0.93, 2 * 3600, 14 * DAY),
]

# Bypass path parameters (unique segment: direct chemical)
Q_BYPASS = 0.82
T_BYPASS = 270 * DAY  # 9-month Hohmann

# Baseline delta-v
DV_RELAY_BASELINE = 0.0 + 0.8 + 0.0 + 1.5  # = 2.3 km/s
DV_BYPASS_BASELINE = 0.9  # km/s


def compute_hull_params():
    """Compute relay vs bypass hull crossover and margin function."""
    Q_relay = 1.0
    T_relay = 0.0
    for _, _, p, lat, dwell in LINKS_RELAY:
        Q_relay *= p
        T_relay += lat + dwell

    nlQ_relay = -math.log(Q_relay)
    nlQ_bypass = -math.log(Q_BYPASS)

    dT = T_relay - T_BYPASS
    dnlQ = nlQ_bypass - nlQ_relay

    lambda_star = dnlQ / dT if abs(dT) > 1e-6 else 0
    tau_half_star = math.log(2) / lambda_star if lambda_star > 0 else float("inf")

    return {
        "Q_relay": Q_relay,
        "Q_bypass": Q_BYPASS,
        "nlQ_relay": nlQ_relay,
        "nlQ_bypass": nlQ_bypass,
        "T_relay_s": T_relay,
        "T_bypass_s": T_BYPASS,
        "T_relay_d": T_relay / DAY,
        "T_bypass_d": T_BYPASS / DAY,
        "dT_s": dT,
        "dnlQ": dnlQ,
        "lambda_star": lambda_star,
        "tau_half_star_d": tau_half_star / DAY,
    }


def hull_margin(hull, lambda_c):
    """Margin = |lambda * dT - dnlQ|."""
    return abs(lambda_c * hull["dT_s"] - hull["dnlQ"])


def main():
    print("=" * 70)
    print("  C2 DELTA-V SENSITIVITY ANALYSIS")
    print("  How robust is the propulsive cost verdict?")
    print("=" * 70)

    hull = compute_hull_params()

    print("\n--- Hull Parameters ---")
    print(
        f"  Relay unique:  Q = {hull['Q_relay']:.4f}, "
        f"-ln(Q) = {hull['nlQ_relay']:.4f}, T = {hull['T_relay_d']:.1f} d"
    )
    print(
        f"  Bypass unique: Q = {hull['Q_bypass']:.4f}, "
        f"-ln(Q) = {hull['nlQ_bypass']:.4f}, T = {hull['T_bypass_d']:.1f} d"
    )
    print(f"  Crossover tau_half: {hull['tau_half_star_d']:.1f} d")
    print(f"  mu (log-odds per km/s): {MU_LOG:.4f}")

    # Margins at key operating points
    margin_hw = hull_margin(hull, 0)
    margin_lh2 = hull_margin(hull, LAMBDA_LH2)

    print(f"\n--- Baseline (dv_relay={DV_RELAY_BASELINE}, dv_bypass={DV_BYPASS_BASELINE}) ---")
    dv_diff_base = abs(DV_RELAY_BASELINE - DV_BYPASS_BASELINE)
    perturb_base = MU_LOG * dv_diff_base
    print(f"  |dv_diff| = {dv_diff_base:.1f} km/s")
    print(f"  Propulsive perturbation = {perturb_base:.4f} nats")
    print(f"  Hull margin at hardware (lambda=0):   {margin_hw:.4f} nats")
    print(f"  Hull margin at LH2 (tau_half=180d):   {margin_lh2:.4f} nats")
    print(
        f"  Ratio at hardware: {perturb_base / margin_hw:.1f}x "
        f"({'FAIL' if perturb_base >= margin_hw else 'PASS'})"
    )
    print(
        f"  Ratio at LH2:     {perturb_base / margin_lh2:.1f}x "
        f"({'FAIL' if perturb_base >= margin_lh2 else 'PASS'})"
    )

    # ── Part 1: Sweep dv_relay with dv_bypass fixed ───────────────────
    print(f"\n--- Part 1: Sweep dv_relay (dv_bypass = {DV_BYPASS_BASELINE} km/s fixed) ---")
    dv_relay_range = np.arange(0.5, 4.1, 0.25)

    print(
        f"  {'dv_relay':>8s} | {'|diff|':>6s} | {'perturb':>8s} | "
        f"{'ratio_hw':>9s} | {'ratio_lh2':>9s} | {'C2_hw':>6s} | {'C2_lh2':>7s}"
    )
    print("  " + "-" * 72)

    for dv_r in dv_relay_range:
        diff = abs(dv_r - DV_BYPASS_BASELINE)
        perturb = MU_LOG * diff
        r_hw = perturb / margin_hw if margin_hw > 0 else float("inf")
        r_lh2 = perturb / margin_lh2 if margin_lh2 > 0 else float("inf")
        p_hw = "PASS" if perturb < margin_hw else "FAIL"
        p_lh2 = "PASS" if perturb < margin_lh2 else "FAIL"
        marker = " <-- baseline" if abs(dv_r - DV_RELAY_BASELINE) < 0.01 else ""
        print(
            f"  {dv_r:8.2f} | {diff:6.2f} | {perturb:8.4f} | "
            f"{r_hw:9.1f}x | {r_lh2:9.1f}x | {p_hw:>6s} | {p_lh2:>7s}{marker}"
        )

    # ── Part 2: Sweep dv_bypass with dv_relay fixed ───────────────────
    print(f"\n--- Part 2: Sweep dv_bypass (dv_relay = {DV_RELAY_BASELINE} km/s fixed) ---")
    dv_bypass_range = np.arange(0.3, 3.1, 0.25)

    print(
        f"  {'dv_bypass':>9s} | {'|diff|':>6s} | {'perturb':>8s} | "
        f"{'ratio_hw':>9s} | {'ratio_lh2':>9s} | {'C2_hw':>6s} | {'C2_lh2':>7s}"
    )
    print("  " + "-" * 72)

    for dv_b in dv_bypass_range:
        diff = abs(DV_RELAY_BASELINE - dv_b)
        perturb = MU_LOG * diff
        r_hw = perturb / margin_hw if margin_hw > 0 else float("inf")
        r_lh2 = perturb / margin_lh2 if margin_lh2 > 0 else float("inf")
        p_hw = "PASS" if perturb < margin_hw else "FAIL"
        p_lh2 = "PASS" if perturb < margin_lh2 else "FAIL"
        marker = " <-- baseline" if abs(dv_b - DV_BYPASS_BASELINE) < 0.01 else ""
        print(
            f"  {dv_b:9.2f} | {diff:6.2f} | {perturb:8.4f} | "
            f"{r_hw:9.1f}x | {r_lh2:9.1f}x | {p_hw:>6s} | {p_lh2:>7s}{marker}"
        )

    # ── Part 3: Critical dv_diff for PASS ─────────────────────────────
    print("\n--- Part 3: Critical delta-v difference for C2 PASS ---")
    dv_crit_hw = margin_hw / MU_LOG
    dv_crit_lh2 = margin_lh2 / MU_LOG

    print(
        f"  At hardware: |dv_diff| < {dv_crit_hw:.2f} km/s "
        f"(current: {dv_diff_base:.1f} km/s, {dv_diff_base / dv_crit_hw:.1f}x over)"
    )
    print(
        f"  At LH2:      |dv_diff| < {dv_crit_lh2:.2f} km/s "
        f"(current: {dv_diff_base:.1f} km/s, {dv_diff_base / dv_crit_lh2:.1f}x over)"
    )

    # ── Part 4: 2D sweep (both vary simultaneously, ±30%) ────────────
    print("\n--- Part 4: 2D sweep — dv_relay +/-30%, dv_bypass +/-30% ---")
    relay_lo, relay_hi = DV_RELAY_BASELINE * 0.7, DV_RELAY_BASELINE * 1.3
    bypass_lo, bypass_hi = DV_BYPASS_BASELINE * 0.7, DV_BYPASS_BASELINE * 1.3

    n_grid = 11
    dv_r_grid = np.linspace(relay_lo, relay_hi, n_grid)
    dv_b_grid = np.linspace(bypass_lo, bypass_hi, n_grid)

    n_pass_hw = 0
    n_pass_lh2 = 0
    n_total = n_grid * n_grid
    min_ratio_hw = float("inf")
    min_ratio_lh2 = float("inf")

    for dv_r in dv_r_grid:
        for dv_b in dv_b_grid:
            diff = abs(dv_r - dv_b)
            perturb = MU_LOG * diff
            r_hw = perturb / margin_hw
            r_lh2 = perturb / margin_lh2
            if perturb < margin_hw:
                n_pass_hw += 1
            if perturb < margin_lh2:
                n_pass_lh2 += 1
            min_ratio_hw = min(min_ratio_hw, r_hw)
            min_ratio_lh2 = min(min_ratio_lh2, r_lh2)

    print(f"  Grid: {n_grid}x{n_grid} = {n_total} points")
    print(f"  Relay range:  [{relay_lo:.2f}, {relay_hi:.2f}] km/s")
    print(f"  Bypass range: [{bypass_lo:.2f}, {bypass_hi:.2f}] km/s")
    print(f"  C2 PASS at hardware: {n_pass_hw}/{n_total} ({100 * n_pass_hw / n_total:.0f}%)")
    print(f"  C2 PASS at LH2:     {n_pass_lh2}/{n_total} ({100 * n_pass_lh2 / n_total:.0f}%)")
    print(f"  Min ratio (best case) at hardware: {min_ratio_hw:.2f}x")
    print(f"  Min ratio (best case) at LH2:     {min_ratio_lh2:.2f}x")

    # ── Conclusion ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CONCLUSION")
    print("=" * 70)

    if min_ratio_hw > 1.0:
        print("  C2 FAILS across the entire +/-30% range.")
        print("  The propulsive perturbation exceeds the hull margin by")
        print(f"  at least {min_ratio_hw:.1f}x (hardware) / {min_ratio_lh2:.1f}x (LH2)")
        print("  even at the most favorable delta-v combination.")
        print("\n  The C2 verdict is ROBUST to trajectory uncertainty.")
        print("  The 1D hull is structurally insufficient; the 2D hull")
        print("  (T, -ln Q, delta-v) is needed for architecture screening.")
    else:
        frac = n_pass_hw / n_total
        print(f"  C2 flips to PASS in {100 * frac:.0f}% of the +/-30% range.")
        print("  The verdict is SENSITIVE to trajectory assumptions.")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "hull": {
            "Q_relay": hull["Q_relay"],
            "Q_bypass": hull["Q_bypass"],
            "T_relay_d": hull["T_relay_d"],
            "T_bypass_d": hull["T_bypass_d"],
            "tau_half_star_d": hull["tau_half_star_d"],
        },
        "baseline": {
            "dv_relay": DV_RELAY_BASELINE,
            "dv_bypass": DV_BYPASS_BASELINE,
            "dv_diff": dv_diff_base,
            "perturbation": perturb_base,
            "margin_hw": margin_hw,
            "margin_lh2": margin_lh2,
            "ratio_hw": perturb_base / margin_hw,
            "ratio_lh2": perturb_base / margin_lh2,
        },
        "critical_dv_diff": {
            "hardware": dv_crit_hw,
            "lh2": dv_crit_lh2,
        },
        "sweep_2d": {
            "range_pct": 30,
            "n_pass_hw": n_pass_hw,
            "n_pass_lh2": n_pass_lh2,
            "n_total": n_total,
            "min_ratio_hw": min_ratio_hw,
            "min_ratio_lh2": min_ratio_lh2,
        },
    }

    outpath = os.path.join("runs", "c2_deltav_sensitivity_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
