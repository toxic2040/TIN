#!/usr/bin/env python3
"""
C4 Finite-Campaign Analysis — "What if I only need 5 years?"

The standard C4 asks: can the architecture sustain itself indefinitely?
A mission planner asks: can I operate for N years with pre-positioned reserves?

This script computes finite-campaign success probabilities for the EMJ
relay architecture, directly addressing the "Deepest Vulnerability"
identified in the grounding audit.

Model: N delivery attempts (one per synodic period), each with probability
DR of success (independent Bernoulli trials). Campaign succeeds if at least
one delivery succeeds within the campaign window.

    P(success | N attempts) = 1 - (1 - DR)^N

Extended model: campaign requires k successful deliveries (not just one).
Uses the binomial survival function.

Grounding audit item: "A mission planner will immediately ask 'okay, but
what if I only need 5 years?'"
"""

import json
import math
import os

from scipy.stats import binom

# ── Constants (from run_pipeline_self_consistency.py) ─────────────────
DAY = 86400.0
YEAR = 365.25 * DAY
SYNODIC_EM = 779.9 * DAY  # Earth-Mars synodic period
SYNODIC_MJ = 398.9 * DAY  # Earth-Jupiter synodic period

EPSILON = 0.01  # Standard risk tolerance


def campaign_success_prob(dr, n_attempts, k_required=1):
    """
    P(at least k successes in n Bernoulli trials with probability dr).

    k=1: P = 1 - (1-dr)^n  (at least one success)
    k>1: uses binomial survival function
    """
    if dr <= 0:
        return 0.0
    if dr >= 1.0:
        return 1.0 if n_attempts >= k_required else 0.0
    if k_required == 1:
        return 1.0 - (1.0 - dr) ** n_attempts
    return float(binom.sf(k_required - 1, n_attempts, dr))


def attempts_for_target(dr, target_prob, k_required=1):
    """Minimum attempts to achieve target success probability."""
    if dr <= 0:
        return float("inf")
    if dr >= 1.0:
        return k_required
    if k_required == 1:
        # 1 - (1-dr)^n >= target  =>  (1-dr)^n <= 1-target
        # n >= log(1-target) / log(1-dr)
        return math.ceil(math.log(1 - target_prob) / math.log(1 - dr))
    # General case: binary search
    lo, hi = k_required, 100_000
    while lo < hi:
        mid = (lo + hi) // 2
        if campaign_success_prob(dr, mid, k_required) >= target_prob:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    print("=" * 70)
    print("  C4 FINITE-CAMPAIGN ANALYSIS")
    print("  'What if I only need N years?'")
    print("=" * 70)

    # Hub configurations from the evaluator
    hubs = {
        "Mars (hardware)": {
            "dr": 0.418,
            "synodic_s": SYNODIC_EM,
            "label": "Mars station, hardware cargo",
        },
        "Mars (LH2)": {
            "dr": 0.027,
            "synodic_s": SYNODIC_EM,
            "label": "Mars station, LH2 propellant",
        },
        "Jupiter (hardware)": {
            "dr": 0.040,
            "synodic_s": SYNODIC_MJ,
            "label": "Jupiter station, hardware cargo",
        },
        "Jupiter (LH2)": {
            "dr": 0.004,
            "synodic_s": SYNODIC_MJ,
            "label": "Jupiter station, LH2 propellant",
        },
    }

    campaign_years = [2, 5, 10, 15, 20, 30, 50, 100]

    # ── Part 1: Success probability vs campaign duration ──────────────
    print("\n--- Part 1: P(at least 1 success) vs campaign duration ---\n")
    header = f"  {'Hub':25s} | {'DR':>6s}"
    for cy in campaign_years:
        header += f" | {cy:>4d}yr"
    print(header)
    print("  " + "-" * (32 + 9 * len(campaign_years)))

    results = {}
    for hub_name, cfg in hubs.items():
        dr = cfg["dr"]
        t_syn = cfg["synodic_s"]
        row = f"  {hub_name:25s} | {dr:6.3f}"
        hub_results = {}
        for cy in campaign_years:
            n = max(1, int(cy * YEAR / t_syn))
            p = campaign_success_prob(dr, n)
            row += f" | {p:5.1%}"
            hub_results[f"{cy}yr"] = {
                "n_attempts": n,
                "p_success": round(p, 4),
            }
        print(row)
        results[hub_name] = {"dr": dr, "campaigns": hub_results}

    # ── Part 2: Years needed for 50%, 90%, 99% confidence ────────────
    print("\n--- Part 2: Campaign duration for target confidence ---\n")
    targets = [0.50, 0.90, 0.99]
    header = f"  {'Hub':25s} | {'DR':>6s}"
    for t in targets:
        header += f" | {t:>7.0%}"
    header += " | R_h_indef"
    print(header)
    print("  " + "-" * (32 + 11 * len(targets) + 12))

    for hub_name, cfg in hubs.items():
        dr = cfg["dr"]
        t_syn = cfg["synodic_s"]
        row = f"  {hub_name:25s} | {dr:6.3f}"
        for t in targets:
            n = attempts_for_target(dr, t)
            years = n * t_syn / YEAR
            if years > 10000:
                row += f" | {'>10000':>6s}yr"
            else:
                row += f" | {years:6.1f}yr"
        # Compare to indefinite R_h_min
        r_indef = attempts_for_target(dr, 1 - EPSILON)
        r_indef_yr = r_indef * t_syn / YEAR
        if r_indef_yr > 10000:
            row += f" | {'>10000':>8s}yr"
        else:
            row += f" | {r_indef_yr:8.1f}yr"
        print(row)

    # ── Part 3: Multiple deliveries required ──────────────────────────
    print("\n--- Part 3: P(success) when k > 1 deliveries needed ---")
    print("  (20-year campaign)\n")

    k_values = [1, 2, 3, 5, 10]
    header = f"  {'Hub':25s} | {'DR':>6s}"
    for k in k_values:
        header += f" | {'k=' + str(k):>6s}"
    print(header)
    print("  " + "-" * (32 + 9 * len(k_values)))

    for hub_name, cfg in hubs.items():
        dr = cfg["dr"]
        t_syn = cfg["synodic_s"]
        n_20yr = max(1, int(20 * YEAR / t_syn))
        row = f"  {hub_name:25s} | {dr:6.3f}"
        for k in k_values:
            p = campaign_success_prob(dr, n_20yr, k)
            row += f" | {p:5.1%}"
        print(row)

    # ── Part 4: The mission planner's table ───────────────────────────
    print("\n--- Part 4: Mission planner's decision table ---")
    print("  Minimum campaign duration (years) for 90% confidence of at least k deliveries\n")

    header = f"  {'Hub':25s}"
    for k in k_values:
        header += f" | {'k=' + str(k):>8s}"
    print(header)
    print("  " + "-" * (25 + 11 * len(k_values)))

    for hub_name, cfg in hubs.items():
        dr = cfg["dr"]
        t_syn = cfg["synodic_s"]
        row = f"  {hub_name:25s}"
        for k in k_values:
            n = attempts_for_target(dr, 0.90, k)
            years = n * t_syn / YEAR
            if years > 10000:
                row += f" | {'>10000':>7s}yr"
            else:
                row += f" | {years:7.1f}yr"
        print(row)

    # ── Part 5: The verdict ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  VERDICT")
    print("=" * 70)

    # Jupiter LH2: compute the campaign at which you get even 50%
    dr_jup = hubs["Jupiter (LH2)"]["dr"]
    t_syn_jup = hubs["Jupiter (LH2)"]["synodic_s"]
    n_50 = attempts_for_target(dr_jup, 0.50)
    yr_50 = n_50 * t_syn_jup / YEAR
    n_90 = attempts_for_target(dr_jup, 0.90)
    yr_90 = n_90 * t_syn_jup / YEAR

    print(f"\n  Jupiter (LH2, DR = {dr_jup}):")
    print(f"    50% confidence of 1 delivery: {yr_50:.0f} years ({n_50} attempts)")
    print(f"    90% confidence of 1 delivery: {yr_90:.0f} years ({n_90} attempts)")
    print(
        f"    5-year campaign success prob:  "
        f"{campaign_success_prob(dr_jup, int(5 * YEAR / t_syn_jup)):.1%}"
    )
    print(
        f"    20-year campaign success prob: "
        f"{campaign_success_prob(dr_jup, int(20 * YEAR / t_syn_jup)):.1%}"
    )

    dr_mars_lh2 = hubs["Mars (LH2)"]["dr"]
    t_syn_mars = hubs["Mars (LH2)"]["synodic_s"]
    n_90_mars = attempts_for_target(dr_mars_lh2, 0.90)
    yr_90_mars = n_90_mars * t_syn_mars / YEAR

    print(f"\n  Mars (LH2, DR = {dr_mars_lh2}):")
    print(f"    90% confidence of 1 delivery: {yr_90_mars:.0f} years")
    print(
        f"    5-year campaign success prob:  "
        f"{campaign_success_prob(dr_mars_lh2, int(5 * YEAR / t_syn_mars)):.1%}"
    )

    dr_mars_hw = hubs["Mars (hardware)"]["dr"]
    n_90_mars_hw = attempts_for_target(dr_mars_hw, 0.90)
    yr_90_mars_hw = n_90_mars_hw * t_syn_mars / YEAR

    print(f"\n  Mars (hardware, DR = {dr_mars_hw}):")
    print(f"    90% confidence of 1 delivery: {yr_90_mars_hw:.1f} years")
    print(
        f"    5-year campaign success prob:  "
        f"{campaign_success_prob(dr_mars_hw, int(5 * YEAR / t_syn_mars)):.1%}"
    )

    print("\n  The finite-campaign model does not rescue Jupiter.")
    print("  Even asking the easier question ('can I get one delivery")
    print(f"  in N years?'), Jupiter LH2 needs {yr_50:.0f} years for a")
    print(f"  coin-flip chance. The steady-state C4 verdict ({yr_90:.0f}-year")
    print("  R_h_min) is not an artifact of the indefinite-horizon")
    print("  assumption.")
    print(f"\n  Mars hardware is viable in finite campaign (90% in {yr_90_mars_hw:.0f} years).")
    print(f"  Mars LH2 is marginal ({yr_90_mars:.0f} years for 90%).")
    print("  This confirms the grounding audit: Mars is in the sensitive")
    print("  regime; Jupiter is structurally unsound.")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "model": "Bernoulli trials, one per synodic period",
        "epsilon": EPSILON,
        "hubs": {},
    }
    for hub_name, cfg in hubs.items():
        dr = cfg["dr"]
        t_syn = cfg["synodic_s"]
        hub_out = {"dr": dr, "synodic_days": t_syn / DAY}
        for cy in campaign_years:
            n = max(1, int(cy * YEAR / t_syn))
            p = campaign_success_prob(dr, n)
            hub_out[f"p_{cy}yr"] = round(p, 4)
        for t in targets:
            n = attempts_for_target(dr, t)
            hub_out[f"years_for_{int(t * 100)}pct"] = round(n * t_syn / YEAR, 1)
        output["hubs"][hub_name] = hub_out

    outpath = os.path.join("runs", "c4_finite_campaign_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
