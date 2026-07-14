#!/usr/bin/env python3
"""Historical C4 reserve-model sweep versus commodity half-life.

This script reproduces an archived synthetic EMJ relay-chain calculation. It
sweeps tau_half from 90 to 3600 days and computes the model quantity R_h_min at
each hub. It produces:

  1. R_h_min(tau_half) curve at each hub — the architectural radius as
     a function of commodity physics, not topology.

  2. The tau_half value where each model row crosses a historical 20-year
     reporting threshold.

  3. A cumulative-exposure heuristic and a cadence-only cycler-count example.

These are archived model outputs, not mission feasibility, ZBO requirements,
fleet-sizing advice, or a universal sustainability rule. The analytical DR is
an oracle-path product under the assumptions encoded below; it is not validated
as an operational architecture assessment.
"""

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DAY = 86400.0
WEEK = 7 * DAY
MONTH = 30.44 * DAY
YEAR = 365.25 * DAY
SYNODIC_EM = 779.9 * DAY
TRANSIT_EM = 243 * DAY
SYNODIC_MJ = 398.9 * DAY
TRANSIT_MJ = 730 * DAY

EPSILON = 0.01

# ── Network topology ───────────────────────────────────────────────────

LINKS = [
    ("earth_surface", "l2_hub", 0.98, "weekly_launch"),
    ("l2_hub", "em_cycler_e", 0.95, "perigee_rendezvous"),
    ("em_cycler_e", "em_cycler_m", 0.99, "em_coast"),
    ("em_cycler_m", "mars_station", 0.93, "mars_rendezvous"),
    ("mars_station", "l4_relay", 0.90, "periodic_transfer"),
    ("l4_relay", "mj_cycler", 0.88, "l4_rendezvous"),
    ("mj_cycler", "jupiter_station", 0.85, "jupiter_rendezvous"),
]

DWELL_AT_NODE = {
    "l2_hub": 3 * DAY,
    "em_cycler_e": 14 * DAY,
    "em_cycler_m": 0,
    "mars_station": 14 * DAY,
    "l4_relay": 60 * DAY,
    "mj_cycler": 0,
}

LATENCY = {
    "weekly_launch": 3600,
    "perigee_rendezvous": 4 * 3600,
    "em_coast": TRANSIT_EM,
    "mars_rendezvous": 2 * 3600,
    "periodic_transfer": DAY,
    "l4_rendezvous": DAY,
    "jupiter_rendezvous": TRANSIT_MJ,
}

RESUPPLY_CADENCE = {
    "l2_hub": WEEK,
    "em_cycler_e": SYNODIC_EM,
    "em_cycler_m": SYNODIC_EM,
    "mars_station": SYNODIC_EM,
    "l4_relay": MONTH,
    "mj_cycler": SYNODIC_MJ,
    "jupiter_station": SYNODIC_MJ,
}

# Nodes in path order, with their hop index
RELAY_PATH = [
    (1, "l2_hub"),
    (2, "em_cycler_e"),
    (3, "em_cycler_m"),
    (4, "mars_station"),
    (5, "l4_relay"),
    (6, "mj_cycler"),
    (7, "jupiter_station"),
]

# Key hubs for detailed analysis
KEY_HUBS = ["mars_station", "l4_relay", "jupiter_station"]


# ── Analytical computation (no engine needed) ──────────────────────────


def compute_per_hop_exposure():
    """
    Compute exposure time (latency + destination dwell) for each hop.
    Returns list of (link_type, to_node, p_hw, exposure_s).
    """
    hops = []
    for _, to_node, p_hw, link_type in LINKS:
        lat = LATENCY[link_type]
        dwell = DWELL_AT_NODE.get(to_node, 0)
        hops.append((link_type, to_node, p_hw, lat + dwell))
    return hops


def compute_dr_analytical(n_hops, tau_half_s):
    """
    Compute DR to hop n_hops analytically.

    For single-path relay chain:
      eta_OPSP = product of (p_hw × D_decay) for each hop
      DR = S_T × eta_OPSP

    S_T is topology-dependent and computed once from the engine.
    eta_OPSP is computed here analytically for speed.
    """
    hops = compute_per_hop_exposure()
    tau = tau_half_s / math.log(2)

    eta = 1.0
    for i in range(n_hops):
        _, _, p_hw, exposure = hops[i]
        decay = math.exp(-exposure / tau)
        eta *= p_hw * decay
    return eta


def r_min(dr, epsilon=EPSILON):
    """Model reserve epochs for a given DR and risk-tolerance parameter."""
    if dr <= 0:
        return float("inf")
    if dr >= 1.0:
        return 1
    return math.ceil(math.log(epsilon) / math.log(1 - dr))


def r_min_years(dr, cadence_s, epsilon=EPSILON):
    """Convert the model reserve epochs to years at the stated cadence."""
    r = r_min(dr, epsilon)
    if r == float("inf"):
        return float("inf")
    return r * cadence_s / YEAR


# ── Engine-based S_T computation (run once) ────────────────────────────


def generate_contacts(window_years=10, seed=42):
    """Generate EMJ contact plan."""
    rng = np.random.default_rng(seed)
    T_end = window_years * YEAR
    contacts = []

    t = 0.0
    while t < T_end:
        contacts.append(
            {
                "from_node": "earth_surface",
                "to_node": "l2_hub",
                "start_s": t,
                "duration_s": 4 * 3600,
                "latency_s": 3600,
                "p_success": 0.98,
                "link_type": "weekly_launch",
            }
        )
        t += WEEK

    em_departures = []
    t = 90 * DAY
    while t < T_end:
        jitter = rng.uniform(-5 * DAY, 5 * DAY)
        dep = t + jitter
        em_departures.append(dep)
        contacts.append(
            {
                "from_node": "l2_hub",
                "to_node": "em_cycler_e",
                "start_s": dep,
                "duration_s": 14 * DAY,
                "latency_s": 4 * 3600,
                "p_success": 0.95,
                "link_type": "perigee_rendezvous",
            }
        )
        t += SYNODIC_EM

    for dep in em_departures:
        contacts.append(
            {
                "from_node": "em_cycler_e",
                "to_node": "em_cycler_m",
                "start_s": dep,
                "duration_s": 30 * DAY,
                "latency_s": TRANSIT_EM,
                "p_success": 0.99,
                "link_type": "em_coast",
            }
        )

    for dep in em_departures:
        arr = dep + TRANSIT_EM
        if arr < T_end:
            contacts.append(
                {
                    "from_node": "em_cycler_m",
                    "to_node": "mars_station",
                    "start_s": arr,
                    "duration_s": 14 * DAY,
                    "latency_s": 2 * 3600,
                    "p_success": 0.93,
                    "link_type": "mars_rendezvous",
                }
            )

    t = 0.0
    while t < T_end:
        contacts.append(
            {
                "from_node": "mars_station",
                "to_node": "l4_relay",
                "start_s": t,
                "duration_s": 2 * DAY,
                "latency_s": DAY,
                "p_success": 0.90,
                "link_type": "periodic_transfer",
            }
        )
        t += MONTH

    mj_departures = []
    t = 200 * DAY
    while t < T_end:
        jitter = rng.uniform(-10 * DAY, 10 * DAY)
        dep = t + jitter
        mj_departures.append(dep)
        contacts.append(
            {
                "from_node": "l4_relay",
                "to_node": "mj_cycler",
                "start_s": dep,
                "duration_s": 7 * DAY,
                "latency_s": DAY,
                "p_success": 0.88,
                "link_type": "l4_rendezvous",
            }
        )
        t += SYNODIC_MJ

    for dep in mj_departures:
        if dep + TRANSIT_MJ < T_end:
            contacts.append(
                {
                    "from_node": "mj_cycler",
                    "to_node": "jupiter_station",
                    "start_s": dep,
                    "duration_s": 14 * DAY,
                    "latency_s": TRANSIT_MJ,
                    "p_success": 0.85,
                    "link_type": "jupiter_rendezvous",
                }
            )

    contacts.sort(key=lambda c: c["start_s"])
    return contacts


def compute_s_t_at_hubs(contacts):
    """Compute S_T from Earth to each hub via engine (hardware contacts)."""
    from tin.core.oracle import earliest_arrival

    t_start = min(c["start_s"] for c in contacts)
    t_end = max(c["start_s"] + c["duration_s"] for c in contacts)
    times = np.arange(t_start, t_end, DAY)

    s_t_by_hub = {}
    for _, dest in RELAY_PATH:
        feasible = 0
        for t in times:
            ok, _ = earliest_arrival("earth_surface", dest, float(t), contacts)
            if ok:
                feasible += 1
        s_t_by_hub[dest] = feasible / len(times)

    return s_t_by_hub


# ── Worker for parallel sweep ──────────────────────────────────────────


def _sweep_worker(args):
    """Compute R_h_min at all hubs for a single tau_half value."""
    tau_half_days, s_t_by_hub = args
    tau_half_s = tau_half_days * DAY

    results = {"tau_half_days": tau_half_days}

    for hop_idx, (n_hops, node) in enumerate(RELAY_PATH):
        eta = compute_dr_analytical(n_hops, tau_half_s)
        s_t = s_t_by_hub[node]
        dr = s_t * eta
        cadence = RESUPPLY_CADENCE[node]
        r_years = r_min_years(dr, cadence)

        results[node] = {
            "eta": eta,
            "dr": dr,
            "r_min": r_min(dr),
            "r_years": r_years,
        }

    return results


# ── Main ───────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  HISTORICAL TAU SENSITIVITY MODEL SWEEP")
    print("  ARCHIVED C4 RESERVE OUTPUTS — NOT MISSION OR DESIGN GUIDANCE")
    print("=" * 70)

    # ── Step 1: Compute S_T at each hub (engine, once) ──
    print("\n  Computing S_T at each hub (engine)...")
    contacts = generate_contacts(window_years=10)
    s_t_by_hub = compute_s_t_at_hubs(contacts)

    print("  S_T values:")
    for _, node in RELAY_PATH:
        print(f"    {node:20s}: S_T = {s_t_by_hub[node]:.4f}")

    # ── Step 2: Sweep tau_half ──
    # Dense near the expected critical region, sparse at extremes
    tau_values = sorted(
        set(
            list(range(90, 360, 15))  # 90-360 days, every 15 days
            + list(range(360, 720, 30))  # 360-720 days, every 30 days
            + list(range(720, 1800, 90))  # 720-1800 days, every 90 days
            + list(range(1800, 3601, 180))  # 1800-3600 days, every 180 days
            + [180, 260, 365, 730, 1095, 1460, 1825, 2190, 2555, 3650]  # key values
        )
    )

    print(
        f"\n  Sweeping {len(tau_values)} tau_half values from "
        f"{min(tau_values)} to {max(tau_values)} days..."
    )

    tasks = [(tau_d, s_t_by_hub) for tau_d in tau_values]
    n_workers = min(os.cpu_count() or 4, len(tasks))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        sweep_results = list(pool.map(_sweep_worker, tasks))

    # ── Step 3: Historical reporting-threshold crossings ──
    print(f"\n\n{'=' * 70}")
    print("  HISTORICAL 20-YEAR REPORTING-THRESHOLD CROSSINGS")
    print(f"{'=' * 70}")
    print("  Archived convention: modeled R_h <= 20 years is below the reporting threshold")
    print("  This convention is not a feasibility, safety, or ZBO requirement.")

    threshold_years = 20

    for node_label in KEY_HUBS:
        # Find the tau_half where R_h_years crosses the threshold
        prev_years = None
        critical_tau = None
        for res in sweep_results:
            hub = res.get(node_label, {})
            curr_years = hub.get("r_years", float("inf"))
            if curr_years is None:
                curr_years = float("inf")
            if curr_years <= threshold_years and (
                prev_years is None or prev_years > threshold_years
            ):
                critical_tau = res["tau_half_days"]
                break
            prev_years = curr_years

        if critical_tau:
            print(
                f"\n  {node_label:20s}: modeled crossing tau_half = {critical_tau} days "
                f"({critical_tau / 365.25:.1f} years)"
            )
            print(f"    Below {critical_tau}d: modeled R_h > {threshold_years} years")
            print(f"    At/above {critical_tau}d: modeled R_h <= {threshold_years} years")
            print("    No ZBO specification or mission requirement is inferred from this crossing.")
        else:
            # Check which side of the historical reporting threshold contains all rows.
            all_pass = all(
                (res.get(node_label, {}).get("r_years", float("inf")) or float("inf"))
                <= threshold_years
                for res in sweep_results
            )
            if all_pass:
                print(
                    f"\n  {node_label:20s}: below the model threshold at all tested tau_half values"
                )
            else:
                last_r = sweep_results[-1].get(node_label, {}).get("r_years", float("inf"))
                last_r_str = f"{last_r:.1f}" if last_r and last_r < 1e6 else "inf"
                print(
                    f"\n  {node_label:20s}: above the model threshold at all tested tau_half values "
                    f"(R_h = {last_r_str} years even at tau_half = {tau_values[-1]}d)"
                )
                print("    This synthetic result does not prescribe a topology or hardware change.")

    # ── Step 4: Detailed table at key tau_half values ──
    key_taus = [90, 120, 180, 260, 365, 540, 730, 1095, 1460, 1825, 2190, 3650]
    key_taus = [t for t in key_taus if t in tau_values]

    print(f"\n\n{'=' * 70}")
    print("  DETAILED TABLE — R_h (years) at Key Tau Values")
    print(f"{'=' * 70}")

    header = f"  {'tau_half':>8s} |"
    for node_label in KEY_HUBS:
        short = node_label.replace("_station", "").replace("_relay", "")
        header += f" {'DR':>7s} {'R(yr)':>8s} |"
    print(f"\n{header}")
    print("  " + "-" * (12 + len(KEY_HUBS) * 19))

    for tau_d in key_taus:
        res = next((r for r in sweep_results if r["tau_half_days"] == tau_d), None)
        if not res:
            continue

        row = f"  {tau_d:6d} d |"
        for node_label in KEY_HUBS:
            hub = res.get(node_label, {})
            dr = hub.get("dr", 0)
            r_yr = hub.get("r_years", float("inf"))
            r_str = (
                f"{r_yr:.1f}"
                if r_yr and r_yr < 10000
                else ("inf" if r_yr is None or r_yr == float("inf") else f"{r_yr:.0f}")
            )
            row += f" {dr:7.4f} {r_str:>8s} |"
        print(row)

    # ── Step 5: Historical cumulative-exposure heuristic ──
    print(f"\n\n{'=' * 70}")
    print("  HISTORICAL CUMULATIVE-EXPOSURE HEURISTIC")
    print(f"{'=' * 70}")

    # Compute cumulative exposure at each hop
    hops = compute_per_hop_exposure()
    cum_exposure = 0.0
    print("\n  Cumulative exposure through relay chain:")
    print(
        f"  {'Hop':>3s} {'Node':20s} | {'this hop':>10s} | {'cumulative':>10s} | {'tau ratio':>10s}"
    )
    print("  " + "-" * 65)

    for i, (lt, to_node, p_hw, exposure) in enumerate(hops):
        cum_exposure += exposure
        # At tau_half = 180d (tau = 260d), the ratio determines decay severity
        tau_ratio = cum_exposure / (180 * DAY / math.log(2))
        print(
            f"  {i + 1:3d} {to_node:20s} | {exposure / DAY:8.1f} d | "
            f"{cum_exposure / DAY:8.1f} d | {tau_ratio:8.2f} tau"
        )

    print("\n  Archived model relation: compare cumulative path exposure with")
    print("  tau_half / ln(2) under the encoded exponential-decay assumptions.")
    print("  For the LH2 parameter row (tau_half = 180d), that reference is ≈260 days.")
    print("  The encoded EM coast is 243d, or 0.94 of that reference interval.")
    print("  No universal sustainability boundary or design rule is established.")

    # ── Step 6: Historical cadence-only cycler-count scaling ──
    print(f"\n\n{'=' * 70}")
    print("  HISTORICAL CYCLER-COUNT SCALING EXAMPLE — NOT FLEET GUIDANCE")
    print(f"{'=' * 70}")

    print("\n  If N cyclers with complementary phasing reduce the")
    print("  resupply cadence by factor N, R_h_years scales by 1/N.")
    print("  (Assumes additional cyclers change neither DR nor exposure.)")
    print("  This arithmetic is not a fleet recommendation or feasibility result.")

    tau_half_s = 180 * DAY
    for node_label in KEY_HUBS:
        hop_idx = next(i for i, (_, n) in enumerate(RELAY_PATH) if n == node_label)
        n_hops = hop_idx + 1
        eta = compute_dr_analytical(n_hops, tau_half_s)
        s_t = s_t_by_hub[node_label]
        dr = s_t * eta
        cadence = RESUPPLY_CADENCE[node_label]

        r = r_min(dr)
        if r == float("inf"):
            print(f"\n  {node_label}: modeled DR = {dr:.6f}; cadence-only scaling is undefined")
            continue

        r_yr_1 = r * cadence / YEAR

        # How many cyclers to get R_h_years under 20?
        if r_yr_1 <= threshold_years:
            n_cyclers = 1
        else:
            n_cyclers = math.ceil(r_yr_1 / threshold_years)

        r_yr_n = r_yr_1 / n_cyclers

        print(f"\n  {node_label} (tau_half=180d, DR={dr:.4f}):")
        print(f"    1 cycler:  R_h = {r} epochs × {cadence / MONTH:.0f} mo = {r_yr_1:.1f} yr")
        print(
            f"    {n_cyclers} cyclers: R_h = {r} epochs × {cadence / MONTH / n_cyclers:.1f} mo = {r_yr_n:.1f} yr"
        )
        if n_cyclers > 1:
            print(
                f"    Under the stated cadence-only scaling, N={n_cyclers} places the "
                f"reported R_h below {threshold_years} years; this is not a fleet requirement."
            )

    # ── Save results ──
    output = {
        "claim_status": (
            "historical synthetic model output; not mission feasibility, ZBO, "
            "fleet-sizing, or universal design guidance"
        ),
        "epsilon": EPSILON,
        "threshold_years": threshold_years,
        "tau_values_days": tau_values,
        "s_t_by_hub": s_t_by_hub,
        "sweep": sweep_results,
        "cumulative_exposure_days": [
            sum(h[3] for h in compute_per_hop_exposure()[: i + 1]) / DAY for i in range(len(LINKS))
        ],
    }

    outpath = os.path.join("runs", "tau_sensitivity_results.json")
    with open(outpath, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
            default=lambda x: (
                float(x)
                if isinstance(x, (np.floating, np.integer))
                else None
                if x == float("inf")
                else x
            ),
        )
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
