#!/usr/bin/env python3
"""run_monotonicity_emj.py — Monotonicity test on the EMJ worked example.

Tests the monotonicity conjecture on the whitepaper's own 7-node
Earth-Mars-Jupiter transport network with physically motivated cargo types.

Two topologies:
  A. Relay-only (7-hop chain)
  B. Relay + bypass (adds direct Earth→Jupiter, creating path multiplicity)

Cargo types swept via tau_half:
  - inf (hardware)
  - 1460d (~4yr, satellite components)
  - 730d (2yr)
  - 365d (1yr, some biologics)
  - 180d (cryogenic propellant, LCH4)
  - 90d (cryogenic propellant, LOX/LH2)
  - 60d (perishable supplies)
  - 30d (extreme perishability)

For each (topology, tau_half): trace oracle paths with per-hop timing,
apply dwell decay, compute phi and lambda. Report per-path decomposition.

This is a FIXED-ROUTING test: the oracle selects paths without knowledge
of cargo type (earliest-arrival oracle = lambda→∞ limit). The dwell decay
is applied post-hoc to measure how path quality changes with perishability.

A SEPARATE adaptive test is also run: the oracle re-selects paths using
commodity-aware utility U_k = log(Q_k) - lambda * T_k.

Provenance
----------
Input:  EMJ contact plan (generated inline, same as run_emj_worked_example.py)
Output: runs/monotonicity_emj_results.json

Usage:
    python -m runs.run_monotonicity_emj
"""

from __future__ import annotations

import heapq
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_INF = float("inf")

# ── Constants (from run_emj_worked_example.py) ────────────────────────
DAY = 86400.0
WEEK = 7 * DAY
MONTH = 30.44 * DAY
YEAR = 365.25 * DAY
SYNODIC_EM = 779.9 * DAY
TRANSIT_EM = 243 * DAY
SYNODIC_MJ = 398.9 * DAY
TRANSIT_MJ = 730 * DAY

DT_INJECT = 7 * DAY  # weekly injection for 10yr window

TAU_HALF_DAYS = [float("inf"), 1460, 730, 365, 180, 90, 60, 30]

DWELL_AT_NODE = {
    "l2_hub": 3 * DAY,
    "em_cycler_e": 14 * DAY,
    "em_cycler_m": 0,
    "mars_station": 14 * DAY,
    "l4_relay": 60 * DAY,
    "mj_cycler": 0,
}


# ── Contact plan generation (verbatim from run_emj_worked_example.py) ─


def _generate_contacts(window_years=10, seed=42):
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
        dep_time = t + jitter
        em_departures.append(dep_time)
        contacts.append(
            {
                "from_node": "l2_hub",
                "to_node": "em_cycler_e",
                "start_s": dep_time,
                "duration_s": 14 * DAY,
                "latency_s": 4 * 3600,
                "p_success": 0.95,
                "link_type": "perigee_rendezvous",
            }
        )
        t += SYNODIC_EM

    for dep_time in em_departures:
        contacts.append(
            {
                "from_node": "em_cycler_e",
                "to_node": "em_cycler_m",
                "start_s": dep_time,
                "duration_s": 30 * DAY,
                "latency_s": TRANSIT_EM,
                "p_success": 0.99,
                "link_type": "em_coast",
            }
        )

    for dep_time in em_departures:
        arrival_time = dep_time + TRANSIT_EM
        if arrival_time < T_end:
            contacts.append(
                {
                    "from_node": "em_cycler_m",
                    "to_node": "mars_station",
                    "start_s": arrival_time,
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
        dep_time = t + jitter
        mj_departures.append(dep_time)
        contacts.append(
            {
                "from_node": "l4_relay",
                "to_node": "mj_cycler",
                "start_s": dep_time,
                "duration_s": 7 * DAY,
                "latency_s": DAY,
                "p_success": 0.88,
                "link_type": "l4_rendezvous",
            }
        )
        t += SYNODIC_MJ

    for dep_time in mj_departures:
        if dep_time + TRANSIT_MJ < T_end:
            contacts.append(
                {
                    "from_node": "mj_cycler",
                    "to_node": "jupiter_station",
                    "start_s": dep_time,
                    "duration_s": 14 * DAY,
                    "latency_s": TRANSIT_MJ,
                    "p_success": 0.85,
                    "link_type": "jupiter_rendezvous",
                }
            )

    contacts.sort(key=lambda c: c["start_s"])
    return contacts


def _add_bypass(contacts, window_years=10, seed=99, transit_years=2.7, p_bypass=0.82):
    """Add direct Earth→Jupiter bypass route.

    Default: original EMJ (2.7yr, p=0.82) — bypass dominates, no crossover.
    For crossover test: transit_years=5.0 makes bypass slower than relay,
    creating a genuine hull crossover at tau_half ≈ 424d.
    """
    rng = np.random.default_rng(seed)
    T_end = window_years * YEAR
    bypass = list(contacts)
    t = 180 * DAY
    while t < T_end:
        jitter = rng.uniform(-10 * DAY, 10 * DAY)
        dep_time = t + jitter
        bypass.append(
            {
                "from_node": "earth_surface",
                "to_node": "jupiter_station",
                "start_s": dep_time,
                "duration_s": 14 * DAY,
                "latency_s": transit_years * YEAR,
                "p_success": p_bypass,
                "link_type": "direct_chemical",
            }
        )
        t += SYNODIC_MJ
    bypass.sort(key=lambda c: c["start_s"])
    return bypass


# ── Commodity-aware Dijkstra ───────────────────────────────────────────


def _trace_paths_commodity(contacts, source, dest, t_end_s, lam_hazard):
    """Commodity-aware oracle: minimize C = -log(p) + lam*latency per edge.

    At lam_hazard=0, this maximizes reliability (prefers high-p paths).
    At lam_hazard→∞, this minimizes total transit time (earliest arrival).
    At intermediate lam, it trades off reliability vs exposure.

    This is the affine parametric shortest-path from Section 3.7.
    """
    adj = {}
    for idx, c in enumerate(contacts):
        fn = c["from_node"]
        ps = c.get("p_success", 0.98)
        ls = c["latency_s"]
        # Edge cost: -log(p) + lambda * latency
        edge_cost = -math.log(max(ps, 1e-300)) + lam_hazard * ls
        adj.setdefault(fn, []).append(
            (c["start_s"], c["start_s"] + c["duration_s"], ls, c["to_node"], ps, idx, edge_cost)
        )

    inj_times = np.arange(0.0, t_end_s, DT_INJECT)
    paths = []

    for t_inj in inj_times:
        # Dijkstra on cumulative cost (not arrival time)
        # But we still need time-feasibility: can't use a contact before it opens
        best_cost = {source: 0.0}
        best_time = {source: float(t_inj)}
        prev = {}
        # heap: (cumulative_cost, arrival_time, node)
        heap = [(0.0, float(t_inj), source)]
        found = False

        while heap:
            cost_cur, t_cur, node = heapq.heappop(heap)
            if node == dest:
                found = True
                break
            if cost_cur > best_cost.get(node, _INF):
                continue
            for ss, es, ls, tn, ps, cidx, ecost in adj.get(node, []):
                if t_cur > es:
                    continue
                t_dep = max(t_cur, ss)
                t_arr = t_dep + ls
                # Add dwell cost: time waiting at current node
                dwell_at_node = t_dep - t_cur
                dwell_cost = lam_hazard * dwell_at_node
                total_cost = cost_cur + ecost + dwell_cost
                if total_cost < best_cost.get(tn, _INF):
                    best_cost[tn] = total_cost
                    best_time[tn] = t_arr
                    prev[tn] = (node, cidx, t_dep, t_arr)
                    heapq.heappush(heap, (total_cost, t_arr, tn))

        if not found:
            continue

        hops = []
        n = dest
        while n in prev:
            parent, cidx, t_dep, t_arr = prev[n]
            c = contacts[cidx]
            hops.append(
                {
                    "from": parent,
                    "to": n,
                    "t_depart": t_dep,
                    "t_arrive": t_arr,
                    "latency_s": c["latency_s"],
                    "p_success": c.get("p_success", 0.98),
                    "link_type": c.get("link_type", ""),
                }
            )
            n = parent
        hops.reverse()

        dwells = []
        for i in range(len(hops) - 1):
            dwell = hops[i + 1]["t_depart"] - hops[i]["t_arrive"]
            dwells.append(max(0.0, dwell))

        total_transit = sum(h["latency_s"] for h in hops)
        total_dwell = sum(dwells)

        paths.append(
            {
                "H": len(hops),
                "hops": hops,
                "dwell_times": dwells,
                "total_dwell_s": total_dwell,
                "total_transit_s": total_transit,
                "total_exposure_s": total_transit + total_dwell,
                "log_p_link": sum(math.log(max(h["p_success"], 1e-300)) for h in hops),
                "route_signature": "→".join(h["from"] for h in hops) + "→" + hops[-1]["to"],
            }
        )

    return paths, len(inj_times)


# ── Earliest-arrival oracle path tracer ───────────────────────────────


def _trace_paths(contacts, source, dest, t_end_s):
    """Trace oracle paths with per-hop dwell times."""
    adj = {}
    for idx, c in enumerate(contacts):
        fn = c["from_node"]
        adj.setdefault(fn, []).append(
            (
                c["start_s"],
                c["start_s"] + c["duration_s"],
                c["latency_s"],
                c["to_node"],
                c.get("p_success", 0.98),
                idx,
            )
        )

    inj_times = np.arange(0.0, t_end_s, DT_INJECT)
    paths = []

    for t_inj in inj_times:
        best = {source: float(t_inj)}
        prev = {}
        heap = [(float(t_inj), source)]
        found = False

        while heap:
            t_cur, node = heapq.heappop(heap)
            if node == dest:
                found = True
                break
            if t_cur > best.get(node, _INF):
                continue
            for ss, es, ls, tn, ps, cidx in adj.get(node, []):
                if t_cur > es:
                    continue
                t_dep = max(t_cur, ss)
                t_arr = t_dep + ls
                if t_arr < best.get(tn, _INF):
                    best[tn] = t_arr
                    prev[tn] = (node, cidx, t_dep, t_arr)
                    heapq.heappush(heap, (t_arr, tn))

        if not found:
            continue

        hops = []
        n = dest
        while n in prev:
            parent, cidx, t_dep, t_arr = prev[n]
            c = contacts[cidx]
            hops.append(
                {
                    "from": parent,
                    "to": n,
                    "t_depart": t_dep,
                    "t_arrive": t_arr,
                    "latency_s": c["latency_s"],
                    "p_success": c.get("p_success", 0.98),
                    "link_type": c.get("link_type", ""),
                }
            )
            n = parent
        hops.reverse()

        dwells = []
        for i in range(len(hops) - 1):
            dwell = hops[i + 1]["t_depart"] - hops[i]["t_arrive"]
            dwells.append(max(0.0, dwell))

        # Total exposure = sum of dwell times + transit latencies
        total_transit = sum(h["latency_s"] for h in hops)
        total_dwell = sum(dwells)
        total_exposure = total_transit + total_dwell

        paths.append(
            {
                "H": len(hops),
                "hops": hops,
                "dwell_times": dwells,
                "total_dwell_s": total_dwell,
                "total_transit_s": total_transit,
                "total_exposure_s": total_exposure,
                "log_p_link": sum(math.log(max(h["p_success"], 1e-300)) for h in hops),
                "route_signature": "→".join(h["from"] for h in hops) + "→" + hops[-1]["to"],
            }
        )

    return paths, len(inj_times)


def _apply_decay(paths, tau_half_s):
    """Apply exponential dwell decay and compute corrected metrics.

    Decay is applied to BOTH intermediate dwell AND transit latency.
    This models cargo that degrades continuously during the entire journey.
    """
    if not paths:
        return None

    all_log_p_eff = []
    per_path = []
    path_products = []

    for path in paths:
        H = path["H"]
        hops = path["hops"]
        dwells = path["dwell_times"]

        hop_effs = []
        for i, hop in enumerate(hops):
            log_p = math.log(max(hop["p_success"], 1e-300))
            # Transit decay
            if tau_half_s != _INF and tau_half_s > 0:
                transit_decay = -hop["latency_s"] / tau_half_s
            else:
                transit_decay = 0.0
            # Dwell decay at intermediate node
            if i < len(dwells) and tau_half_s != _INF and tau_half_s > 0:
                dwell_decay = -dwells[i] / tau_half_s
            else:
                dwell_decay = 0.0
            log_p_eff = log_p + transit_decay + dwell_decay
            hop_effs.append(log_p_eff)

        all_log_p_eff.extend(hop_effs)
        log_prod = sum(hop_effs)
        per_path.append((H, log_prod, path["route_signature"], path["total_exposure_s"]))
        path_products.append(math.exp(max(log_prod, -700)))

    hops_arr = np.array([H for H, _, _, _ in per_path])
    E_H = float(np.mean(hops_arr))
    var_H = float(np.var(hops_arr))
    lambda_phys = float(np.mean(all_log_p_eff))
    eta_opsp = float(np.mean(path_products))
    eta_lyap = math.exp(E_H * lambda_phys) if lambda_phys > -500 else 0.0
    phi = eta_opsp / eta_lyap if eta_lyap > 1e-15 else float("nan")
    var_log_p = float(np.var(all_log_p_eff))

    # Per-route-signature breakdown
    route_stats = defaultdict(list)
    for H, lp, sig, exp_s in per_path:
        route_stats[sig].append((H, lp, exp_s))

    route_breakdown = {}
    for sig, entries in route_stats.items():
        hs = [e[0] for e in entries]
        lps = [e[1] for e in entries]
        exps = [e[2] for e in entries]
        route_breakdown[sig] = {
            "count": len(entries),
            "H": hs[0],
            "frac": len(entries) / len(per_path),
            "mean_log_p_path": float(np.mean(lps)),
            "mean_p_path": float(np.mean([math.exp(max(lp, -700)) for lp in lps])),
            "mean_exposure_days": float(np.mean(exps)) / DAY,
        }

    return {
        "E_H": E_H,
        "var_H": var_H,
        "lambda_phys": lambda_phys,
        "eta_opsp": eta_opsp,
        "eta_lyap": eta_lyap,
        "phi": phi,
        "var_log_p": var_log_p,
        "n_paths": len(paths),
        "route_breakdown": route_breakdown,
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    print("=" * 72)
    print("MONOTONICITY TEST — EMJ Worked Example")
    print("=" * 72)

    t0 = time.time()

    # Generate contact plans
    relay_contacts = _generate_contacts(window_years=10)
    bypass_fast = _add_bypass(relay_contacts, window_years=10, transit_years=2.7, p_bypass=0.82)
    bypass_slow = _add_bypass(relay_contacts, window_years=10, transit_years=5.0, p_bypass=0.82)
    print(f"\n  Relay contacts: {len(relay_contacts)}")
    print(f"  Fast bypass (2.7yr, p=0.82): {len(bypass_fast)} contacts — DOMINATES relay")
    print(f"  Slow bypass (5.0yr, p=0.82): {len(bypass_slow)} contacts — CROSSOVER at ~424d")

    source = "earth_surface"
    dest = "jupiter_station"
    t_end = 10 * YEAR

    topologies = {
        "relay_only": relay_contacts,
        "bypass_slow_crossover": bypass_slow,
    }

    all_results = {}

    for topo_name, contacts in topologies.items():
        print(f"\n{'=' * 72}")
        print(f"  Topology: {topo_name}")
        print(f"{'=' * 72}")

        # Trace paths (fixed routing — same for all tau_half)
        paths, n_inject = _trace_paths(contacts, source, dest, t_end)
        n_feasible = len(paths)
        s_t = n_feasible / n_inject if n_inject > 0 else 0

        print(f"  Injections: {n_inject}  Feasible: {n_feasible}  S_T={s_t:.4f}")

        hop_hist = Counter(p["H"] for p in paths)
        print(f"  Hop distribution: {dict(sorted(hop_hist.items()))}")

        route_sigs = Counter(p["route_signature"] for p in paths)
        print(f"  Route signatures: {len(route_sigs)} distinct")
        for sig, cnt in route_sigs.most_common(5):
            frac = cnt / len(paths) * 100
            print(f"    {sig}: {cnt} ({frac:.1f}%)")

        # Dwell statistics
        all_dwells = [d for p in paths for d in p["dwell_times"]]
        all_exposures = [p["total_exposure_s"] for p in paths]
        if all_dwells:
            print(
                f"  Dwell: mean={np.mean(all_dwells) / DAY:.1f}d  max={max(all_dwells) / DAY:.1f}d"
            )
        print(
            f"  Total exposure: mean={np.mean(all_exposures) / DAY:.0f}d  "
            f"max={max(all_exposures) / DAY:.0f}d"
        )

        # Sweep tau_half
        sweep = []
        for tau_d in TAU_HALF_DAYS:
            tau_s = tau_d * DAY if tau_d != _INF else _INF
            metrics = _apply_decay(paths, tau_s)
            if metrics:
                metrics["tau_half_days"] = tau_d
                sweep.append(metrics)

        # Print sweep table
        print(
            f"\n  {'tau_half':>10s}  {'lambda':>10s}  {'eta_opsp':>10s}  "
            f"{'eta_lyap':>10s}  {'phi':>10s}  {'var_log_p':>10s}"
        )
        baseline_phi = sweep[0]["phi"] if sweep else None
        for s in sweep:
            marker = ""
            if baseline_phi and s["phi"] > baseline_phi * 1.001:
                marker = " *** VIOLATION"
            print(
                f"  {str(s['tau_half_days']):>10s}  {s['lambda_phys']:10.6f}  "
                f"{s['eta_opsp']:10.6f}  {s['eta_lyap']:10.6f}  "
                f"{s['phi']:10.6f}  {s['var_log_p']:10.6f}{marker}"
            )

        # Route breakdown at key tau_half values
        for tau_d in [float("inf"), 180, 60]:
            match = [s for s in sweep if s["tau_half_days"] == tau_d]
            if match:
                rb = match[0]["route_breakdown"]
                print(f"\n  Route breakdown at tau_half={tau_d}d:")
                for sig in sorted(rb.keys()):
                    r = rb[sig]
                    print(
                        f"    H={r['H']}  frac={r['frac']:.3f}  "
                        f"mean_p={r['mean_p_path']:.6f}  "
                        f"exposure={r['mean_exposure_days']:.0f}d  "
                        f"{sig[:60]}"
                    )

        # Detect violations
        violations = []
        if baseline_phi:
            for s in sweep[1:]:
                if s["phi"] > baseline_phi * 1.001:
                    violations.append(
                        {
                            "tau_half_days": s["tau_half_days"],
                            "phi": s["phi"],
                            "baseline_phi": baseline_phi,
                            "delta": s["phi"] - baseline_phi,
                        }
                    )

        all_results[topo_name] = {
            "n_contacts": len(contacts),
            "n_inject": n_inject,
            "n_feasible": n_feasible,
            "s_t": s_t,
            "hop_histogram": {str(k): v for k, v in sorted(hop_hist.items())},
            "n_routes": len(route_sigs),
            "sweep": sweep,
            "n_violations": len(violations),
            "violations": violations,
        }

        if violations:
            print(f"\n  *** {len(violations)} MONOTONICITY VIOLATIONS ***")
            for v in violations:
                print(
                    f"    tau_half={v['tau_half_days']}d  "
                    f"phi={v['phi']:.6f} vs {v['baseline_phi']:.6f}  "
                    f"delta={v['delta']:+.6f}"
                )
        else:
            print("\n  No monotonicity violations.")

    # ── Part 2: Commodity-aware oracle on slow-bypass topology ─────────
    print(f"\n{'=' * 72}")
    print("  COMMODITY-AWARE ORACLE — Slow bypass (crossover at ~424d)")
    print("  (Section 3.7: minimize C = -log Q + lambda * T)")
    print(f"{'=' * 72}")

    # Lambda hazard values: ln(2)/tau_half in /second (for the Dijkstra)
    lam_sweep = []
    for tau_d in [float("inf"), 1460, 730, 500, 424, 365, 180, 90, 60, 30]:
        if tau_d == _INF:
            lam = 0.0
        else:
            lam = math.log(2) / (tau_d * DAY)
        lam_sweep.append((tau_d, lam))

    adaptive_results = []
    for tau_d, lam in lam_sweep:
        paths, n_inj = _trace_paths_commodity(bypass_slow, source, dest, t_end, lam)
        n_feas = len(paths)
        s_t_adap = n_feas / n_inj if n_inj > 0 else 0

        hop_hist = Counter(p["H"] for p in paths)
        route_sigs = Counter(p["route_signature"] for p in paths)

        # Apply decay at this tau_half to compute corrected metrics
        tau_s = tau_d * DAY if tau_d != _INF else _INF
        metrics = _apply_decay(paths, tau_s) if paths else None

        rec = {
            "tau_half_days": tau_d,
            "lambda_hazard": lam,
            "n_feasible": n_feas,
            "s_t": s_t_adap,
            "hop_histogram": {str(k): v for k, v in sorted(hop_hist.items())},
            "n_routes": len(route_sigs),
            "route_fractions": {sig: cnt / n_feas for sig, cnt in route_sigs.most_common()},
        }
        if metrics:
            rec.update(
                {
                    "eta_opsp": metrics["eta_opsp"],
                    "phi": metrics["phi"],
                    "lambda_phys": metrics["lambda_phys"],
                    "E_H": metrics["E_H"],
                    "var_H": metrics["var_H"],
                    "route_breakdown": metrics["route_breakdown"],
                }
            )
        adaptive_results.append(rec)

    # Print adaptive results
    print(
        f"\n  {'tau_half':>10s}  {'lam':>12s}  {'S_T':>6s}  {'n_feas':>6s}  "
        f"{'E_H':>6s}  {'var_H':>8s}  {'eta':>10s}  {'phi':>10s}  routes"
    )
    for r in adaptive_results:
        routes_str = "  ".join(
            f"H={k}:{v:.0%}" for k, v in sorted(r.get("hop_histogram", {}).items())
        )
        print(
            f"  {str(r['tau_half_days']):>10s}  {r['lambda_hazard']:12.2e}  "
            f"{r['s_t']:6.3f}  {r['n_feasible']:6d}  "
            f"{r.get('E_H', 0):6.2f}  {r.get('var_H', 0):8.3f}  "
            f"{r.get('eta_opsp', 0):10.6f}  {r.get('phi', 0):10.6f}  "
            f"{routes_str}"
        )

    # Route breakdown at key points
    for r in adaptive_results:
        rb = r.get("route_breakdown", {})
        if len(rb) > 1:
            print(
                f"\n  Route breakdown at tau_half={r['tau_half_days']}d "
                f"(lambda={r['lambda_hazard']:.2e}):"
            )
            for sig in sorted(rb.keys()):
                info = rb[sig]
                print(
                    f"    H={info['H']}  frac={info['frac']:.3f}  "
                    f"mean_p={info['mean_p_path']:.6f}  "
                    f"exposure={info['mean_exposure_days']:.0f}d  "
                    f"{sig[:60]}"
                )

    all_results["adaptive_bypass"] = adaptive_results

    wall = time.time() - t0
    print(f"\n  Total wall time: {wall:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    output = {
        "provenance": {
            "script": "runs/run_monotonicity_emj.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_s": wall,
            "dt_inject_days": DT_INJECT / DAY,
            "window_years": 10,
            "tau_half_sweep_days": [d if d != _INF else None for d in TAU_HALF_DAYS],
        },
        "results": all_results,
    }

    out_path = _HERE / "monotonicity_emj_results.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize(output), f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
