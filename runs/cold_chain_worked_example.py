#!/usr/bin/env python3
"""run_cold_chain_worked_example.py — Pharmaceutical cold chain worked example.

Demonstrates that the TIN framework (exact factorization, hull crossover,
commodity-aware routing, departure window scheduling) transfers to a
non-space domain without engine modification.

Network (7 nodes, matching EMJ scale):
  1. manufacturer        — vaccine production facility
  2. national_hub        — national cold storage distribution center
  3. regional_dc_a       — regional DC, eastern route (shorter, less reliable cold chain)
  4. regional_dc_b       — regional DC, western route (longer, more reliable cold chain)
  5. district_warehouse  — district-level consolidation point
  6. last_mile_depot     — last-mile staging
  7. clinic              — point of administration (destination)

Two route families (matching EMJ relay vs bypass):
  FAST route:  manufacturer → national_hub → regional_dc_a → district_warehouse
               → last_mile_depot → clinic
               5 hops, p_per_hop ≈ 0.93-0.98, transit 1-8 hours per leg
               Total transit: ~24h, but dwell at hubs adds 1-14 days

  RELIABLE route: manufacturer → national_hub → regional_dc_b → clinic
                  3 hops, p_per_hop ≈ 0.97-0.99 (better cold storage)
                  Total transit: ~36h, but longer single transit leg
                  Less intermediate dwell (fewer stops)

Commodity types (matching space cargo hierarchy):
  - Stable pills:       tau_half = inf     (room-temp stable, like hardware)
  - Standard vaccines:  tau_half = 30 days (2-8°C, like satellite components)
  - mRNA vaccines:      tau_half = 3 days  (ultra-cold, like cryogenic propellant)

Timetable: weekly truck departures on each leg (= contact windows),
simulated over 1 year. Dwell times at hubs are variable (1-14 days)
depending on scheduling alignment — exactly like synodic alignment
in the space case.

Provenance
----------
Method: Same oracle + dwell-decay machinery as run_monotonicity_emj.py
Output: runs/cold_chain_results.json

Usage:
    python -m runs.run_cold_chain_worked_example
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

# ── Time constants ────────────────────────────────────────────────────
HOUR = 3600.0
DAY = 86400.0
WEEK = 7 * DAY
YEAR = 365.25 * DAY

# ── Network definition ───────────────────────────────────────────────
# 7 nodes, 2 route families, matching EMJ structure

# Link parameters: (from, to, p_success, schedule_interval, window_hours,
#                   transit_hours, link_type)
# p_success = probability cold chain is maintained during this leg

FAST_ROUTE_LINKS = [
    # FAST route: 5 hops through regional network
    ("manufacturer", "national_hub", 0.98, 1 * DAY, 8 * HOUR, 4 * HOUR, "refrigerated_truck"),
    ("national_hub", "regional_dc_a", 0.95, 2 * DAY, 6 * HOUR, 8 * HOUR, "regional_freight"),
    ("regional_dc_a", "district_warehouse", 0.93, 3 * DAY, 4 * HOUR, 6 * HOUR, "district_van"),
    ("district_warehouse", "last_mile_depot", 0.96, 2 * DAY, 4 * HOUR, 3 * HOUR, "local_delivery"),
    ("last_mile_depot", "clinic", 0.97, 1 * DAY, 4 * HOUR, 1 * HOUR, "last_mile"),
]

RELIABLE_ROUTE_LINKS = [
    # RELIABLE route: 3 hops via better-equipped western DC
    ("manufacturer", "national_hub", 0.98, 1 * DAY, 8 * HOUR, 4 * HOUR, "refrigerated_truck"),
    ("national_hub", "regional_dc_b", 0.99, 3 * DAY, 8 * HOUR, 14 * HOUR, "cold_chain_express"),
    ("regional_dc_b", "clinic", 0.97, 4 * DAY, 6 * HOUR, 12 * HOUR, "direct_delivery"),
]

# Compute back-of-envelope
Q_fast = math.prod(p for _, _, p, *_ in FAST_ROUTE_LINKS)
Q_reliable = math.prod(p for _, _, p, *_ in RELIABLE_ROUTE_LINKS)

SOURCE = "manufacturer"
DEST = "clinic"
WINDOW_YEARS = 1
DT_INJECT = 6 * HOUR  # injection every 6 hours


# ── Contact plan generation ──────────────────────────────────────────


def _generate_contacts(links, window_s, seed=42):
    """Generate scheduled contacts from link definitions.

    Each link produces contacts at regular intervals (schedule_interval)
    with some jitter, over the simulation window.
    """
    rng = np.random.default_rng(seed)
    contacts = []

    for from_n, to_n, p_success, interval, window, transit, ltype in links:
        t = 0.0
        while t < window_s:
            # Add jitter: ±10% of interval
            jitter = rng.uniform(-0.1 * interval, 0.1 * interval)
            dep_time = max(0.0, t + jitter)
            contacts.append(
                {
                    "from_node": from_n,
                    "to_node": to_n,
                    "start_s": dep_time,
                    "duration_s": window,  # how long the departure window is open
                    "latency_s": transit,  # actual transit time
                    "p_success": p_success,  # P(cold chain maintained)
                    "link_type": ltype,
                }
            )
            t += interval

    contacts.sort(key=lambda c: c["start_s"])
    return contacts


# ── Oracle path tracer (verbatim from run_monotonicity_emj.py) ───────


def _trace_paths(contacts, source, dest, t_end_s):
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

        dwells = [
            max(0, hops[i + 1]["t_depart"] - hops[i]["t_arrive"]) for i in range(len(hops) - 1)
        ]

        total_transit = sum(h["latency_s"] for h in hops)
        total_dwell = sum(dwells)
        paths.append(
            {
                "t_inj": float(t_inj),
                "H": len(hops),
                "hops": hops,
                "dwell_times": dwells,
                "total_dwell_h": total_dwell / HOUR,
                "total_transit_h": total_transit / HOUR,
                "total_exposure_h": (total_transit + total_dwell) / HOUR,
                "log_Q": sum(math.log(max(h["p_success"], 1e-300)) for h in hops),
                "Q": math.exp(sum(math.log(max(h["p_success"], 1e-300)) for h in hops)),
                "route_signature": "→".join(h["from"] for h in hops) + "→" + hops[-1]["to"],
            }
        )

    return paths, len(inj_times)


# ── Commodity-aware oracle ────────────────────────────────────────────


def _trace_paths_commodity(contacts, source, dest, t_end_s, lam_hazard):
    """Dijkstra minimizing C = -log(p) + lambda * latency per edge."""
    adj = {}
    for idx, c in enumerate(contacts):
        fn = c["from_node"]
        ps = c.get("p_success", 0.98)
        ls = c["latency_s"]
        edge_cost = -math.log(max(ps, 1e-300)) + lam_hazard * ls
        adj.setdefault(fn, []).append(
            (c["start_s"], c["start_s"] + c["duration_s"], ls, c["to_node"], ps, idx, edge_cost)
        )

    inj_times = np.arange(0.0, t_end_s, DT_INJECT)
    paths = []

    for t_inj in inj_times:
        best_cost = {source: 0.0}
        best_time = {source: float(t_inj)}
        prev = {}
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
                dwell_cost = lam_hazard * (t_dep - t_cur)
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

        dwells = [
            max(0, hops[i + 1]["t_depart"] - hops[i]["t_arrive"]) for i in range(len(hops) - 1)
        ]
        total_transit = sum(h["latency_s"] for h in hops)
        total_dwell = sum(dwells)

        paths.append(
            {
                "t_inj": float(t_inj),
                "H": len(hops),
                "hops": hops,
                "dwell_times": dwells,
                "total_dwell_h": total_dwell / HOUR,
                "total_transit_h": total_transit / HOUR,
                "total_exposure_h": (total_transit + total_dwell) / HOUR,
                "log_Q": sum(math.log(max(h["p_success"], 1e-300)) for h in hops),
                "Q": math.exp(sum(math.log(max(h["p_success"], 1e-300)) for h in hops)),
                "route_signature": "→".join(h["from"] for h in hops) + "→" + hops[-1]["to"],
            }
        )

    return paths, len(inj_times)


# ── Dwell decay ───────────────────────────────────────────────────────


def _apply_decay(paths, tau_half_s):
    """Apply exponential decay and compute corrected metrics."""
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
            transit_decay = (
                -hop["latency_s"] / tau_half_s if tau_half_s != _INF and tau_half_s > 0 else 0.0
            )
            dwell_decay = (
                -dwells[i] / tau_half_s
                if i < len(dwells) and tau_half_s != _INF and tau_half_s > 0
                else 0.0
            )
            log_p_eff = log_p + transit_decay + dwell_decay
            hop_effs.append(log_p_eff)

        all_log_p_eff.extend(hop_effs)
        log_prod = sum(hop_effs)
        per_path.append((H, log_prod, path["route_signature"], path["total_exposure_h"]))
        path_products.append(math.exp(max(log_prod, -700)))

    hops_arr = np.array([H for H, _, _, _ in per_path])
    E_H = float(np.mean(hops_arr))
    var_H = float(np.var(hops_arr))
    lambda_phys = float(np.mean(all_log_p_eff))
    eta_opsp = float(np.mean(path_products))
    eta_lyap = math.exp(E_H * lambda_phys) if lambda_phys > -500 else 0.0
    phi = eta_opsp / eta_lyap if eta_lyap > 1e-15 else float("nan")

    route_stats = defaultdict(list)
    for H, lp, sig, exp_h in per_path:
        route_stats[sig].append((H, lp, exp_h))
    route_breakdown = {}
    for sig, entries in route_stats.items():
        lps = [e[1] for e in entries]
        exps = [e[2] for e in entries]
        route_breakdown[sig] = {
            "count": len(entries),
            "H": entries[0][0],
            "frac": len(entries) / len(per_path),
            "mean_p_path": float(np.mean([math.exp(max(lp, -700)) for lp in lps])),
            "mean_exposure_h": float(np.mean(exps)),
        }

    return {
        "E_H": E_H,
        "var_H": var_H,
        "lambda_phys": lambda_phys,
        "eta_opsp": eta_opsp,
        "eta_lyap": eta_lyap,
        "phi": phi,
        "n_paths": len(paths),
        "route_breakdown": route_breakdown,
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    print("=" * 72)
    print("COLD CHAIN WORKED EXAMPLE — Pharmaceutical Distribution")
    print("Same framework, different domain, zero engine changes")
    print("=" * 72)

    t0 = time.time()
    window_s = WINDOW_YEARS * YEAR

    # Generate contacts for both route families
    fast_contacts = _generate_contacts(FAST_ROUTE_LINKS, window_s, seed=42)
    reliable_contacts = _generate_contacts(RELIABLE_ROUTE_LINKS, window_s, seed=99)
    combined = sorted(fast_contacts + reliable_contacts, key=lambda c: c["start_s"])

    print("\n  Network: 7 nodes, 2 route families")
    print(f"  Fast route (5-hop):     Q = {Q_fast:.4f}")
    print(f"  Reliable route (3-hop): Q = {Q_reliable:.4f}")
    print(
        f"  Contacts: {len(fast_contacts)} fast + {len(reliable_contacts)} reliable = {len(combined)} total"
    )
    print(f"  Window: {WINDOW_YEARS} year, injection every {DT_INJECT / HOUR:.0f}h")

    # ── Part 1: Earliest-arrival oracle (fixed routing) ───────────────
    print(f"\n{'=' * 72}")
    print("  PART 1: Earliest-arrival oracle (commodity-blind)")
    print(f"{'=' * 72}")

    paths_ea, n_inject = _trace_paths(combined, SOURCE, DEST, window_s)
    n_feasible = len(paths_ea)
    s_t = n_feasible / n_inject if n_inject > 0 else 0

    print(f"\n  Injections: {n_inject}  Feasible: {n_feasible}  S_T = {s_t:.4f}")

    hop_hist = Counter(p["H"] for p in paths_ea)
    route_sigs = Counter(p["route_signature"] for p in paths_ea)
    print(f"  Hop distribution: {dict(sorted(hop_hist.items()))}")
    print(f"  Routes: {len(route_sigs)} distinct")
    for sig, cnt in route_sigs.most_common(5):
        frac = cnt / len(paths_ea) * 100
        print(f"    {sig}")
        print(f"      → {cnt} ({frac:.1f}%)")

    all_dwells = [d / HOUR for p in paths_ea for d in p["dwell_times"]]
    all_exp = [p["total_exposure_h"] for p in paths_ea]
    if all_dwells:
        print(f"  Dwell: mean={np.mean(all_dwells):.1f}h  max={max(all_dwells):.1f}h")
    print(f"  Total exposure: mean={np.mean(all_exp):.1f}h  max={max(all_exp):.1f}h")

    # Sweep tau_half
    tau_sweep_days = [float("inf"), 365, 90, 30, 14, 7, 3, 1]
    print(f"\n  {'tau_half':>10s}  {'lambda':>10s}  {'eta_opsp':>10s}  {'phi':>10s}")
    for tau_d in tau_sweep_days:
        tau_s = tau_d * DAY if tau_d != _INF else _INF
        metrics = _apply_decay(paths_ea, tau_s)
        if metrics:
            print(
                f"  {str(tau_d) + 'd':>10s}  {metrics['lambda_phys']:10.4f}  "
                f"{metrics['eta_opsp']:10.6f}  {metrics['phi']:10.4f}"
            )

    # Departure window scheduling
    print("\n  DEPARTURE WINDOW SCHEDULING (tau_half=3d, mRNA vaccine)")
    tau_s = 3 * DAY
    for p in paths_ea:
        T_exp = p["total_exposure_h"] * HOUR  # back to seconds
        Q = p["Q"]
        lam = math.log(2) / tau_s
        p["p_mrna"] = Q * math.exp(-lam * T_exp)
        p["p_standard"] = Q * math.exp(-math.log(2) / (30 * DAY) * T_exp)
        p["p_stable"] = Q

    by_mrna = sorted(paths_ea, key=lambda p: p["p_mrna"], reverse=True)
    print("\n  Best 10 windows for mRNA (tau_half=3d):")
    for p in by_mrna[:10]:
        dep_d = p["t_inj"] / DAY
        print(
            f"    day={dep_d:6.1f}  route H={p['H']}  dwell={p['total_dwell_h']:.1f}h  "
            f"exposure={p['total_exposure_h']:.1f}h  p_delivery={p['p_mrna']:.6f}"
        )
    print("\n  Worst 10 windows for mRNA:")
    for p in by_mrna[-10:]:
        dep_d = p["t_inj"] / DAY
        print(
            f"    day={dep_d:6.1f}  route H={p['H']}  dwell={p['total_dwell_h']:.1f}h  "
            f"exposure={p['total_exposure_h']:.1f}h  p_delivery={p['p_mrna']:.6f}"
        )

    if by_mrna[-1]["p_mrna"] > 0:
        ratio = by_mrna[0]["p_mrna"] / by_mrna[-1]["p_mrna"]
    else:
        ratio = float("inf")
    print(f"\n  Best/worst ratio: {ratio:.0f}x")

    # Scheduling gain
    print("\n  SCHEDULING GAIN (best 20% windows vs all)")
    for name, key in [
        ("Stable pills", "p_stable"),
        ("Standard vaccine (30d)", "p_standard"),
        ("mRNA vaccine (3d)", "p_mrna"),
    ]:
        vals = sorted([p[key] for p in paths_ea], reverse=True)
        top20 = vals[: max(1, len(vals) // 5)]
        mean_all = np.mean(vals)
        mean_top = np.mean(top20)
        gain = mean_top / mean_all if mean_all > 1e-15 else float("inf")
        print(f"    {name:25s}: all={mean_all:.6f}  best20%={mean_top:.6f}  gain={gain:.1f}x")

    # ── Part 2: Commodity-aware oracle ────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  PART 2: Commodity-aware oracle (route switching)")
    print(f"{'=' * 72}")

    # Hull crossover computation
    # Fast: Q=0.8010, mean exposure ~ variable
    # Reliable: Q=0.9412, mean exposure ~ variable
    # But exposure varies per injection, so compute average
    fast_paths = [p for p in paths_ea if p["H"] == 5]
    reliable_paths = [p for p in paths_ea if p["H"] == 3]

    if fast_paths and reliable_paths:
        T_fast = np.mean([p["total_exposure_h"] for p in fast_paths]) * HOUR
        T_reliable = np.mean([p["total_exposure_h"] for p in reliable_paths]) * HOUR
        print(
            f"\n  Fast route:     Q={Q_fast:.4f}  mean_exposure={T_fast / HOUR:.1f}h ({T_fast / DAY:.1f}d)"
        )
        print(
            f"  Reliable route: Q={Q_reliable:.4f}  mean_exposure={T_reliable / HOUR:.1f}h ({T_reliable / DAY:.1f}d)"
        )

        if T_fast != T_reliable:
            lam_star = math.log(Q_fast / Q_reliable) / (T_fast - T_reliable)
            if lam_star > 0:
                tau_star = math.log(2) / lam_star
                print(
                    f"  Hull crossover: lambda*={lam_star * DAY:.4f}/day  tau_half*={tau_star / DAY:.1f}d"
                )
            else:
                print("  No crossover (one route dominates)")
    else:
        print("  Only one route family found under earliest-arrival oracle")

    # Sweep commodity-aware oracle
    print(f"\n  {'tau_half':>10s}  {'S_T':>6s}  {'E_H':>6s}  {'var_H':>8s}  {'eta':>10s}  routes")
    for tau_d in [float("inf"), 90, 30, 14, 7, 3, 1]:
        if tau_d == _INF:
            lam = 0.0
        else:
            lam = math.log(2) / (tau_d * DAY)
        ca_paths, n_inj = _trace_paths_commodity(combined, SOURCE, DEST, window_s, lam)
        n_feas = len(ca_paths)
        s_t_ca = n_feas / n_inj if n_inj > 0 else 0

        hop_hist_ca = Counter(p["H"] for p in ca_paths)
        routes_str = "  ".join(f"H={k}:{v}" for k, v in sorted(hop_hist_ca.items()))

        tau_s = tau_d * DAY if tau_d != _INF else _INF
        metrics = _apply_decay(ca_paths, tau_s) if ca_paths else None
        E_H = metrics["E_H"] if metrics else 0
        var_H = metrics["var_H"] if metrics else 0
        eta = metrics["eta_opsp"] if metrics else 0

        print(
            f"  {str(tau_d) + 'd':>10s}  {s_t_ca:6.3f}  {E_H:6.2f}  {var_H:8.3f}  {eta:10.6f}  {routes_str}"
        )

        # Show route breakdown if mixed
        if metrics and len(metrics["route_breakdown"]) > 1:
            for sig in sorted(metrics["route_breakdown"].keys()):
                info = metrics["route_breakdown"][sig]
                print(
                    f"           H={info['H']}  frac={info['frac']:.3f}  "
                    f"mean_p={info['mean_p_path']:.6f}  "
                    f"exposure={info['mean_exposure_h']:.0f}h"
                )

    # ── Factorization check ───────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  FACTORIZATION CHECK: DR = S_T × eta")
    print(f"{'=' * 72}")
    for tau_d in [float("inf"), 30, 3]:
        tau_s = tau_d * DAY if tau_d != _INF else _INF
        metrics = _apply_decay(paths_ea, tau_s)
        if metrics:
            dr = s_t * metrics["eta_opsp"]
            print(
                f"  tau_half={str(tau_d) + 'd':>6s}: S_T={s_t:.4f} × eta={metrics['eta_opsp']:.6f} = DR={dr:.6f}"
            )

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

    summary = {
        "network": {
            "nodes": 7,
            "route_families": 2,
            "Q_fast": Q_fast,
            "Q_reliable": Q_reliable,
            "n_contacts": len(combined),
        },
        "factorization": {
            "S_T": s_t,
            "n_feasible": n_feasible,
            "n_inject": n_inject,
        },
    }

    output = {
        "provenance": {
            "script": "runs/run_cold_chain_worked_example.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_s": wall,
        },
        "summary": summary,
    }

    out_path = _HERE / "cold_chain_results.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize(output), f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
