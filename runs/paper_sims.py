#!/usr/bin/env python3
"""run_paper_sims.py — TIN Paper Simulation Suite

Generates quantitative results for conference paper:
  1. Halo Impact Comparison (with/without EM-L2)
  2. Route Selection Distribution (emergency/normal/bulk)
  3. Fragment Partial Delivery Demonstration

Contact plan: static snapshot of lunar_default 10-node constellation
with realistic contact windows per study §N8.

Run:  python3 run_paper_sims.py
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

from tin.config.lunar_default import get_lunar_constellation
from tin.core.dtn import (
    PARTIAL_AGGREGATE,
    PRIORITY_BULK,
    PRIORITY_EMERGENCY,
    PRIORITY_NORMAL,
    Bundle,
    CustodyNode,
    CustodyState,
    DTNNetwork,
)
from tin.core.routing import RW_CGR, Contact

# ═══════════════════════════════════════════════════════════════════════
# CONTACT PLAN BUILDER
# ═══════════════════════════════════════════════════════════════════════


def build_contact_plan(include_halo: bool = True, seed: int = 42) -> RW_CGR:
    """Build a realistic static contact plan for 28-day sim window.

    Assumptions (per study §2.1, §N8, §2.3):
    - 8 polar sats at 400 km, 89.5° inc, ~118 min period
    - Each polar sat sees Earth for ~30% of orbit when on near-side
    - Polar-polar cross-links: available when co-visible (~40% duty cycle)
    - ELFO hub: ~6h loiter at apolune with Earth visibility, 12h period
    - EM-L2 Halo: permanent Earth visibility, 14.5d period
    - Surface nodes: South Pole (good polar coverage), Far-side (needs halo)

    Contact plan is a single-epoch snapshot (t=0 to t=86400s = 1 day).
    Contacts repeat but we model one representative period.
    """
    rng = np.random.default_rng(seed)
    router = RW_CGR(seed=seed)

    cfg = get_lunar_constellation()
    polar_ids = [s.sat_id for s in cfg.satellites]  # Polar-0..Polar-7
    elfo_id = cfg.relay_hubs[0].sat_id  # ELFO-HUB

    # Surface nodes
    surface_sp = "Surface-SouthPole"
    surface_fs = "Surface-FarSide"

    # Earth ground
    earth = "DSN-Earth"

    # ── Surface → Polar contacts ──
    # South Pole sees ~4 polar sats at any time (good geometry)
    # Far-side sees ~2 polar sats briefly per orbit (poor geometry)
    polar_period_s = 7080  # ~118 min

    for i, pid in enumerate(polar_ids):
        # South Pole → each polar: ~35 min per orbit, high reliability
        phase_offset = i * (polar_period_s / 8)
        router.add_contact(
            Contact(
                surface_sp,
                pid,
                start_s=phase_offset,
                duration_s=2100,  # 35 min
                p_success=0.95,
                data_rate_kbps=256,
                latency_s=0.005,
            )
        )
        router.add_contact(
            Contact(
                pid,
                surface_sp,
                start_s=phase_offset,
                duration_s=2100,
                p_success=0.95,
                data_rate_kbps=256,
                latency_s=0.005,
            )
        )

        # Far-side → only Polar-3,4 have brief windows (~5 min, low reliability)
        # Far-side geometry is poor without halo — this is the key differentiator
        if i in [3, 4]:
            router.add_contact(
                Contact(
                    surface_fs,
                    pid,
                    start_s=phase_offset + 3540,  # half orbit later
                    duration_s=300,  # 5 min only
                    p_success=0.65,
                    data_rate_kbps=64,
                    latency_s=0.012,
                )
            )
            router.add_contact(
                Contact(
                    pid,
                    surface_fs,
                    start_s=phase_offset + 3540,
                    duration_s=300,
                    p_success=0.65,
                    data_rate_kbps=64,
                    latency_s=0.012,
                )
            )

    # ── Polar → Polar cross-links ──
    # Adjacent polars (RAAN diff = 45°) can cross-link ~40% of orbit
    for i in range(8):
        j = (i + 1) % 8
        router.add_contact(
            Contact(
                polar_ids[i],
                polar_ids[j],
                start_s=0,
                duration_s=2832,  # ~40% of period
                p_success=0.92,
                data_rate_kbps=512,
                latency_s=0.003,
            )
        )
        router.add_contact(
            Contact(
                polar_ids[j],
                polar_ids[i],
                start_s=0,
                duration_s=2832,
                p_success=0.92,
                data_rate_kbps=512,
                latency_s=0.003,
            )
        )

    # ── Polar → ELFO-HUB ──
    # ELFO at apolune (~6h window), sees all polars during loiter
    for pid in polar_ids:
        router.add_contact(
            Contact(
                pid,
                elfo_id,
                start_s=0,
                duration_s=21600,  # 6h
                p_success=0.93,
                data_rate_kbps=1000,
                latency_s=0.02,
            )
        )
        router.add_contact(
            Contact(
                elfo_id,
                pid,
                start_s=0,
                duration_s=21600,
                p_success=0.93,
                data_rate_kbps=1000,
                latency_s=0.02,
            )
        )

    # ── Polar → Earth (direct, intermittent) ──
    # Only near-side polars (0,1,6,7) see Earth; ~30% duty cycle, 1.3s light-time
    for i in [0, 1, 6, 7]:
        router.add_contact(
            Contact(
                polar_ids[i],
                earth,
                start_s=0,
                duration_s=2124,  # ~30% of period
                p_success=0.88,
                data_rate_kbps=500,
                latency_s=1.3,
            )
        )

    # ── ELFO → Earth (during apolune loiter, good geometry) ──
    router.add_contact(
        Contact(
            elfo_id,
            earth,
            start_s=0,
            duration_s=21600,  # 6h at apolune
            p_success=0.92,
            data_rate_kbps=2000,
            latency_s=1.3,
        )
    )

    # ── EM-L2 Halo contacts (only if included) ──
    if include_halo:
        halo_id = "EM-L2-HALO"

        # Halo → Earth: PERMANENT visibility, high reliability, optical-capable
        router.add_contact(
            Contact(
                halo_id,
                earth,
                start_s=0,
                duration_s=86400,  # full day
                p_success=0.99,
                data_rate_kbps=10000,
                latency_s=1.4,
            )
        )

        # Polar → Halo: all polars can reach halo when on far-side (~50% of orbit)
        for pid in polar_ids:
            router.add_contact(
                Contact(
                    pid,
                    halo_id,
                    start_s=0,
                    duration_s=3540,  # ~50% of period
                    p_success=0.96,
                    data_rate_kbps=2000,
                    latency_s=0.4,
                )
            )

        # ELFO → Halo
        router.add_contact(
            Contact(
                elfo_id,
                halo_id,
                start_s=0,
                duration_s=21600,
                p_success=0.97,
                data_rate_kbps=5000,
                latency_s=0.35,
            )
        )

        # Surface far-side → Halo (direct, limited but available per §N8)
        router.add_contact(
            Contact(
                surface_fs,
                halo_id,
                start_s=0,
                duration_s=72000,  # ~83% visibility from far-side
                p_success=0.90,
                data_rate_kbps=64,
                latency_s=0.5,
            )
        )

    return router


def build_dtn_network(include_halo: bool = True) -> DTNNetwork:
    """Create DTN network nodes matching the contact plan."""
    net = DTNNetwork()
    cfg = get_lunar_constellation()

    # Surface nodes
    net.add_node(CustodyNode("Surface-SouthPole", node_type="surface", storage_bytes=500_000_000))
    net.add_node(CustodyNode("Surface-FarSide", node_type="surface", storage_bytes=500_000_000))

    # Polar sats (100 MB storage each)
    for s in cfg.satellites:
        net.add_node(CustodyNode(s.sat_id, node_type="polar", storage_bytes=100_000_000))

    # ELFO hub (500 MB)
    net.add_node(CustodyNode(cfg.relay_hubs[0].sat_id, node_type="elfo", storage_bytes=500_000_000))

    # Halo
    if include_halo:
        net.add_node(CustodyNode("EM-L2-HALO", node_type="halo", storage_bytes=1_000_000_000))

    # Earth DSN (unlimited)
    net.add_node(CustodyNode("DSN-Earth", node_type="ground", storage_bytes=10_000_000_000))

    return net


# ═══════════════════════════════════════════════════════════════════════
# SIMULATION 1: HALO IMPACT COMPARISON
# ═══════════════════════════════════════════════════════════════════════


def sim1_halo_impact(n_bundles: int = 300, seed: int = 42):
    """Compare delivery ratio, mean latency, and fragment completion
    for emergency/normal/bulk traffic WITH and WITHOUT EM-L2 halo."""

    print("\n" + "=" * 72)
    print("  SIMULATION 1: Halo Impact Comparison")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    sources = ["Surface-SouthPole", "Surface-FarSide"]
    priorities = [PRIORITY_EMERGENCY, PRIORITY_NORMAL, PRIORITY_BULK]
    priority_names = {0: "EMERGENCY", 2: "NORMAL", 3: "BULK"}
    priority_sizes = {0: 256, 2: 4096, 3: 65000}  # bytes

    results = {}  # (halo_mode, priority) → {delivered, total, latencies, frag_complete}

    for include_halo in [True, False]:
        mode = "WITH_HALO" if include_halo else "NO_HALO"
        router = build_contact_plan(include_halo=include_halo, seed=seed)
        net = build_dtn_network(include_halo=include_halo)

        for prio in priorities:
            pname = priority_names[prio]
            delivered = 0
            total = 0
            latencies = []
            frag_complete = 0
            frag_total = 0

            bundles_per_source = n_bundles // (len(sources) * len(priorities))

            for src in sources:
                for b_idx in range(bundles_per_source):
                    total += 1
                    t_s = rng.uniform(0, 3600)  # random start within first hour
                    is_frag = prio == PRIORITY_BULK and b_idx % 3 == 0
                    size = priority_sizes[prio]

                    bundle = net.create_bundle(
                        source=src,
                        destination="DSN-Earth",
                        priority=prio,
                        size_bytes=size,
                        payload_type="telemetry",
                        t_s=t_s,
                        fragmented=is_frag,
                        num_groups=5 if is_frag else 1,
                    )
                    if is_frag:
                        frag_total += 1

                    # Route via RW-CGR
                    route = router.route_bundle(bundle, src, "DSN-Earth", t_s=t_s)
                    if route and len(route) >= 2:
                        # Simulate stochastic link failure along route
                        # Compute cumulative p_success for the path
                        path_p = 1.0
                        for k in range(len(route) - 1):
                            fn, tn = route[k], route[k + 1]
                            cs = [c for c in router.contact_graph.get(fn, []) if c.to_node == tn]
                            if cs:
                                path_p *= cs[0].p_success
                            else:
                                path_p *= 0.5  # unknown link

                        # Stochastic delivery: bundle fails with probability 1-path_p
                        if rng.random() > path_p:
                            # Link failure — bundle lost
                            bundle.custody_state = CustodyState.X
                            bundle.deleted = True
                            if total % 20 == 0:
                                router.clear_cache()
                            continue

                        # Determine link types
                        link_types = []
                        for k in range(len(route) - 1):
                            if "EM-L2" in route[k] or "EM-L2" in route[k + 1]:
                                link_types.append("optical")
                            elif "DSN" in route[k + 1]:
                                link_types.append("rf-deep")
                            else:
                                link_types.append("rf")

                        # Use appropriate data rates from the contact
                        contacts = router.contact_graph
                        rate = 256.0  # default
                        for k in range(len(route) - 1):
                            fn, tn = route[k], route[k + 1]
                            cs = [c for c in contacts.get(fn, []) if c.to_node == tn]
                            if cs:
                                rate = min(rate, cs[0].data_rate_kbps)

                        ok = net.route_along_path(
                            bundle,
                            route,
                            link_types=link_types,
                            t_s=t_s,
                            data_rate_kbps=rate,
                        )
                        if ok:
                            delivered += 1
                            latencies.append(bundle.total_latency_s())
                            if is_frag:
                                if bundle.aggregate_state() == CustodyState.R.value:
                                    frag_complete += 1

                    # Clear cache periodically to simulate time-varying contacts
                    if total % 20 == 0:
                        router.clear_cache()

            results[(mode, pname)] = {
                "delivered": delivered,
                "total": total,
                "ratio": delivered / max(total, 1),
                "mean_latency_s": np.mean(latencies) if latencies else float("nan"),
                "p50_latency_s": np.percentile(latencies, 50) if latencies else float("nan"),
                "p95_latency_s": np.percentile(latencies, 95) if latencies else float("nan"),
                "frag_complete": frag_complete,
                "frag_total": frag_total,
            }

    # ── Print Table 1: Delivery Ratio ──
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  Table 1: Delivery Ratio by Priority and Halo Configuration    │")
    print("├───────────┬────────────┬────────────┬──────────────────────────┤")
    print("│ Priority  │ With Halo  │  No Halo   │ Improvement (Δ)          │")
    print("├───────────┼────────────┼────────────┼──────────────────────────┤")
    for pname in ["EMERGENCY", "NORMAL", "BULK"]:
        r_h = results[("WITH_HALO", pname)]
        r_n = results[("NO_HALO", pname)]
        delta = r_h["ratio"] - r_n["ratio"]
        print(
            f"│ {pname:<9} │  {r_h['ratio']:>7.1%}   │  {r_n['ratio']:>7.1%}   │  {delta:>+7.1%}  ({r_h['delivered']}/{r_h['total']} vs {r_n['delivered']}/{r_n['total']}) │"
        )
    print("└───────────┴────────────┴────────────┴──────────────────────────┘")

    # ── Print Table 2: Latency ──
    print("\n┌───────────────────────────────────────────────────────────────────────┐")
    print("│  Table 2: End-to-End Latency (seconds) — Delivered Bundles Only      │")
    print("├───────────┬──────────────────────┬──────────────────────┬─────────────┤")
    print("│ Priority  │  With Halo (mean/p95)│  No Halo (mean/p95) │  Δ mean     │")
    print("├───────────┼──────────────────────┼──────────────────────┼─────────────┤")
    for pname in ["EMERGENCY", "NORMAL", "BULK"]:
        r_h = results[("WITH_HALO", pname)]
        r_n = results[("NO_HALO", pname)]
        dm = (
            r_h["mean_latency_s"] - r_n["mean_latency_s"]
            if not (math.isnan(r_h["mean_latency_s"]) or math.isnan(r_n["mean_latency_s"]))
            else float("nan")
        )
        print(
            f"│ {pname:<9} │  {r_h['mean_latency_s']:>7.2f} / {r_h['p95_latency_s']:>7.2f}   │  {r_n['mean_latency_s']:>7.2f} / {r_n['p95_latency_s']:>7.2f}   │  {dm:>+8.2f}   │"
        )
    print("└───────────┴──────────────────────┴──────────────────────┴─────────────┘")

    # ── Print Table 3: Fragment Completion ──
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  Table 3: Fragment Group Completion (BULK only)     │")
    print("├───────────────┬──────────────┬──────────────────────┤")
    print("│ Configuration │ Complete/Tot │ Completion Rate      │")
    print("├───────────────┼──────────────┼──────────────────────┤")
    for mode in ["WITH_HALO", "NO_HALO"]:
        r = results[(mode, "BULK")]
        rate = r["frag_complete"] / max(r["frag_total"], 1)
        label = "With Halo" if mode == "WITH_HALO" else "No Halo  "
        print(
            f"│ {label}     │   {r['frag_complete']:>3}/{r['frag_total']:<3}     │  {rate:>7.1%}               │"
        )
    print("└───────────────┴──────────────┴──────────────────────┘")

    return results


# ═══════════════════════════════════════════════════════════════════════
# SIMULATION 2: ROUTE SELECTION DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════


def sim2_route_distribution(n_bundles: int = 500, seed: int = 42):
    """Analyze what fraction of traffic routes through halo vs. direct paths."""

    print("\n" + "=" * 72)
    print("  SIMULATION 2: Route Selection Distribution (With Halo)")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    router = build_contact_plan(include_halo=True, seed=seed)

    sources = ["Surface-SouthPole", "Surface-FarSide"]
    priorities = [PRIORITY_EMERGENCY, PRIORITY_NORMAL, PRIORITY_BULK]
    priority_names = {0: "EMERGENCY", 2: "NORMAL", 3: "BULK"}

    # Track route types
    route_stats = defaultdict(lambda: defaultdict(int))
    # route_stats[pname][route_type] = count

    def classify_route(route):
        if route is None:
            return "NO_ROUTE"
        route_str = " → ".join(route)
        if "EM-L2-HALO" in route:
            return "VIA_HALO"
        elif "ELFO-HUB" in route:
            return "VIA_ELFO"
        else:
            return "DIRECT_POLAR"

    for prio in priorities:
        pname = priority_names[prio]
        for src in sources:
            bundles_per = n_bundles // (len(sources) * len(priorities))
            for b_idx in range(bundles_per):
                t_s = rng.uniform(0, 7200)
                size = {0: 256, 2: 4096, 3: 65000}[prio]
                bundle = Bundle(
                    bundle_id=f"s2-{pname}-{src[:3]}-{b_idx}",
                    priority=prio,
                    size_bytes=size,
                )

                # Get top-k routes
                routes = router.find_routes(
                    src, "DSN-Earth", bundle, t_s=t_s, n_samples=300, top_k=3
                )
                router.clear_cache()  # force re-sample each time

                if routes:
                    best = routes[0]
                    rtype = classify_route(best)
                    route_stats[pname][rtype] += 1

                    # Also track source-specific
                    src_key = f"{pname}_{src.split('-')[-1]}"
                    route_stats[src_key][rtype] += 1
                else:
                    route_stats[pname]["NO_ROUTE"] += 1

    # ── Print Table 4: Route Distribution ──
    print("\n┌───────────────────────────────────────────────────────────────────┐")
    print("│  Table 4: Best-Route Selection Distribution (% of routed)       │")
    print("├───────────┬──────────┬──────────┬──────────────┬────────────────┤")
    print("│ Priority  │ Via Halo │ Via ELFO │ Direct Polar │ No Route Found │")
    print("├───────────┼──────────┼──────────┼──────────────┼────────────────┤")
    for pname in ["EMERGENCY", "NORMAL", "BULK"]:
        stats = route_stats[pname]
        total = sum(stats.values())
        if total == 0:
            continue
        halo_pct = stats.get("VIA_HALO", 0) / total
        elfo_pct = stats.get("VIA_ELFO", 0) / total
        direct_pct = stats.get("DIRECT_POLAR", 0) / total
        noroute_pct = stats.get("NO_ROUTE", 0) / total
        print(
            f"│ {pname:<9} │  {halo_pct:>5.1%}  │  {elfo_pct:>5.1%}  │    {direct_pct:>5.1%}     │     {noroute_pct:>5.1%}      │"
        )
    print("└───────────┴──────────┴──────────┴──────────────┴────────────────┘")

    # ── Table 5: By Source ──
    print("\n┌───────────────────────────────────────────────────────────────────────┐")
    print("│  Table 5: Route Selection by Source (Emergency traffic only)         │")
    print("├────────────────────┬──────────┬──────────┬──────────────┬────────────┤")
    print("│ Source             │ Via Halo │ Via ELFO │ Direct Polar │  No Route  │")
    print("├────────────────────┼──────────┼──────────┼──────────────┼────────────┤")
    for src_label in ["EMERGENCY_SouthPole", "EMERGENCY_FarSide"]:
        if src_label not in route_stats:
            continue
        stats = route_stats[src_label]
        total = sum(stats.values())
        if total == 0:
            continue
        halo_pct = stats.get("VIA_HALO", 0) / total
        elfo_pct = stats.get("VIA_ELFO", 0) / total
        direct_pct = stats.get("DIRECT_POLAR", 0) / total
        noroute_pct = stats.get("NO_ROUTE", 0) / total
        nice_name = src_label.replace("EMERGENCY_", "")
        print(
            f"│ {nice_name:<18} │  {halo_pct:>5.1%}  │  {elfo_pct:>5.1%}  │    {direct_pct:>5.1%}     │   {noroute_pct:>5.1%}     │"
        )
    print("└────────────────────┴──────────┴──────────┴──────────────┴────────────┘")

    return route_stats


# ═══════════════════════════════════════════════════════════════════════
# SIMULATION 3: FRAGMENT PARTIAL DELIVERY DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════


def sim3_fragment_partial_delivery(seed: int = 42):
    """Demonstrate PARTIAL aggregate state with a concrete fragmented bundle.

    Scenario: A 6.5 MB science bundle from far-side, fragmented into 10 groups.
    Groups 0-6 are delivered via halo; groups 7-9 are still in transit
    (simulating the realistic case where some relay paths complete before others).
    """

    print("\n" + "=" * 72)
    print("  SIMULATION 3: Fragment Partial Delivery Demonstration")
    print("=" * 72)

    net = DTNNetwork()

    # Build minimal network for the scenario
    nodes = {
        "Surface-FarSide": CustodyNode("Surface-FarSide", storage_bytes=500_000_000),
        "Polar-3": CustodyNode("Polar-3", storage_bytes=100_000_000),
        "Polar-5": CustodyNode("Polar-5", storage_bytes=100_000_000),
        "EM-L2-HALO": CustodyNode("EM-L2-HALO", storage_bytes=1_000_000_000),
        "ELFO-HUB": CustodyNode("ELFO-HUB", storage_bytes=500_000_000),
        "DSN-Earth": CustodyNode("DSN-Earth", storage_bytes=10_000_000_000),
    }
    for n in nodes.values():
        net.add_node(n)

    # Create a 6.5 MB science bundle, fragmented into 10 groups
    bundle = net.create_bundle(
        source="Surface-FarSide",
        destination="DSN-Earth",
        priority=PRIORITY_NORMAL,
        size_bytes=6_500_000,
        payload_type="science_image",
        t_s=0.0,
        fragmented=True,
        num_groups=10,
    )

    print(f"\n  Bundle: {bundle.bundle_id}")
    print(f"  Size: {bundle.size_bytes:,} bytes ({bundle.size_bytes / 1e6:.1f} MB)")
    print(f"  Fragment groups: {len(bundle.fragment_groups)}")
    print(f"  Initial aggregate state: {bundle.aggregate_state()}")

    # ── Phase 1: Source accepts custody ──
    nodes["Surface-FarSide"].accept_custody(bundle, t_s=0.0)
    print("\n  Phase 1 — Source accepts custody")
    print(f"    State: {bundle.aggregate_state()}, Custodian: {bundle.current_custodian}")

    # ── Phase 2: Forward to Polar-3 (all groups) ──
    nodes["Surface-FarSide"].forward_bundle(
        bundle,
        nodes["Polar-3"],
        t_s=10.0,
        link_type="rf",
        data_rate_kbps=128,
        propagation_delay_s=0.008,
    )
    print("\n  Phase 2 — Forward to Polar-3")
    print(f"    State: {bundle.aggregate_state()}, Custodian: {bundle.current_custodian}")

    # ── Phase 3: Polar-3 forwards to Halo (all groups) ──
    nodes["Polar-3"].forward_bundle(
        bundle,
        nodes["EM-L2-HALO"],
        t_s=500.0,
        link_type="rf",
        data_rate_kbps=2000,
        propagation_delay_s=0.4,
    )
    print("\n  Phase 3 — Forward to EM-L2 Halo")
    print(f"    State: {bundle.aggregate_state()}, Custodian: {bundle.current_custodian}")

    # ── Phase 4: Halo forwards to DSN-Earth, but simulate partial completion ──
    # We manually simulate: groups 0-6 complete delivery, groups 7-9 still in transit
    # This models a realistic scenario where the optical link drops mid-transfer

    # First, transition all groups to OUTSTANDING (as if halo started forwarding)
    for g in bundle.fragment_groups:
        g.state = CustodyState.O

    # Groups 0-6: successfully received at Earth
    for i in range(7):
        bundle.fragment_groups[i].state = CustodyState.R
        bundle.fragment_groups[i].custody_record.append(
            {
                "node": "DSN-Earth",
                "time_s": 560.0 + i * 0.5,
            }
        )

    # Groups 7-9: still OUTSTANDING (in transit / link interrupted)
    # (they remain CustodyState.O from above)

    agg = bundle.aggregate_state()
    print("\n  Phase 4 — Partial delivery (7/10 groups received at Earth)")
    print(f"    Aggregate state: {agg}")
    print(f"    Expected: PARTIAL → {PARTIAL_AGGREGATE}")
    assert agg == PARTIAL_AGGREGATE, f"Expected PARTIAL, got {agg}"

    # ── Per-group state table ──
    print("\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │  Table 6: Fragment Group State (Bundle {bundle.bundle_id})  │")
    print("  ├──────────┬────────────┬────────────┬───────────────┤")
    print("  │ Group    │ Offset     │ Size (B)   │ State         │")
    print("  ├──────────┼────────────┼────────────┼───────────────┤")
    for g in bundle.fragment_groups:
        size = g.offset_end - g.offset_start
        print(f"  │ {g.group_id:<8} │ {g.offset_start:>10,} │ {size:>10,} │ {g.state.value:<13} │")
    print("  └──────────┴────────────┴────────────┴───────────────┘")

    # ── Phase 5: Remaining groups delivered → full RECEIVED ──
    for i in range(7, 10):
        bundle.fragment_groups[i].state = CustodyState.R
        bundle.fragment_groups[i].custody_record.append(
            {
                "node": "DSN-Earth",
                "time_s": 620.0 + i * 0.5,
            }
        )
    bundle.custody_state = CustodyState.R
    bundle.delivered = True
    bundle.delivery_time_s = 625.0

    agg_final = bundle.aggregate_state()
    print("\n  Phase 5 — All groups delivered")
    print(f"    Final aggregate state: {agg_final}")
    print(f"    Bundle delivered: {bundle.delivered}")

    # Summary
    print("\n  ┌──────────────────────────────────────────┐")
    print("  │  Custody Chain:                           │")
    for entry in bundle.custody_chain:
        print(f"  │    t={entry['time_s']:>8.3f}s  node={entry['node']:<20}│")
    print(f"  │  Total hops: {bundle.hop_count}                           │")
    print("  └──────────────────────────────────────────┘")

    return bundle


# ═══════════════════════════════════════════════════════════════════════
# SIMULATION 4: COMPREHENSIVE METRICS SUMMARY (JSON for plotting)
# ═══════════════════════════════════════════════════════════════════════


def sim4_generate_plot_data(n_bundles: int = 600, seed: int = 42):
    """Generate structured data for Plotly visualization."""

    print("\n" + "=" * 72)
    print("  SIMULATION 4: Generating Plot Data")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    priorities = [PRIORITY_EMERGENCY, PRIORITY_NORMAL, PRIORITY_BULK]
    priority_names = {0: "EMERGENCY", 2: "NORMAL", 3: "BULK"}
    sources = ["Surface-SouthPole", "Surface-FarSide"]

    plot_data = {
        "delivery_ratios": [],
        "latency_distributions": [],
        "route_types": [],
    }

    for include_halo in [True, False]:
        mode = "With Halo" if include_halo else "No Halo"
        router = build_contact_plan(include_halo=include_halo, seed=seed)
        net = build_dtn_network(include_halo=include_halo)

        for prio in priorities:
            pname = priority_names[prio]
            delivered = 0
            total = 0
            latencies = []
            halo_routes = 0
            elfo_routes = 0
            direct_routes = 0
            no_routes = 0

            bundles_per = n_bundles // (len(sources) * len(priorities))

            for src in sources:
                for b_idx in range(bundles_per):
                    total += 1
                    t_s = rng.uniform(0, 5000)
                    size = {0: 256, 2: 4096, 3: 65000}[prio]

                    bundle = net.create_bundle(
                        source=src,
                        destination="DSN-Earth",
                        priority=prio,
                        size_bytes=size,
                        payload_type="telemetry",
                        t_s=t_s,
                    )

                    route = router.route_bundle(bundle, src, "DSN-Earth", t_s=t_s)
                    if route and len(route) >= 2:
                        link_types = ["rf"] * (len(route) - 1)
                        ok = net.route_along_path(
                            bundle, route, link_types=link_types, t_s=t_s, data_rate_kbps=256.0
                        )
                        if ok:
                            delivered += 1
                            latencies.append(bundle.total_latency_s())

                        if "EM-L2-HALO" in route:
                            halo_routes += 1
                        elif "ELFO-HUB" in route:
                            elfo_routes += 1
                        else:
                            direct_routes += 1
                    else:
                        no_routes += 1

                    if total % 30 == 0:
                        router.clear_cache()

            ratio = delivered / max(total, 1)
            plot_data["delivery_ratios"].append(
                {
                    "config": mode,
                    "priority": pname,
                    "ratio": round(ratio, 4),
                    "delivered": delivered,
                    "total": total,
                }
            )
            plot_data["latency_distributions"].append(
                {
                    "config": mode,
                    "priority": pname,
                    "mean": round(np.mean(latencies), 3) if latencies else None,
                    "p50": round(np.percentile(latencies, 50), 3) if latencies else None,
                    "p95": round(np.percentile(latencies, 95), 3) if latencies else None,
                    "min": round(min(latencies), 3) if latencies else None,
                    "max": round(max(latencies), 3) if latencies else None,
                    "n": len(latencies),
                }
            )
            if include_halo:
                routed_total = halo_routes + elfo_routes + direct_routes + no_routes
                if routed_total > 0:
                    plot_data["route_types"].append(
                        {
                            "priority": pname,
                            "via_halo_pct": round(halo_routes / routed_total, 4),
                            "via_elfo_pct": round(elfo_routes / routed_total, 4),
                            "direct_polar_pct": round(direct_routes / routed_total, 4),
                            "no_route_pct": round(no_routes / routed_total, 4),
                        }
                    )

    # Save JSON
    json_path = os.path.join(os.path.dirname(__file__), "paper_sim_data.json")
    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    print(f"\n  Plot data saved to: {json_path}")

    return plot_data


# ═══════════════════════════════════════════════════════════════════════
# PLOTLY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════


def generate_plots(plot_data: dict):
    """Generate Plotly HTML visualizations from simulation data."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("\n  ⚠ Plotly not installed. Skipping visualizations.")
        print("    Install with: pip install plotly")
        return None

    print("\n" + "=" * 72)
    print("  Generating Plotly Visualizations")
    print("=" * 72)

    # ── Figure 1: Delivery Ratio Comparison (grouped bar) ──
    fig1 = go.Figure()

    priorities_order = ["EMERGENCY", "NORMAL", "BULK"]
    colors = {"With Halo": "#2196F3", "No Halo": "#FF5722"}

    for config in ["With Halo", "No Halo"]:
        ratios = []
        for pname in priorities_order:
            entry = [
                d
                for d in plot_data["delivery_ratios"]
                if d["config"] == config and d["priority"] == pname
            ]
            ratios.append(entry[0]["ratio"] * 100 if entry else 0)

        fig1.add_trace(
            go.Bar(
                name=config,
                x=priorities_order,
                y=ratios,
                marker_color=colors[config],
                text=[f"{r:.1f}%" for r in ratios],
                textposition="outside",
            )
        )

    fig1.update_layout(
        title="Fig. 1: Delivery Ratio by Priority — Halo Impact",
        yaxis_title="Delivery Ratio (%)",
        xaxis_title="Traffic Priority",
        barmode="group",
        yaxis_range=[0, 105],
        template="plotly_white",
        font=dict(size=14),
        width=800,
        height=500,
    )

    fig1_path = os.path.join(os.path.dirname(__file__), "fig1_delivery_ratio.html")
    fig1.write_html(fig1_path)
    print(f"  → {fig1_path}")

    # ── Figure 2: Latency Comparison ──
    fig2 = make_subplots(rows=1, cols=1)

    for config in ["With Halo", "No Halo"]:
        means = []
        p95s = []
        for pname in priorities_order:
            entry = [
                d
                for d in plot_data["latency_distributions"]
                if d["config"] == config and d["priority"] == pname
            ]
            if entry and entry[0]["mean"] is not None:
                means.append(entry[0]["mean"])
                p95s.append(entry[0]["p95"])
            else:
                means.append(0)
                p95s.append(0)

        fig2.add_trace(
            go.Bar(
                name=f"{config} (mean)",
                x=priorities_order,
                y=means,
                marker_color=colors[config],
                text=[f"{m:.2f}s" for m in means],
                textposition="outside",
            )
        )
        fig2.add_trace(
            go.Scatter(
                name=f"{config} (p95)",
                x=priorities_order,
                y=p95s,
                mode="markers+text",
                marker=dict(size=12, symbol="diamond", color=colors[config]),
                text=[f"p95: {p:.2f}s" for p in p95s],
                textposition="top center",
            )
        )

    fig2.update_layout(
        title="Fig. 2: End-to-End Latency — Mean and P95",
        yaxis_title="Latency (seconds)",
        xaxis_title="Traffic Priority",
        template="plotly_white",
        font=dict(size=14),
        width=800,
        height=500,
    )

    fig2_path = os.path.join(os.path.dirname(__file__), "fig2_latency.html")
    fig2.write_html(fig2_path)
    print(f"  → {fig2_path}")

    # ── Figure 3: Route Distribution (stacked bar) ──
    if plot_data["route_types"]:
        fig3 = go.Figure()

        route_colors = {
            "via_halo_pct": "#4CAF50",
            "via_elfo_pct": "#FFC107",
            "direct_polar_pct": "#2196F3",
            "no_route_pct": "#F44336",
        }
        route_labels = {
            "via_halo_pct": "Via EM-L2 Halo",
            "via_elfo_pct": "Via ELFO Hub",
            "direct_polar_pct": "Direct Polar→Earth",
            "no_route_pct": "No Route",
        }

        for rtype in ["via_halo_pct", "via_elfo_pct", "direct_polar_pct", "no_route_pct"]:
            values = []
            for pname in priorities_order:
                entry = [d for d in plot_data["route_types"] if d["priority"] == pname]
                values.append(entry[0][rtype] * 100 if entry else 0)

            fig3.add_trace(
                go.Bar(
                    name=route_labels[rtype],
                    x=priorities_order,
                    y=values,
                    marker_color=route_colors[rtype],
                )
            )

        fig3.update_layout(
            title="Fig. 3: Route Selection Distribution by Priority (With Halo)",
            yaxis_title="Percentage of Traffic (%)",
            xaxis_title="Traffic Priority",
            barmode="stack",
            template="plotly_white",
            font=dict(size=14),
            width=800,
            height=500,
        )

        fig3_path = os.path.join(os.path.dirname(__file__), "fig3_route_distribution.html")
        fig3.write_html(fig3_path)
        print(f"  → {fig3_path}")

    return [fig1_path, fig2_path, fig3_path] if plot_data["route_types"] else [fig1_path, fig2_path]


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  TIN — Paper Simulation Suite                                  ║")
    print("║  Contact plan: lunar_default 10-node constellation              ║")
    print("║  Seed: 42 (reproducible)                                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Run all simulations
    r1 = sim1_halo_impact(n_bundles=300, seed=42)
    r2 = sim2_route_distribution(n_bundles=500, seed=42)
    r3 = sim3_fragment_partial_delivery(seed=42)
    r4 = sim4_generate_plot_data(n_bundles=600, seed=42)

    # Generate plots
    plot_paths = generate_plots(r4)

    print("\n" + "=" * 72)
    print("  ALL SIMULATIONS COMPLETE")
    print("=" * 72)
    print("\n  Files generated:")
    print("    • paper_sim_data.json  — raw results for paper tables")
    if plot_paths:
        for p in plot_paths:
            print(f"    • {os.path.basename(p)}")
    print()
