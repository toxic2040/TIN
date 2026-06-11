"""run_validation.py — Standalone validation (no pytest needed).

Run: python3 run_validation.py
Maps to: B1, N1, N5, B2, N8, A5
"""

import sys

from tin.config.lunar_default import LUNAR_BODY, get_lunar_constellation

# Add project root to path
from tin.core.dtn import (
    PARTIAL_AGGREGATE,
    PRIORITY_EMERGENCY,
    PRIORITY_NORMAL,
    Bundle,
    CustodyNode,
    CustodyState,
    DTNNetwork,
    FragmentGroup,
)
from tin.core.routing import RW_CGR, Contact

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  ✗ {name} — {detail}")


def main():
    global passed, failed, errors

    # ── B1: 5-State Custody FSM ──────────────────────────────────────────
    print("\n═══ B1: 5-State Custody FSM ═══")

    b = Bundle(bundle_id="b1")
    check("Initial state = HOLDING", b.custody_state == CustodyState.H)

    b.accept_custody("node-A", 100.0)
    check("accept_custody → H", b.custody_state == CustodyState.H)
    check("custodian recorded", b.current_custodian == "node-A")
    check("chain length", len(b.custody_chain) == 1)

    node = CustodyNode("n1")
    b_exp = Bundle(bundle_id="b-exp", created_s=0.0, lifetime_s=100.0)
    accepted = node.accept_custody(b_exp, 200.0)
    check("Expired → X, refused", not accepted and b_exp.custody_state == CustodyState.X)

    node_small = CustodyNode("n-small", storage_bytes=100)
    b_big = Bundle(bundle_id="b-big", size_bytes=200)
    accepted2 = node_small.accept_custody(b_big, 0.0)
    check("Overflow → P, refused", not accepted2 and b_big.custody_state == CustodyState.P)

    n1 = CustodyNode("n1-fwd")
    n2 = CustodyNode("n2-fwd")
    b_fwd = Bundle(bundle_id="b-fwd", size_bytes=1000, created_s=0.0)
    n1.accept_custody(b_fwd, 0.0)
    n1.forward_bundle(b_fwd, n2, t_s=1.0, data_rate_kbps=256.0)
    check("Forward → accepted by n2 → H", b_fwd.custody_state == CustodyState.H)
    check("Custodian updated", b_fwd.current_custodian == "n2-fwd")

    b_del = Bundle(bundle_id="b-del")
    b_del.mark_delivered(100.0)
    check("mark_delivered → R", b_del.custody_state == CustodyState.R and b_del.delivered)

    node_prio = CustodyNode("n-prio")
    b_norm = Bundle(bundle_id="bn", priority=PRIORITY_NORMAL)
    b_emrg = Bundle(bundle_id="be", priority=PRIORITY_EMERGENCY)
    node_prio.accept_custody(b_norm, 0.0)
    node_prio.accept_custody(b_emrg, 1.0)
    check("Emergency inserted at front", node_prio.bundles[0].bundle_id == "be")

    # ── N1: Fragment-Group Custody ────────────────────────────────────────
    print("\n═══ N1: Fragment-Group Custody ═══")

    b_nf = Bundle(bundle_id="bnf")
    check("Non-frag aggregate = H", b_nf.aggregate_state() == CustodyState.H.value)

    b_frag_h = Bundle(
        bundle_id="bfh",
        is_fragmented=True,
        fragment_groups=[
            FragmentGroup("g0", 0, 100, state=CustodyState.H),
            FragmentGroup("g1", 100, 200, state=CustodyState.H),
        ],
    )
    check("All H → aggregate H", b_frag_h.aggregate_state() == CustodyState.H.value)

    b_frag_r = Bundle(
        bundle_id="bfr",
        is_fragmented=True,
        fragment_groups=[
            FragmentGroup("g0", 0, 100, state=CustodyState.R),
            FragmentGroup("g1", 100, 200, state=CustodyState.R),
        ],
    )
    check("All R → aggregate R", b_frag_r.aggregate_state() == CustodyState.R.value)

    b_frag_p = Bundle(
        bundle_id="bfp",
        is_fragmented=True,
        fragment_groups=[
            FragmentGroup("g0", 0, 100, state=CustodyState.R),
            FragmentGroup("g1", 100, 200, state=CustodyState.H),
        ],
    )
    check("Mixed R+H → PARTIAL", b_frag_p.aggregate_state() == PARTIAL_AGGREGATE)

    b_frag_x = Bundle(
        bundle_id="bfx",
        is_fragmented=True,
        fragment_groups=[
            FragmentGroup("g0", 0, 100, state=CustodyState.R),
            FragmentGroup("g1", 100, 200, state=CustodyState.X),
        ],
    )
    check("Any X → aggregate X", b_frag_x.aggregate_state() == CustodyState.X.value)

    b_frag_ac = Bundle(
        bundle_id="bfac",
        is_fragmented=True,
        fragment_groups=[
            FragmentGroup("g0", 0, 100, state=CustodyState.O),
            FragmentGroup("g1", 100, 200, state=CustodyState.O),
        ],
    )
    b_frag_ac.accept_custody("node-A", 50.0)
    check(
        "accept_custody propagates to fragments",
        all(g.state == CustodyState.H for g in b_frag_ac.fragment_groups),
    )

    net = DTNNetwork()
    net.add_node(CustodyNode("src"))
    b_net_frag = net.create_bundle("src", "dst", size_bytes=6500000, fragmented=True, num_groups=10)
    check("Network creates 10 groups", len(b_net_frag.fragment_groups) == 10)
    check(
        "Fragment offsets span bundle",
        b_net_frag.fragment_groups[0].offset_start == 0
        and b_net_frag.fragment_groups[-1].offset_end == 6500000,
    )

    # ── N5: SPE Pre-Flush ────────────────────────────────────────────────
    print("\n═══ N5: SPE Pre-Flush ═══")

    node_spe = CustodyNode("relay-spe")
    b_spe1 = Bundle(bundle_id="bs1")
    b_spe2 = Bundle(bundle_id="bs2")
    node_spe.accept_custody(b_spe1, 0.0)
    node_spe.accept_custody(b_spe2, 0.0)
    b_spe1.custody_state = CustodyState.O
    b_spe2.custody_state = CustodyState.P
    node_spe.pre_flush_to_holding()
    check("Pre-flush O→H", b_spe1.custody_state == CustodyState.H)
    check("Pre-flush P→H", b_spe2.custody_state == CustodyState.H)

    # ── DTN Network Integration ──────────────────────────────────────────
    print("\n═══ DTN Network Integration ═══")

    net2 = DTNNetwork()
    for nid in ["surface", "polar-0", "elfo", "dsn_earth"]:
        net2.add_node(CustodyNode(nid))

    b_route = net2.create_bundle("surface", "dsn_earth", priority=PRIORITY_EMERGENCY)
    ok = net2.route_along_path(
        b_route,
        ["surface", "polar-0", "elfo", "dsn_earth"],
        link_types=["rf", "rf", "optical"],
        t_s=0.0,
        data_rate_kbps=1000.0,
    )
    check("3-hop route delivers", ok and b_route.delivered)
    check("3 hops recorded", b_route.hop_count == 3)
    check("Final state R", b_route.custody_state == CustodyState.R)
    check("Latency > 0", b_route.total_latency_s() > 0)

    s = net2.summary()
    check("Summary counts correct", s["delivered"] == 1 and s["total_bundles"] == 1)

    # ── B2: RW-CGR Stochastic Routing ────────────────────────────────────
    print("\n═══ B2: RW-CGR Stochastic Routing ═══")

    router = RW_CGR(seed=42)
    router.add_contact(
        Contact(
            "lunar_surface",
            "polar-0",
            start_s=0,
            duration_s=600,
            p_success=0.95,
            data_rate_kbps=256,
            latency_s=0.01,
        )
    )
    router.add_contact(
        Contact(
            "polar-0",
            "EM-L2-HALO",
            start_s=0,
            duration_s=3600,
            p_success=0.98,
            data_rate_kbps=1000,
            latency_s=0.5,
        )
    )
    router.add_contact(
        Contact(
            "EM-L2-HALO",
            "earth_dsn",
            start_s=0,
            duration_s=86400,
            p_success=0.99,
            data_rate_kbps=10000,
            latency_s=1.3,
        )
    )
    router.add_contact(
        Contact(
            "polar-0",
            "earth_dsn",
            start_s=0,
            duration_s=300,
            p_success=0.85,
            data_rate_kbps=500,
            latency_s=1.3,
        )
    )

    b_rw = Bundle(bundle_id="brw", priority=PRIORITY_EMERGENCY, size_bytes=256)
    routes = router.find_routes("lunar_surface", "earth_dsn", b_rw, t_s=0, n_samples=500)
    check("Found ≥1 route", len(routes) >= 1)

    halo_routes = [r for r in routes if "EM-L2-HALO" in r]
    check("Halo in top routes (N8 integration)", len(halo_routes) >= 1, f"got routes: {routes}")

    best = router.route_bundle(b_rw, "lunar_surface", "earth_dsn", t_s=0)
    check("route_bundle returns path", best is not None and best[0] == "lunar_surface")

    for route in routes:
        check(f"No loops in {route}", len(route) == len(set(route)))

    router_empty = RW_CGR()
    result = router_empty.route_bundle(Bundle(bundle_id="x"), "nowhere", "earth", t_s=0)
    check("No route → None", result is None)

    router.clear_cache()
    check("Cache cleared", len(router.route_cache) == 0)

    # ── N8: Lunar Config ─────────────────────────────────────────────────
    print("\n═══ N8: Lunar Constellation Config ═══")

    check("Moon radius", LUNAR_BODY.radius_km == 1737.4)
    check("Moon μ", LUNAR_BODY.mu_km3s2 == 4902.8)

    cfg = get_lunar_constellation()
    total_nodes = len(cfg.satellites) + len(cfg.relay_hubs) + len(cfg.halo_satellites)
    check("10 total nodes", total_nodes == 10)
    check("8 polar sats", len(cfg.satellites) == 8)
    check("1 ELFO hub", len(cfg.relay_hubs) == 1)
    check("1 halo sat", len(cfg.halo_satellites) == 1)

    elfo = cfg.relay_hubs[0]
    check("ELFO e=0.58", abs(elfo.e - 0.58) < 1e-6)
    check("ELFO type", elfo.sat_type == "elfo")

    halo = cfg.halo_satellites[0]
    check("Halo z=8000 km", halo.z_amplitude_km == 8000.0)
    check("Halo period 14.5d", halo.period_days == 14.5)

    for i, sat in enumerate(cfg.satellites):
        check(
            f"Polar-{i} RAAN={i * 45}°",
            abs(sat.raan_deg - i * 45) < 1e-6 and abs(sat.i_deg - 89.5) < 1e-6,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'═' * 60}")
    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("\n  All validations passed. Ready for CI integration.")
        sys.exit(0)


if __name__ == "__main__":
    main()
