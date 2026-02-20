#!/usr/bin/env python3
"""tin.cli.demo — Unified TIN v0.4.0 demonstration runner

Usage:
    python -m tin.cli.demo --body lunar
    python -m tin.cli.demo --body lunar --days 28 --emergency
    python -m tin.cli.demo --body mars
    python -m tin.cli.demo --body mars --conjunction
    python -m tin.cli.demo --body mars --trade
    python -m tin.cli.demo --body both
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Ensure package importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tin.core.dtn import PRIORITY_EMERGENCY, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_BULK


def run_lunar(args):
    """Run full lunar TIN demonstration."""
    from tin.lunar import (
        LunarConstellation, make_lunar_dtn_network,
        LUNAR_CUSTODY_CHAIN, LUNAR_LINK_TYPES, LUNAR_DATA_RATES, LUNAR_DELAYS,
    )

    print(f"\n{'='*65}")
    print(f"  TIN LUNAR v0.4.0 — {args.days}-day simulation")
    print(f"{'='*65}")

    # Build constellation
    const = LunarConstellation(n_sats=8, alt_km=400, include_elfo=True)

    # Coverage at multiple latitudes
    print(f"\n  Coverage Analysis (8x400km + ELFO):")
    for lat in [-89.5, -75.0, -60.0, 0.0]:
        cov = const.compute_coverage(lat_deg=lat, sim_days=args.days)
        layers = cov["layer_breakdown"]
        print(f"    lat={lat:>6.1f}°: {cov['coverage_pct']:>5.1f}% cov, "
              f"{cov['worst_gap_min']:>5.1f}min gap  "
              f"[polar={layers['polar']}% +elfo={layers['elfo']}%]")

    # Trade study sweep
    if args.trade:
        print(f"\n  N-Sat Trade Study:")
        print(f"    {'Config':<18} {'Cov%':>6} {'Gap':>8} {'Cost$M':>8}")
        print(f"    {'-'*42}")
        for n in [4, 6, 8, 10, 12]:
            for elfo in [False, True]:
                c = LunarConstellation(n_sats=n, alt_km=400, include_elfo=elfo)
                r = c.compute_coverage(lat_deg=-89.5, sim_days=28)
                cost = n * 24 * 1.0  # $M
                tag = f"{n}x400km{'+ELFO' if elfo else ''}"
                print(f"    {tag:<18} {r['coverage_pct']:>5.1f}% "
                      f"{r['worst_gap_min']:>7.1f}m {cost:>7.1f}")

    # BPv7 Emergency routing
    if args.emergency:
        print(f"\n  BPv7 Emergency Custody Chain:")
        dtn = make_lunar_dtn_network()

        # Create emergency bundle at t=300s (EVA astronaut)
        bundle = dtn.create_bundle(
            source="Rover-Alpha", destination="dsn_earth",
            priority=PRIORITY_EMERGENCY, t_s=300.0,
            payload_type="emergency_medical", size_bytes=512,
        )

        # Route through canonical chain
        success = dtn.route_along_path(
            bundle, LUNAR_CUSTODY_CHAIN, LUNAR_LINK_TYPES, t_s=300.0,
            data_rates=LUNAR_DATA_RATES, delays=LUNAR_DELAYS,
        )

        summary = dtn.summary()
        print(f"    Bundles: {summary['total_bundles']} | "
              f"Delivered: {summary['delivered']} | "
              f"Transfers: {summary['total_transfers']}")

        if summary["latency_mean_s"] is not None:
            print(f"    Latency: {summary['latency_mean_s']:.2f}s "
                  f"({summary['latency_mean_min']:.2f} min)")

        # Print chain
        for b in summary["bundles"]:
            chain = " → ".join(c["node"] for c in b["custody_chain"])
            lat_s = f"{b['latency_s']:.3f}s" if b["latency_s"] else "?"
            print(f"    [{b['priority']}] {b['bundle_id']}: {lat_s}")
            print(f"      {chain}")
            for hop in b["hop_log"]:
                print(f"        {hop['from']:>22} → {hop['to']:<22} "
                      f"{hop['link_type']} ({hop['arrive_s'] - hop['depart_s']:.3f}s)")

    # Final report
    baseline = const.compute_coverage(lat_deg=-89.5, sim_days=args.days)
    print(f"\n  ✅ TIN LUNAR v0.4.0 — {args.days}-day run complete")
    print(f"     Coverage: {baseline['coverage_pct']}% (Shackleton)")
    print(f"     Worst-case latency: {baseline['worst_gap_min']} min")
    return baseline


def run_mars(args):
    """Run full Mars TIN demonstration."""
    from tin.mars import (
        MarsConstellation, make_mars_dtn_network, simulate_conjunction,
        MARS_CUSTODY_CHAIN, MARS_LINK_TYPES, MARS_DATA_RATES, MARS_DELAYS,
    )

    print(f"\n{'='*65}")
    print(f"  TIN MARS v0.4.0 — {args.days}-sol simulation")
    print(f"{'='*65}")

    const = MarsConstellation(n_polar=6, polar_alt_km=350, n_areo=2,
                               use_aerostats=True)

    # Coverage at multiple latitudes
    print(f"\n  Coverage Analysis (6pol+2areo+24aero + Phobos + Deimos):")
    for lat in [0.0, -30.0, -60.0, -89.5]:
        cov = const.compute_coverage(lat_deg=lat, sim_days=args.days)
        layers = cov["layer_breakdown"]
        print(f"    lat={lat:>6.1f}°: {cov['coverage_pct']:>5.1f}% cov, "
              f"{cov['worst_gap_min']:>5.1f}min gap  "
              f"[aero={layers['aerostat']}% polar={layers['polar']}% "
              f"areo={layers['areo']}%]")

    # Trade study
    if args.trade:
        print(f"\n  Mars Trade Study:")
        print(f"    {'Config':<26} {'Lat':>5} {'Cov%':>6} {'Gap':>7} {'Cost$M':>8}")
        print(f"    {'-'*55}")
        configs = [
            (4, 2, False, "4pol+2areo"),
            (6, 2, False, "6pol+2areo"),
            (6, 2, True,  "6pol+2areo+24aero"),
            (8, 2, True,  "8pol+2areo+24aero"),
            (6, 4, True,  "6pol+4areo+24aero"),
        ]
        for n_p, n_a, aero, label in configs:
            c = MarsConstellation(n_p, 350, n_a, use_aerostats=aero)
            for lat in [0.0, -89.5]:
                r = c.compute_coverage(lat_deg=lat)
                cost = (n_p * 24 + n_a * 200) * 2.0 + 130 + (84 if aero else 0)
                print(f"    {label:<26} {lat:>5.1f} {r['coverage_pct']:>5.1f}% "
                      f"{r['worst_gap_min']:>6.1f}m {cost:>7.0f}")

    # Conjunction
    if args.conjunction:
        print(f"\n  Solar Conjunction Simulation:")
        conj = simulate_conjunction(const)
        print(f"    Blackout: {conj['blackout_days']} days")
        print(f"    Local Mars coverage: {conj['local_coverage_pct']}% "
              f"(gap {conj['local_worst_gap_min']} min)")
        print(f"    Emergency chain:")
        for node in conj["custody_chain"]:
            print(f"      → {node}")
        print(f"    Total latency: {conj['total_latency_sols']} sols "
              f"({conj['total_latency_days']} days)")
        print(f"    {conj['clinical_outcome']}")

    # BPv7 routing
    if args.emergency:
        print(f"\n  BPv7 Emergency Custody Chain (nominal, no conjunction):")
        dtn = make_mars_dtn_network()
        bundle = dtn.create_bundle(
            source="Mars-Habitat", destination="dsn_earth",
            priority=PRIORITY_EMERGENCY, t_s=300.0,
            payload_type="emergency_medical", size_bytes=512,
        )
        success = dtn.route_along_path(
            bundle, MARS_CUSTODY_CHAIN, MARS_LINK_TYPES, t_s=300.0,
            data_rates=MARS_DATA_RATES, delays=MARS_DELAYS,
        )
        summary = dtn.summary()
        for b in summary["bundles"]:
            chain = " → ".join(c["node"] for c in b["custody_chain"])
            lat_s = f"{b['latency_s']:.1f}s" if b["latency_s"] else "?"
            print(f"    [{b['priority']}] {lat_s}: {chain}")

    baseline = const.compute_coverage(lat_deg=0.0, sim_days=args.days)
    print(f"\n  ✅ TIN MARS v0.4.0 — {args.days}-sol run complete")
    print(f"     Coverage: {baseline['coverage_pct']}% (equatorial)")
    print(f"     Worst-case gap: {baseline['worst_gap_min']} min")
    return baseline


def main():
    parser = argparse.ArgumentParser(description="TIN v0.4.0 Unified Demo")
    parser.add_argument("--body", choices=["lunar", "mars", "both"], default="lunar")
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--emergency", action="store_true", default=True,
                        help="Run BPv7 emergency routing")
    parser.add_argument("--no-emergency", dest="emergency", action="store_false")
    parser.add_argument("--trade", action="store_true", default=False,
                        help="Run N-sat trade study")
    parser.add_argument("--conjunction", action="store_true", default=False,
                        help="Run Mars conjunction sim (Mars only)")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Enable all analyses")
    args = parser.parse_args()

    if args.all:
        args.emergency = True
        args.trade = True
        args.conjunction = True

    print(f"\n  TIN v0.4.0 — Tolerant Interplanetary Network")
    print(f"  Planet-agnostic DTN architecture with AI-assisted routing")
    print(f"  MIT License | github.com/toxic2040/TIN-v0.3.1")

    if args.body in ("lunar", "both"):
        run_lunar(args)
    if args.body in ("mars", "both"):
        run_mars(args)

    if args.body == "both":
        print(f"\n{'='*65}")
        print(f"  Dual-body run complete — lunar + Mars from shared core.")
        print(f"  Same base classes, same BPv7 engine, same CLI.")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
