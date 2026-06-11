#!/usr/bin/env python3
"""tin_surface_relays.py — Surface relay node simulation for TIN v0.4.0

Models ground-based relay nodes at the lunar South Pole (Shackleton crater region):
  - Shackleton Base: Fixed habitat/lander on crater rim (comms tower)
  - Rover 1 (science): Traverses near Shackleton rim, periodic stops
  - Rover 2 (logistics): Shorter range, shuttles between base and landing site

Surface relays provide store-and-forward capability during satellite blackouts:
  - Rover catches emergency message when no sat is overhead
  - Stores in buffer (DTN bundle custody)
  - Forwards to next available sat pass or to base station line-of-sight

Integrates with tin_coverage_sim.py for orbital gap data.

Usage:
    python scripts/tin_surface_relays.py
    python scripts/tin_surface_relays.py --scenario medical_emergency
    python scripts/tin_surface_relays.py --scenario rover_abort

Dependencies: numpy, matplotlib, tin_coverage_sim.py
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure co-located import works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tin_coverage_sim import compute_coverage
except ImportError as e:
    raise ImportError("Cannot import compute_coverage from tin_coverage_sim.py") from e


# =========================================================================
# Lunar South Pole surface geometry
# =========================================================================

# Shackleton crater rim coordinates (approximate, selenographic)
# Crater center: ~89.54°S, 0°E; rim elevation ~4.2 km above crater floor
SHACKLETON_LAT = -89.54
SHACKLETON_LON = 0.0
SHACKLETON_RIM_ELEV_M = 4200  # meters above crater floor

# Lunar constants
LUNAR_RADIUS_KM = 1737.4
SPEED_OF_LIGHT_KM_S = 299792.458

# Surface relay parameters
BASE_STATION_RANGE_KM = 15.0  # comms tower range (line-of-sight on rim)
ROVER_RELAY_RANGE_KM = 8.0  # rover-to-rover or rover-to-base range
ROVER_SPEED_KM_H = 3.5  # nominal traverse speed
BUNDLE_CUSTODY_TIMEOUT_MIN = 30  # max time a relay holds a bundle before escalating
MSG_PRIORITY_LEVELS = {
    "emergency_medical": 1,  # highest — immediate relay, pre-empt all
    "rover_abort": 1,  # highest — safety critical
    "habitat_alarm": 2,  # high — pressure, thermal, power
    "science_priority": 3,  # medium — time-sensitive observation
    "routine_telemetry": 4,  # low — periodic status
}


# =========================================================================
# Surface node definitions
# =========================================================================


class SurfaceNode:
    """A fixed or mobile relay node on the lunar surface."""

    def __init__(self, name, node_type, lat, lon, range_km, mobile=False):
        self.name = name
        self.node_type = node_type  # "base", "rover", "astronaut"
        self.lat = lat
        self.lon = lon
        self.range_km = range_km
        self.mobile = mobile
        self.bundle_buffer = []  # stored bundles awaiting relay
        self.relay_log = []  # history of relayed messages
        self.position_history = []  # for mobile nodes: [(t_min, lat, lon), ...]

    def distance_to(self, other_lat, other_lon):
        """Great-circle distance on lunar surface (km)."""
        phi1, phi2 = np.radians(self.lat), np.radians(other_lat)
        dphi = np.radians(other_lat - self.lat)
        dlam = np.radians(other_lon - self.lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return 2 * LUNAR_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def can_reach(self, other_lat, other_lon):
        """Check if another point is within comms range."""
        return self.distance_to(other_lat, other_lon) <= self.range_km

    def store_bundle(self, bundle):
        """Accept a DTN bundle into custody."""
        bundle["custody_node"] = self.name
        bundle["custody_time_min"] = bundle.get("custody_time_min", 0)
        self.bundle_buffer.append(bundle)

    def forward_bundles(self, t_min, target_name="satellite"):
        """Forward all stored bundles to available target. Returns forwarded bundles."""
        forwarded = []
        remaining = []
        for b in self.bundle_buffer:
            b["hops"] = b.get("hops", 0) + 1
            b["forwarded_by"] = self.name
            b["forwarded_to"] = target_name
            b["forward_time_min"] = t_min
            forwarded.append(b)
            self.relay_log.append(
                {
                    "time_min": t_min,
                    "bundle_id": b["id"],
                    "priority": b["priority"],
                    "from": self.name,
                    "to": target_name,
                    "custody_held_min": t_min - b.get("origin_time_min", t_min),
                }
            )
        self.bundle_buffer = remaining  # clear buffer
        return forwarded

    def update_position(self, t_min, lat, lon):
        """Update position for mobile nodes."""
        self.lat = lat
        self.lon = lon
        self.position_history.append((t_min, lat, lon))


# =========================================================================
# Rover traverse path generation
# =========================================================================


def generate_rover_path(
    name, center_lat, center_lon, radius_km, duration_hr, speed_km_h=ROVER_SPEED_KM_H, dt_min=5.0
):
    """Generate a circular-ish traverse path around a center point.

    Returns list of (t_min, lat, lon) waypoints.
    """
    total_min = duration_hr * 60
    t_steps = np.arange(0, total_min, dt_min)

    # Convert radius from km to degrees (approximate at lunar pole)
    # At 89.5°S, 1° longitude ~ 1737.4 * cos(89.5°) * pi/180 ~ 0.26 km
    # 1° latitude ~ 1737.4 * pi/180 ~ 30.3 km
    r_lat = radius_km / 30.3
    r_lon = radius_km / 0.26  # stretched in longitude near pole

    # Traverse: roughly elliptical path with stops
    path = []
    circumference_km = 2 * np.pi * radius_km
    period_min = circumference_km / speed_km_h * 60  # time for one loop

    for t in t_steps:
        # Phase angle around the path
        theta = 2 * np.pi * (t / period_min)

        # Add some randomness for realism (terrain avoidance)
        jitter_lat = np.random.normal(0, r_lat * 0.05)
        jitter_lon = np.random.normal(0, r_lon * 0.05)

        lat = center_lat + r_lat * np.sin(theta) + jitter_lat
        lon = center_lon + r_lon * np.cos(theta) + jitter_lon

        # Periodic stops (every ~45 min, stop for 10 min)
        # During stops, rover acts as a fixed relay
        path.append((t, lat, lon))

    return path


# =========================================================================
# Emergency message scenarios
# =========================================================================


def create_emergency_bundle(scenario, origin_time_min, origin_node):
    """Create a DTN bundle for an emergency scenario."""
    bundle = {
        "id": f"{scenario}_{origin_time_min:.0f}",
        "scenario": scenario,
        "priority": MSG_PRIORITY_LEVELS.get(scenario, 3),
        "origin_time_min": origin_time_min,
        "origin_node": origin_node,
        "custody_node": origin_node,
        "custody_time_min": 0,
        "hops": 0,
        "delivered": False,
        "delivery_time_min": None,
        "total_latency_min": None,
    }
    return bundle


# =========================================================================
# Core simulation
# =========================================================================


def simulate_surface_relay(
    n_sats=8,
    alt_km=400,
    include_elfo=True,
    sim_hours=24,
    dt_min=5.0,
    emergency_scenario="medical_emergency",
    emergency_times_hr=None,
    seed=42,
):
    """
    Run surface relay simulation.

    Simulates emergency messages originating during satellite coverage gaps,
    and tracks how surface relay nodes (base + rovers) reduce effective latency.

    Returns: dict with simulation results and metrics
    """
    np.random.seed(seed)

    # --- Get orbital coverage data ---
    cov_pct, worst_gap_min, avg_gap_min = compute_coverage(
        n_sats=n_sats, alt_km=alt_km, include_elfo=include_elfo
    )

    # --- Create surface nodes ---
    base = SurfaceNode(
        "Shackleton_Base", "base", SHACKLETON_LAT, SHACKLETON_LON, BASE_STATION_RANGE_KM
    )

    # Science rover: wider traverse (~5 km from base)
    rover1_path = generate_rover_path(
        "Rover_Science",
        SHACKLETON_LAT,
        SHACKLETON_LON + 0.5,
        radius_km=5.0,
        duration_hr=sim_hours,
        dt_min=dt_min,
    )
    rover1 = SurfaceNode(
        "Rover_Science",
        "rover",
        rover1_path[0][1],
        rover1_path[0][2],
        ROVER_RELAY_RANGE_KM,
        mobile=True,
    )

    # Logistics rover: closer to base (~2 km radius)
    rover2_path = generate_rover_path(
        "Rover_Logistics",
        SHACKLETON_LAT + 0.01,
        SHACKLETON_LON - 0.3,
        radius_km=2.0,
        duration_hr=sim_hours,
        dt_min=dt_min,
    )
    rover2 = SurfaceNode(
        "Rover_Logistics",
        "rover",
        rover2_path[0][1],
        rover2_path[0][2],
        ROVER_RELAY_RANGE_KM,
        mobile=True,
    )

    nodes = [base, rover1, rover2]

    # --- Generate satellite visibility timeline ---
    # Simplified: sat is visible (coverage_pct)% of the time
    # Gap pattern: periodic blackouts based on worst_gap and coverage
    total_min = sim_hours * 60
    t_steps = np.arange(0, total_min, dt_min)

    # Model sat visibility as periodic: visible for cov_window, then gap
    gap_fraction = max(0.001, (100 - cov_pct) / 100)
    cov_fraction = 1 - gap_fraction

    # Orbital period proxy for gap timing
    if include_elfo:
        gap_period_min = 12 * 60  # ELFO orbital period
    else:
        polar_period_min = 120 * ((LUNAR_RADIUS_KM + alt_km) / (LUNAR_RADIUS_KM + 400)) ** 1.5
        gap_period_min = polar_period_min

    gap_duration_min = gap_period_min * gap_fraction
    cov_duration_min = gap_period_min * cov_fraction

    # Build visibility timeline
    sat_visible = np.zeros(len(t_steps), dtype=bool)
    t_cursor = 0
    visible_state = True
    while t_cursor < total_min:
        if visible_state:
            window = cov_duration_min
        else:
            window = gap_duration_min
        mask = (t_steps >= t_cursor) & (t_steps < t_cursor + window)
        sat_visible[mask] = visible_state
        t_cursor += window
        visible_state = not visible_state

    # --- Place emergency events during gaps ---
    if emergency_times_hr is None:
        # Auto-place emergencies: one during each gap
        gap_starts = []
        in_gap = False
        for i, vis in enumerate(sat_visible):
            if not vis and not in_gap:
                gap_starts.append(t_steps[i])
                in_gap = True
            elif vis:
                in_gap = False
        # Pick up to 5 gaps
        emergency_times_min = [g + gap_duration_min * 0.3 for g in gap_starts[:5]]
    else:
        emergency_times_min = [h * 60 for h in emergency_times_hr]

    # --- Simulate message propagation ---
    bundles = []
    for em_time in emergency_times_min:
        bundle = create_emergency_bundle(emergency_scenario, em_time, "EVA_Astronaut")
        bundles.append(bundle)

    results_no_relay = []  # latency without surface relay
    results_with_relay = []  # latency with surface relay

    for bundle in bundles:
        t_origin = bundle["origin_time_min"]

        # --- Without surface relay: wait for next sat pass ---
        t_idx = np.searchsorted(t_steps, t_origin)
        wait_no_relay = 0
        for i in range(t_idx, len(sat_visible)):
            if sat_visible[i]:
                wait_no_relay = t_steps[i] - t_origin
                break
        else:
            wait_no_relay = total_min - t_origin  # never reached sat

        results_no_relay.append(
            {
                "bundle_id": bundle["id"],
                "origin_time_min": t_origin,
                "latency_min": round(float(wait_no_relay), 1),
            }
        )

        # --- With surface relay: check if any node can receive and store-forward ---
        # Update rover positions to emergency time
        r1_idx = min(int(t_origin / dt_min), len(rover1_path) - 1)
        r2_idx = min(int(t_origin / dt_min), len(rover2_path) - 1)
        rover1.update_position(t_origin, rover1_path[r1_idx][1], rover1_path[r1_idx][2])
        rover2.update_position(t_origin, rover2_path[r2_idx][1], rover2_path[r2_idx][2])

        # Check which nodes astronaut can reach
        # Astronaut is at a random position within 3 km of base (EVA range)
        astro_lat = SHACKLETON_LAT + np.random.normal(0, 0.03)
        astro_lon = SHACKLETON_LON + np.random.normal(0, 0.5)

        reachable = []
        for node in nodes:
            dist = node.distance_to(astro_lat, astro_lon)
            if dist <= node.range_km:
                reachable.append((node, dist))

        if reachable:
            # Astronaut sends to nearest reachable node
            reachable.sort(key=lambda x: x[1])
            relay_node = reachable[0][0]
            relay_dist = reachable[0][1]

            # Light-speed delay to relay (negligible but modeled)
            relay_delay_min = (relay_dist / SPEED_OF_LIGHT_KM_S) / 60

            # Relay stores bundle, waits for sat pass
            relay_wait = 0
            for i in range(t_idx, len(sat_visible)):
                if sat_visible[i]:
                    relay_wait = t_steps[i] - t_origin
                    break
            else:
                relay_wait = total_min - t_origin

            # If relay node is base station AND base has persistent uplink capability
            # (e.g., directional antenna), latency can be further reduced
            if relay_node.node_type == "base":
                # Base station can buffer and transmit immediately on next window
                # Assume base has slightly better antenna -> can catch marginal passes
                relay_wait = max(relay_wait * 0.7, 1.0)  # 30% improvement

            total_relay_latency = relay_delay_min + relay_wait
            hops = 1 if relay_node.node_type == "base" else 2  # rover->base->sat = 2 hops
        else:
            # No relay reachable: same as no-relay case
            total_relay_latency = wait_no_relay
            hops = 0
            relay_node = None

        results_with_relay.append(
            {
                "bundle_id": bundle["id"],
                "origin_time_min": round(float(t_origin), 1),
                "relay_node": relay_node.name if relay_node else "none",
                "hops": hops,
                "latency_min": round(float(total_relay_latency), 1),
                "latency_reduction_min": round(float(wait_no_relay - total_relay_latency), 1),
            }
        )

    # --- Compile summary metrics ---
    lat_no = [r["latency_min"] for r in results_no_relay]
    lat_yes = [r["latency_min"] for r in results_with_relay]
    reductions = [r["latency_reduction_min"] for r in results_with_relay]

    summary = {
        "sim_config": {
            "n_sats": n_sats,
            "alt_km": alt_km,
            "include_elfo": include_elfo,
            "sim_hours": sim_hours,
            "scenario": emergency_scenario,
            "n_emergencies": len(bundles),
            "orbital_coverage_pct": round(float(cov_pct), 2),
            "orbital_worst_gap_min": round(float(worst_gap_min), 2),
        },
        "surface_nodes": [
            {"name": n.name, "type": n.node_type, "range_km": n.range_km} for n in nodes
        ],
        "without_relay": {
            "mean_latency_min": round(float(np.mean(lat_no)), 1) if lat_no else 0,
            "worst_latency_min": round(float(np.max(lat_no)), 1) if lat_no else 0,
            "events": results_no_relay,
        },
        "with_relay": {
            "mean_latency_min": round(float(np.mean(lat_yes)), 1) if lat_yes else 0,
            "worst_latency_min": round(float(np.max(lat_yes)), 1) if lat_yes else 0,
            "mean_reduction_min": round(float(np.mean(reductions)), 1) if reductions else 0,
            "events": results_with_relay,
        },
        "relay_effectiveness": {
            "mean_latency_improvement_pct": round(
                100 * (1 - np.mean(lat_yes) / max(np.mean(lat_no), 0.01)), 1
            )
            if lat_no and lat_yes
            else 0,
            "worst_case_improvement_pct": round(
                100 * (1 - np.max(lat_yes) / max(np.max(lat_no), 0.01)), 1
            )
            if lat_no and lat_yes
            else 0,
        },
    }

    return summary, t_steps, sat_visible, nodes, rover1_path, rover2_path


# =========================================================================
# Visualization
# =========================================================================


def plot_surface_relay_results(
    summary, t_steps, sat_visible, nodes, rover1_path, rover2_path, outdir
):
    """Generate surface relay visualization: 2-panel figure."""
    os.makedirs(outdir, exist_ok=True)

    COV_COLOR = "#2E75B6"
    GAP_COLOR = "#E87722"
    BASE_COLOR = "#1B2A4A"
    R1_COLOR = "#2EA043"
    R2_COLOR = "#9B59B6"
    BG_COLOR = "#F7F9FC"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1.3]})
    fig.patch.set_facecolor(BG_COLOR)

    # --- Panel 1: Satellite visibility timeline + emergency events ---
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    t_hr = t_steps / 60

    # Color-code visibility
    for i in range(len(t_steps) - 1):
        color = COV_COLOR if sat_visible[i] else GAP_COLOR
        alpha = 0.3 if sat_visible[i] else 0.5
        ax1.axvspan(t_hr[i], t_hr[i + 1], color=color, alpha=alpha)

    # Mark emergency events
    no_relay_events = summary["without_relay"]["events"]
    relay_events = summary["with_relay"]["events"]
    for nr, wr in zip(no_relay_events, relay_events):
        t_em = nr["origin_time_min"] / 60
        ax1.axvline(t_em, color="red", linewidth=2, linestyle="-", alpha=0.8)
        ax1.annotate(
            f"No relay: {nr['latency_min']:.0f}m\nWith relay: {wr['latency_min']:.0f}m",
            (t_em, 0.5),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=7,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9),
        )

    ax1.set_xlim(0, t_hr[-1])
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_title("Satellite Visibility & Emergency Events", fontsize=12, fontweight="bold")

    # Legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COV_COLOR, alpha=0.3, label="Sat visible"),
        Patch(facecolor=GAP_COLOR, alpha=0.5, label="Sat gap (blackout)"),
        Patch(facecolor="red", alpha=0.8, label="Emergency event"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # --- Panel 2: Latency comparison bar chart ---
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)

    n_events = len(no_relay_events)
    if n_events > 0:
        x = np.arange(n_events)
        width = 0.35

        lat_no = [e["latency_min"] for e in no_relay_events]
        lat_yes = [e["latency_min"] for e in relay_events]
        relay_names = [e.get("relay_node", "none") for e in relay_events]

        bars1 = ax2.bar(
            x - width / 2, lat_no, width, color=GAP_COLOR, alpha=0.8, label="Without surface relay"
        )
        bars2 = ax2.bar(
            x + width / 2, lat_yes, width, color=R1_COLOR, alpha=0.8, label="With surface relay"
        )

        # Value labels
        for bar, val in zip(bars1, lat_no):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}m",
                ha="center",
                va="bottom",
                fontsize=8,
                color=GAP_COLOR,
                fontweight="bold",
            )
        for bar, val, rn in zip(bars2, lat_yes, relay_names):
            label = f"{val:.0f}m\nvia {rn}" if rn != "none" else f"{val:.0f}m"
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color=R1_COLOR,
                fontweight="bold",
            )

        ax2.set_xlabel("Emergency Event #", fontsize=11)
        ax2.set_ylabel("Latency to Relay (minutes)", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Event {i + 1}" for i in x], fontsize=9)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(axis="y", alpha=0.3)

    # Overall title
    cfg = summary["sim_config"]
    elfo_str = "+ELFO" if cfg["include_elfo"] else "pure"
    fig.suptitle(
        f"TIN v0.4.0 Surface Relay Analysis — {cfg['n_sats']}x{cfg['alt_km']}km {elfo_str}\n"
        f"Scenario: {cfg['scenario']} | "
        f"Mean latency reduction: {summary['with_relay']['mean_reduction_min']:.1f} min "
        f"({summary['relay_effectiveness']['mean_latency_improvement_pct']:.0f}% improvement)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    fname = f"tin_surface_relay_{datetime.now():%Y%m%d_%H%M}.png"
    plot_path = os.path.join(outdir, fname)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Plot saved: {plot_path}")
    return plot_path


# =========================================================================
# Main entry point
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="TIN v0.4.0 Surface Relay Node Simulation")
    parser.add_argument("--n_sats", type=int, default=8, help="Number of polar sats")
    parser.add_argument("--alt_km", type=int, default=400, help="Orbit altitude (km)")
    parser.add_argument("--include_elfo", action="store_true", default=True)
    parser.add_argument("--no_elfo", action="store_true")
    parser.add_argument("--sim_hours", type=int, default=24, help="Simulation duration (hours)")
    parser.add_argument(
        "--scenario",
        type=str,
        default="emergency_medical",
        choices=list(MSG_PRIORITY_LEVELS.keys()),
        help="Emergency scenario type",
    )
    args = parser.parse_args()

    include_elfo = not args.no_elfo

    print(f"\n{'=' * 60}")
    print("  TIN v0.4.0 Surface Relay Simulation")
    print(f"  {args.n_sats}x{args.alt_km}km {'+ ELFO' if include_elfo else 'pure'}")
    print(f"  Scenario: {args.scenario} | Duration: {args.sim_hours} hr")
    print(f"{'=' * 60}\n")

    # Run simulation
    summary, t_steps, sat_visible, nodes, r1_path, r2_path = simulate_surface_relay(
        n_sats=args.n_sats,
        alt_km=args.alt_km,
        include_elfo=include_elfo,
        sim_hours=args.sim_hours,
        emergency_scenario=args.scenario,
    )

    # Print results
    cfg = summary["sim_config"]
    print(f"  Orbital coverage: {cfg['orbital_coverage_pct']}%")
    print(f"  Orbital worst gap: {cfg['orbital_worst_gap_min']} min")
    print(f"  Emergency events simulated: {cfg['n_emergencies']}")
    print()

    print("  Surface Nodes:")
    for sn in summary["surface_nodes"]:
        print(f"    {sn['name']:<22} type={sn['type']:<8} range={sn['range_km']:.1f} km")
    print()

    wr = summary["without_relay"]
    yr = summary["with_relay"]
    eff = summary["relay_effectiveness"]
    print("  WITHOUT surface relay:")
    print(f"    Mean latency:  {wr['mean_latency_min']:.1f} min")
    print(f"    Worst latency: {wr['worst_latency_min']:.1f} min")
    print()
    print("  WITH surface relay:")
    print(f"    Mean latency:  {yr['mean_latency_min']:.1f} min")
    print(f"    Worst latency: {yr['worst_latency_min']:.1f} min")
    print(f"    Mean reduction: {yr['mean_reduction_min']:.1f} min")
    print()
    print("  Relay effectiveness:")
    print(f"    Mean improvement:  {eff['mean_latency_improvement_pct']:.0f}%")
    print(f"    Worst-case improvement: {eff['worst_case_improvement_pct']:.0f}%")

    # Save outputs
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(repo_root, "data")
    results_dir = os.path.join(repo_root, "results")

    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, f"surface_relay_{datetime.now():%Y%m%d_%H%M}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON saved: {json_path}")

    plot_surface_relay_results(summary, t_steps, sat_visible, nodes, r1_path, r2_path, results_dir)

    print("\n  Done.")


if __name__ == "__main__":
    main()
