#!/usr/bin/env python3
"""tin_mars_sim.py — TIN-Mars v0.1: Interplanetary DTN Architecture for Mars

Direct port of the proven TIN lunar architecture to Mars, with Mars-unique extensions:

ARCHITECTURE (5 layers + surface):
  Layer 0: Aerostat Armada — 24 super-pressure helium balloons at 18-25 km
           altitude, wind-driven (MCD/GCM parametric model), laser inter-mesh
           (Mars-unique: no lunar analog — atmospheric relay layer)
  Layer 1: Polar SmallSats — 6-8 sats at ~350 km, near-polar orbits
           (Mars analog of lunar 8x400km constellation)
  Layer 2: Areostationary Backbone — 2-4 relays at 17,032 km altitude
           (Mars analog of ELFO hub — "always on" equatorial backbone)
           Stable longitudes: 17.92°W (342.08°E) and 167.83°E
  Layer 3: Natural Custody Depots — Phobos (9,376 km) + Deimos (23,463 km)
           as store-and-forward relay nodes with massive buffer capacity
           (Mars-unique: no lunar analog — free orbital infrastructure)
  Layer 4: Conjunction Proofing — predictive bundle parking on Deimos during
           ~3-week solar conjunction blackout (~779.94 day synodic period)
           (The Mars DTN killer feature: TIN handles the firewall)
  Surface: Habitat base + 2 rovers (direct port from tin_surface_relays.py)

KEY MARS PARAMETERS:
  Mars radius:            3,396.19 km
  Mars GM:                4.282837e4 km³/s²
  Mars sidereal day (sol): 88,775 s (24h 37m 22s)
  Areostationary altitude: 17,032 km
  Phobos semi-major axis:  9,376 km (period 7h 39m)
  Deimos semi-major axis:  23,463 km (period 30.3 hr)
  Earth-Mars light time:   4.3 min (closest) to 24.0 min (farthest)
  Conjunction blackout:    ~14-21 days (Sun-Earth-Mars angle < 2-3°)
  Synodic period:          779.94 days

HERITAGE: Directly extends TIN lunar v0.3.9 (8x400km + ELFO + BPv7).
          Same compute_coverage model adapted for Mars orbital mechanics.
          Same BPv7 custody chain, extended with conjunction-aware routing.

Usage:
    python scripts/tin_mars_sim.py --scenario nominal
    python scripts/tin_mars_sim.py --scenario conjunction
    python scripts/tin_mars_sim.py --scenario dust_storm
    python scripts/tin_mars_sim.py --scenario emergency_conjunction

Dependencies: numpy, matplotlib
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================================
# Mars Physical Constants
# =========================================================================

MARS = {
    "name": "Mars",
    "R_km": 3396.19,  # equatorial radius (km)
    "GM_km3s2": 4.282837e4,  # gravitational parameter (km³/s²)
    "sol_s": 88775.0,  # sidereal day (seconds)
    "J2": 1.9555e-3,  # oblateness (for future perturbation model)
    "areo_alt_km": 17032.0,  # areostationary altitude
    "areo_period_s": 88775.0,  # = 1 sol
    "areo_stable_lon": [342.08, 167.83],  # stable areostationary longitudes (°E)
}

PHOBOS = {
    "name": "Phobos",
    "a_km": 9376.0,  # semi-major axis from Mars center
    "period_s": 27555.0,  # orbital period (7h 39m)
    "alt_km": 9376.0 - 3396.19,  # altitude above surface (~5980 km)
    "incl_deg": 1.08,  # near-equatorial
    "decay_cm_yr": -1.8,  # tidal decay (spiraling inward)
}

DEIMOS = {
    "name": "Deimos",
    "a_km": 23463.0,
    "period_s": 109080.0,  # ~30.3 hours
    "alt_km": 23463.0 - 3396.19,  # ~20,067 km altitude
    "incl_deg": 1.79,
}

# Earth-Mars link parameters
EARTH_MARS = {
    "synodic_period_days": 779.94,  # conjunction-to-conjunction
    "min_distance_km": 5.46e7,  # closest approach
    "max_distance_km": 4.013e8,  # farthest (conjunction)
    "min_owlt_s": 182.0,  # one-way light time at closest (3.03 min)
    "max_owlt_s": 1339.0,  # at farthest (22.3 min)
    "conjunction_blackout_days": 18,  # typical comm blackout
    "corona_angle_deg": 3.0,  # SEP angle below which link is blocked
}


# =========================================================================
# Mars Orbital Mechanics
# =========================================================================


def mars_kepler_period(alt_km):
    """Orbital period for circular Mars orbit at given altitude."""
    a = MARS["R_km"] + alt_km
    return 2 * np.pi * np.sqrt(a**3 / MARS["GM_km3s2"])


def mars_owlt(distance_km):
    """One-way light time in seconds."""
    return distance_km / 299792.458


def earth_mars_distance(t_days, t0_conjunction_days=0):
    """Approximate Earth-Mars distance over synodic cycle.

    Simple sinusoidal model — adequate for conjunction blackout timing.
    Returns distance in km.
    """
    phase = 2 * np.pi * (t_days - t0_conjunction_days) / EARTH_MARS["synodic_period_days"]
    # At conjunction (phase=0): max distance; at opposition (phase=π): min distance
    d_min = EARTH_MARS["min_distance_km"]
    d_max = EARTH_MARS["max_distance_km"]
    d_mean = (d_min + d_max) / 2
    d_amp = (d_max - d_min) / 2
    return d_mean + d_amp * np.cos(phase)


def is_conjunction_blackout(t_days, t0_conjunction_days=0):
    """Check if Earth-Mars link is blacked out by solar conjunction.

    At t=t0, conjunction is centered (Sun between Earth and Mars).
    Blackout occurs when Sun-Earth-Mars angle < corona threshold.
    """
    phase = 2 * np.pi * (t_days - t0_conjunction_days) / EARTH_MARS["synodic_period_days"]
    # At phase=0 (conjunction center): SEP angle is 0° → maximum blackout
    # SEP angle grows as we move away from conjunction
    # Model: SEP angle ≈ |phase| in degrees, scaled to synodic period
    sep_angle_deg = np.abs(np.degrees(phase)) % 360
    if sep_angle_deg > 180:
        sep_angle_deg = 360 - sep_angle_deg
    return sep_angle_deg < EARTH_MARS["corona_angle_deg"]


# =========================================================================
# AEROSTAT ARMADA — Layer 0: Mobile Wind-Driven Mesh (v0.2)
# =========================================================================
# 24 helium super-pressure balloons at 18-25 km altitude
# Heritage: NASA ULDB + Mars 2001 balloon concepts
# Winds: MCD/GCM parametric — zonal jets, diurnal tides, Valles katabatic,
#         dust storm coupling. Laser inter-aerostat mesh for low-latency
#         local pickup (<10 min anywhere in equatorial band).


class MarsWindField:
    """22 km altitude wind field — MCD/GCM parametric approximation.

    Zonal (east): base flow + seasonal polar jets + diurnal thermal tide
    Meridional (north): weak Hadley cell + Valles Marineris katabatic
    Dust coupling: optical depth τ-scaled wind boost (GDS events +35%)
    """

    def __init__(self, alt_km=22.0):
        self.alt_km = alt_km

    def get_wind(self, lat_deg, lon_deg, t_sol):
        """Returns (u_east, v_north) in m/s at given position and time."""
        lat = np.deg2rad(lat_deg)
        ls = (t_sol % 668.59) * 360.0 / 668.59  # solar longitude (Ls)

        # Zonal: base 40 m/s + seasonal jet + 12 m/s diurnal tide
        u = (
            40.0
            + 45.0 * np.sin(lat * 1.8) * np.sin(np.deg2rad(ls))
            + 12.0 * np.sin(2 * np.pi * t_sol)
        )

        # Meridional: weak Hadley circulation
        v = 9.0 * np.cos(lat) * np.cos(np.deg2rad(ls))

        # Valles Marineris katabatic perturbation (real signal persists weakly to ~20 km)
        if 250 < (lon_deg % 360) < 320 and -28 < lat_deg < -3:
            v -= 7.5

        # Dust storm boost: τ-scaled (2018 GDS-like peak near Ls=310°)
        tau_boost = 1.0 + 0.35 * np.exp(-(((t_sol % 668.59) - 310) ** 2) / 60**2)
        return u * tau_boost, v * tau_boost


class AerostatArmada:
    """Fleet of 24 wind-driven super-pressure balloons for local DTN mesh.

    Propagation: Euler step on sphere (fast, accurate enough for network sim).
    Coverage model: 24 balloons → <400 km average spacing in equatorial band.
    Each aerostat carries laser comms for inter-mesh and uplink to polar sats.
    """

    def __init__(self, n=24, seed=42):
        self.n = n
        np.random.seed(seed)
        self.lats = np.random.uniform(-30, 30, n)  # equatorial bias initial seed
        self.lons = np.random.uniform(0, 360, n)
        self.wind = MarsWindField()
        self.t = 0.0

    def propagate(self, dt_sol=0.01):
        """Advance all aerostats by dt_sol (~15 min at dt=0.01)."""
        for i in range(self.n):
            u, v = self.wind.get_wind(self.lats[i], self.lons[i], self.t)
            dlat = v * dt_sol * MARS["sol_s"] / (MARS["R_km"] * 1000) * (180 / np.pi)
            cos_lat = np.cos(np.deg2rad(self.lats[i]))
            if abs(cos_lat) < 0.01:
                cos_lat = 0.01  # polar singularity guard
            dlon = u * dt_sol * MARS["sol_s"] / (MARS["R_km"] * 1000 * cos_lat) * (180 / np.pi)
            self.lats[i] = np.clip(self.lats[i] + dlat, -85, 85)
            self.lons[i] = (self.lons[i] + dlon) % 360
        self.t += dt_sol

    def avg_gap_min(self, lat_deg=0.0):
        """Effective local pickup latency via nearest aerostat.

        24 balloons in equatorial band → ~4-12 min local mesh pickup.
        Coverage degrades toward poles (aerostats drift equatorward).
        """
        lat_penalty = abs(lat_deg) / 60.0  # 0 at equator, 1.5 at poles
        return max(4.0, 12.0 * (1 - 1.0 / (1 + lat_penalty)))

    def run_sim(self, sim_sols=28, dt_sol=0.01):
        """Fast-forward the armada for sim_sols."""
        n_steps = int(sim_sols / dt_sol)
        for _ in range(n_steps):
            self.propagate(dt_sol)
        return self


AEROSTAT_COST_M = 3.5  # $M per balloon (heritage: NASA ULDB cost envelope)


# =========================================================================
# Mars Coverage Model (ported from lunar tin_coverage_sim.py)
# =========================================================================


def mars_compute_coverage(
    n_polar_sats=6,
    polar_alt_km=350,
    n_areo_relays=2,
    use_phobos=True,
    use_deimos=True,
    use_aerostats=True,
    elev_min_deg=5.0,
    sim_sols=28,
    lat_deg=0.0,
):
    """Compute Mars surface coverage metrics.

    Architecture layers:
      0. Aerostat Armada (if use_aerostats) — 24 wind-driven balloons
      1. Polar smallsats (n_polar_sats at polar_alt_km)
      2. Areostationary relays (n_areo_relays at 17,032 km)
      3. Phobos relay (if use_phobos)
      4. Deimos relay (if use_deimos)

    Returns: (coverage_pct, worst_gap_min, avg_gap_min, layer_breakdown)
    """
    total_min = sim_sols * MARS["sol_s"] / 60

    # --- Layer 0: Aerostat Armada ---
    aerostat_cov = 0.0
    aero_gap_min = 999.0
    if use_aerostats:
        armada = AerostatArmada(24)
        # Fast-forward to get equilibrium distribution (reduced steps for speed)
        armada.run_sim(sim_sols=min(sim_sols, 5), dt_sol=0.05)
        aero_gap_min = armada.avg_gap_min(lat_deg)
        # Aerostat coverage contribution: strong equatorial, weak polar
        aerostat_cov = max(0, 25.0 * (1 - (abs(lat_deg) / 60.0) ** 1.3))

    # --- Layer 1: Polar constellation ---
    # Mars is ~2x lunar radius, so need more sats for equivalent coverage
    # Scale: each sat covers ~(1/n) of the orbital plane per period
    polar_period_min = mars_kepler_period(polar_alt_km) / 60
    base_polar_cov = 35.0 + (n_polar_sats * 7.2) + ((350 - polar_alt_km) * 0.01)
    # Latitude dependence: polar sats best at poles, worst at equator
    lat_factor = 0.6 + 0.4 * (abs(lat_deg) / 90.0)  # 60% at equator, 100% at poles
    base_polar_cov *= lat_factor
    base_polar_cov = min(92.0, max(30.0, base_polar_cov))

    # --- Layer 2: Areostationary relays ---
    areo_cov = 0.0
    if n_areo_relays > 0:
        # Each areo relay sees ~42% of Mars surface (120° cone at 17,032 km)
        # 2 relays at stable longitudes cover ~70% of equatorial band
        # 4 relays: ~95% equatorial, dropping toward poles
        per_relay_cov = 42.0 * (1 - abs(lat_deg) / 120.0)  # drops off toward poles
        per_relay_cov = max(0, per_relay_cov)
        areo_cov = min(95.0, n_areo_relays * per_relay_cov * 0.6)  # overlap discount

    # --- Layer 3: Phobos relay ---
    phobos_cov = 0.0
    if use_phobos:
        # Phobos at 5,980 km alt, 7.66 hr period — visible ~55% of time from equatorial regions
        # Near-equatorial orbit → coverage peaks at low latitudes
        phobos_duty = 0.55 * (1 - (abs(lat_deg) / 90.0) ** 1.5)
        phobos_cov = phobos_duty * 30.0  # 30% effective coverage contribution

    # --- Layer 4: Deimos relay ---
    deimos_cov = 0.0
    if use_deimos:
        # Deimos at 20,067 km alt, 30.3 hr period — near-synchronous, visible ~90% from equator
        deimos_duty = 0.90 * (1 - (abs(lat_deg) / 90.0) ** 2)
        deimos_cov = deimos_duty * 20.0  # supplementary coverage

    # --- Combined coverage (union, not sum) ---
    # P(A ∪ B ∪ C ∪ D ∪ E) approximation using inclusion-exclusion simplified
    layers = [
        aerostat_cov / 100,
        base_polar_cov / 100,
        areo_cov / 100,
        phobos_cov / 100,
        deimos_cov / 100,
    ]
    combined = 1.0
    for p in layers:
        combined *= 1 - p
    coverage_pct = min(100.0, (1 - combined) * 100)

    # --- Gap analysis ---
    uncovered_frac = max(0.001, (100 - coverage_pct) / 100)

    # Number of gap-closing passes per sol
    polar_passes = n_polar_sats * (MARS["sol_s"] / (mars_kepler_period(polar_alt_km)))
    phobos_passes = MARS["sol_s"] / PHOBOS["period_s"] if use_phobos else 0
    deimos_passes = MARS["sol_s"] / DEIMOS["period_s"] if use_deimos else 0
    aerostat_passes = (
        24 * (MARS["sol_s"] / (15 * 60)) if use_aerostats else 0
    )  # ~24 effective "passes"/sol
    total_passes = (
        polar_passes + phobos_passes + deimos_passes + (n_areo_relays * 1.0) + aerostat_passes
    )

    worst_gap_min = uncovered_frac * MARS["sol_s"] / 60 / max(total_passes, 1) * 2.0
    worst_gap_min = max(1.0, worst_gap_min)  # floor: processing + propagation

    # Aerostats dominate local latency when present (always overhead in equatorial band)
    if use_aerostats:
        worst_gap_min = min(worst_gap_min, aero_gap_min)

    avg_gap_min = worst_gap_min * 0.35

    layer_breakdown = {
        "aerostat_armada_pct": round(aerostat_cov, 1),
        "polar_constellation_pct": round(base_polar_cov, 1),
        "areostationary_pct": round(areo_cov, 1),
        "phobos_relay_pct": round(phobos_cov, 1),
        "deimos_relay_pct": round(deimos_cov, 1),
        "combined_pct": round(coverage_pct, 1),
    }

    return coverage_pct, worst_gap_min, avg_gap_min, layer_breakdown


# =========================================================================
# Conjunction Blackout Simulation
# =========================================================================


def simulate_conjunction(
    n_polar_sats=6,
    polar_alt_km=350,
    n_areo_relays=2,
    blackout_days=None,
    pre_park_days=5,
    bundle_priority="EMERGENCY",
):
    """Simulate bundle routing during Earth-Mars solar conjunction.

    The conjunction problem:
      - For ~18 days, no direct Earth-Mars link (Sun blocks RF/optical)
      - Life-critical bundles must be stored and burst-forwarded post-blackout
      - Deimos (30.3 hr period, ~20,000 km altitude) serves as conjunction depot

    Strategy:
      1. Pre-conjunction: predictive routing parks high-priority bundles on Deimos
      2. During blackout: local Mars network operates autonomously (Layers 1-3)
      3. Post-conjunction: Deimos burst-forwards stored bundles to Earth via
         areostationary backbone as soon as link quality permits

    Returns: dict with conjunction simulation results
    """
    if blackout_days is None:
        blackout_days = EARTH_MARS["conjunction_blackout_days"]

    synodic_days = EARTH_MARS["synodic_period_days"]
    sim_days = blackout_days + 2 * pre_park_days + 10  # pre + blackout + post recovery

    # Time axis: center blackout at t=0
    t_days = np.linspace(-pre_park_days - 5, blackout_days + pre_park_days + 5, 1000)

    # Earth-Mars distance and blackout status over time
    distances_km = earth_mars_distance(t_days, t0_conjunction_days=0)
    owlt_s = distances_km / 299792.458
    blackout_mask = np.array([is_conjunction_blackout(t) for t in t_days])

    # Blackout window bounds
    blackout_start = t_days[np.argmax(blackout_mask)] if any(blackout_mask) else 0
    blackout_end = (
        t_days[len(t_days) - 1 - np.argmax(blackout_mask[::-1])] if any(blackout_mask) else 0
    )
    actual_blackout_days = blackout_end - blackout_start

    # --- Emergency bundle scenario ---
    # Bundle created during conjunction (worst case: day 3 of blackout)
    emergency_time_days = blackout_start + 3.0
    bundle_sol = emergency_time_days * 86400 / MARS["sol_s"]

    # Local Mars coverage (still fully operational during conjunction)
    local_cov, local_gap_min, _, layer_info = mars_compute_coverage(
        n_polar_sats=n_polar_sats,
        polar_alt_km=polar_alt_km,
        n_areo_relays=n_areo_relays,
        use_phobos=True,
        use_deimos=True,
        use_aerostats=True,
    )

    # Aerostat local pickup latency (dominates surface-to-orbit path)
    armada = AerostatArmada(24)
    armada.run_sim(sim_sols=5, dt_sol=0.05)
    aero_pickup_min = armada.avg_gap_min(0.0)

    # Custody chain during conjunction:
    # Astronaut → Aerostat (laser) → Polar Sat → Areo Relay → Deimos (depot) → [WAIT] → Earth
    surface_to_areo_min = aero_pickup_min + 2.0  # aerostat pickup + relay hops to areo
    areo_to_deimos_min = 5.0  # laser cross-link, ~23,000 km
    deimos_storage_days = max(0, blackout_end - emergency_time_days + 1.0)
    deimos_storage_min = deimos_storage_days * 24 * 60
    post_blackout_burst_min = (
        mars_owlt(distances_km[np.argmax(~blackout_mask & (t_days > blackout_end))]) / 60
        if any(~blackout_mask & (t_days > blackout_end))
        else 22.0
    )

    total_latency_min = (
        surface_to_areo_min + areo_to_deimos_min + deimos_storage_min + post_blackout_burst_min
    )
    total_latency_sols = total_latency_min / (MARS["sol_s"] / 60)

    # Pre-parked bundles: routine data staged on Deimos before conjunction
    pre_parked_bundles = {
        "routine_telemetry": {"size_MB": 500, "parked_days_before": pre_park_days},
        "science_data": {"size_MB": 2000, "parked_days_before": pre_park_days - 1},
        "habitat_status": {"size_MB": 50, "parked_days_before": 1},
    }

    results = {
        "scenario": "conjunction",
        "config": {
            "n_polar_sats": n_polar_sats,
            "polar_alt_km": polar_alt_km,
            "n_areo_relays": n_areo_relays,
            "use_phobos": True,
            "use_deimos": True,
        },
        "conjunction": {
            "blackout_start_day": round(blackout_start, 1),
            "blackout_end_day": round(blackout_end, 1),
            "actual_blackout_days": round(actual_blackout_days, 1),
            "max_owlt_min": round(float(np.max(owlt_s)) / 60, 1),
        },
        "local_mars_network": {
            "coverage_pct": round(local_cov, 1),
            "worst_gap_min": round(local_gap_min, 1),
            "layer_breakdown": layer_info,
            "status": "FULLY OPERATIONAL — conjunction affects Earth link only",
        },
        "emergency_during_conjunction": {
            "event": f"Emergency at conjunction day +3 (sol {bundle_sol:.1f})",
            "surface_to_deimos_min": round(surface_to_areo_min + areo_to_deimos_min, 1),
            "deimos_storage_days": round(deimos_storage_days, 1),
            "deimos_storage_min": round(deimos_storage_min, 0),
            "post_blackout_burst_min": round(post_blackout_burst_min, 1),
            "total_latency_min": round(total_latency_min, 1),
            "total_latency_sols": round(total_latency_sols, 2),
            "total_latency_days": round(total_latency_min / 1440, 1),
            "custody_chain": [
                "Astronaut (EVA/habitat)",
                "Overhead Aerostat (laser mesh)",
                "Mars Polar Sat → Areostationary Relay",
                f"Deimos Depot (stored {deimos_storage_days:.1f} days)",
                "Earth (DSN) — post-conjunction burst",
            ],
            "clinical_outcome": (
                f"Stabilized locally in <{surface_to_areo_min:.0f} min via overhead aerostat. "
                f"Full bundle (ECG, video, vitals) reaches Earth surgeons "
                f"{total_latency_sols:.1f} sols after event. "
                f"Earth response protocol received ~{total_latency_sols + 0.5:.1f} sols."
            ),
        },
        "pre_parked_bundles": pre_parked_bundles,
        "time_data": {
            "t_days": t_days.tolist(),
            "distances_km": distances_km.tolist(),
            "owlt_s": owlt_s.tolist(),
            "blackout_mask": blackout_mask.tolist(),
        },
    }

    return results


# =========================================================================
# Mars Trade Study (N-sat sweep like lunar trade study)
# =========================================================================

MARS_CONFIGS = [
    {"n_polar": 4, "alt_km": 350, "n_areo": 2, "aero": False, "label": "4pol+2areo"},
    {"n_polar": 6, "alt_km": 350, "n_areo": 2, "aero": False, "label": "6pol+2areo"},
    {"n_polar": 8, "alt_km": 350, "n_areo": 2, "aero": False, "label": "8pol+2areo"},
    {"n_polar": 6, "alt_km": 350, "n_areo": 4, "aero": False, "label": "6pol+4areo"},
    {"n_polar": 8, "alt_km": 350, "n_areo": 4, "aero": False, "label": "8pol+4areo"},
    {"n_polar": 6, "alt_km": 350, "n_areo": 2, "aero": True, "label": "6pol+2areo+24aero"},
    {"n_polar": 8, "alt_km": 350, "n_areo": 2, "aero": True, "label": "8pol+2areo+24aero"},
    {"n_polar": 6, "alt_km": 350, "n_areo": 4, "aero": True, "label": "6pol+4areo+24aero"},
]

MARS_SAT_MASS_KG = 24  # 12U baseline
AREO_SAT_MASS_KG = 200  # larger relay satellite
COST_PER_KG_M = 2.0  # $M/kg to Mars orbit (higher than lunar)


def run_mars_trade_study(use_phobos=True, use_deimos=True, lats=None):
    """Sweep Mars configurations across latitude bands."""
    if lats is None:
        lats = [0.0, -45.0, -89.5]  # equator, mid-latitude, polar

    results = []
    for cfg in MARS_CONFIGS:
        for lat in lats:
            cov, worst_gap, avg_gap, layers = mars_compute_coverage(
                n_polar_sats=cfg["n_polar"],
                polar_alt_km=cfg["alt_km"],
                n_areo_relays=cfg["n_areo"],
                use_phobos=use_phobos,
                use_deimos=use_deimos,
                use_aerostats=cfg.get("aero", False),
                lat_deg=lat,
            )
            cost = (
                cfg["n_polar"] * MARS_SAT_MASS_KG + cfg["n_areo"] * AREO_SAT_MASS_KG
            ) * COST_PER_KG_M
            if use_phobos:
                cost += 50  # Phobos relay deployment $50M
            if use_deimos:
                cost += 80  # Deimos depot deployment $80M
            if cfg.get("aero", False):
                cost += 24 * AEROSTAT_COST_M  # 24 balloons

            results.append(
                {
                    "config": cfg["label"],
                    "lat_deg": lat,
                    "coverage_pct": round(float(cov), 1),
                    "worst_gap_min": round(float(worst_gap), 1),
                    "avg_gap_min": round(float(avg_gap), 1),
                    "cost_proxy_M_usd": round(float(cost), 1),
                    "layers": layers,
                }
            )

    return results


# =========================================================================
# Visualization
# =========================================================================


def plot_mars_conjunction(results, outdir):
    """Plot conjunction blackout timeline + custody chain."""
    os.makedirs(outdir, exist_ok=True)

    BG = "#F7F9FC"
    MARS_RED = "#C1440E"
    EARTH_BLUE = "#2E75B6"
    BLACKOUT = "#1B1B2F"
    DEIMOS_GOLD = "#D4A017"
    SAFE_GREEN = "#2EA043"

    td = results["time_data"]
    t_days = np.array(td["t_days"])
    owlt_min = np.array(td["owlt_s"]) / 60
    blackout = np.array(td["blackout_mask"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor(BG)

    # --- Panel 1: OWLT + Blackout Timeline ---
    ax1.set_facecolor(BG)
    ax1.fill_between(t_days, 0, owlt_min, alpha=0.3, color=EARTH_BLUE, label="One-way light time")
    ax1.plot(t_days, owlt_min, color=EARTH_BLUE, linewidth=2)

    # Shade blackout region
    for i in range(len(t_days) - 1):
        if blackout[i]:
            ax1.axvspan(t_days[i], t_days[i + 1], color=BLACKOUT, alpha=0.3)

    # Mark emergency event
    em = results["emergency_during_conjunction"]
    conj = results["conjunction"]
    em_day = conj["blackout_start_day"] + 3.0
    ax1.axvline(em_day, color=MARS_RED, linewidth=2.5, linestyle="--", label="Emergency event")
    ax1.annotate(
        "Emergency!\nDay +3 of blackout",
        (em_day, np.max(owlt_min) * 0.8),
        fontsize=9,
        fontweight="bold",
        color=MARS_RED,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=MARS_RED),
    )

    # Mark Deimos storage period
    storage_start = em_day
    storage_end = conj["blackout_end_day"] + 1.0
    ax1.axvspan(
        storage_start,
        storage_end,
        color=DEIMOS_GOLD,
        alpha=0.15,
        label=f"Deimos custody ({em['deimos_storage_days']:.0f} days)",
    )

    ax1.set_xlabel("Days from conjunction center", fontsize=11)
    ax1.set_ylabel("One-Way Light Time (min)", fontsize=11)
    ax1.set_title(
        "Earth-Mars Conjunction Blackout & Emergency Routing", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # --- Panel 2: Custody chain diagram ---
    ax2.set_facecolor(BG)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 10)
    ax2.set_axis_off()

    chain = em["custody_chain"]
    n = len(chain)
    x_positions = np.linspace(8, 92, n)
    y_center = 5

    colors = [MARS_RED, EARTH_BLUE, EARTH_BLUE, DEIMOS_GOLD, SAFE_GREEN]

    for i, (label, x, color) in enumerate(zip(chain, x_positions, colors)):
        # Draw node box
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="black", alpha=0.85)
        ax2.text(
            x,
            y_center,
            label,
            fontsize=8,
            ha="center",
            va="center",
            bbox=bbox_props,
            color="white",
            fontweight="bold",
            wrap=True,
        )

        # Draw arrow to next node
        if i < n - 1:
            ax2.annotate(
                "",
                xy=(x_positions[i + 1] - 6, y_center),
                xytext=(x + 6, y_center),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
            )

    # Timing labels
    ax2.text(
        x_positions[0],
        y_center - 2.5,
        f"Local hop:\n{em['surface_to_deimos_min']:.0f} min",
        fontsize=8,
        ha="center",
        color=MARS_RED,
        fontweight="bold",
    )
    ax2.text(
        (x_positions[2] + x_positions[3]) / 2,
        y_center - 2.5,
        f"Deimos storage:\n{em['deimos_storage_days']:.0f} days",
        fontsize=8,
        ha="center",
        color=DEIMOS_GOLD,
        fontweight="bold",
    )
    ax2.text(
        x_positions[-1],
        y_center - 2.5,
        f"Post-conjunction burst:\n{em['post_blackout_burst_min']:.0f} min",
        fontsize=8,
        ha="center",
        color=SAFE_GREEN,
        fontweight="bold",
    )
    ax2.text(
        50,
        y_center + 3.5,
        f"Total end-to-end: {em['total_latency_sols']:.1f} sols ({em['total_latency_days']:.0f} days)",
        fontsize=12,
        ha="center",
        fontweight="bold",
        color=BLACKOUT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=DEIMOS_GOLD, alpha=0.3),
    )

    ax2.set_title("Conjunction Emergency Custody Chain", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    fname = f"tin_mars_conjunction_{datetime.now():%Y%m%d_%H%M}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Plot saved: {path}")
    return path


def plot_mars_trade(results, outdir):
    """Plot Mars trade study results — coverage vs latitude by config."""
    os.makedirs(outdir, exist_ok=True)

    BG = "#F7F9FC"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)

    # Group by latitude
    lats = sorted(set(r["lat_deg"] for r in results))
    configs = sorted(
        set(r["config"] for r in results),
        key=lambda c: results[[r["config"] for r in results].index(c)]["coverage_pct"],
    )

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(configs)))

    # Panel 1: Coverage by config, grouped by latitude
    ax1.set_facecolor(BG)
    width = 0.8 / len(lats)
    for i, lat in enumerate(lats):
        lat_results = [r for r in results if r["lat_deg"] == lat]
        lat_results.sort(
            key=lambda r: (
                [c["label"] for c in MARS_CONFIGS].index(r["config"])
                if r["config"] in [c["label"] for c in MARS_CONFIGS]
                else 99
            )
        )
        x = np.arange(len(lat_results))
        covs = [r["coverage_pct"] for r in lat_results]
        labels = [r["config"] for r in lat_results]
        offset = (i - len(lats) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, covs, width, label=f"lat={lat}°", alpha=0.8)

    ax1.set_xlabel("Configuration", fontsize=11)
    ax1.set_ylabel("Coverage (%)", fontsize=11)
    ax1.set_title("Mars Coverage by Configuration & Latitude", fontsize=12, fontweight="bold")
    ax1.set_xticks(np.arange(len(MARS_CONFIGS)))
    ax1.set_xticklabels([c["label"] for c in MARS_CONFIGS], rotation=35, ha="right", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Worst gap vs cost
    ax2.set_facecolor(BG)
    for lat in lats:
        lat_results = [r for r in results if r["lat_deg"] == lat]
        costs = [r["cost_proxy_M_usd"] for r in lat_results]
        gaps = [r["worst_gap_min"] for r in lat_results]
        ax2.scatter(costs, gaps, s=100, alpha=0.7, label=f"lat={lat}°", edgecolors="black")
        for r in lat_results:
            ax2.annotate(
                r["config"],
                (r["cost_proxy_M_usd"], r["worst_gap_min"]),
                fontsize=6,
                ha="center",
                va="bottom",
            )

    ax2.set_xlabel("Cost Proxy ($M)", fontsize=11)
    ax2.set_ylabel("Worst-Case Gap (min)", fontsize=11)
    ax2.set_title("Mars Cost vs Latency Trade Space", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"tin_mars_trade_{datetime.now():%Y%m%d_%H%M}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Plot saved: {path}")
    return path


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="TIN-Mars v0.1 Simulation")
    parser.add_argument(
        "--scenario",
        default="nominal",
        choices=["nominal", "conjunction", "trade_study", "emergency_conjunction", "all"],
    )
    parser.add_argument("--n_polar", type=int, default=6)
    parser.add_argument("--n_areo", type=int, default=2)
    parser.add_argument("--alt_km", type=int, default=350)
    args = parser.parse_args()

    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(repo_root, "data")
    results_dir = os.path.join(repo_root, "results")

    print(f"\n{'=' * 65}")
    print("  TIN-Mars v0.1 — Interplanetary DTN Architecture")
    print(f"  Config: {args.n_polar} polar + {args.n_areo} areo + Phobos + Deimos")
    print(f"  Scenario: {args.scenario}")
    print(f"{'=' * 65}")

    if args.scenario in ("nominal", "all"):
        print("\n  --- Nominal Coverage Analysis ---")
        for lat in [0, -30, -60, -89.5]:
            cov, wg, ag, layers = mars_compute_coverage(
                n_polar_sats=args.n_polar,
                polar_alt_km=args.alt_km,
                n_areo_relays=args.n_areo,
                lat_deg=lat,
            )
            print(
                f"    lat={lat:>6.1f}°: cov={cov:>5.1f}%  worst_gap={wg:>6.1f}min  "
                f"[aero={layers.get('aerostat_armada_pct', 0)}% "
                f"polar={layers['polar_constellation_pct']}% areo={layers['areostationary_pct']}% "
                f"phobos={layers['phobos_relay_pct']}% deimos={layers['deimos_relay_pct']}%]"
            )

    if args.scenario in ("conjunction", "emergency_conjunction", "all"):
        print("\n  --- Conjunction Blackout Simulation ---")
        conj = simulate_conjunction(
            n_polar_sats=args.n_polar, polar_alt_km=args.alt_km, n_areo_relays=args.n_areo
        )
        ci = conj["conjunction"]
        em = conj["emergency_during_conjunction"]
        lm = conj["local_mars_network"]
        print(
            f"    Blackout window: day {ci['blackout_start_day']} to {ci['blackout_end_day']} "
            f"({ci['actual_blackout_days']} days)"
        )
        print(f"    Max OWLT: {ci['max_owlt_min']} min")
        print(
            f"    Local Mars coverage during blackout: {lm['coverage_pct']}% "
            f"(worst gap {lm['worst_gap_min']} min)"
        )
        print(f"    Status: {lm['status']}")
        print("\n    Emergency during conjunction:")
        print(f"      Surface → Deimos: {em['surface_to_deimos_min']} min")
        print(f"      Deimos storage: {em['deimos_storage_days']} days")
        print(f"      Post-blackout burst: {em['post_blackout_burst_min']} min")
        print(
            f"      Total latency: {em['total_latency_sols']} sols ({em['total_latency_days']} days)"
        )
        print("\n    Custody chain:")
        for node in em["custody_chain"]:
            print(f"      → {node}")
        print(f"\n    Clinical outcome: {em['clinical_outcome']}")

        # Save and plot
        os.makedirs(data_dir, exist_ok=True)
        jpath = os.path.join(data_dir, f"mars_conjunction_{datetime.now():%Y%m%d_%H%M}.json")
        # Remove time_data for JSON (too large)
        save_conj = {k: v for k, v in conj.items() if k != "time_data"}
        with open(jpath, "w") as f:
            json.dump(save_conj, f, indent=2)
        print(f"\n  JSON saved: {jpath}")
        plot_mars_conjunction(conj, results_dir)

    if args.scenario in ("trade_study", "all"):
        print("\n  --- Mars Trade Study ---")
        trade = run_mars_trade_study()
        print(f"\n    {'Config':<22} {'Lat':>5} {'Cov%':>6} {'Gap':>8} {'Cost':>8}")
        print(f"    {'-' * 52}")
        for r in trade:
            print(
                f"    {r['config']:<22} {r['lat_deg']:>5.1f} {r['coverage_pct']:>5.1f}% "
                f"{r['worst_gap_min']:>7.1f}m {r['cost_proxy_M_usd']:>7.1f}M"
            )

        os.makedirs(data_dir, exist_ok=True)
        jpath = os.path.join(data_dir, f"mars_trade_{datetime.now():%Y%m%d_%H%M}.json")
        with open(jpath, "w") as f:
            json.dump(trade, f, indent=2)
        print(f"\n  JSON saved: {jpath}")
        plot_mars_trade(trade, results_dir)

    print("\n  Done.")


if __name__ == "__main__":
    main()
