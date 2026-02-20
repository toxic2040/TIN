"""tin.mars — Mars specialization for TIN v0.4.0

Defines Mars as a CelestialBody, the polar + areostationary constellation,
Phobos/Deimos custody depots, Aerostat Armada, and conjunction handling.
All inheriting from tin.core.base.
"""

from tin.core import (
    BodyConfig, SatConfig, ConstellationConfig, CelestialBody, Constellation,
)
from tin.core.dtn import CustodyNode, DTNNetwork, Bundle, PRIORITY_EMERGENCY
from typing import Dict, List, Optional
import numpy as np


# =========================================================================
# Mars Body
# =========================================================================

MARS_BODY_CONFIG = BodyConfig(
    name="Mars",
    radius_km=3396.19,
    mu_km3s2=4.282837e4,
    rotation_period_s=88775.0,
    elevation_mask_deg=5.0,
    j2=1.9555e-3,
)

EARTH_MARS_SYNODIC_DAYS = 779.94
CONJUNCTION_BLACKOUT_DAYS = 18
CORONA_ANGLE_DEG = 3.0


class MarsBody(CelestialBody):
    """Mars — inherits core geometry."""
    def __init__(self):
        super().__init__(MARS_BODY_CONFIG)


# =========================================================================
# Phobos / Deimos (natural custody depots)
# =========================================================================

PHOBOS = {"name": "Phobos", "a_km": 9376.0, "period_s": 27555.0, "alt_km": 5979.8}
DEIMOS = {"name": "Deimos", "a_km": 23463.0, "period_s": 109080.0, "alt_km": 20066.8}


# =========================================================================
# Aerostat Armada (Layer 0 — atmospheric mesh)
# =========================================================================

class MarsWindField:
    """22 km altitude wind field — MCD/GCM parametric approximation."""
    def get_wind(self, lat_deg, lon_deg, t_sol):
        lat = np.deg2rad(lat_deg)
        ls = (t_sol % 668.59) * 360.0 / 668.59
        u = (40.0 + 45.0 * np.sin(lat * 1.8) * np.sin(np.deg2rad(ls))
             + 12.0 * np.sin(2 * np.pi * t_sol))
        v = 9.0 * np.cos(lat) * np.cos(np.deg2rad(ls))
        if 250 < (lon_deg % 360) < 320 and -28 < lat_deg < -3:
            v -= 7.5
        tau = 1.0 + 0.35 * np.exp(-((t_sol % 668.59) - 310)**2 / 60**2)
        return u * tau, v * tau


class AerostatArmada:
    """24 wind-driven super-pressure balloons at 22 km altitude."""
    def __init__(self, n=24, seed=42):
        self.n = n
        np.random.seed(seed)
        self.lats = np.random.uniform(-30, 30, n)
        self.lons = np.random.uniform(0, 360, n)
        self.wind = MarsWindField()
        self.t = 0.0

    def propagate(self, dt_sol=0.01):
        R = MARS_BODY_CONFIG.radius_km * 1000
        sol_s = MARS_BODY_CONFIG.rotation_period_s
        for i in range(self.n):
            u, v = self.wind.get_wind(self.lats[i], self.lons[i], self.t)
            self.lats[i] += v * dt_sol * sol_s / R * (180 / np.pi)
            cos_lat = max(0.01, np.cos(np.deg2rad(self.lats[i])))
            self.lons[i] = (self.lons[i] + u * dt_sol * sol_s / (R * cos_lat) * (180 / np.pi)) % 360
            self.lats[i] = np.clip(self.lats[i], -85, 85)
        self.t += dt_sol

    def run_sim(self, sim_sols=5, dt_sol=0.05):
        for _ in range(int(sim_sols / dt_sol)):
            self.propagate(dt_sol)
        return self

    def avg_gap_min(self, lat_deg=0.0):
        return max(4.0, 12.0 * (1 - 1.0 / (1 + abs(lat_deg) / 60.0)))


# =========================================================================
# Mars Constellation
# =========================================================================

def make_mars_constellation_config(
    n_polar: int = 6, polar_alt_km: float = 350.0,
    n_areo: int = 2,
) -> ConstellationConfig:
    """Build Mars polar + areostationary constellation."""
    sats = []
    a_polar = MARS_BODY_CONFIG.radius_km + polar_alt_km
    for i in range(n_polar):
        sats.append(SatConfig(
            sat_id=f"MARS-POLAR-{i+1:02d}", a_km=a_polar,
            e=0.001, i_deg=89.5,
            raan_deg=i * (360.0 / n_polar), M0_deg=i * (360.0 / n_polar),
            sat_type="polar",
        ))

    relays = []
    areo_alt = 17032.0
    stable_lons = [342.08, 167.83, 75.0, 255.0]  # 2 stable + 2 drifting
    for i in range(n_areo):
        relays.append(SatConfig(
            sat_id=f"AREO-{i+1:02d}",
            a_km=MARS_BODY_CONFIG.radius_km + areo_alt,
            e=0.0, i_deg=0.0,
            raan_deg=stable_lons[i % len(stable_lons)],
            M0_deg=stable_lons[i % len(stable_lons)],
            sat_type="areo",
        ))

    return ConstellationConfig(
        body=MARS_BODY_CONFIG, satellites=sats, relay_hubs=relays, sim_days=28,
    )


class MarsConstellation(Constellation):
    """TIN Mars constellation with 5-layer coverage model."""

    def __init__(self, n_polar=6, polar_alt_km=350.0, n_areo=2,
                 use_phobos=True, use_deimos=True, use_aerostats=True):
        body = MarsBody()
        config = make_mars_constellation_config(n_polar, polar_alt_km, n_areo)
        super().__init__(body, config)
        self.n_polar = n_polar
        self.polar_alt_km = polar_alt_km
        self.n_areo = n_areo
        self.use_phobos = use_phobos
        self.use_deimos = use_deimos
        self.use_aerostats = use_aerostats

    def compute_coverage(self, lat_deg=0.0, sim_days=28, **kwargs) -> Dict:
        """5-layer Mars coverage model (ported from tin_mars_sim.py v0.2)."""
        # Layer 0: Aerostats
        aerostat_cov = 0.0
        aero_gap = 999.0
        if self.use_aerostats:
            armada = AerostatArmada(24)
            armada.run_sim(sim_sols=min(sim_days, 5))
            aero_gap = armada.avg_gap_min(lat_deg)
            aerostat_cov = max(0, 25.0 * (1 - (abs(lat_deg) / 60.0) ** 1.3))

        # Layer 1: Polar
        lat_factor = 0.6 + 0.4 * (abs(lat_deg) / 90.0)
        polar_cov = min(92.0, max(30.0, (35.0 + self.n_polar * 7.2) * lat_factor))

        # Layer 2: Areostationary
        areo_cov = 0.0
        if self.n_areo > 0:
            per = max(0, 42.0 * (1 - abs(lat_deg) / 120.0))
            areo_cov = min(95.0, self.n_areo * per * 0.6)

        # Layer 3: Phobos
        phobos_cov = 0.0
        if self.use_phobos:
            phobos_cov = 0.55 * (1 - (abs(lat_deg) / 90.0) ** 1.5) * 30.0

        # Layer 4: Deimos
        deimos_cov = 0.0
        if self.use_deimos:
            deimos_cov = 0.90 * (1 - (abs(lat_deg) / 90.0) ** 2) * 20.0

        # Union
        layers = [aerostat_cov / 100, polar_cov / 100, areo_cov / 100,
                  phobos_cov / 100, deimos_cov / 100]
        combined = 1.0
        for p in layers:
            combined *= (1 - p)
        coverage_pct = min(100.0, (1 - combined) * 100)

        # Gap analysis
        uncov = max(0.001, (100 - coverage_pct) / 100)
        R = self.body.radius_km
        sol_s = MARS_BODY_CONFIG.rotation_period_s
        polar_period = self.body.kepler_period(self.polar_alt_km)
        total_passes = (self.n_polar * sol_s / polar_period
                        + (sol_s / PHOBOS["period_s"] if self.use_phobos else 0)
                        + (sol_s / DEIMOS["period_s"] if self.use_deimos else 0)
                        + self.n_areo
                        + (24 * sol_s / 900 if self.use_aerostats else 0))
        worst_gap = max(1.0, uncov * sol_s / 60 / max(total_passes, 1) * 2.0)
        if self.use_aerostats:
            worst_gap = min(worst_gap, aero_gap)
        avg_gap = worst_gap * 0.35

        return {
            "coverage_pct": round(coverage_pct, 1),
            "worst_gap_min": round(worst_gap, 1),
            "avg_gap_min": round(avg_gap, 1),
            "layer_breakdown": {
                "aerostat": round(aerostat_cov, 1),
                "polar": round(polar_cov, 1),
                "areo": round(areo_cov, 1),
                "phobos": round(phobos_cov, 1),
                "deimos": round(deimos_cov, 1),
            },
        }


# =========================================================================
# Conjunction Simulation
# =========================================================================

def is_conjunction_blackout(t_days, t0=0):
    phase = 2 * np.pi * (t_days - t0) / EARTH_MARS_SYNODIC_DAYS
    sep = np.abs(np.degrees(phase)) % 360
    if sep > 180:
        sep = 360 - sep
    return sep < CORONA_ANGLE_DEG


def simulate_conjunction(constellation: MarsConstellation, emergency_day_offset=3.0):
    """Simulate emergency routing during solar conjunction."""
    t_days = np.linspace(-10, 30, 1000)
    blackout = np.array([is_conjunction_blackout(t) for t in t_days])

    bl_start = t_days[np.argmax(blackout)] if any(blackout) else 0
    bl_end = t_days[len(t_days) - 1 - np.argmax(blackout[::-1])] if any(blackout) else 0

    local = constellation.compute_coverage(lat_deg=0.0)

    em_day = bl_start + emergency_day_offset
    aero_pickup = AerostatArmada(24).run_sim(5).avg_gap_min(0.0) + 2.0
    deimos_days = max(0, bl_end - em_day + 1.0)
    burst_min = 22.3
    total_min = aero_pickup + 5.0 + deimos_days * 1440 + burst_min
    total_sols = total_min / (MARS_BODY_CONFIG.rotation_period_s / 60)

    return {
        "blackout_days": round(bl_end - bl_start, 1),
        "local_coverage_pct": local["coverage_pct"],
        "local_worst_gap_min": local["worst_gap_min"],
        "surface_to_deimos_min": round(aero_pickup + 5.0, 1),
        "deimos_storage_days": round(deimos_days, 1),
        "total_latency_sols": round(total_sols, 1),
        "total_latency_days": round(total_min / 1440, 1),
        "custody_chain": [
            "Astronaut", "Aerostat (laser mesh)",
            "Polar Sat → Areo Relay",
            f"Deimos Depot ({deimos_days:.0f} days)",
            "Earth (DSN)",
        ],
        "clinical_outcome": (
            f"Stabilized locally in <{aero_pickup:.0f} min. "
            f"Full bundle reaches Earth {total_sols:.1f} sols after event."
        ),
    }


# =========================================================================
# Mars DTN Network
# =========================================================================

def make_mars_dtn_network(n_polar=6, n_areo=2) -> DTNNetwork:
    nodes = []
    # Surface
    nodes.append(CustodyNode("Mars-Habitat", "habitat", 1_000_000_000, True))
    nodes.append(CustodyNode("Rover-1", "rover", 50_000_000))
    # Aerostats (aggregate node)
    nodes.append(CustodyNode("Aerostat-Mesh", "aerostat", 100_000_000, True))
    # Orbital
    for i in range(n_polar):
        nodes.append(CustodyNode(f"MARS-POLAR-{i+1:02d}", "sat", 100_000_000))
    for i in range(n_areo):
        nodes.append(CustodyNode(f"AREO-{i+1:02d}", "areo", 200_000_000, True))
    # Moon depots
    nodes.append(CustodyNode("Phobos-Relay", "moon_depot", 500_000_000, True))
    nodes.append(CustodyNode("Deimos-Depot", "moon_depot", 2_000_000_000, True))
    # Earth
    nodes.append(CustodyNode("dsn_earth", "dsn", 10_000_000_000))
    return DTNNetwork(nodes)


# Mars custody chain
MARS_CUSTODY_CHAIN = [
    "Aerostat-Mesh", "MARS-POLAR-01", "AREO-01", "Deimos-Depot", "dsn_earth"
]
MARS_LINK_TYPES = [
    "surface_to_aero", "aero_to_sat", "sat_to_areo", "areo_to_deimos", "deimos_to_dsn"
]
MARS_DATA_RATES = [512, 2048, 4096, 4096, 8192]
MARS_DELAYS = [0.001, 0.003, 0.01, 0.08, 1200.0]  # last = Earth-Mars OWLT
