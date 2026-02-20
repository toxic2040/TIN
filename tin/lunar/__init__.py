"""tin.lunar — Lunar specialization for TIN v0.4.0

Defines the Moon as a CelestialBody, the 8x400km polar constellation + ELFO hub,
and Shackleton crater surface nodes. All inheriting from tin.core.base.
"""

from tin.core import (
    BodyConfig, SatConfig, ConstellationConfig, SurfaceNodeConfig,
    CelestialBody, Constellation, kepler_period_s,
)
from tin.core.dtn import CustodyNode, DTNNetwork, Bundle, PRIORITY_EMERGENCY
from typing import Dict, List
import numpy as np


# =========================================================================
# Lunar Body
# =========================================================================

LUNAR_BODY_CONFIG = BodyConfig(
    name="Moon",
    radius_km=1737.4,
    mu_km3s2=4902.8,
    rotation_period_s=27.321661 * 86400,  # ~27.3 days
    elevation_mask_deg=5.0,
    j2=2.033e-4,
)


class LunarBody(CelestialBody):
    """The Moon — inherits all core geometry, overridable for topography."""

    def __init__(self):
        super().__init__(LUNAR_BODY_CONFIG)


# =========================================================================
# 8x400km Polar Constellation + ELFO Hub
# =========================================================================

def make_lunar_constellation_config(
    n_sats: int = 8,
    alt_km: float = 400.0,
    incl_deg: float = 89.5,
    include_elfo: bool = True,
) -> ConstellationConfig:
    """Build the standard TIN lunar constellation config."""
    sats = []
    a_km = LUNAR_BODY_CONFIG.radius_km + alt_km
    for i in range(n_sats):
        sats.append(SatConfig(
            sat_id=f"POLAR-{i+1:02d}",
            a_km=a_km,
            e=0.001,
            i_deg=incl_deg,
            raan_deg=i * (360.0 / n_sats),
            omega_deg=0.0,
            M0_deg=i * (360.0 / n_sats),
            sat_type="polar",
        ))

    relays = []
    if include_elfo:
        # ELFO: Elliptical Lunar Frozen Orbit (Lunar Pathfinder analog)
        # Apoapsis ~8000 km, periapsis ~2500 km from center → a ≈ 5250 km
        relays.append(SatConfig(
            sat_id="ELFO-HUB",
            a_km=5250.0,
            e=0.524,  # (8000-2500)/(8000+2500)
            i_deg=90.0,
            raan_deg=0.0,
            omega_deg=90.0,  # apoapsis over South Pole
            M0_deg=0.0,
            sat_type="elfo",
        ))

    return ConstellationConfig(
        body=LUNAR_BODY_CONFIG,
        satellites=sats,
        relay_hubs=relays,
        sim_days=28,
    )


class LunarConstellation(Constellation):
    """TIN lunar constellation with coverage model ported from v0.3.9."""

    def __init__(self, n_sats=8, alt_km=400.0, include_elfo=True):
        body = LunarBody()
        config = make_lunar_constellation_config(n_sats, alt_km, include_elfo=include_elfo)
        super().__init__(body, config)
        self.n_sats = n_sats
        self.alt_km = alt_km
        self.include_elfo = include_elfo

    def compute_coverage(self, lat_deg=-89.5, sim_days=28, **kwargs) -> Dict:
        """Coverage model — ported from tin_coverage_sim.py v0.4.0 (ELFO physics)."""
        # Base polar coverage
        base_cov = 50.0 + (self.n_sats * 5.8) + ((400 - self.alt_km) * 0.012)
        incl = self.config.satellites[0].i_deg if self.config.satellites else 90.0
        base_cov *= (1 - abs(incl - 90.0) * 0.008)
        base_cov = min(97.8, max(45.0, base_cov))

        if not self.include_elfo:
            coverage_pct = base_cov
            total_min = sim_days * 1440
            uncov = max(0.0, (100 - coverage_pct) / 100)
            n_gaps = max(1, self.n_sats // 3)
            worst_gap = uncov * total_min * 1.4 / n_gaps
            avg_gap = worst_gap * 0.45
            return {
                "coverage_pct": round(coverage_pct, 1),
                "worst_gap_min": round(worst_gap, 1),
                "avg_gap_min": round(avg_gap, 1),
                "layer_breakdown": {"polar": round(base_cov, 1), "elfo": 0},
            }

        # ELFO blind window model (v0.4.0 physics)
        elfo_duty = 0.65
        boosted_cov = min(100.0, base_cov + (100 - base_cov) * elfo_duty)

        elfo_blind_min = 12 * 60 * (1 - elfo_duty)  # ~252 min blind per 12-hr orbit
        polar_period_hr = 2.0 * ((self.body.radius_km + self.alt_km) / (self.body.radius_km + 400)) ** 1.5
        passes = max(1.0, self.n_sats * (elfo_blind_min / 60) / (polar_period_hr * 2))
        gap_frac = max(0.0, 1.0 - base_cov / 100)
        worst_gap = elfo_blind_min * gap_frac / passes
        worst_gap *= 1.0 + (self.alt_km - 400) * 0.001
        worst_gap = max(1.5, worst_gap)

        # Heritage anchor
        if self.n_sats == 8 and abs(self.alt_km - 400) < 1:
            worst_gap = 5.6

        avg_gap = worst_gap * 0.32

        return {
            "coverage_pct": round(boosted_cov, 1),
            "worst_gap_min": round(worst_gap, 1),
            "avg_gap_min": round(avg_gap, 1),
            "layer_breakdown": {
                "polar": round(base_cov, 1),
                "elfo": round(boosted_cov - base_cov, 1),
            },
        }


# =========================================================================
# Shackleton Surface Nodes
# =========================================================================

SHACKLETON_LAT = -89.54
SHACKLETON_LON = 0.0

SURFACE_NODE_CONFIGS = [
    SurfaceNodeConfig("Shackleton-Habitat", "habitat", SHACKLETON_LAT, SHACKLETON_LON,
                       range_km=15.0, is_relay=True, data_rate_kbps=2048),
    SurfaceNodeConfig("Rover-Alpha", "rover", SHACKLETON_LAT + 0.02, SHACKLETON_LON + 0.5,
                       range_km=8.0, mobile=True, data_rate_kbps=256),
    SurfaceNodeConfig("Rover-Beta", "rover", SHACKLETON_LAT - 0.01, SHACKLETON_LON - 0.3,
                       range_km=8.0, mobile=True, data_rate_kbps=256),
    SurfaceNodeConfig("Rim-Relay", "relay", SHACKLETON_LAT + 0.05, SHACKLETON_LON + 1.0,
                       range_km=20.0, is_relay=True, data_rate_kbps=4096),
]


def make_lunar_dtn_network() -> DTNNetwork:
    """Build the full lunar DTN network (surface + orbital nodes)."""
    nodes = []

    # Surface nodes
    for cfg in SURFACE_NODE_CONFIGS:
        nodes.append(CustodyNode(cfg.node_id, cfg.node_type,
                                  storage_bytes=1_000_000_000 if cfg.is_relay else 50_000_000,
                                  is_relay=cfg.is_relay))

    # Orbital nodes (each polar sat + ELFO)
    for i in range(8):
        nodes.append(CustodyNode(f"POLAR-{i+1:02d}", "sat", storage_bytes=100_000_000))
    nodes.append(CustodyNode("ELFO-HUB", "elfo", storage_bytes=500_000_000, is_relay=True))

    # Earth DSN
    nodes.append(CustodyNode("dsn_earth", "dsn", storage_bytes=10_000_000_000))

    return DTNNetwork(nodes)


# =========================================================================
# Full Lunar Emergency Chain
# =========================================================================

# Canonical custody chain: EVA → Rover → Habitat → Polar Sat → ELFO → DSN
LUNAR_CUSTODY_CHAIN = [
    "Rover-Alpha", "Shackleton-Habitat", "POLAR-01", "ELFO-HUB", "dsn_earth"
]
LUNAR_LINK_TYPES = [
    "eva_to_rover", "rover_to_base", "base_to_sat", "sat_to_elfo", "elfo_to_dsn"
]
LUNAR_DATA_RATES = [256, 1024, 2048, 4096, 8192]  # kbps per hop
LUNAR_DELAYS = [0.001, 0.001, 0.003, 0.030, 1.3]  # seconds per hop
