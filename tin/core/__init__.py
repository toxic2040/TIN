"""tin.core.base — Planet-agnostic base classes for TIN v0.4.0

Layer 0: Physical constants, orbital mechanics primitives, visibility geometry.
All body-specific parameters injected via config — no hardcoded lunar/Mars values.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# =========================================================================
# Configuration dataclasses
# =========================================================================

@dataclass
class BodyConfig:
    """Physical parameters for any celestial body."""
    name: str
    radius_km: float
    mu_km3s2: float                        # gravitational parameter
    rotation_period_s: float               # sidereal rotation period
    elevation_mask_deg: float = 5.0        # minimum elevation for LOS
    j2: float = 0.0                        # oblateness (for perturbation model)

@dataclass
class SatConfig:
    """Orbital elements for a single satellite."""
    sat_id: str
    a_km: float                            # semi-major axis
    e: float = 0.001                       # eccentricity
    i_deg: float = 90.0                    # inclination
    raan_deg: float = 0.0                  # right ascension of ascending node
    omega_deg: float = 0.0                 # argument of periapsis
    M0_deg: float = 0.0                    # mean anomaly at epoch
    sat_type: str = "polar"                # polar, areo, elfo, relay

@dataclass
class ConstellationConfig:
    """Full constellation specification."""
    body: BodyConfig
    satellites: List[SatConfig] = field(default_factory=list)
    relay_hubs: List[SatConfig] = field(default_factory=list)
    sim_days: int = 28
    dt_s: float = 300.0                    # time step for propagation

@dataclass
class SurfaceNodeConfig:
    """A surface node (habitat, rover, fixed relay)."""
    node_id: str
    node_type: str                         # "habitat", "rover", "relay", "eva"
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    range_km: float = 15.0                 # comms range
    is_relay: bool = False
    mobile: bool = False
    data_rate_kbps: float = 256.0


# =========================================================================
# Core orbital mechanics (planet-agnostic)
# =========================================================================

def kepler_period_s(a_km: float, mu: float) -> float:
    """Orbital period in seconds for given semi-major axis and GM."""
    return 2 * np.pi * np.sqrt(a_km**3 / mu)


def kepler_position(sat: SatConfig, t_s: float, mu: float) -> np.ndarray:
    """Compute satellite position (body-centered inertial) at time t.

    Simplified two-body Keplerian propagation (J2-ready stub).
    Returns [x, y, z] in km.
    """
    period = kepler_period_s(sat.a_km, mu)
    # Mean anomaly at time t
    M = np.deg2rad(sat.M0_deg) + 2 * np.pi * (t_s % period) / period
    # For near-circular orbits (e << 1), true anomaly ≈ mean anomaly
    theta = M  # TODO: Kepler's equation solver for eccentric orbits
    r = sat.a_km * (1 - sat.e**2) / (1 + sat.e * np.cos(theta))

    # Position in orbital plane
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)

    # Rotate to body-centered inertial frame
    i = np.deg2rad(sat.i_deg)
    raan = np.deg2rad(sat.raan_deg)
    omega = np.deg2rad(sat.omega_deg)

    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)

    x = (x_orb * (cos_raan * cos_omega - sin_raan * sin_omega * cos_i)
         - y_orb * (cos_raan * sin_omega + sin_raan * cos_omega * cos_i))
    y = (x_orb * (sin_raan * cos_omega + cos_raan * sin_omega * cos_i)
         - y_orb * (sin_raan * sin_omega - cos_raan * cos_omega * cos_i))
    z = x_orb * sin_omega * sin_i + y_orb * cos_omega * sin_i

    return np.array([x, y, z])


def surface_position(lat_deg: float, lon_deg: float, radius_km: float) -> np.ndarray:
    """Convert lat/lon to body-centered Cartesian (km)."""
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)
    return np.array([
        radius_km * np.cos(phi) * np.cos(lam),
        radius_km * np.cos(phi) * np.sin(lam),
        radius_km * np.sin(phi),
    ])


def elevation_angle(surface_pos: np.ndarray, sat_pos: np.ndarray,
                     radius_km: float) -> float:
    """Compute elevation angle (degrees) of satellite from surface point."""
    range_vec = sat_pos - surface_pos
    range_norm = np.linalg.norm(range_vec)
    if range_norm < 1e-6:
        return 90.0
    # Dot product of surface normal (= surface_pos for sphere) and range vector
    dot = np.dot(surface_pos, range_vec)
    sin_elev = dot / (radius_km * range_norm)
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


# =========================================================================
# Abstract base classes
# =========================================================================

class CelestialBody(ABC):
    """Abstract celestial body — subclass for Moon, Mars, etc."""

    def __init__(self, config: BodyConfig):
        self.config = config
        self.name = config.name
        self.radius_km = config.radius_km
        self.mu = config.mu_km3s2

    def kepler_period(self, alt_km: float) -> float:
        """Orbital period at altitude (seconds)."""
        return kepler_period_s(self.radius_km + alt_km, self.mu)

    def surface_pos(self, lat_deg: float, lon_deg: float) -> np.ndarray:
        """Surface position in body-centered frame (km)."""
        return surface_position(lat_deg, lon_deg, self.radius_km)

    def is_visible(self, sat_pos: np.ndarray, surface_pos: np.ndarray) -> bool:
        """Check line-of-sight (base: spherical, override for topography)."""
        elev = elevation_angle(surface_pos, sat_pos, self.radius_km)
        return elev >= self.config.elevation_mask_deg


class Constellation(ABC):
    """Abstract constellation — subclass for polar, areostationary, etc."""

    def __init__(self, body: CelestialBody, config: ConstellationConfig):
        self.body = body
        self.config = config
        self.satellites = config.satellites
        self.relay_hubs = config.relay_hubs
        self.contact_log: List[Dict] = []

    def sat_position(self, sat: SatConfig, t_s: float) -> np.ndarray:
        """Position of satellite at time t (seconds from epoch)."""
        return kepler_position(sat, t_s, self.body.mu)

    @abstractmethod
    def compute_coverage(self, lat_deg: float, sim_days: int, **kwargs) -> Dict:
        """Compute coverage metrics for a surface point. Returns dict with
        coverage_pct, worst_gap_min, avg_gap_min, layer_breakdown."""
        pass

    def get_visible_sats(self, surface_pos: np.ndarray, t_s: float) -> List[SatConfig]:
        """Return all satellites visible from a surface point at time t."""
        visible = []
        for sat in self.satellites + self.relay_hubs:
            sat_pos = self.sat_position(sat, t_s)
            if self.body.is_visible(sat_pos, surface_pos):
                visible.append(sat)
        return visible

    def propagate(self, sim_days: int, dt_s: float = 300.0,
                  eval_lat: float = -89.5, eval_lon: float = 0.0) -> Dict:
        """Run full propagation and return coverage report."""
        print(f"  Propagating {len(self.satellites)} sats + "
              f"{len(self.relay_hubs)} relay(s) for {sim_days} days...")

        total_steps = int(sim_days * 86400 / dt_s)
        surface_pos = self.body.surface_pos(eval_lat, eval_lon)
        visible_count = 0

        gaps_min = []
        current_gap_s = 0

        for step in range(total_steps):
            t_s = step * dt_s
            sats = self.get_visible_sats(surface_pos, t_s)
            if sats:
                visible_count += 1
                if current_gap_s > 0:
                    gaps_min.append(current_gap_s / 60)
                    current_gap_s = 0
            else:
                current_gap_s += dt_s

        if current_gap_s > 0:
            gaps_min.append(current_gap_s / 60)

        cov_pct = 100.0 * visible_count / max(total_steps, 1)
        worst_gap = max(gaps_min) if gaps_min else 0.0
        avg_gap = float(np.mean(gaps_min)) if gaps_min else 0.0

        print(f"  ✓ Propagation complete — cov={cov_pct:.1f}%, "
              f"worst_gap={worst_gap:.1f}min")

        return {
            "coverage_pct": round(cov_pct, 1),
            "worst_gap_min": round(worst_gap, 1),
            "avg_gap_min": round(avg_gap, 1),
            "total_steps": total_steps,
            "eval_lat": eval_lat,
        }
