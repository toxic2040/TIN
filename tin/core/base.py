# SPDX-License-Identifier: MIT
"""tin.core.base — Shared dataclasses for body, satellite, and constellation configs.

These are the types referenced by tin.config.lunar_default and any future
body-specific configs (Mars, heliocentric).
"""

from dataclasses import dataclass, field


@dataclass
class BodyConfig:
    """Celestial body parameters."""

    name: str
    radius_km: float
    mu_km3s2: float
    rotation_period_s: float
    elevation_mask_deg: float = 5.0
    j2: float = 0.0  # zonal harmonic (stub until propagator uses it)


@dataclass
class SatConfig:
    """Orbital element set for a constellation satellite."""

    sat_id: str
    a_km: float  # semi-major axis
    i_deg: float = 90.0  # inclination
    raan_deg: float = 0.0  # RAAN
    e: float = 0.0  # eccentricity (needed for ELFO, default circular)
    aop_deg: float = 0.0  # argument of periapsis
    ta_deg: float = 0.0  # true anomaly at epoch
    sat_type: str = "polar"  # polar | elfo | areo | relay


@dataclass
class HaloConfig:
    """EM-L2 (or other libration-point) halo orbit config (N8)."""

    sat_id: str
    z_amplitude_km: float  # out-of-plane amplitude
    period_days: float  # halo period
    family: str = "northern"  # northern | southern
    station_keeping_ms_yr: float = 12.0  # m/s per year
    insertion_dv_kms: float = 3.2


@dataclass
class HeliocentricOrbitConfig:
    """Heliocentric (Sun-centred) orbit for interplanetary relay or planet.

    Uses Keplerian elements in the ecliptic frame.  The J2 propagator in
    contact_gen treats these identically to SatConfig elements but referenced
    to the Sun's mu and (negligible) J2.

    Attributes
    ----------
    body_id       : label (e.g. "Earth", "Mars", "relay-1")
    a_au          : semi-major axis in AU
    e             : eccentricity
    i_deg         : inclination to ecliptic
    raan_deg      : longitude of ascending node (ecliptic)
    aop_deg       : argument of perihelion
    ta_deg        : true anomaly at epoch
    body_config   : optional BodyConfig if this is a planet (for surface contacts)
    """

    body_id: str
    a_au: float
    e: float = 0.0
    i_deg: float = 0.0
    raan_deg: float = 0.0
    aop_deg: float = 0.0
    ta_deg: float = 0.0
    body_config: BodyConfig | None = None

    @property
    def a_km(self) -> float:
        """Semi-major axis in km (for propagator compatibility)."""
        return self.a_au * 149_597_870.7


@dataclass
class ConstellationConfig:
    """Full constellation definition for a given body."""

    body: BodyConfig
    satellites: list[SatConfig] = field(default_factory=list)
    relay_hubs: list[SatConfig] = field(default_factory=list)
    halo_satellites: list[HaloConfig] = field(default_factory=list)
    sim_days: int = 28
    dt_s: float = 300.0
