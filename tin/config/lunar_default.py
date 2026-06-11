# SPDX-License-Identifier: MIT
"""tin.config.lunar_default — Official lunar baseline per 2026-02-23 study (N8 halo)."""

from tin.core.base import BodyConfig, ConstellationConfig, HaloConfig, SatConfig

LUNAR_BODY = BodyConfig(
    name="Moon",
    radius_km=1737.4,
    mu_km3s2=4902.8,
    rotation_period_s=2.3606e6,
    elevation_mask_deg=5.0,
    j2=2.034e-4,
)


def get_lunar_constellation() -> ConstellationConfig:
    """Official TIN lunar baseline: 8 polar + ELFO + 1 EM-L2 halo.

    Study N8: 10-node constellation achieving 99.95% South Pole,
    96-98% far-side, ~99% global.
    """
    return ConstellationConfig(
        body=LUNAR_BODY,
        satellites=[
            SatConfig(
                sat_id=f"Polar-{i}",
                a_km=1737.4 + 400,
                i_deg=89.5,
                raan_deg=i * 45,
                sat_type="polar",
            )
            for i in range(8)
        ],
        relay_hubs=[
            SatConfig(
                sat_id="ELFO-HUB",
                a_km=5740,
                e=0.58,
                i_deg=56.0,
                sat_type="elfo",
            )
        ],
        halo_satellites=[
            HaloConfig(
                sat_id="EM-L2-HALO",
                z_amplitude_km=8000.0,
                period_days=14.5,
            )
        ],
        sim_days=28,
        dt_s=300.0,
    )
