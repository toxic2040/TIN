# SPDX-License-Identifier: MIT
"""tin.scenarios.mars_baseline — Mars baseline structural feasibility oracle.

6 J2-perturbed polar orbiters at 400 km altitude, 93° inclination.
Surface station at lat=0°, lon=0° (equatorial, prime meridian).
DSN contacts: intermittent, based on Earth hemisphere visibility + 12 h/day duty cycle.

Run:
    python -m tin.scenarios.mars_baseline
"""

import datetime
import json
import math

import numpy as np

from tin.core.base import BodyConfig, SatConfig
from tin.core.oracle import earliest_arrival

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT_KM_S = 299_792.458  # km/s
AU_KM = 149_597_870.7  # km per AU
EARTH_DIST_KM = 2.25 * AU_KM  # Mars–Earth near-opposition average (~336.6 M km)

# ---------------------------------------------------------------------------
# Mars body configuration
# ---------------------------------------------------------------------------
MARS = BodyConfig(
    name="Mars",
    radius_km=3389.5,
    mu_km3s2=42828.37,
    rotation_period_s=88775.0,  # 1 Mars sidereal sol
    j2=0.00196045,
    elevation_mask_deg=5.0,
)

T_SOL = MARS.rotation_period_s  # simulation horizon, seconds

# ---------------------------------------------------------------------------
# Constellation — 6 polar orbiters
# ---------------------------------------------------------------------------
_ALT_KM = 400.0
_A_KM = MARS.radius_km + _ALT_KM  # 3789.5 km

ORBITERS = [
    SatConfig(
        sat_id=f"MRO-{i + 1}",
        a_km=_A_KM,
        i_deg=93.0,
        raan_deg=30.0 * i,  # evenly spaced: 0, 30, 60, 90, 120, 150 deg
        e=0.0,
        aop_deg=0.0,
        ta_deg=0.0,
        sat_type="areo",
    )
    for i in range(6)
]

# ---------------------------------------------------------------------------
# J2-perturbed Keplerian propagator  (secular terms only)
# ---------------------------------------------------------------------------


def _sat_pos(sat: SatConfig, t_s: float, body: BodyConfig) -> np.ndarray:
    """MCI position vector (km) at time t_s.

    Secular J2 drift applied to RAAN and AoP; circular orbit (e=0) assumed.
    Uses argument of latitude u = AoP(t) + ta(t) for compact evaluation.
    """
    a = sat.a_km
    e = sat.e
    i = math.radians(sat.i_deg)
    R = body.radius_km
    mu = body.mu_km3s2
    J2 = body.j2

    n = math.sqrt(mu / a**3)  # mean motion, rad/s
    fac = 1.5 * n * J2 * (R / a) ** 2  # shared prefactor
    denom = (1.0 - e**2) ** 2  # = 1 for circular

    raan_dot = -fac * math.cos(i) / denom
    aop_dot = fac * (2.0 - 2.5 * math.sin(i) ** 2) / denom

    raan = math.radians(sat.raan_deg) + raan_dot * t_s
    aop = math.radians(sat.aop_deg) + aop_dot * t_s
    ta = math.radians(sat.ta_deg) + n * t_s  # circular: ta = M
    u = aop + ta  # argument of latitude

    # Standard perifocal-to-inertial rotation; r = a for e=0
    ci, si = math.cos(i), math.sin(i)
    cos_r, sin_r = math.cos(raan), math.sin(raan)
    cos_u, sin_u = math.cos(u), math.sin(u)

    x = a * (cos_r * cos_u - sin_r * sin_u * ci)
    y = a * (sin_r * cos_u + cos_r * sin_u * ci)
    z = a * (sin_u * si)

    return np.array([x, y, z])


def _station_pos(lat_deg: float, lon_deg: float, t_s: float, body: BodyConfig) -> np.ndarray:
    """MCI position vector (km) of a body-fixed surface station at time t_s.

    Mars rotates about the z-axis with its sidereal period.
    """
    omega = 2.0 * math.pi / body.rotation_period_s
    lon_i = math.radians(lon_deg) + omega * t_s  # inertial longitude
    lat = math.radians(lat_deg)
    R = body.radius_km

    return np.array(
        [
            R * math.cos(lat) * math.cos(lon_i),
            R * math.cos(lat) * math.sin(lon_i),
            R * math.sin(lat),
        ]
    )


def _elevation_deg(r_gs: np.ndarray, r_sat: np.ndarray) -> float:
    """Elevation angle (degrees) of r_sat above the local horizon at r_gs.

    el = arcsin( dot(r_sat - r_gs, r_gs_hat) / |r_sat - r_gs| )
    """
    v = r_sat - r_gs
    v_mag = np.linalg.norm(v)
    gs_mag = np.linalg.norm(r_gs)
    if v_mag < 1e-9 or gs_mag < 1e-9:
        return -90.0
    sin_el = float(np.clip(float(np.dot(v, r_gs)) / (v_mag * gs_mag), -1.0, 1.0))
    return math.degrees(math.asin(sin_el))


def _earth_visibility_arr(
    r_sat_arr: np.ndarray,
    mars_radius_km: float,
) -> tuple:
    """Vectorised Earth visibility check for a (N, 3) array of MCI positions.

    Earth is fixed at [EARTH_DIST_KM, 0, 0].

    Conditions:
      1. Hemisphere: x_sat > 0  (orbiter on Earth-facing side)
      2. No occultation: LOS from orbiter to Earth does not pass inside Mars

    Returns
    -------
    visible  : (N,) bool array
    range_km : (N,) float array  (distance to Earth; meaningful only where visible)
    """
    r_earth = np.array([EARTH_DIST_KM, 0.0, 0.0])

    # --- Condition 1: hemisphere ---
    hemi = r_sat_arr[:, 0] > 0.0

    # --- Condition 2: LOS occultation ---
    # p(t) = r_sat + t*(r_earth - r_sat),  t in [0, 1]
    # Closest approach to Mars centre (origin): t* = -r_sat·D / |D|²
    D = r_earth - r_sat_arr  # (N, 3)
    D_sq = np.einsum("ij,ij->i", D, D)  # |D|²  (N,)
    t_star = -np.einsum("ij,ij->i", r_sat_arr, D) / D_sq  # (N,)
    # Closest point (clamped to segment)
    tc = np.clip(t_star, 0.0, 1.0)[:, np.newaxis]
    closest = r_sat_arr + tc * D
    d_mars = np.linalg.norm(closest, axis=1)
    occluded = (t_star > 0.0) & (t_star < 1.0) & (d_mars < mars_radius_km)

    visible = hemi & ~occluded
    range_km = np.linalg.norm(r_earth - r_sat_arr, axis=1)

    return visible, range_km


# ---------------------------------------------------------------------------
# Contact window extraction
# ---------------------------------------------------------------------------


def build_contacts(
    station_lat: float,
    station_lon: float,
    orbiters: list,
    body: BodyConfig,
    t_end_s: float,
    dt_s: float = 60.0,
    dsn_on_s: float = 43200.0,
    dsn_cycle_s: float = 86400.0,
) -> list:
    """Extract surface↔orbiter and orbiter→Earth contact windows.

    Returns a list of contact dicts compatible with tin.core.oracle:
        from_node, to_node, start_s, duration_s, latency_s

    Surface contacts are bidirectional; latency computed from midpoint range.
    Single-sample visibility flickers (duration=0) are discarded.

    DSN contacts (orbiter→Earth) are intermittent:
      - Orbiter must be on Earth-facing hemisphere (x_sat > 0) with clear LOS
      - DSN complex availability: (t % dsn_cycle_s) < dsn_on_s
      - Latency = range_to_Earth / c  (~1123 s at 2.25 AU)
    """
    mask_deg = body.elevation_mask_deg
    times = np.arange(0.0, t_end_s, dt_s)
    n_t = len(times)
    contacts = []

    for sat in orbiters:
        # Vectorised position arrays over the full horizon
        r_gs_arr = np.array([_station_pos(station_lat, station_lon, float(t), body) for t in times])
        r_sat_arr = np.array([_sat_pos(sat, float(t), body) for t in times])

        # Elevation at every timestep
        v_arr = r_sat_arr - r_gs_arr
        v_mag = np.linalg.norm(v_arr, axis=1)
        gs_mag = np.linalg.norm(r_gs_arr, axis=1)
        dots = np.einsum("ij,ij->i", v_arr, r_gs_arr)
        sin_el = np.clip(dots / (v_mag * gs_mag), -1.0, 1.0)
        el_arr = np.degrees(np.arcsin(sin_el))
        visible = el_arr >= mask_deg

        # Scan for contact windows with a two-pointer walk
        i = 0
        while i < n_t:
            if visible[i]:
                i_start = i
                while i < n_t and visible[i]:
                    i += 1
                i_end = i  # one past last visible index

                if i_end - i_start >= 2:  # skip single-sample flickers
                    t_s = float(times[i_start])
                    t_e = float(times[i_end - 1])
                    duration = t_e - t_s
                    i_mid = (i_start + i_end - 1) // 2
                    range_km = float(np.linalg.norm(r_sat_arr[i_mid] - r_gs_arr[i_mid]))
                    latency = range_km / C_LIGHT_KM_S

                    for fn, tn in [("surface", sat.sat_id), (sat.sat_id, "surface")]:
                        contacts.append(
                            {
                                "from_node": fn,
                                "to_node": tn,
                                "start_s": t_s,
                                "duration_s": duration,
                                "latency_s": latency,
                            }
                        )
            else:
                i += 1

        # --- DSN contact windows: orbiter → Earth ---
        # Earth hemisphere visibility + DSN duty cycle (12 h on / 12 h off)
        earth_vis, range_arr = _earth_visibility_arr(r_sat_arr, body.radius_km)
        dsn_avail = (times % dsn_cycle_s) < dsn_on_s
        dsn_visible = earth_vis & dsn_avail

        i = 0
        while i < n_t:
            if dsn_visible[i]:
                i_start = i
                while i < n_t and dsn_visible[i]:
                    i += 1
                i_end = i
                if i_end - i_start >= 2:
                    t_s = float(times[i_start])
                    t_e = float(times[i_end - 1])
                    duration = t_e - t_s
                    i_mid = (i_start + i_end - 1) // 2
                    latency = float(range_arr[i_mid]) / C_LIGHT_KM_S
                    contacts.append(
                        {
                            "from_node": sat.sat_id,
                            "to_node": "earth",
                            "start_s": t_s,
                            "duration_s": duration,
                            "latency_s": latency,
                        }
                    )
            else:
                i += 1

    return contacts


# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------


def run_config(
    name: str,
    active_orbiters: list,
    station_lat: float = 0.0,
    station_lon: float = 0.0,
    station_label: str = "Equatorial",
    dsn_on_s: float = 43200.0,
    dsn_cycle_s: float = 86400.0,
) -> dict:
    """Build contact plan and run oracle sweep for a given orbiter set and station.

    Returns a result dict:
        name, station, n_sats, S_T_full, n_surf, n_dsn, n_inject, n_gaps
    """
    contacts = build_contacts(
        station_lat=station_lat,
        station_lon=station_lon,
        orbiters=active_orbiters,
        body=MARS,
        t_end_s=T_SOL,
        dt_s=60.0,
        dsn_on_s=dsn_on_s,
        dsn_cycle_s=dsn_cycle_s,
    )

    n_surf = sum(1 for c in contacts if c["from_node"] == "surface" or c["to_node"] == "surface")
    n_dsn = sum(1 for c in contacts if c["to_node"] == "earth")

    oracle_dt = 300.0
    t_injections = np.arange(0.0, T_SOL, oracle_dt)
    n_inject = len(t_injections)

    n_reachable = sum(
        earliest_arrival("surface", "earth", float(tk), contacts)[0] for tk in t_injections
    )

    return {
        "name": name,
        "station": station_label,
        "n_sats": len(active_orbiters),
        "S_T_full": n_reachable / n_inject,
        "n_surf": n_surf,
        "n_dsn": n_dsn,
        "n_inject": n_inject,
        "n_gaps": n_inject - n_reachable,
    }


# ---------------------------------------------------------------------------
# DSN duty-cycle sweep
# ---------------------------------------------------------------------------


def run_dsn_sweep(
    station_lat: float,
    station_lon: float,
    orbiters: list,
    dsn_hours_range=None,
) -> list:
    """Sweep DSN on-time (hours per Mars sol) and return oracle results.

    DSN available when  (t_s % T_SOL) < (dsn_hours * 3600).
    Default range: 2, 4, 6, … 22 h/sol.

    Returns a list of dicts with keys:
        dsn_hours, S_T_full, n_gaps, n_surf, n_dsn
    """
    if dsn_hours_range is None:
        dsn_hours_range = np.arange(2, 24, 2)  # 2, 4, …, 22

    rows = []
    for dsn_h in dsn_hours_range:
        r = run_config(
            name="Mars-C",
            active_orbiters=orbiters,
            station_lat=station_lat,
            station_lon=station_lon,
            dsn_on_s=float(dsn_h) * 3600.0,
            dsn_cycle_s=T_SOL,  # one Mars sol
        )
        rows.append(
            {
                "dsn_hours": float(dsn_h),
                "S_T_full": r["S_T_full"],
                "n_gaps": r["n_gaps"],
                "n_surf": r["n_surf"],
                "n_dsn": r["n_dsn"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Results serialiser
# ---------------------------------------------------------------------------


def save_dsn_sweep(
    sweep_rows: list,
    station_lat: float,
    num_orbiters: int,
    path: str = "mars_oracle_sweep.json",
) -> None:
    """Write DSN sweep results to *path* as a JSON array.

    Each element contains:
        dsn_hours, s_full, gaps, dsn_contacts, surface_contacts,
        station_lat, num_orbiters, timestamp  (ISO-8601 UTC)
    """
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [
        {
            "dsn_hours": row["dsn_hours"],
            "s_full": row["S_T_full"],
            "gaps": row["n_gaps"],
            "dsn_contacts": row["n_dsn"],
            "surface_contacts": row["n_surf"],
            "station_lat": station_lat,
            "num_orbiters": num_orbiters,
            "timestamp": ts,
        }
        for row in sweep_rows
    ]
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"  Saved {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Main — six-configuration sweep (3 orbiter sets × 2 stations)
# ---------------------------------------------------------------------------


def main() -> None:
    print("Mars Baseline — Structural Feasibility Oracle Sweep  [realistic DSN]")
    print(f"  Horizon : {T_SOL:.0f} s (1 Mars sol)  |  Propagation dt=60 s  |  Oracle dt=300 s")
    print()

    # ---- orbiter configurations ----
    mars_a = [s for s in ORBITERS if s.raan_deg in (0.0, 90.0)]  # 2 sats
    mars_b = [
        SatConfig(
            sat_id=f"MRO-B{i + 1}",
            a_km=_A_KM,
            i_deg=93.0,
            raan_deg=raan,
            e=0.0,
            aop_deg=0.0,
            ta_deg=0.0,
            sat_type="areo",
        )
        for i, raan in enumerate([0.0, 45.0, 90.0, 135.0])
    ]  # 4 sats
    mars_c = ORBITERS  # 6 sats

    orb_configs = [
        ("Mars-A", mars_a),
        ("Mars-B", mars_b),
        ("Mars-C", mars_c),
    ]

    # ---- station configurations ----
    stations = [
        ("Stn-A", 0.0, 0.0, "Equatorial  (lat=0°)"),
        ("Stn-B", 70.0, 0.0, "High-Lat    (lat=70°N)"),
    ]

    # ---- run all six combinations ----
    print("  Running …")
    results = []
    for stn_tag, stn_lat, stn_lon, stn_label in stations:
        for orb_name, orbiters in orb_configs:
            tag = f"{orb_name}/{stn_tag}"
            print(f"    {tag:<12}  {stn_label:<26}  {len(orbiters)} sats …", end=" ", flush=True)
            r = run_config(
                name=orb_name,
                active_orbiters=orbiters,
                station_lat=stn_lat,
                station_lon=stn_lon,
                station_label=stn_label,
            )
            results.append(r)
            print(f"S_T_full = {r['S_T_full']:.4f}")

    # ---- results table ----
    C = dict(cfg=7, stn=27, sats=5, st=8, surf=6, dsn=5, gaps=5)
    hdr = (
        f"  {'Config':<{C['cfg']}}  {'Station':<{C['stn']}}"
        f"  {'Sats':>{C['sats']}}  {'S_T_full':>{C['st']}}"
        f"  {'S.Cnt':>{C['surf']}}  {'D.Cnt':>{C['dsn']}}"
        f"  {'Gaps':>{C['gaps']}}"
    )
    bar = "  " + "─" * (len(hdr) - 2)

    print()
    print(hdr)
    print(bar)
    for r in results:
        print(
            f"  {r['name']:<{C['cfg']}}  {r['station']:<{C['stn']}}"
            f"  {r['n_sats']:>{C['sats']}}"
            f"  {r['S_T_full']:>{C['st']}.4f}"
            f"  {r['n_surf']:>{C['surf']}}"
            f"  {r['n_dsn']:>{C['dsn']}}"
            f"  {r['n_gaps']:>{C['gaps']}}"
        )
        if r["station"].startswith("Equatorial") and r["name"] == "Mars-C":
            print(bar)  # separator between station blocks
    print(bar)
    print()
    print("  S.Cnt = surface↔orbiter contact windows  |  D.Cnt = orbiter→Earth DSN windows")

    # ---- DSN duty-cycle sweep: High-Lat station, 6 orbiters ----
    print()
    print("DSN Duty-Cycle Sweep — Station B (lat=70°N), 6 orbiters (Mars-C)")
    print(f"  Cycle = 1 Mars sol ({T_SOL:.0f} s)  |  DSN on when (t % T_SOL) < (DSN_h × 3600)")
    print()

    sweep_results = run_dsn_sweep(
        station_lat=70.0,
        station_lon=0.0,
        orbiters=mars_c,
    )

    SW = dict(h=12, st=8, gaps=5, surf=6, dsn=5)
    shdr = (
        f"  {'DSN Hours/Sol':>{SW['h']}}  {'S_T_full':>{SW['st']}}"
        f"  {'Gaps':>{SW['gaps']}}  {'S.Cnt':>{SW['surf']}}"
        f"  {'D.Cnt':>{SW['dsn']}}"
    )
    sbar = "  " + "─" * (len(shdr) - 2)

    print(shdr)
    print(sbar)
    for row in sweep_results:
        print(
            f"  {row['dsn_hours']:>{SW['h']}.0f}"
            f"  {row['S_T_full']:>{SW['st']}.4f}"
            f"  {row['n_gaps']:>{SW['gaps']}}"
            f"  {row['n_surf']:>{SW['surf']}}"
            f"  {row['n_dsn']:>{SW['dsn']}}"
        )
    print(sbar)
    print()
    print(
        "  S.Cnt = surface↔orbiter windows (unchanged)  |  "
        "D.Cnt = orbiter→Earth windows (scales with DSN on-time)"
    )
    print()
    save_dsn_sweep(sweep_results, station_lat=70.0, num_orbiters=len(mars_c))


if __name__ == "__main__":
    main()
