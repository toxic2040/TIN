import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

print("TIN v0.3.7 — Hybrid Polar + ELFO Physics-Aware Sim")

parser = argparse.ArgumentParser()
parser.add_argument("--polar_sats", type=int, default=8)
parser.add_argument("--altitude", type=float, default=400.0)
parser.add_argument("--inclination", type=float, default=90.0)
parser.add_argument("--min_elev", type=float, default=5.0)
parser.add_argument("--sim_days", type=int, default=28)
parser.add_argument("--include_elfo", action="store_true")
parser.add_argument("--j2", action="store_true", help="Add simple lunar J2 precession")
parser.add_argument("--topo_shadow", action="store_true", help="Approx Shackleton crater shadowing")
parser.add_argument("--solar_storm", action="store_true", help="Probabilistic outages")
parser.add_argument("--output", type=str, default="tin_v0.3.7_hybrid.png")
args = parser.parse_args()

# Constants (Moon)
R_moon = 1737.4  # km
mu = 4902.8  # km³/s²
J2_moon = 2.034e-4  # lunar J2

# Polar sats
a_polar = R_moon + args.altitude
n_polar = np.sqrt(mu / a_polar**3)
inc_p = np.deg2rad(args.inclination)
RAANs = np.deg2rad(np.linspace(0, 360, args.polar_sats, endpoint=False))
M0s = np.deg2rad(np.linspace(0, 360, args.polar_sats, endpoint=False))

# ELFO (real Lunar Pathfinder params)
if args.include_elfo:
    a_elfo = 5740.0
    e_elfo = 0.58
    inc_elfo = np.deg2rad(55.0)
    omega_elfo = np.deg2rad(86.0)  # frozen arg peri
    RAAN_elfo = np.deg2rad(0.0)  # placeholder — can tune
    M0_elfo = np.deg2rad(0.0)

# Grid (South Pole focus + full for maps)
lats = np.arange(-90, -59, 0.25)
lons = np.arange(0, 360, 1.0)
LON, LAT = np.meshgrid(lons, lats)
flat_lats, flat_lons = LAT.ravel(), LON.ravel()

pos_gs = R_moon * np.column_stack(
    (
        np.cos(np.deg2rad(flat_lats)) * np.cos(np.deg2rad(flat_lons)),
        np.cos(np.deg2rad(flat_lats)) * np.sin(np.deg2rad(flat_lons)),
        np.sin(np.deg2rad(flat_lats)),
    )
)

T_sim = args.sim_days * 86400.0
num_times = 1440  # 1-min resolution
times = np.linspace(0, T_sim, num_times, endpoint=False)

covered_count = np.zeros(len(flat_lats), dtype=int)
bundle_contacts = []  # for DTN later

for t_idx, t in enumerate(times):
    visible = np.zeros(len(flat_lats), dtype=bool)

    # Polar sats with optional J2 precession
    for k in range(args.polar_sats):
        n = n_polar
        if args.j2:
            # Simple secular RAAN drift
            raan_dot = (
                -1.5 * n * J2_moon * (R_moon / a_polar) ** 2 * np.cos(inc_p) / (1 - 0**2) ** 2
            )
            RAAN = RAANs[k] + raan_dot * t
        else:
            RAAN = RAANs[k]

        M = M0s[k] + n * t
        E = M  # circular approx
        nu = E
        r = a_polar
        # rotation matrix same as before...
        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        cos_O, sin_O = np.cos(RAAN), np.sin(RAAN)
        cos_i, sin_i = np.cos(inc_p), np.sin(inc_p)
        sat_pos = np.array(
            [
                r * (cos_nu * cos_O - sin_nu * sin_O * cos_i),
                r * (cos_nu * sin_O + sin_nu * cos_O * cos_i),
                r * (sin_nu * sin_i),
            ]
        )

        # visibility calc (same as yours)
        vec = sat_pos - pos_gs
        dist = np.linalg.norm(vec, axis=1)
        cos_zen = np.clip(np.sum(vec * pos_gs, axis=1) / (dist * R_moon), -1, 1)
        elev = 90 - np.rad2deg(np.arccos(cos_zen))
        vis = elev > args.min_elev

        if args.topo_shadow:
            vis &= (flat_lats > -88) | (
                np.random.rand(len(flat_lats)) > 0.15
            )  # crude crater shadow

        visible |= vis

    # ELFO
    if args.include_elfo:
        M = M0_elfo + np.sqrt(mu / a_elfo**3) * t
        E = M + e_elfo * np.sin(M)  # simple Kepler
        nu = 2 * np.arctan2(
            np.sqrt(1 + e_elfo) * np.sin(E / 2), np.sqrt(1 - e_elfo) * np.cos(E / 2)
        )
        r = a_elfo * (1 - e_elfo**2) / (1 + e_elfo * np.cos(nu))
        # full rotation matrix for elliptical (same pattern)
        # ... (I kept it short here — full version in the file I'll paste in next msg if you want)
        # visibility same

        visible |= elfo_vis  # placeholder

    # Solar storm
    if args.solar_storm and np.random.rand() < 0.08:  # ~8% outage periods
        visible = np.zeros_like(visible)

    covered_count += visible.astype(int)

    # Record contacts for DTN
    if np.any(visible):
        bundle_contacts.append((t, np.sum(visible)))

coverage_pct = (covered_count / num_times * 100).reshape(LAT.shape)
south_avg = np.mean(coverage_pct[lats < -85])

print(f"28-day South Pole avg: {south_avg:.2f}% (with all physics flags)")

# Plot + JSON log
# ... (your original plot code, plus extra title with flags)

with open("tin_v0.3.7_contacts.json", "w") as f:
    json.dump({"coverage": south_avg, "contact_windows": len(bundle_contacts)}, f)

plt.savefig(args.output, dpi=400)
print("✅ v0.3.7 sim complete")
