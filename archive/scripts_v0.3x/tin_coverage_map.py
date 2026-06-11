import argparse

import matplotlib.pyplot as plt
import numpy as np

print("🚀 TIN v0.3.5 – Hybrid Polar + ELFO Coverage Map CLI")

parser = argparse.ArgumentParser(description="TIN v0.3.5 Hybrid Polar + ELFO Coverage Map CLI")
parser.add_argument("--polar_sats", type=int, default=6, help="Number of polar satellites (6 or 8)")
parser.add_argument(
    "--region", choices=["south", "farside"], default="south", help="Region: south or farside"
)
parser.add_argument("--include_elfo", action="store_true", help="Include Lunar Pathfinder ELFO hub")
parser.add_argument("--altitude", type=float, default=400.0, help="Polar sats altitude km")
parser.add_argument("--inclination", type=float, default=90.0)
parser.add_argument("--min_elev", type=float, default=5.0)
parser.add_argument("--sim_days", type=int, default=28)
parser.add_argument("--output", type=str, default=None)

args = parser.parse_args()

if args.output is None:
    elfo_str = "_elfo" if args.include_elfo else ""
    args.output = f"tin_hybrid_{args.polar_sats}polar{elfo_str}_{args.region}.png"

# Constants
R_moon = 1737.4
mu = 4902.8
min_elev = args.min_elev
T_sim = args.sim_days * 86400.0
num_times = 720
times = np.linspace(0, T_sim, num_times, endpoint=False)

# Grid
if args.region == "south":
    lats = np.arange(-90.0, -59.5, 0.5)
    title_region = "South Pole"
    avg_threshold = -85
    pole_marker = True
    farside_line = False
else:
    lats = np.arange(-80.0, 81.0, 1.0)
    title_region = "Far-Side"
    avg_threshold = None
    pole_marker = False
    farside_line = True

lons = np.arange(0.0, 360.0, 2.0)
LON, LAT = np.meshgrid(lons, lats)
flat_lats = LAT.ravel()
flat_lons = LON.ravel()

pos_gs = R_moon * np.column_stack(
    (
        np.cos(np.deg2rad(flat_lats)) * np.cos(np.deg2rad(flat_lons)),
        np.cos(np.deg2rad(flat_lats)) * np.sin(np.deg2rad(flat_lons)),
        np.sin(np.deg2rad(flat_lats)),
    )
)

covered_count = np.zeros(len(flat_lats), dtype=int)

# Polar sats
a = R_moon + args.altitude
n = np.sqrt(mu / a**3)
inc = np.deg2rad(args.inclination)
e = 0.0
RAANs = np.deg2rad(np.linspace(0, 360, args.polar_sats, endpoint=False))
M0s = np.deg2rad(np.linspace(0, 360, args.polar_sats, endpoint=False))

for t in times:
    visible_any = np.zeros(len(flat_lats), dtype=bool)
    # Polar sats
    for k in range(args.polar_sats):
        M = M0s[k] + n * t
        E = M + e * np.sin(M)
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        cos_O, sin_O = np.cos(RAANs[k]), np.sin(RAANs[k])
        cos_i, sin_i = np.cos(inc), np.sin(inc)
        sat_x = r * (cos_nu * cos_O - sin_nu * sin_O * cos_i)
        sat_y = r * (cos_nu * sin_O + sin_nu * cos_O * cos_i)
        sat_z = r * (sin_nu * sin_i)
        sat_pos = np.array([sat_x, sat_y, sat_z])
        vec = sat_pos - pos_gs
        dist = np.linalg.norm(vec, axis=1)
        dot_prod = np.sum(vec * pos_gs, axis=1)
        cos_zen = np.clip(dot_prod / (dist * R_moon), -1.0, 1.0)
        zen_ang = np.arccos(cos_zen)
        elev = np.rad2deg(np.pi / 2 - zen_ang)
        visible_any |= elev > min_elev
    # ELFO hub
    if args.include_elfo:
        a_elfo = 5740.0
        e_elfo = 0.58
        inc_elfo = 55.0
        omega_elfo = 86.0
        RAAN_elfo = 0.0
        n_elfo = np.sqrt(mu / a_elfo**3)
        M_elfo = n_elfo * t  # perilune timing
        E_elfo = M_elfo + e_elfo * np.sin(M_elfo)
        nu_elfo = 2 * np.arctan2(
            np.sqrt(1 + e_elfo) * np.sin(E_elfo / 2), np.sqrt(1 - e_elfo) * np.cos(E_elfo / 2)
        )
        r_elfo = a_elfo * (1 - e_elfo**2) / (1 + e_elfo * np.cos(nu_elfo))
        cos_nu, sin_nu = (
            np.cos(nu_elfo + np.deg2rad(omega_elfo)),
            np.sin(nu_elfo + np.deg2rad(omega_elfo)),
        )
        cos_O, sin_O = np.cos(RAAN_elfo), np.sin(RAAN_elfo)
        cos_i, sin_i = np.cos(np.deg2rad(inc_elfo)), np.sin(np.deg2rad(inc_elfo))
        sat_x = r_elfo * (cos_nu * cos_O - sin_nu * sin_O * cos_i)
        sat_y = r_elfo * (cos_nu * sin_O + sin_nu * cos_O * cos_i)
        sat_z = r_elfo * (sin_nu * sin_i)
        sat_pos = np.array([sat_x, sat_y, sat_z])
        vec = sat_pos - pos_gs
        dist = np.linalg.norm(vec, axis=1)
        dot_prod = np.sum(vec * pos_gs, axis=1)
        cos_zen = np.clip(dot_prod / (dist * R_moon), -1.0, 1.0)
        zen_ang = np.arccos(cos_zen)
        elev = np.rad2deg(np.pi / 2 - zen_ang)
        visible_any |= elev > min_elev
    covered_count += visible_any.astype(int)

coverage_pct = (covered_count / num_times * 100.0).reshape(LAT.shape)

# Average
if args.region == "south":
    avg = np.mean(coverage_pct[lats < avg_threshold])
    print(f" South-pole average (lat < {avg_threshold}°): {avg:.1f}%")
else:
    farside_mask = (flat_lons > 90) & (flat_lons < 270)
    avg = np.mean(coverage_pct.ravel()[farside_mask])
    print(f" Far-side average (rough 90-270° long): {avg:.1f}%")

# Plot
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.pcolormesh(LON, LAT, coverage_pct, cmap="plasma", shading="auto", vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label="Coverage Availability (%)")
ax.set_xlabel("Lunar Longitude (°E)")
ax.set_ylabel("Lunar Latitude (°N)")
elfo_str = " + Pathfinder ELFO Hub" if args.include_elfo else ""
title = f"TIN v0.3.5 Hybrid — {args.polar_sats} Polar{elfo_str}"
ax.set_title(title + f"\n{title_region} Coverage (28-day avg, elev >{min_elev}°)")
if pole_marker:
    ax.plot(0, -90, "k+", markersize=12, label="South Pole")
    ax.contour(LON, LAT, LAT, levels=[-88], colors="white", linestyles="--", linewidths=1.5)
    ax.text(5, -89.5, "Shackleton PSR", color="white", fontsize=10, fontweight="bold")
if farside_line:
    ax.axvline(175, color="white", linestyle="--", linewidth=2, label="Far-Side Center")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(args.output, dpi=300, bbox_inches="tight")
print(f"✅ Saved: {args.output}")
plt.show()
