"""Generate five figures retained from the historical working-paper pipeline.

Outputs (all to ../figures/):
  fig5_contact_density_bound.pdf  — Contact-Density Bound (ergodic vs non-ergodic)
  fig6_mars_dsn_sweep.pdf         — Mars DSN sweep: S_T mean + min/max band
  fig7_mars_ttl_sweep.pdf         — Mars TTL sweep: single vs dual station DR
  fig8_sol_phase.pdf              — Sol phase structure: bar chart of per-sol S_T
  fig9_architecture.pdf           — Architecture comparison: orbiter count + station lat
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "figure.figsize": (7, 4),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

_HERE = Path(__file__).parent
_FIGS = _HERE.parent / "figures"
_FIGS.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────


def _load(name):
    p = _HERE / name
    if not p.exists():
        print(f"ERROR: {p} not found")
        sys.exit(1)
    with open(p) as fh:
        return json.load(fh)


conj = _load("conjecture_figure_data.json")
erg = _load("ergodic_results.json")
ttl = _load("ttl_sweep_results.json")
arch = _load("architecture_sweep_results.json")

print("All data loaded.")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Contact-Density Bound
# ══════════════════════════════════════════════════════════════════════════════

thinning = conj["thinning"]
deletion = conj["deletion"]

c_thin = np.array([p["c"] for p in thinning])
dr_thin = np.array([p["dr"] for p in thinning])
n_thin = np.array([p.get("n_polars", 0) for p in thinning])

c_del = np.array([p["c"] for p in deletion])
dr_del = np.array([p["dr"] for p in deletion])
dr_del_std = np.array([p["dr_std"] for p in deletion])

# sort thinning by C for the bound line
idx_t = np.argsort(c_thin)
c_thin_s = c_thin[idx_t]
dr_thin_s = dr_thin[idx_t]

fig, ax = plt.subplots()

# Non-ergodic: deletion (red triangles, error bars)
ax.errorbar(
    c_del,
    dr_del,
    yerr=dr_del_std,
    fmt="^",
    color="#d62728",
    markersize=7,
    zorder=3,
    ecolor="#d62728",
    elinewidth=1.0,
    capsize=3,
    label="Deletion sweep (non-ergodic)",
)
ax.plot(c_del, dr_del, color="#d62728", linewidth=0.8, alpha=0.4, linestyle=":", zorder=2)

# Ergodic: thinning (blue circles)
ax.scatter(
    c_thin, dr_thin, marker="o", s=60, color="#1f77b4", zorder=5, label="Thinning sweep (ergodic)"
)

# Dashed black ergodic bound f(C) connecting thinning points
ax.plot(
    c_thin_s,
    dr_thin_s,
    color="black",
    linewidth=1.4,
    linestyle="--",
    zorder=4,
    label="Ergodic bound $f(C)$",
)

ax.set_xlabel("Contact hours/day $C$")
ax.set_ylabel(r"Delivery ratio $\mathrm{DR}$")
ax.set_xlim(0, 105)
ax.set_ylim(0.35, 1.08)
ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.2, alpha=0.2)
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
ax.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

fig.savefig(_FIGS / "fig5_contact_density_bound.pdf")
plt.close(fig)
print("  fig5_contact_density_bound.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Mars DSN Sweep  (exp4_dsn_per_sol: min/max band + mean line)
# ══════════════════════════════════════════════════════════════════════════════

dsn_rows = erg["exp4_dsn_per_sol"]

dsn_h = np.array([r["dsn_hours"] for r in dsn_rows])
st_mean = np.array([r["S_T_full"] for r in dsn_rows])
st_min = np.array([r["S_T_min"] for r in dsn_rows])
st_max = np.array([r["S_T_max"] for r in dsn_rows])

fig, ax = plt.subplots()

ax.fill_between(dsn_h, st_min, st_max, alpha=0.20, color="#1f77b4", label="$S_T$ min/max band")
ax.plot(dsn_h, st_mean, color="#1f77b4", linewidth=1.8, label=r"$\bar{S}_T$ (mean over sols)")
ax.scatter(dsn_h, st_mean, s=40, color="#1f77b4", zorder=4)

ax.set_xlabel("DSN hours per sol")
ax.set_ylabel(r"Structural feasibility $S_T^{\mathrm{full}}$")
ax.set_xlim(0, dsn_h.max() + 1)
ax.set_ylim(0.88, 1.01)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
ax.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

fig.savefig(_FIGS / "fig6_mars_dsn_sweep.pdf")
plt.close(fig)
print("  fig6_mars_dsn_sweep.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Mars TTL Sweep
# ══════════════════════════════════════════════════════════════════════════════

sweep = ttl["sweep"]
ttl_h = np.array([r["ttl_h"] for r in sweep])
dr_sng = np.array([r["single"]["dr"] for r in sweep])
dr_dual = np.array([r["dual"]["dr"] for r in sweep])

fig, ax = plt.subplots()

ax.plot(
    ttl_h,
    dr_sng,
    color="#1f77b4",
    linewidth=1.8,
    linestyle="-",
    label=r"Single station 70$^\circ$N",
)
ax.plot(
    ttl_h,
    dr_dual,
    color="#d62728",
    linewidth=1.8,
    linestyle="--",
    label=r"Dual stations 70$^\circ$N + 20$^\circ$N",
)
ax.scatter(ttl_h, dr_sng, s=35, color="#1f77b4", zorder=4)
ax.scatter(ttl_h, dr_dual, s=35, color="#d62728", marker="^", zorder=4)

ax.set_xlabel("Emergency TTL (hours)")
ax.set_ylabel(r"Delivery ratio $\mathrm{DR}$")
ax.set_xlim(0, ttl_h.max() + 0.5)
ax.set_ylim(0, 1.04)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
ax.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

fig.savefig(_FIGS / "fig7_mars_ttl_sweep.pdf")
plt.close(fig)
print("  fig7_mars_ttl_sweep.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Sol Phase Structure
# ══════════════════════════════════════════════════════════════════════════════

per_sol = erg["exp3_per_sol"]
sols = np.array([r["sol"] for r in per_sol], dtype=int)
st_sol = np.array([r["S_T_full"] for r in per_sol])
st_avg = st_sol.mean()

fig, ax = plt.subplots()

ax.bar(sols, st_sol, color="#1f77b4", alpha=0.80, width=0.55, edgecolor="white", linewidth=0.6)
ax.axhline(
    st_avg,
    color="black",
    linewidth=1.3,
    linestyle="--",
    label=f"Mean $\\bar{{S}}_T = {st_avg:.4f}$",
)

ax.set_xlabel("Sol number")
ax.set_ylabel(r"Structural feasibility $S_T^{\mathrm{full}}$")
ax.set_xticks(sols)
ax.set_ylim(0.92, 1.005)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.4)
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
ax.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

fig.savefig(_FIGS / "fig8_sol_phase.pdf")
plt.close(fig)
print("  fig8_sol_phase.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Architecture Comparison (two subplots)
# ══════════════════════════════════════════════════════════════════════════════

orb_rows = arch["exp1_orbiter_scaling"]
stn_rows = arch["exp2_second_station"]

n_sats = np.array([r["n_sats"] for r in orb_rows])
st_omean = np.array([r["S_T_mean"] for r in orb_rows])
st_omin = np.array([r["S_T_min"] for r in orb_rows])
st_omax = np.array([r["S_T_max"] for r in orb_rows])

b_lats = np.array([r["b_lat_deg"] for r in stn_rows])
st_comb = np.array([r["st_C_mean"] for r in stn_rows])

# sort station sweep by latitude for a smooth curve
idx2 = np.argsort(b_lats)
b_lats_s = b_lats[idx2]
st_comb_s = st_comb[idx2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left — orbiter count vs S_T
ax1.fill_between(n_sats, st_omin, st_omax, alpha=0.20, color="#1f77b4", label="$S_T$ min/max band")
ax1.plot(
    n_sats,
    st_omean,
    "o-",
    color="#1f77b4",
    linewidth=1.8,
    markersize=6,
    label=r"$\bar{S}_T$ (mean)",
)
ax1.set_xlabel("Orbiter count")
ax1.set_ylabel(r"Structural feasibility $S_T^{\mathrm{full}}$")
ax1.set_ylim(0.92, 1.005)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax1.grid(True, which="major", linewidth=0.4, alpha=0.4)
ax1.tick_params(axis="both", direction="in", top=True, right=True)
ax1.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

# Right — Station B latitude vs combined S_T
ax2.plot(
    b_lats_s,
    st_comb_s,
    "^-",
    color="#d62728",
    linewidth=1.8,
    markersize=6,
    label=r"Combined $S_T^{\mathrm{full}}$ (A $\cup$ B)",
)
ax2.set_xlabel(r"Station B latitude ($^\circ$)")
ax2.set_ylabel(r"Combined $S_T^{\mathrm{full}}$ (A $\cup$ B)")
ax2.set_ylim(0.96, 1.005)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
ax2.grid(True, which="major", linewidth=0.4, alpha=0.4)
ax2.tick_params(axis="both", direction="in", top=True, right=True)
ax2.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

fig.tight_layout(pad=1.5)
fig.savefig(_FIGS / "fig9_architecture.pdf")
plt.close(fig)
print("  fig9_architecture.pdf")

print("\nAll 5 figures generated successfully.")
