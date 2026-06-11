#!/usr/bin/env python3
"""P2 Fig 5: N_eff collapse — phase-coverage diversity as universal predictor.

Four panels:
  A: |delta_gamma/gamma| vs mean N_eff_src (8 bodies + 4 CRAWDAD)
  B: Predictor comparison bar chart (Spearman |rho| for each candidate)
  C: N_eff_src distribution per body (strip plot showing regime separation)
  D: S_T vs N_eff_src per config (regime diagram with catastrophe threshold)
"""

import json

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tin_figure_style import BODY_COLORS, apply_style, figsize_double, save_fig

apply_style("pre")

RUNS = Path(__file__).parent

# ── Load data ────────────────────────────────────────────────────────
with open(RUNS / "neff_results.json") as f:
    neff_data = json.load(f)

with open(RUNS / "time_reversal_results.json") as f:
    tr_data = json.load(f)

with open(RUNS / "crawdad_neff.json") as f:
    crawdad_neff = json.load(f)

# ── Join neff with per-config time-reversal data ─────────────────────
pc_lookup = {}
for r in tr_data["per_config"]:
    key = (r["target"], r["n_orb"], r["epoch_day"])
    pc_lookup[key] = r

joined = []
for nr in neff_data:
    key = (nr["target"], nr["n_orb"], nr["epoch_day"])
    pc = pc_lookup.get(key)
    if pc is None:
        continue
    st = pc["forward"]["S_T"]
    lyap = pc["forward"]["lyapunov"]
    if st is None or lyap is None:
        continue
    joined.append(
        {
            "target": nr["target"],
            "neff_src": nr["neff_src"],
            "neff_all": nr["neff_all"],
            "gini": nr["gini"],
            "nc": nr["nc"],
            "S_T": st,
            "gamma": -lyap,
        }
    )

print(f"Joined: {len(joined)} / {len(neff_data)} neff records")

# ── Body-level summaries for Panel A ─────────────────────────────────
body_names = []
body_neff = []
body_asym = []

for body_info in tr_data["orbital_summary"]:
    t = body_info["target"]
    fwd_g = body_info["forward"]["gamma"]
    dg = body_info["delta_gamma"]
    if fwd_g == 0:
        continue
    neffs = [r["neff_src"] for r in neff_data if r["target"] == t]
    body_names.append(t.capitalize())
    body_neff.append(np.mean(neffs))
    body_asym.append(abs(dg / fwd_g))

# CRAWDAD summaries
crawdad_names = []
crawdad_neff_vals = []
crawdad_asym = []

for cs in tr_data["crawdad_summary"]:
    trace = cs["trace"]
    if trace not in crawdad_neff:
        continue
    fwd_g = cs["forward"]["gamma"]
    dg = cs["delta_gamma"]
    if fwd_g == 0:
        continue
    crawdad_names.append(trace)
    crawdad_neff_vals.append(crawdad_neff[trace]["neff"])
    crawdad_asym.append(abs(dg / fwd_g))

# ── Correlations ─────────────────────────────────────────────────────
# Body-level Spearman (n=8)
rho_body, p_body = stats.spearmanr(body_neff, body_asym)
print(f"Body-level: rho={rho_body:.3f}, p={p_body:.4f}")

# Including CRAWDAD (n=12)
all_neff_a = body_neff + crawdad_neff_vals
all_asym_a = body_asym + crawdad_asym
rho_all, p_all = stats.spearmanr(all_neff_a, all_asym_a)
print(f"With CRAWDAD: rho={rho_all:.3f}, p={p_all:.4f}")

# ── Predictor comparison (Panel B) ───────────────────────────────────
predictor_names = [
    "$N_{\\mathrm{eff,src}}$",
    "$N_{\\mathrm{eff,all}}$",
    "Gini",
    "$N_c$",
]
predictor_keys = ["neff_src", "neff_all", "gini", "nc"]

st_vals = np.array([r["S_T"] for r in joined])
rho_bars = []
for key in predictor_keys:
    vals = np.array([r[key] for r in joined])
    rho, _ = stats.spearmanr(vals, st_vals)
    rho_bars.append(abs(rho))
    print(f"  rho(S_T, {key}) = {rho:.3f}")

# ── Figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=figsize_double("pre", height_ratio=0.65))
ax_a, ax_b, ax_c, ax_d = axes.flat

# ── Panel A: Asymmetry vs N_eff_src ──────────────────────────────────
for name, nv, av in zip(body_names, body_neff, body_asym):
    color = BODY_COLORS.get(name, "#888888")
    ax_a.scatter(
        nv, av, s=40, color=color, zorder=3, edgecolors="white", linewidths=0.5, label=name
    )

for i, (name, nv, av) in enumerate(zip(crawdad_names, crawdad_neff_vals, crawdad_asym)):
    ax_a.scatter(
        nv,
        av,
        s=30,
        color="#666666",
        marker="D",
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
        label="CRAWDAD" if i == 0 else None,
    )

ax_a.set_xlabel("$N_{\\mathrm{eff,src}}$")
ax_a.set_ylabel("$|\\Delta\\gamma / \\gamma|$")
ax_a.set_yscale("log")
ax_a.text(
    0.97,
    0.97,
    f"$\\rho = {rho_body:.3f}$\n$p = {p_body:.3f}$",
    transform=ax_a.transAxes,
    fontsize=7,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
)
ax_a.legend(fontsize=5, loc="center left", framealpha=0.88, handletextpad=0.3, borderpad=0.4)
ax_a.set_title("(a) Asymmetry vs phase diversity", fontsize=8, loc="left")

# ── Panel B: Predictor comparison ────────────────────────────────────
colors_b = ["#4477AA", "#33BBEE", "#EE7733", "#CC3311"]
ax_b.barh(range(len(predictor_names)), rho_bars, color=colors_b, edgecolor="white", linewidth=0.5)
ax_b.set_yticks(range(len(predictor_names)))
ax_b.set_yticklabels(predictor_names)
ax_b.set_xlabel("$|\\rho|$ with $S_T$")
ax_b.set_xlim(0, 1)
for i, v in enumerate(rho_bars):
    ax_b.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=6)
ax_b.set_title("(b) Predictor comparison", fontsize=8, loc="left")

# ── Panel C: N_eff_src distribution per body ─────────────────────────
body_order = ["Jupiter", "Saturn", "Ceres", "Mars", "Venus", "Europa", "Titan", "Mercury"]
rng = np.random.default_rng(42)

for i, bname in enumerate(body_order):
    target = bname.lower()
    vals = [r["neff_src"] for r in neff_data if r["target"] == target]
    if not vals:
        continue
    color = BODY_COLORS.get(bname, "#888888")
    jitter = rng.uniform(-0.2, 0.2, len(vals))
    ax_c.scatter(
        vals,
        np.full(len(vals), i) + jitter,
        s=8,
        color=color,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )

# CRAWDAD at bottom
crawdad_all_neff = [crawdad_neff[k]["neff"] for k in crawdad_neff]
jitter_c = rng.uniform(-0.15, 0.15, len(crawdad_all_neff))
ax_c.scatter(
    crawdad_all_neff,
    np.full(len(crawdad_all_neff), -1) + jitter_c,
    s=15,
    color="#666666",
    marker="D",
    edgecolors="white",
    linewidths=0.3,
)

# Threshold line
ax_c.axvline(5, color="#CC3311", ls="--", lw=0.8, alpha=0.7, zorder=1)
ax_c.text(6, -1, "$N_{\\mathrm{eff}} = 5$", fontsize=6, color="#CC3311")

ytick_labels = ["CRAWDAD"] + body_order
ytick_pos = [-1] + list(range(len(body_order)))
ax_c.set_yticks(ytick_pos)
ax_c.set_yticklabels(ytick_labels, fontsize=6)
ax_c.set_xlabel("$N_{\\mathrm{eff,src}}$")
ax_c.set_title("(c) Phase diversity by body", fontsize=8, loc="left")

# ── Panel D: Regime diagram (S_T vs N_eff_src) ──────────────────────
for target in sorted(set(r["target"] for r in joined)):
    recs = [r for r in joined if r["target"] == target]
    neffs = [r["neff_src"] for r in recs]
    sts = [r["S_T"] for r in recs]
    bname = target.capitalize()
    color = BODY_COLORS.get(bname, "#888888")
    ax_d.scatter(
        neffs, sts, s=10, color=color, alpha=0.5, edgecolors="none", label=bname, rasterized=True
    )

# Thresholds
ax_d.axvline(5, color="#CC3311", ls="--", lw=0.8, alpha=0.7)
ax_d.axhline(0.5, color="#888888", ls=":", lw=0.6, alpha=0.5)
ax_d.text(6, 0.08, "Catastrophe\nthreshold", fontsize=6, color="#CC3311")

ax_d.set_xlabel("$N_{\\mathrm{eff,src}}$")
ax_d.set_ylabel("$S_T$")
ax_d.set_ylim(-0.05, 1.05)
ax_d.legend(
    fontsize=5, loc="lower right", framealpha=0.88, ncol=2, handletextpad=0.3, columnspacing=0.6
)
ax_d.set_title("(d) Regime diagram", fontsize=8, loc="left")

# ── Save ─────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.5)
save_fig(fig, "p2_neff_collapse")
plt.close(fig)
