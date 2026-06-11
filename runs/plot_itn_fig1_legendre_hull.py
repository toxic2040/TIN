#!/usr/bin/env python3
"""ITN Fig 1: Commodity-dependent path selection via convex hull.

The (T, -log Q) plane, the lower convex hull, and supporting lines at
different commodity hazard rates lambda.  Shows how different cargoes
"see" different effective networks through the affine parametric
shortest-path structure (Whitepaper Section 3.7).

    Cost_k(lambda) = -ln Q_k  +  lambda * T_k

Each commodity hazard lambda defines a supporting line of slope -lambda
to the lower hull.  The tangent point is the optimal path for that
commodity.  Paths above the hull are never optimal for any commodity.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tin_figure_style import apply_style, figsize_single, save_fig

apply_style("pre")

# ── Path families (EMJ-inspired, embellished for pedagogy) ─────────
# (name, T_days total exposure, Q hardware reliability product)
PATHS = [
    ("Direct chemical", 400, 0.65),  # A — fast, risky
    ("Cycler + relay", 750, 0.83),  # B — moderate
    ("Full relay chain", 1300, 0.92),  # C — long, safe
    ("Slow safe relay", 1900, 0.965),  # D — very safe
    ("Congested bypass", 550, 0.52),  # E — dominated
    ("Detour via Venus", 1100, 0.74),  # F — dominated
]

labels = list("ABCDEF")
T = np.array([p[1] for p in PATHS], dtype=float)
Q = np.array([p[2] for p in PATHS], dtype=float)
nLQ = -np.log(Q)
names = [p[0] for p in PATHS]

# ── Lower convex hull (Andrew's monotone chain) ───────────────────
order = np.argsort(T)
hull = []
for i in order:
    while len(hull) >= 2:
        x0, y0 = T[hull[-2]], nLQ[hull[-2]]
        x1, y1 = T[hull[-1]], nLQ[hull[-1]]
        cross = (x1 - x0) * (nLQ[i] - y0) - (y1 - y0) * (T[i] - x0)
        if cross >= 0:
            break
        hull.pop()
    hull.append(i)

on_hull = set(hull)
hull_T = T[hull]
hull_nLQ = nLQ[hull]

# ── Crossover lambda values between hull facets ────────────────────
crossovers = []
for k in range(len(hull) - 1):
    i0, i1 = hull[k], hull[k + 1]
    lam = (nLQ[i0] - nLQ[i1]) / (T[i1] - T[i0])
    tau = np.log(2) / lam
    crossovers.append((lam, tau, k))

# ── Three commodity hazard rates ───────────────────────────────────
commodities = [
    (0.0015, "Cryogenic LH$_2$", "#CC3311"),
    (0.00040, "Perishable supplies", "#EE7733"),
    (0.00005, "Durable hardware", "#4477AA"),
]

# ── Figure ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=figsize_single("pre", height_ratio=1.05))

T_line = np.linspace(50, 2400, 600)

# Dominated region shading
hull_interp = np.interp(T_line, hull_T, hull_nLQ)
mask = (T_line >= hull_T[0]) & (T_line <= hull_T[-1])
ax.fill_between(T_line[mask], hull_interp[mask], 0.82, alpha=0.04, color="#888888", zorder=0)
ax.text(900, 0.58, "\\textit{dominated}", fontsize=6.5, color="#BBBBBB", ha="center")

# Hull line
ax.plot(hull_T, hull_nLQ, "-", color="#333333", lw=1.8, zorder=3)

# ── Supporting lines ───────────────────────────────────────────────
for lam, comm_label, color in commodities:
    # Optimal path at this lambda
    costs = nLQ + lam * T
    opt = int(np.argmin(costs))
    C_opt = costs[opt]

    # Supporting line: -logQ = C_opt - lambda * T
    y_line = C_opt - lam * T_line
    vis = (y_line >= -0.10) & (y_line <= 0.82)
    ax.plot(T_line[vis], y_line[vis], "--", color=color, lw=0.7, alpha=0.55, zorder=2)

    # Tangent-point halo
    ax.plot(T[opt], nLQ[opt], "o", color=color, markersize=11, alpha=0.18, zorder=5)

# ── Path markers ──────────────────────────────────────────────────
for i in range(len(PATHS)):
    if i in on_hull:
        ax.plot(
            T[i],
            nLQ[i],
            "s",
            color="#4477AA",
            markeredgecolor="#222222",
            markeredgewidth=0.7,
            markersize=5.5,
            zorder=7,
        )
    else:
        ax.plot(
            T[i],
            nLQ[i],
            "o",
            color="#CCCCCC",
            markeredgecolor="#999999",
            markeredgewidth=0.6,
            markersize=4.5,
            zorder=4,
        )

# ── Path labels ────────────────────────────────────────────────────
# Manual offsets: (dx, dy, ha, va) tuned for readability
offsets = {
    "A": (45, 0.020, "left", "bottom"),
    "B": (45, 0.012, "left", "bottom"),
    "C": (45, 0.012, "left", "bottom"),
    "D": (-45, 0.012, "right", "bottom"),
    "E": (45, -0.010, "left", "top"),
    "F": (45, 0.012, "left", "bottom"),
}
for i, lab in enumerate(labels):
    dx, dy, ha, va = offsets[lab]
    c = "#333333" if i in on_hull else "#AAAAAA"
    ax.text(
        T[i] + dx,
        nLQ[i] + dy,
        "\\textbf{%s} %s" % (lab, names[i]),
        fontsize=4.8,
        color=c,
        ha=ha,
        va=va,
    )

# ── Crossover annotations on hull ─────────────────────────────────
# Offset each crossover label to avoid overlapping path labels
cross_offsets = [
    (-120, 0.09),  # A→B crossover: shift left and up
    (120, 0.07),  # B→C crossover: shift right and up
    (100, 0.055),  # C→D crossover: shift right and up
]
for idx, (lam_c, tau_c, k) in enumerate(crossovers):
    mid_T = (hull_T[k] + hull_T[k + 1]) / 2
    mid_y = (hull_nLQ[k] + hull_nLQ[k + 1]) / 2
    if tau_c < 400:
        s = "$\\tau^*_{1/2}$=%d\\,d" % int(tau_c)
    elif tau_c < 3650:
        s = "$\\tau^*_{1/2}$=%.1f\\,yr" % (tau_c / 365.25)
    else:
        s = "$\\tau^*_{1/2}$=%.0f\\,yr" % (tau_c / 365.25)
    dx, dy = cross_offsets[idx] if idx < len(cross_offsets) else (40, 0.065)
    ax.annotate(
        s,
        xy=(mid_T, mid_y),
        xytext=(mid_T + dx, mid_y + dy),
        fontsize=4.5,
        color="#888888",
        ha="center",
        arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.4, shrinkB=2),
    )

# ── Commodity labels (near tangent points) ─────────────────────────
# Cryogenic → tangent at A
ax.text(
    250,
    0.50,
    "Cryogenic\nLH$_2$",
    fontsize=6,
    color="#CC3311",
    ha="center",
    va="bottom",
    linespacing=1.1,
)
# Perishable → tangent at B
ax.text(
    920,
    0.24,
    "Perishable\nsupplies",
    fontsize=6,
    color="#EE7733",
    ha="center",
    va="bottom",
    linespacing=1.1,
)
# Hardware → tangent at D
ax.text(
    2050,
    0.085,
    "Durable\nhardware",
    fontsize=6,
    color="#4477AA",
    ha="center",
    va="bottom",
    linespacing=1.1,
)

# ── Slope indicator ────────────────────────────────────────────────
ax.annotate(
    "",
    xy=(170, 0.10),
    xytext=(170, 0.38),
    arrowprops=dict(arrowstyle="<-", color="#AAAAAA", lw=0.5),
)
ax.text(
    135,
    0.24,
    "slope $= -\\lambda_c$",
    fontsize=5.5,
    color="#AAAAAA",
    rotation=90,
    ha="center",
    va="center",
)

# ── Axes ───────────────────────────────────────────────────────────
ax.set_xlabel("Total exposure time $T_k$ (days)")
ax.set_ylabel("Path cost $-\\!\\ln Q_k$")
ax.set_xlim(80, 2300)
ax.set_ylim(-0.08, 0.82)

save_fig(fig, "itn_legendre_hull")
plt.close(fig)
print("Done.")
