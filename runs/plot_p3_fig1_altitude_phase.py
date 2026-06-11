#!/usr/bin/env python3
"""P3 Fig 1: Altitude phase diagram — DR vs altitude for 3 families.

Shows non-monotonic peak for polar, sun-sync, frozen elliptical.
Two-panel (Moon | Mars), NaN-filtered, peak-annotated.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tin_figure_style import FAMILY_STYLES, apply_style, figsize_double, save_fig

apply_style("ieee")

RUNS = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(RUNS, "epyc_results", "campaign_2026_03_11", "campaign_bucket_3_results.json")
) as f:
    records = json.load(f)

# Aggregate: mean DR by (body, family, alt_km), skip NaN
agg = defaultdict(list)
for r in records:
    dr = r["DR"]
    if dr is not None and not (isinstance(dr, float) and np.isnan(dr)):
        key = (r["body"], r["family"], r["alt_km"])
        agg[key].append(dr)

styles = {
    "polar": {**FAMILY_STYLES["polar"], "label": "Polar"},
    "sunsync": {**FAMILY_STYLES["sunsync"], "label": "Sun-sync"},
    "frozen_elliptical": {**FAMILY_STYLES["frozen_elliptical"], "label": "Frozen ellip."},
}

fig, axes = plt.subplots(
    1, 2, figsize=figsize_double("ieee", height_ratio=0.45), gridspec_kw={"wspace": 0.3}
)

for ax, body in zip(axes, ["Moon", "Mars"]):
    for fam, style in styles.items():
        alts, drs = [], []
        for (b, f, a), vals in sorted(agg.items()):
            if b == body and f == fam:
                mean_dr = np.mean(vals)
                if not np.isnan(mean_dr) and mean_dr > 0:
                    alts.append(a)
                    drs.append(mean_dr)
        if not alts:
            continue

        ax.plot(
            alts,
            drs,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            markersize=4.5,
            lw=1.3,
            alpha=0.85,
        )

        # Annotate peak
        peak_idx = int(np.argmax(drs))
        peak_alt = alts[peak_idx]
        peak_dr = drs[peak_idx]
        ax.plot(
            peak_alt,
            peak_dr,
            marker=style["marker"],
            color=style["color"],
            markersize=9,
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        # Label with altitude and DR — offset varies by family to avoid overlap
        if peak_alt < 500:
            alt_str = f"{peak_alt:.0f} km"
        else:
            alt_str = f"{peak_alt / 1000:.1f}k km"
        offsets = {"polar": (12, 14), "sunsync": (12, -20), "frozen_elliptical": (-45, -22)}
        ofs = offsets.get(fam, (12, 8))
        ax.annotate(
            f"{alt_str}\nDR={peak_dr:.3f}",
            xy=(peak_alt, peak_dr),
            xytext=ofs,
            textcoords="offset points",
            fontsize=6,
            color=style["color"],
            arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.5),
        )

    ax.set_xlabel("Altitude (km)")
    if ax == axes[0]:
        ax.set_ylabel("Delivery ratio DR")
    ax.set_title(body, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2)
    ax.set_xscale("log")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.88)

    # Consistent y range
    ax.set_ylim(0.48, 1.02)

save_fig(fig, "p3_altitude_phase_diagram")
plt.close(fig)
