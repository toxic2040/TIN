"""run_conjecture_figure.py — Conjecture figure: ergodic vs non-ergodic contact generation.

Reads cislunar_purepolar_results.json (no ELFO, no halo, pure polar).
Extracts thinning sweep and duty-cycle deletion sweep data.

Outputs
-------
  conjecture_figure_data.json  — two arrays: thinning and deletion, each with
                                  {c, s_t, eta, dr} per operating point.
  conjecture_figure.png        — matplotlib figure at 300 DPI.

Figure design
-------------
  Blue filled circles  : thinning sweep  (ergodic — remove whole satellites)
  Red filled triangles : deletion sweep  (non-ergodic — remove random windows)
  Dashed blue line     : linear envelope fit through thinning points, showing
                          the ergodic achievability band across C.
  X axis               : C  (polar→Earth contact hours per day)
  Y axis               : DR_norm  (delivery ratio, NORMAL priority)
  Title                : "Contact-Density Bound: Ergodic vs Non-Ergodic
                          Contact Generation"
"""

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------


def main():
    src_path = _HERE / "cislunar_purepolar_results.json"
    if not src_path.exists():
        print(f"ERROR: {src_path} not found.")
        print("Run run_cislunar_purepolar.py first.")
        sys.exit(1)

    with open(src_path) as fh:
        raw = json.load(fh)

    # ---------------------------------------------------------------------------
    # Extract thinning sweep
    # ---------------------------------------------------------------------------

    thinning = []
    for r in raw["exp2_thinning"]:
        thinning.append(
            {
                "n_polars": r["n_polars"],
                "c": r["c_pe_h"],
                "s_t": r["s_t_full"],
                "eta": r["eta_norm"],
                "dr": r["dr_norm"],
            }
        )

    # Sort by C ascending (for fit and display)
    thinning.sort(key=lambda x: x["c"])

    # ---------------------------------------------------------------------------
    # Extract deletion sweep (per-seed DR std for error bars)
    # ---------------------------------------------------------------------------

    deletion = []
    for r in raw["exp3_duty_cycle"]:
        # DR std from per-seed results
        seed_drs = [s["dr_n"] for s in r["seed_results"]]
        dr_std = float(np.std(seed_drs, ddof=1)) if len(seed_drs) > 1 else 0.0

        deletion.append(
            {
                "p_delete": r["p_delete"],
                "c": r["c_pe_h_mean"],
                "s_t": r["s_t_full_mean"],
                "s_t_std": r["s_t_full_std"],
                "eta": r["eta_norm_mean"],
                "dr": r["dr_norm_mean"],
                "dr_std": dr_std,
            }
        )

    deletion.sort(key=lambda x: x["c"])

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------

    out_json = _HERE / "conjecture_figure_data.json"
    with open(out_json, "w") as fh:
        json.dump({"thinning": thinning, "deletion": deletion}, fh, indent=2)
    print(f"  Data saved → {out_json.name}")

    # Confirm the axes diverge as expected
    print()
    print("  Cross-axis divergence check (matched C pairs):")
    print(f"  {'axis':>8}  {'C_pe_h':>7}  {'S_T':>7}  {'DR':>7}  {'η':>7}")
    for pt in thinning:
        print(
            f"  {'thin':>8}  {pt['c']:>7.1f}  {pt['s_t']:>7.4f}  {pt['dr']:>7.4f}  {pt['eta']:>7.4f}"
        )
    print()
    for pt in deletion:
        print(
            f"  {'del':>8}  {pt['c']:>7.1f}  {pt['s_t']:>7.4f}  {pt['dr']:>7.4f}  {pt['eta']:>7.4f}"
        )

    # ---------------------------------------------------------------------------
    # Matplotlib figure
    # ---------------------------------------------------------------------------

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("\n  matplotlib not available — skipping figure generation.")
        print("  Install with: pip install matplotlib")
        sys.exit(0)

    c_thin = np.array([p["c"] for p in thinning])
    dr_thin = np.array([p["dr"] for p in thinning])
    n_thin = np.array([p["n_polars"] for p in thinning])

    c_del = np.array([p["c"] for p in deletion])
    dr_del = np.array([p["dr"] for p in deletion])
    dr_del_std = np.array([p["dr_std"] for p in deletion])
    p_del = np.array([p["p_delete"] for p in deletion])

    # Linear envelope fit through thinning points
    coeffs = np.polyfit(c_thin, dr_thin, 1)
    c_fit = np.linspace(0.0, c_thin.max() * 1.05, 200)
    dr_fit = np.polyval(coeffs, c_fit)

    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Deletion: plot first so thinning points sit on top
    ax.errorbar(
        c_del,
        dr_del,
        yerr=dr_del_std,
        fmt="^",
        color="#d62728",
        markersize=8,
        ecolor="#d62728",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        label="Deletion sweep (non-ergodic)",
        zorder=3,
    )
    # Faint connecting line for deletion to show monotone trend
    ax.plot(c_del, dr_del, color="#d62728", linewidth=0.8, alpha=0.45, linestyle=":", zorder=2)

    # Thinning: ergodic regime
    ax.scatter(
        c_thin,
        dr_thin,
        marker="o",
        s=72,
        color="#1f77b4",
        zorder=5,
        label="Thinning sweep (ergodic)",
    )

    # Annotate each thinning point with n=
    for n, c, dr in zip(n_thin, c_thin, dr_thin):
        offset_x = -4.5 if c < 50 else 2.0
        offset_y = 0.008
        ax.annotate(
            f"n={n}",
            xy=(c, dr),
            xytext=(c + offset_x, dr + offset_y),
            fontsize=7.5,
            color="#1f77b4",
            ha="right" if offset_x < 0 else "left",
        )

    # Dashed envelope fit through thinning
    ax.plot(
        c_fit,
        dr_fit,
        color="#1f77b4",
        linewidth=1.6,
        linestyle="--",
        zorder=4,
        label=f"Ergodic envelope (linear fit,  slope={coeffs[0] * 100:.2f}%/100h)",
    )

    # Reference: DR = S_T (η = 1 line — cosmetic reference)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.text(97, 1.005, "DR = 1", fontsize=7.5, color="gray", ha="right")

    # Annotation: divergence gap at low C
    low_c_del = deletion[0]  # lowest C in deletion (p=0.9)
    low_c_thn = thinning[0]  # lowest C in thinning (n=1)
    gap = low_c_thn["dr"] - low_c_del["dr"]
    ax.annotate(
        "",
        xy=(low_c_del["c"], low_c_del["dr"]),
        xytext=(low_c_thn["c"], low_c_thn["dr"]),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2),
    )
    mid_c = (low_c_del["c"] + low_c_thn["c"]) / 2
    mid_dr = (low_c_del["dr"] + low_c_thn["dr"]) / 2
    ax.text(
        mid_c + 3.5,
        mid_dr,
        f"ΔDR = {gap:.2f}\n(same C)",
        fontsize=8,
        color="#333333",
        va="center",
    )

    # Axes
    ax.set_xlabel("C  (polar → Earth contact hours per day)", fontsize=11)
    ax.set_ylabel("DR$_\\mathrm{norm}$  (delivery ratio)", fontsize=11)
    ax.set_title(
        "Contact-Density Bound: Ergodic vs Non-Ergodic Contact Generation",
        fontsize=11.5,
        pad=9,
    )
    ax.set_xlim(0, 105)
    ax.set_ylim(0.35, 1.08)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.2)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    legend = ax.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor="#cccccc")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(pad=1.4)

    out_png = _HERE / "conjecture_figure.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\n  Figure saved → {out_png.name}")
    print(f"  Size: {out_png.stat().st_size // 1024} kB")
    plt.close(fig)


if __name__ == "__main__":
    main()
