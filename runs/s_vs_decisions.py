#!/usr/bin/env python3
"""
s_vs_decisions.py — Test whether s = η/(1-η) follows a power law
in "routing decisions" = hops × graph_degree.

Two plots (log-log):
  1. s vs (hops × mean_graph_degree)   — degree from contact plan
  2. s vs (hops × n_satellites)         — simpler proxy

Power-law fit: log(s) = a - b·log(decisions) → s ∝ decisions^{-b}

Reads all original result files for contact counts and architecture.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

RUNS = Path(__file__).resolve().parent

# ── Load helpers ───────────────────────────────────────────────


def load_json(name):
    p = RUNS / name
    if not p.exists():
        print(f"WARNING: {p} not found, skipping")
        return None
    with open(p) as f:
        return json.load(f)


def compute_s(eta):
    if eta >= 1.0:
        return float("inf")
    if eta <= 0.0:
        return 0.0
    return eta / (1.0 - eta)


# ── Collect annotated configs ──────────────────────────────────


def main():
    rows = []  # list of dicts with all metrics

    # ---------- Mars (2-hop, no ISL) ----------
    # Contact plan: station↔orbiter + orbiter→earth.  No ISL.
    # n_source_nodes = 1 (station) + n_orbiters.  Earth is sink.
    # Horizon: 3 sols = 3 × 88775 s

    mars = load_json("mars_eta_universality_results.json")
    MARS_HORIZON_DAYS = 3 * 88775.0 / 86400.0  # ~3.08 days

    if mars:
        seen = set()
        for exp_name in ["exp1_architecture", "exp2_dsn_sweep", "exp3_latitude"]:
            for entry in mars["experiments"].get(exp_name, []):
                label = entry["label"]
                if label in seen:
                    continue
                seen.add(label)

                eta = entry["estimates"]["NORMAL"]["eta_mean"]
                st = entry["oracle"]["S_T_full"]
                if st < 0.95:
                    continue

                n_orb = entry["config"]["n_orbiters"]
                n_contacts = entry["n_contacts"]
                n_source = 1 + n_orb  # station + orbiters
                n_sats = n_orb  # satellites only
                hops = 2

                degree_per_day = n_contacts / n_source / MARS_HORIZON_DAYS

                rows.append(
                    {
                        "label": f"Mars {label}",
                        "body": "Mars",
                        "arch": "Mars (no ISL)",
                        "eta": eta,
                        "s": compute_s(eta),
                        "S_T_full": st,
                        "hops": hops,
                        "n_sats": n_sats,
                        "n_contacts": n_contacts,
                        "n_source_nodes": n_source,
                        "degree_per_day": degree_per_day,
                        "decisions_degree": hops * degree_per_day,
                        "decisions_sats": hops * n_sats,
                    }
                )

    # ---------- Cislunar pure-polar (2-hop + ISL) ----------
    # Contact plan: station↔polar + polar↔polar (ISL) + polar→DSN.
    # n_source_nodes = 1 (surface) + n_polars.
    # Horizon: 1 day = 86400 s.
    # Contact count estimation: baseline has 1056 contacts / 1005.0 c_total_h
    #   → mean_dur ≈ 0.951 h/contact.

    PUREPOLAR_MEAN_DUR_H = 1005.0166666666667 / 1056.0  # calibrated from baseline

    purepolar = load_json("cislunar_purepolar_results.json")
    if purepolar:
        for entry in purepolar.get("exp2_thinning", []):
            eta = entry["eta_norm"]
            st = entry["s_t_full"]
            if st < 0.95:
                continue

            n_pol = entry["n_polars"]
            c_total_h = entry["c_total_h"]
            n_contacts_est = round(c_total_h / PUREPOLAR_MEAN_DUR_H)
            n_source = 1 + n_pol
            n_sats = n_pol
            hops = 2

            degree_per_day = n_contacts_est / n_source  # horizon = 1 day

            rows.append(
                {
                    "label": f"Pure-polar n={n_pol}",
                    "body": "Moon",
                    "arch": "Cislunar polar+ISL",
                    "eta": eta,
                    "s": compute_s(eta),
                    "S_T_full": st,
                    "hops": hops,
                    "n_sats": n_sats,
                    "n_contacts": n_contacts_est,
                    "n_source_nodes": n_source,
                    "degree_per_day": degree_per_day,
                    "decisions_degree": hops * degree_per_day,
                    "decisions_sats": hops * n_sats,
                }
            )

    # ---------- Cislunar nohalo (3-hop: polar + ELFO, ISL) ----------
    # Contact plan: station↔polar + polar↔polar (ISL) + polar↔ELFO + polar/ELFO→DSN.
    # n_source_nodes = 1 (surface) + n_polars + 1 (ELFO) = n_polars + 2.
    # Horizon: 1 day.

    NOHALO_MEAN_DUR_H = 1290.15 / 1234.0  # calibrated from baseline

    nohalo = load_json("cislunar_nohalo_results.json")
    if nohalo:
        for entry in nohalo.get("exp2_thinning", []):
            eta = entry["eta_norm"]
            st = entry["s_t_full"]
            if st < 0.95:
                continue

            n_pol = entry["n_polars"]
            c_total_h = entry["c_total_h"]
            n_contacts_est = round(c_total_h / NOHALO_MEAN_DUR_H)
            n_source = 1 + n_pol + 1  # surface + polars + ELFO
            n_sats = n_pol + 1  # polars + ELFO
            hops = 3

            degree_per_day = n_contacts_est / n_source  # horizon = 1 day

            rows.append(
                {
                    "label": f"Nohalo n={n_pol}",
                    "body": "Moon",
                    "arch": "Cislunar polar+ELFO+ISL",
                    "eta": eta,
                    "s": compute_s(eta),
                    "S_T_full": st,
                    "hops": hops,
                    "n_sats": n_sats,
                    "n_contacts": n_contacts_est,
                    "n_source_nodes": n_source,
                    "degree_per_day": degree_per_day,
                    "decisions_degree": hops * degree_per_day,
                    "decisions_sats": hops * n_sats,
                }
            )

    # ---------- Full relay (4-hop: polar + ELFO + halo, paper reference) ----------
    # Estimated contact counts from nohalo baseline + ~150 halo contacts.
    # n_source_nodes = 1 + n_polars + 1 (ELFO) + 1 (halo) = n_polars + 3.

    # The paper values come from the TIN physics engine (different contact gen).
    # We estimate degree from the no-halo baseline scaled up.

    full_relay_configs = [
        {
            "label": "Full relay 8-polar+relay",
            "n_pol": 8,
            "eta": 0.681,
            # nohalo 8P: 1234 contacts + ~150 for halo links
            "n_contacts_est": 1384,
        },
        {
            "label": "Full relay 1-polar+relay",
            "n_pol": 1,
            # nohalo 1P: ~89 contacts + ~50 for halo
            "eta": 0.706,
            "n_contacts_est": 139,
        },
    ]

    for cfg in full_relay_configs:
        n_pol = cfg["n_pol"]
        n_source = 1 + n_pol + 1 + 1  # surface + polars + ELFO + halo
        n_sats = n_pol + 2  # polars + ELFO + halo
        hops = 4
        degree_per_day = cfg["n_contacts_est"] / n_source

        rows.append(
            {
                "label": cfg["label"],
                "body": "Moon",
                "arch": "Cislunar full relay",
                "eta": cfg["eta"],
                "s": compute_s(cfg["eta"]),
                "S_T_full": 1.0,
                "hops": hops,
                "n_sats": n_sats,
                "n_contacts": cfg["n_contacts_est"],
                "n_source_nodes": n_source,
                "degree_per_day": degree_per_day,
                "decisions_degree": hops * degree_per_day,
                "decisions_sats": hops * n_sats,
            }
        )

    # ── Filter: only finite s ─────────────────────────────────────

    rows = [r for r in rows if 0 < r["s"] < float("inf")]
    rows.sort(key=lambda r: r["decisions_degree"])

    # ── Print table ────────────────────────────────────────────────

    print()
    print("=" * 130)
    print(
        f"{'Config':<30} {'η':>7} {'s':>8} {'hops':>5} {'n_sat':>5} "
        f"{'n_cont':>6} {'n_src':>5} {'deg/d':>7} "
        f"{'D=h×deg':>8} {'h×n_s':>6}"
    )
    print("=" * 130)

    for r in rows:
        print(
            f"{r['label']:<30} {r['eta']:>7.4f} {r['s']:>8.1f} "
            f"{r['hops']:>5} {r['n_sats']:>5} "
            f"{r['n_contacts']:>6} {r['n_source_nodes']:>5} "
            f"{r['degree_per_day']:>7.1f} "
            f"{r['decisions_degree']:>8.1f} {r['decisions_sats']:>6}"
        )

    print("=" * 130)

    # ── Power-law fits ─────────────────────────────────────────────

    def power_law_fit(x, y, label):
        """Fit log(y) = a + b·log(x).  Returns (a, b, r²)."""
        mask = (np.array(x) > 0) & (np.array(y) > 0)
        lx = np.log10(np.array(x)[mask])
        ly = np.log10(np.array(y)[mask])
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(lx, ly)
        print(f"\n  {label}:")
        print(f"    log₁₀(s) = {intercept:.3f} + {slope:.3f} · log₁₀(x)")
        print(f"    s ∝ x^{slope:.3f}")
        print(f"    R² = {r_value**2:.4f},  p = {p_value:.2e},  SE(slope) = {std_err:.3f}")
        return intercept, slope, r_value**2

    print("\n" + "=" * 60)
    print("POWER-LAW FITS")
    print("=" * 60)

    x_deg = [r["decisions_degree"] for r in rows]
    x_sat = [r["decisions_sats"] for r in rows]
    y_s = [r["s"] for r in rows]

    a1, b1, r2_1 = power_law_fit(x_deg, y_s, "s vs (hops × degree/day)")
    a2, b2, r2_2 = power_law_fit(x_sat, y_s, "s vs (hops × n_satellites)")

    # ── Also try: separate fits by body ────────────────────────────

    print("\n--- Per-body fits (s vs hops × degree/day) ---")
    for body_name in ["Mars", "Moon"]:
        sub = [r for r in rows if r["body"] == body_name]
        if len(sub) < 3:
            continue
        x_sub = [r["decisions_degree"] for r in sub]
        y_sub = [r["s"] for r in sub]
        power_law_fit(x_sub, y_sub, f"  {body_name} only")

    # ── Plot ───────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    ARCH_COLORS = {
        "Mars (no ISL)": "#2196F3",
        "Cislunar polar+ISL": "#FF9800",
        "Cislunar polar+ELFO+ISL": "#4CAF50",
        "Cislunar full relay": "#E91E63",
    }
    ARCH_MARKERS = {
        "Mars (no ISL)": "o",
        "Cislunar polar+ISL": "s",
        "Cislunar polar+ELFO+ISL": "^",
        "Cislunar full relay": "D",
    }

    def scatter_by_arch(ax, x_key, rows, xlabel, fit_a, fit_b, fit_r2):
        """Scatter plot colored/shaped by architecture, with power-law line."""
        plotted_archs = set()
        for r in rows:
            arch = r["arch"]
            lbl = arch if arch not in plotted_archs else None
            plotted_archs.add(arch)
            ax.scatter(
                r[x_key],
                r["s"],
                c=ARCH_COLORS[arch],
                marker=ARCH_MARKERS[arch],
                s=70,
                alpha=0.75,
                edgecolors="k",
                linewidth=0.5,
                label=lbl,
            )

        # Power-law fit line
        x_all = np.array([r[x_key] for r in rows])
        x_fit = np.logspace(np.log10(x_all.min() * 0.7), np.log10(x_all.max() * 1.4), 100)
        y_fit = 10**fit_a * x_fit**fit_b
        ax.plot(
            x_fit,
            y_fit,
            "k--",
            alpha=0.6,
            linewidth=1.5,
            label=f"s ∝ x$^{{{fit_b:.2f}}}$  (R²={fit_r2:.3f})",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("s = η / (1 − η)", fontsize=12)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, which="both")

    # Left panel: s vs hops × degree/day
    scatter_by_arch(ax1, "decisions_degree", rows, "Decisions = hops × degree/day", a1, b1, r2_1)
    ax1.set_title(
        "s vs Routing Decisions\n(hops × mean graph degree)", fontsize=12, fontweight="bold"
    )

    # Right panel: s vs hops × n_sats
    scatter_by_arch(ax2, "decisions_sats", rows, "Decisions = hops × n_satellites", a2, b2, r2_2)
    ax2.set_title(
        "s vs Routing Decisions\n(hops × n_satellites, simpler proxy)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save
    for ext in ("pdf", "png"):
        out = RUNS / f"s_vs_decisions.{ext}"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"\nFigure saved: {out}")

    # ── JSON output ────────────────────────────────────────────────

    out_json = RUNS / "s_vs_decisions_results.json"
    results = {
        "configs": rows,
        "power_law_fits": {
            "s_vs_hops_x_degree": {
                "intercept": a1,
                "exponent": b1,
                "r_squared": r2_1,
                "formula": f"s = 10^{a1:.3f} × (hops×deg)^{b1:.3f}",
            },
            "s_vs_hops_x_nsats": {
                "intercept": a2,
                "exponent": b2,
                "r_squared": r2_2,
                "formula": f"s = 10^{a2:.3f} × (hops×n_sat)^{b2:.3f}",
            },
        },
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"JSON saved: {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
