#!/usr/bin/env python3
"""Whitespace analysis: two non-obvious structural findings from provenance audit.

Finding 2: The alive/dead transition as Layer -1.
  - Does var_log_p predict proximity to the percolation edge?
  - Does the phi_decompose skip rate correlate with gamma per body?
  - What separates feasible (S_T > 0) from infeasible (S_T = 0) configs?

Finding 3: R² strengthens with more data (0.846 → 0.903).
  - Is this structural (tails obey better) or statistical (more data = less noise)?
  - Bootstrap: sample 17K from 82K repeatedly, measure R² distribution.
  - Per-panel R²: which subsets drive the relationship?
  - Tail vs core: R² in quantile bands.

Output: runs/whitespace_analysis_results.json
"""

import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).parent
PROD_DIR = RUNS_DIR / "epyc_results" / "production_2026_03_11"

PANELS = [
    "production_P1_results.json",
    "production_P3_results.json",
    "production_P4_results.json",
    "production_P5P6_results.json",
    "production_P7_results.json",
    "production_P8_results.json",
    "production_P12_results.json",
    "production_P14a_results.json",
    "production_P14b_results.json",
    "production_P14c_results.json",
    "production_P14d_results.json",
]


def ols_r2(x, y):
    """OLS R² for y = ax + b."""
    n = len(x)
    if n < 5 or np.std(x) < 1e-15:
        return float("nan"), float("nan"), float("nan"), n
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return float(r2), float(slope), float(intercept), n


def load_all_panels():
    """Load all production panels with panel labels."""
    panels = {}
    for panel_file in PANELS:
        path = PROD_DIR / panel_file
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("results", [])
        label = panel_file.replace("production_", "").replace("_results.json", "")
        panels[label] = records
    return panels


def extract_record(r):
    """Extract (var_log_p, gamma, S_T, body, n_paths) from a record."""
    vlp = r.get("var_log_p")
    lyap = r.get("lyapunov")
    st = r.get("S_T", 0.0)
    body = r.get("body") or r.get("target", "unknown")
    n_paths = r.get("n_paths", 0)

    if vlp is not None and lyap is not None:
        if not math.isnan(vlp) and not math.isnan(lyap):
            return vlp, -lyap, st, body, n_paths
    return None, None, st, body, n_paths


def _bootstrap_r2(args):
    """Worker for bootstrap R² computation."""
    vlp, gam, sample_size, seed = args
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(vlp), size=sample_size, replace=False)
    r2, _, _, _ = ols_r2(vlp[idx], gam[idx])
    return r2


def main():
    print("=" * 70)
    print("WHITESPACE ANALYSIS")
    print("=" * 70)

    # Load data
    panels = load_all_panels()
    all_records = []
    panel_records = {}
    for label, recs in panels.items():
        all_records.extend(recs)
        panel_records[label] = recs

    print(f"\nTotal records: {len(all_records)}")

    # =========================================================================
    # FINDING 2: Layer -1 (alive/dead transition)
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("FINDING 2: Layer -1 — Alive/Dead Transition")
    print(f"{'=' * 70}")

    feasible = []  # S_T > 0, has valid var_log_p
    infeasible = []  # S_T = 0
    feasible_no_vlp = []  # S_T > 0 but no var_log_p (n_paths=0 edge case)

    by_body_alive = defaultdict(int)
    by_body_dead = defaultdict(int)
    by_body_vlp = defaultdict(list)
    by_body_gamma = defaultdict(list)

    for r in all_records:
        vlp, gam, st, body, n_paths = extract_record(r)
        if st <= 0 or n_paths == 0:
            infeasible.append(r)
            by_body_dead[body] += 1
        elif vlp is not None:
            feasible.append((vlp, gam, st, body))
            by_body_alive[body] += 1
            by_body_vlp[body].append(vlp)
            by_body_gamma[body].append(gam)
        else:
            feasible_no_vlp.append(r)
            by_body_alive[body] += 1

    print(f"\n  Feasible (S_T > 0, valid vlp): {len(feasible)}")
    print(f"  Infeasible (S_T = 0 or n_paths = 0): {len(infeasible)}")
    print(f"  Feasible but no vlp: {len(feasible_no_vlp)}")

    # Death rate per body
    print(
        f"\n  {'Body':<12} {'Alive':>7} {'Dead':>7} {'Death%':>8} "
        f"{'mean_vlp':>10} {'mean_gamma':>12}"
    )
    print("  " + "-" * 62)

    body_death_rates = {}
    body_mean_gammas = {}
    for body in sorted(set(list(by_body_alive.keys()) + list(by_body_dead.keys()))):
        alive = by_body_alive[body]
        dead = by_body_dead[body]
        total = alive + dead
        death_pct = 100.0 * dead / total if total > 0 else 0
        mean_vlp = np.mean(by_body_vlp[body]) if by_body_vlp[body] else float("nan")
        mean_gam = np.mean(by_body_gamma[body]) if by_body_gamma[body] else float("nan")
        body_death_rates[body] = death_pct
        body_mean_gammas[body] = mean_gam
        print(
            f"  {body:<12} {alive:>7} {dead:>7} {death_pct:>7.1f}% "
            f"{mean_vlp:>10.6f} {mean_gam:>+12.6f}"
        )

    # Correlation: death rate vs mean gamma per body
    bodies_both = [
        b for b in body_death_rates if not math.isnan(body_mean_gammas.get(b, float("nan")))
    ]
    if len(bodies_both) >= 3:
        dr_arr = np.array([body_death_rates[b] for b in bodies_both])
        gam_arr = np.array([body_mean_gammas[b] for b in bodies_both])
        r2_dg, slope_dg, _, _ = ols_r2(dr_arr, gam_arr)
        # Pearson
        r_dg = np.corrcoef(dr_arr, gam_arr)[0, 1]
        print("\n  Corr(death_rate, mean_gamma) across bodies:")
        print(f"    Pearson r = {r_dg:.4f}, R² = {r2_dg:.4f}")
        print(
            f"    Higher death rate → {'more negative' if slope_dg < 0 else 'more positive'} gamma"
        )

    # Distance from edge: S_T distribution for feasible configs
    st_vals = np.array([f[2] for f in feasible])
    vlp_vals = np.array([f[0] for f in feasible])
    gam_vals = np.array([f[1] for f in feasible])

    # Quartile analysis: does var_log_p predict S_T?
    r2_vlp_st, slope_vlp_st, _, _ = ols_r2(vlp_vals, st_vals)
    r_vlp_st = float(np.corrcoef(vlp_vals, st_vals)[0, 1])
    print("\n  var_log_p → S_T (edge proximity):")
    print(f"    Pearson r = {r_vlp_st:.4f}, R² = {r2_vlp_st:.4f}")

    # S_T percentile bins: what happens to gamma near the edge?
    st_percentiles = np.percentile(st_vals, [10, 25, 50, 75, 90])
    print(
        f"\n  S_T distribution: p10={st_percentiles[0]:.3f}, p25={st_percentiles[1]:.3f}, "
        f"median={st_percentiles[2]:.3f}, p75={st_percentiles[3]:.3f}, p90={st_percentiles[4]:.3f}"
    )

    # Gamma in S_T bands
    print("\n  Gamma by S_T band:")
    bands = [
        (0, 0.25, "near-dead"),
        (0.25, 0.50, "struggling"),
        (0.50, 0.75, "moderate"),
        (0.75, 0.95, "healthy"),
        (0.95, 1.01, "full-reach"),
    ]
    layer_minus1_results = []
    for lo, hi, label in bands:
        mask = (st_vals >= lo) & (st_vals < hi)
        if mask.sum() < 5:
            continue
        mg = float(np.mean(gam_vals[mask]))
        mv = float(np.mean(vlp_vals[mask]))
        sg = float(np.std(gam_vals[mask]))
        n = int(mask.sum())
        print(
            f"    S_T [{lo:.2f}, {hi:.2f}) '{label}': n={n:>6}, "
            f"mean_gamma={mg:>+.5f}±{sg:.5f}, mean_vlp={mv:.6f}"
        )
        layer_minus1_results.append(
            {
                "band": label,
                "st_lo": lo,
                "st_hi": hi,
                "n": n,
                "mean_gamma": mg,
                "std_gamma": sg,
                "mean_var_log_p": mv,
            }
        )

    # =========================================================================
    # FINDING 3: R² strengthens with data
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("FINDING 3: R² Strengthens With Data — Structural or Statistical?")
    print(f"{'=' * 70}")

    # Per-panel R²
    print("\n  Per-panel R² (var_log_p → gamma):")
    print(f"  {'Panel':<8} {'n_valid':>8} {'R²':>8} {'slope':>10}")
    print("  " + "-" * 38)
    panel_r2s = []
    for label in sorted(panel_records.keys()):
        recs = panel_records[label]
        vs, gs = [], []
        for r in recs:
            vlp, gam, st, body, np_ = extract_record(r)
            if vlp is not None and np_ > 0:
                vs.append(vlp)
                gs.append(gam)
        if len(vs) >= 10:
            r2, slope, _, n = ols_r2(np.array(vs), np.array(gs))
            print(f"  {label:<8} {n:>8} {r2:>8.4f} {slope:>10.3f}")
            panel_r2s.append({"panel": label, "n": n, "r2": r2, "slope": slope})

    # Bootstrap: sample 17K from 82K, 1000 times
    print(f"\n  Bootstrap R²: sampling n=17,000 from {len(vlp_vals)} valid records, 1000 reps...")
    n_bootstrap = 1000
    sample_size = 17000

    tasks = [(vlp_vals, gam_vals, sample_size, 42 + i * 7) for i in range(n_bootstrap)]
    n_workers = min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        boot_r2s = list(executor.map(_bootstrap_r2, tasks))

    boot_r2s = np.array(boot_r2s)
    boot_r2s = boot_r2s[~np.isnan(boot_r2s)]
    print(f"    Bootstrap R² (n=17K): mean={np.mean(boot_r2s):.4f}, std={np.std(boot_r2s):.4f}")
    print(f"    95% CI: [{np.percentile(boot_r2s, 2.5):.4f}, {np.percentile(boot_r2s, 97.5):.4f}]")
    print(
        f"    phi_sweep R² = 0.846 {'WITHIN' if np.percentile(boot_r2s, 2.5) <= 0.846 <= np.percentile(boot_r2s, 97.5) else 'OUTSIDE'} bootstrap CI"
    )
    print("    Full-data R² = 0.903")

    # Quantile-band R²: is the relationship stronger in the tails?
    print("\n  Quantile-band R² (is the relationship stronger in the tails?):")
    vlp_pcts = np.percentile(vlp_vals, [0, 20, 40, 60, 80, 100])
    quantile_r2s = []
    for i in range(5):
        mask = (vlp_vals >= vlp_pcts[i]) & (vlp_vals < vlp_pcts[i + 1] + 1e-15)
        if mask.sum() < 20:
            continue
        r2, slope, _, n = ols_r2(vlp_vals[mask], gam_vals[mask])
        label = f"Q{i + 1} [{vlp_pcts[i]:.5f}, {vlp_pcts[i + 1]:.5f})"
        print(f"    {label}: n={n:>6}, R²={r2:.4f}, slope={slope:.3f}")
        quantile_r2s.append(
            {
                "quintile": i + 1,
                "lo": float(vlp_pcts[i]),
                "hi": float(vlp_pcts[i + 1]),
                "n": n,
                "r2": r2,
                "slope": slope,
            }
        )

    # Cumulative R²: how does R² change as we add data from most to least extreme?
    print("\n  Cumulative R² (adding data from most to least extreme var_log_p):")
    sort_idx = np.argsort(vlp_vals)[::-1]  # descending var_log_p
    cumulative_r2s = []
    for frac in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
        n_use = int(len(vlp_vals) * frac)
        if n_use < 20:
            continue
        idx = sort_idx[:n_use]
        r2, slope, _, n = ols_r2(vlp_vals[idx], gam_vals[idx])
        print(f"    Top {frac * 100:>5.1f}% (n={n:>6}): R²={r2:.4f}")
        cumulative_r2s.append({"fraction": frac, "n": n, "r2": r2})

    # =========================================================================
    # Persist results
    # =========================================================================
    results = {
        "description": "Whitespace analysis: Layer -1 and R² structural test",
        "finding_2": {
            "total_feasible": len(feasible),
            "total_infeasible": len(infeasible),
            "death_rate_pct": 100.0 * len(infeasible) / len(all_records),
            "corr_death_rate_gamma": {
                "pearson_r": float(r_dg) if len(bodies_both) >= 3 else None,
                "r_squared": float(r2_dg) if len(bodies_both) >= 3 else None,
            },
            "corr_vlp_st": {
                "pearson_r": r_vlp_st,
                "r_squared": r2_vlp_st,
            },
            "gamma_by_st_band": layer_minus1_results,
        },
        "finding_3": {
            "full_data_r2": 0.903,
            "phi_sweep_r2": 0.846,
            "bootstrap": {
                "n_reps": n_bootstrap,
                "sample_size": sample_size,
                "mean_r2": float(np.mean(boot_r2s)),
                "std_r2": float(np.std(boot_r2s)),
                "ci_025": float(np.percentile(boot_r2s, 2.5)),
                "ci_975": float(np.percentile(boot_r2s, 97.5)),
                "phi_sweep_in_ci": bool(
                    np.percentile(boot_r2s, 2.5) <= 0.846 <= np.percentile(boot_r2s, 97.5)
                ),
            },
            "per_panel_r2": panel_r2s,
            "quantile_r2": quantile_r2s,
            "cumulative_r2": cumulative_r2s,
        },
    }

    outpath = RUNS_DIR / "whitespace_analysis_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: {outpath}")


if __name__ == "__main__":
    main()
