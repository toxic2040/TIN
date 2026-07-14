"""Historical fluctuation-scaling diagnostic with two reference exponents.

This script preserves an archived comparison of fitted Var(ln eta) exponents
with numerical references 2/3 and 1. The comparison does not establish a DPRM
isomorphism, KPZ universality, a novel exponent, or a trap/cluster classifier.

The historical comparison used the form:

    Var(ln DR) ∝ E[H]^{2β}

The displayed 2/3 and 1 exponents are references only.

This script:
  1. Loads phi_decompose_results.json (230,400 configs, 30 seeds each)
  2. Groups by (target, n_orb, p_ref, alpha) — each group has 30 seeds
  3. For each group, computes Var(ln η_sim) where η_sim = eta_normal
  4. Bins by E[H] and fits the power-law exponent
  5. Reports orbital and social/vehicular groups separately

Data source: phi_decompose_results.json (orbital rows)
             crawdad_contacts.Exp{1,2,3,6}_results.json (social rows)

Reads:  runs/phi_decompose_results.json, runs/crawdad_contacts.Exp*_results.json
Writes: runs/kpz_scaling_results.json
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def _load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def _safe_log(x):
    """Return ln(x) if x > 0, else NaN."""
    return math.log(x) if x > 0 else float("nan")


def _power_law_fit(x, y):
    """Fit y = a * x^b in log-log space. Returns (a, b, R2)."""
    mask = (np.array(x) > 0) & (np.array(y) > 0) & np.isfinite(x) & np.isfinite(y)
    lx = np.log(np.array(x)[mask])
    ly = np.log(np.array(y)[mask])
    if len(lx) < 3:
        return float("nan"), float("nan"), float("nan")
    # OLS in log-log space
    A = np.vstack([lx, np.ones_like(lx)]).T
    result = np.linalg.lstsq(A, ly, rcond=None)
    b, log_a = result[0]
    a = math.exp(log_a)
    # R²
    ly_pred = b * lx + log_a
    ss_res = np.sum((ly - ly_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def analyze_orbital(phi_data):
    """Analyze the orbital rows from phi_decompose."""
    results = phi_data["results"]

    # Group by config key: (target, n_orb, p_ref, alpha, epoch_day)
    # Each group has seeds with different DR realizations
    groups = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue  # skip blackout epochs
        key = (r["target"], r["n_orb"], r["p_ref"], r["alpha"], r.get("epoch_day", 0))
        eta = r["eta_normal"]
        if eta is not None and eta > 0:
            groups[key].append({"eta": eta, "E_H": r["E_H"], "seed": r["seed"]})

    # For each config, compute Var(ln eta) across seeds
    config_stats = []
    for key, records in groups.items():
        if len(records) < 5:
            continue
        target, n_orb, p_ref, alpha, epoch_day = key
        ln_etas = [_safe_log(r["eta"]) for r in records if not math.isnan(_safe_log(r["eta"]))]
        if len(ln_etas) < 5:
            continue
        mean_E_H = np.mean([r["E_H"] for r in records])
        var_ln_eta = np.var(ln_etas, ddof=1)
        mean_ln_eta = np.mean(ln_etas)

        config_stats.append(
            {
                "target": target,
                "n_orb": n_orb,
                "p_ref": p_ref,
                "alpha": alpha,
                "epoch_day": epoch_day,
                "E_H": float(mean_E_H),
                "var_ln_eta": float(var_ln_eta),
                "mean_ln_eta": float(mean_ln_eta),
                "n_seeds": len(ln_etas),
                "assigned_group": "orbital",
            }
        )

    return config_stats


def analyze_crawdad(crawdad_data, trace_name):
    """Analyze the CRAWDAD social-trace rows."""
    results = crawdad_data["results"]

    # Group by (source, dest, ttl, p_eff) — each group has seeds
    groups = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue
        key = (r["source"], r["dest"], r["ttl"], r["p_eff"])
        eta = r["eta_normal"]
        if eta is not None and eta > 0:
            groups[key].append({"eta": eta, "E_H": r["E_H"], "seed": r["seed"]})

    config_stats = []
    for key, records in groups.items():
        if len(records) < 3:
            continue
        source, dest, ttl, p_eff = key
        ln_etas = [_safe_log(r["eta"]) for r in records if not math.isnan(_safe_log(r["eta"]))]
        if len(ln_etas) < 3:
            continue
        mean_E_H = np.mean([r["E_H"] for r in records])
        var_ln_eta = np.var(ln_etas, ddof=1)
        mean_ln_eta = np.mean(ln_etas)

        config_stats.append(
            {
                "trace": trace_name,
                "source": source,
                "dest": dest,
                "ttl": ttl,
                "p_eff": p_eff,
                "E_H": float(mean_E_H),
                "var_ln_eta": float(var_ln_eta),
                "mean_ln_eta": float(mean_ln_eta),
                "n_seeds": len(ln_etas),
                "assigned_group": "social",
            }
        )

    return config_stats


def bin_and_fit(stats, label, n_bins=15):
    """Bin by E[H], compute mean Var(ln eta) per bin, fit power law."""
    if not stats:
        return {"label": label, "exponent": float("nan"), "R2": float("nan"), "n_points": 0}

    eh_vals = np.array([s["E_H"] for s in stats])
    var_vals = np.array([s["var_ln_eta"] for s in stats])

    # Filter out zeros and NaN
    mask = (eh_vals > 0.5) & (var_vals > 0) & np.isfinite(var_vals)
    eh_vals = eh_vals[mask]
    var_vals = var_vals[mask]

    if len(eh_vals) < 5:
        return {
            "label": label,
            "exponent": float("nan"),
            "R2": float("nan"),
            "n_points": int(len(eh_vals)),
        }

    # Bin by E[H]
    bin_edges = np.linspace(eh_vals.min(), eh_vals.max(), n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (eh_vals >= bin_edges[i]) & (eh_vals < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (eh_vals >= bin_edges[i]) & (eh_vals <= bin_edges[i + 1])
        if np.sum(in_bin) >= 3:
            bin_centers.append(float(np.mean(eh_vals[in_bin])))
            bin_means.append(float(np.mean(var_vals[in_bin])))
            bin_stds.append(float(np.std(var_vals[in_bin], ddof=1)) if np.sum(in_bin) > 1 else 0.0)
            bin_counts.append(int(np.sum(in_bin)))

    if len(bin_centers) < 3:
        # Fall back to raw data fit
        a, b, r2 = _power_law_fit(eh_vals, var_vals)
        return {
            "label": label,
            "exponent": float(b),
            "R2": float(r2),
            "n_points": int(len(eh_vals)),
            "n_bins_used": 0,
            "method": "raw",
        }

    a, b, r2 = _power_law_fit(np.array(bin_centers), np.array(bin_means))

    return {
        "label": label,
        "exponent": float(b),
        "prefactor": float(a),
        "R2": float(r2),
        "n_points": int(len(eh_vals)),
        "n_bins_used": len(bin_centers),
        "method": "binned",
        "bins": [
            {"E_H": c, "var_ln_eta": m, "std": s, "count": n}
            for c, m, s, n in zip(bin_centers, bin_means, bin_stds, bin_counts)
        ],
    }


def compute_fluctuation_stats(stats):
    """Compute distribution shape statistics for ln(eta) fluctuations."""
    var_vals = [s["var_ln_eta"] for s in stats if s["var_ln_eta"] > 0]
    if not var_vals:
        return {}
    arr = np.array(var_vals)
    return {
        "mean_var": float(np.mean(arr)),
        "median_var": float(np.median(arr)),
        "std_var": float(np.std(arr)),
        "min_var": float(np.min(arr)),
        "max_var": float(np.max(arr)),
        "n_configs": len(var_vals),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("HISTORICAL FLUCTUATION-SCALING DIAGNOSTIC — UNIVERSALITY CLAIM RETIRED")
    print("=" * 70)

    # --- Load orbital data ---
    phi_path = _HERE / "phi_decompose_results.json"
    print(f"\nLoading {phi_path.name} ...")
    phi_data = _load_json(phi_path)
    print(f"  {phi_data['n_configs']:,} configs loaded")

    orbital_stats = analyze_orbital(phi_data)
    print(f"  {len(orbital_stats):,} valid orbital config groups")

    # --- Load CRAWDAD data ---
    crawdad_stats = []
    for trace in ["Exp1", "Exp2", "Exp3", "Exp6"]:
        cpath = _HERE / f"crawdad_contacts.{trace}_results.json"
        if cpath.exists():
            print(f"\nLoading {cpath.name} ...")
            cdata = _load_json(cpath)
            cstats = analyze_crawdad(cdata, trace)
            crawdad_stats.extend(cstats)
            print(f"  {len(cstats):,} valid {trace} config groups")

    # --- Load vehicular data ---
    veh_path = _HERE / "vehicular_gamma_results.json"
    vehicular_stats = []
    if veh_path.exists():
        print(f"\nLoading {veh_path.name} ...")
        vdata = _load_json(veh_path)
        vresults = vdata["results"]
        vgroups = defaultdict(list)
        for r in vresults:
            if r["S_T"] < 0.01:
                continue
            key = (r["source"], r["dest"], r["ttl"], r["p_eff"])
            eta = r["eta_greedy"]
            if eta is not None and eta > 0:
                vgroups[key].append({"eta": eta, "E_H": r["E_H"], "seed": r["seed"]})

        for key, records in vgroups.items():
            if len(records) < 3:
                continue
            source, dest, ttl, p_eff = key
            ln_etas = [_safe_log(r["eta"]) for r in records if not math.isnan(_safe_log(r["eta"]))]
            if len(ln_etas) < 3:
                continue
            vehicular_stats.append(
                {
                    "trace": "sf_cab",
                    "source": source,
                    "dest": dest,
                    "ttl": ttl,
                    "p_eff": p_eff,
                    "E_H": float(np.mean([r["E_H"] for r in records])),
                    "var_ln_eta": float(np.var(ln_etas, ddof=1)),
                    "mean_ln_eta": float(np.mean(ln_etas)),
                    "n_seeds": len(ln_etas),
                    "assigned_group": "vehicular",
                }
            )
        print(f"  {len(vehicular_stats):,} valid vehicular config groups")

    all_social = crawdad_stats + vehicular_stats

    # --- Per-target orbital fits ---
    print("\n" + "=" * 70)
    print("HISTORICAL ORBITAL-GROUP SCALING FITS — NO CLASSIFIER")
    print("=" * 70)

    targets = sorted(set(s["target"] for s in orbital_stats))
    per_target_fits = {}
    for target in targets:
        t_stats = [s for s in orbital_stats if s["target"] == target]
        fit = bin_and_fit(t_stats, f"orbital_{target}")
        per_target_fits[target] = fit
        print(
            f"\n  {target:10s}: β_eff = {fit['exponent']:.3f}  (R² = {fit['R2']:.3f}, "
            f"n = {fit['n_points']:,}, bins = {fit.get('n_bins_used', 0)})"
        )

    # Aggregate orbital
    orbital_fit = bin_and_fit(orbital_stats, "orbital_all")
    print(
        f"\n  {'ALL ORBIT':10s}: β_eff = {orbital_fit['exponent']:.3f}  "
        f"(R² = {orbital_fit['R2']:.3f}, n = {orbital_fit['n_points']:,})"
    )

    # --- Per-trace social/vehicular fits ---
    print("\n" + "=" * 70)
    print("HISTORICAL SOCIAL/VEHICULAR-GROUP FITS — NO CLASSIFIER")
    print("=" * 70)

    traces = sorted(set(s.get("trace", "unknown") for s in all_social))
    per_trace_fits = {}
    for trace in traces:
        t_stats = [s for s in all_social if s.get("trace") == trace]
        fit = bin_and_fit(t_stats, f"social_{trace}")
        per_trace_fits[trace] = fit
        print(
            f"\n  {trace:10s}: β_eff = {fit['exponent']:.3f}  (R² = {fit['R2']:.3f}, "
            f"n = {fit['n_points']:,}, bins = {fit.get('n_bins_used', 0)})"
        )

    # Aggregate social/vehicular group
    social_fit = bin_and_fit(all_social, "social_all")
    print(
        f"\n  {'ALL SOCIAL':10s}: β_eff = {social_fit['exponent']:.3f}  "
        f"(R² = {social_fit['R2']:.3f}, n = {social_fit['n_points']:,})"
    )

    # --- Numerical reference comparison ---
    print("\n" + "=" * 70)
    print("HISTORICAL REFERENCE-EXPONENT COMPARISON — DESCRIPTIVE ONLY")
    print("=" * 70)
    print("\n  KPZ prediction:  2β = 2/3 ≈ 0.667  (β = 1/3)")
    print("  CLT prediction:  2β = 1.0           (β = 1/2)")
    print("  No scaling:      2β = 0             (β = 0)")
    print(f"\n  Observed (orbital aggregate): 2β = {orbital_fit['exponent']:.3f}")
    print(f"  Observed (social aggregate):  2β = {social_fit['exponent']:.3f}")

    orbital_exp = orbital_fit["exponent"]
    social_exp = social_fit["exponent"]

    if not math.isnan(orbital_exp):
        if abs(orbital_exp - 0.667) < 0.15:
            orbital_comparison = "within 0.15 of the 2/3 reference"
        elif abs(orbital_exp - 1.0) < 0.15:
            orbital_comparison = "within 0.15 of the 1.0 reference"
        else:
            orbital_comparison = "outside both 0.15 reference bands"
        print(f"\n  Orbital-group comparison: {orbital_comparison}")

    if not math.isnan(social_exp):
        if abs(social_exp - 0.667) < 0.15:
            social_comparison = "within 0.15 of the 2/3 reference"
        elif abs(social_exp - 1.0) < 0.15:
            social_comparison = "within 0.15 of the 1.0 reference"
        else:
            social_comparison = "outside both 0.15 reference bands"
        print(f"  Social-group comparison:  {social_comparison}")
    print("  Reference proximity is not evidence of universality or a mechanism.")

    # --- Save results ---
    elapsed = time.time() - t0
    output = {
        "experiment": "kpz_scaling_test",
        "description": "Historical Var(ln eta) vs E[H] power-law diagnostic",
        "claim_status": "archived diagnostic; universality and classifier claims retired",
        "historical_kpz_reference": "2*beta = 2/3 ≈ 0.667",
        "clt_reference": "2*beta = 1.0",
        "elapsed_s": round(elapsed, 1),
        "orbital_group": {
            "aggregate_fit": orbital_fit,
            "per_target_fits": per_target_fits,
            "fluctuation_stats": compute_fluctuation_stats(orbital_stats),
        },
        "social_vehicular_group": {
            "aggregate_fit": social_fit,
            "per_trace_fits": per_trace_fits,
            "fluctuation_stats": compute_fluctuation_stats(all_social),
        },
    }

    out_path = _HERE / "kpz_scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
