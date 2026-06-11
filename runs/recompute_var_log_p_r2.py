#!/usr/bin/env python3
"""Recompute canonical R²(var_log_p, gamma) from 89K production dataset.

This script establishes the authoritative correlation statistics for Paper 2.
It reads all production panel files, filters invalid records, and computes:
  - Pearson r and R² for var_log_p vs gamma (= -lyapunov)
  - Spearman rho
  - OLS regression coefficients
  - Sample size after filtering

Output: runs/var_log_p_canonical_results.json
"""

import json
import math
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


def load_production():
    """Load all production panels, return list of records."""
    records = []
    for panel in PANELS:
        path = PROD_DIR / panel
        if not path.exists():
            print(f"WARNING: missing {panel}")
            continue
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict) and "results" in data:
            records.extend(data["results"])
        else:
            print(f"WARNING: unexpected format in {panel}")
    return records


def extract_fields(records):
    """Extract var_log_p and gamma arrays, filtering NaN/invalid."""
    var_log_p = []
    gamma = []
    skipped_nan = 0
    skipped_zero_paths = 0

    for r in records:
        vlp = r.get("var_log_p")
        lyap = r.get("lyapunov")
        n_paths = r.get("n_paths", 0)

        # Skip infeasible configs (no oracle paths found)
        if n_paths == 0:
            skipped_zero_paths += 1
            continue

        # Skip NaN values
        if vlp is None or lyap is None:
            skipped_nan += 1
            continue
        if math.isnan(vlp) or math.isnan(lyap):
            skipped_nan += 1
            continue

        var_log_p.append(vlp)
        gamma.append(-lyap)  # gamma = -lyapunov (lambda <= 0, gamma >= 0)

    return (
        np.array(var_log_p),
        np.array(gamma),
        skipped_nan,
        skipped_zero_paths,
    )


def ols_regression(x, y):
    """OLS regression y = slope*x + intercept."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return slope, intercept, r_squared


def pearson_r(x, y):
    """Pearson correlation coefficient."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.sum((x - x_mean) * (y - y_mean)) / (n - 1)
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    return cov / (sx * sy)


def spearman_rho(x, y):
    """Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    # Rank arrays
    x_order = np.argsort(np.argsort(x)).astype(float)
    y_order = np.argsort(np.argsort(y)).astype(float)
    return pearson_r(x_order, y_order)


def main():
    print("Loading production panels...")
    records = load_production()
    print(f"  Total records loaded: {len(records)}")

    print("Extracting var_log_p and gamma...")
    vlp, gam, skipped_nan, skipped_zero = extract_fields(records)
    print(f"  Valid records: {len(vlp)}")
    print(f"  Skipped (NaN): {skipped_nan}")
    print(f"  Skipped (zero paths): {skipped_zero}")

    # Compute statistics
    slope, intercept, r2 = ols_regression(vlp, gam)
    r = pearson_r(vlp, gam)
    rho = spearman_rho(vlp, gam)

    # Also compute correlation (var_log_p vs lyapunov, i.e. negative gamma)
    r_lyap = -r  # sign flip since gamma = -lyapunov

    print(f"\n{'=' * 60}")
    print("CANONICAL var_log_p STATISTICS (89K production)")
    print(f"{'=' * 60}")
    print("  Dataset: production_2026_03_11")
    print(f"  Total configs:        {len(records)}")
    print(f"  Valid (non-NaN, paths>0): {len(vlp)}")
    print()
    print(f"  OLS: gamma = {slope:.6f} * var_log_p + {intercept:.6f}")
    print(f"  R²  = {r2:.6f}")
    print(f"  Pearson r(var_log_p, gamma)    = {r:.6f}")
    print(f"  Pearson r(var_log_p, lyapunov) = {r_lyap:.6f}")
    print(f"  Spearman rho                   = {rho:.6f}")
    print()
    print(
        f"  var_log_p: mean={np.mean(vlp):.6f}, median={np.median(vlp):.6f}, std={np.std(vlp):.6f}"
    )
    print(
        f"  gamma:     mean={np.mean(gam):.6f}, median={np.median(gam):.6f}, std={np.std(gam):.6f}"
    )
    print(f"{'=' * 60}")

    # Persist results
    result = {
        "description": "Canonical var_log_p correlation from 89K production dataset",
        "dataset": "production_2026_03_11",
        "total_configs": len(records),
        "valid_configs": int(len(vlp)),
        "skipped_nan": skipped_nan,
        "skipped_zero_paths": skipped_zero,
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "r_squared": float(r2),
        "pearson_r_vlp_gamma": float(r),
        "pearson_r_vlp_lyapunov": float(r_lyap),
        "spearman_rho": float(rho),
        "var_log_p_mean": float(np.mean(vlp)),
        "var_log_p_median": float(np.median(vlp)),
        "var_log_p_std": float(np.std(vlp)),
        "gamma_mean": float(np.mean(gam)),
        "gamma_median": float(np.median(gam)),
        "gamma_std": float(np.std(gam)),
    }

    outpath = RUNS_DIR / "var_log_p_canonical_results.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten: {outpath}")


if __name__ == "__main__":
    main()
