#!/usr/bin/env python3
"""Recompute gamma_retry (oracle reclassification) from phi_decompose data.

Paper 2 claims Mercury gamma_oracle = +0.228 and delta_policy = +1.042.
These were computed interactively on 2026-03-09 but never persisted.

This script recomputes gamma_retry = d ln(phi_retry) / d E[H] for all
bodies using OLS regression, establishing the canonical values.

Also computes gamma_myopic and delta_policy = gamma_myopic - gamma_retry.

Output: runs/gamma_oracle_canonical_results.json
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).parent


def ols_gamma(ehs, ln_phis):
    """OLS regression ln(phi) = gamma * E[H] + intercept. Returns (gamma, R², n)."""
    n = len(ehs)
    if n < 5:
        return float("nan"), float("nan"), n
    if np.std(ehs) < 1e-12:
        return float("nan"), float("nan"), n
    slope, intercept = np.polyfit(ehs, ln_phis, 1)
    pred = slope * ehs + intercept
    ss_res = float(np.sum((ln_phis - pred) ** 2))
    ss_tot = float(np.sum((ln_phis - np.mean(ln_phis)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return float(slope), r2, n


def main():
    # Load phi_decompose data (153 MB)
    path = RUNS_DIR / "phi_decompose_results.json"
    print(f"Loading {path.name}...")
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        records = data["results"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected format in {path.name}")

    print(f"  Total records: {len(records)}")

    # Group by body (target), filtering aggregation records and invalid entries
    by_body = defaultdict(
        lambda: {"E_H": [], "ln_phi_normal": [], "ln_phi_myopic": [], "ln_phi_retry": []}
    )
    skipped = 0
    for r in records:
        target = r.get("target")
        if target is None:
            skipped += 1
            continue

        eh = r.get("E_H")
        phi_n = r.get("phi_normal")
        phi_m = r.get("phi_myopic")
        phi_r = r.get("phi_retry")

        # Skip invalid
        if any(v is None for v in (eh, phi_n, phi_m, phi_r)):
            skipped += 1
            continue
        if any(math.isnan(v) for v in (eh, phi_n, phi_m, phi_r)):
            skipped += 1
            continue
        if any(v <= 0 for v in (phi_n, phi_m, phi_r)):
            skipped += 1
            continue
        if eh < 1e-6:
            skipped += 1
            continue

        by_body[target]["E_H"].append(eh)
        by_body[target]["ln_phi_normal"].append(math.log(phi_n))
        by_body[target]["ln_phi_myopic"].append(math.log(phi_m))
        by_body[target]["ln_phi_retry"].append(math.log(phi_r))

    print(f"  Skipped (invalid/aggregation): {skipped}")
    print(f"  Bodies found: {sorted(by_body.keys())}")

    # Compute gamma for each body and phi variant
    results = []
    print(
        f"\n{'Body':<12} {'n':>6}  {'gamma_normal':>14} {'gamma_myopic':>14} "
        f"{'gamma_retry':>14} {'delta_policy':>14}"
    )
    print("-" * 82)

    for body in sorted(by_body.keys()):
        d = by_body[body]
        ehs = np.array(d["E_H"])
        n = len(ehs)

        g_normal, r2_n, _ = ols_gamma(ehs, np.array(d["ln_phi_normal"]))
        g_myopic, r2_m, _ = ols_gamma(ehs, np.array(d["ln_phi_myopic"]))
        g_retry, r2_r, _ = ols_gamma(ehs, np.array(d["ln_phi_retry"]))

        delta = (
            g_myopic - g_retry
            if not (math.isnan(g_myopic) or math.isnan(g_retry))
            else float("nan")
        )

        print(
            f"{body:<12} {n:>6}  {g_normal:>+14.6f} {g_myopic:>+14.6f} "
            f"{g_retry:>+14.6f} {delta:>+14.6f}"
        )

        results.append(
            {
                "body": body,
                "n_configs": n,
                "gamma_normal": g_normal,
                "gamma_normal_r2": r2_n,
                "gamma_myopic": g_myopic,
                "gamma_myopic_r2": r2_m,
                "gamma_retry": g_retry,
                "gamma_retry_r2": r2_r,
                "delta_policy": delta,
            }
        )

    output = {
        "description": "Canonical gamma decomposition (normal/myopic/retry) per body",
        "source": "phi_decompose_results.json",
        "total_records": len(records),
        "skipped": skipped,
        "bodies": results,
    }

    outpath = RUNS_DIR / "gamma_oracle_canonical_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten: {outpath}")


if __name__ == "__main__":
    main()
