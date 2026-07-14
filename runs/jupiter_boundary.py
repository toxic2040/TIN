"""Historical Jupiter and Saturn gamma-by-distance diagnostic.

This script reproduces an archived analysis of fitted gamma slopes across
distance and link-reliability bins:

1. Fine-grained distance-binned γ for Jupiter (from phi_decompose + phi_sweep)
2. Comparison with Saturn (also near-zero in the archived aggregate)
3. Per-p_ref gamma analysis across the encoded link-reliability values
4. f_fwd proxy and Φ evolution across the synodic cycle
5. Descriptive comparison across the eight orbital targets

The former classification and phase-transition-boundary interpretations are
retired. Near-zero values and sign changes are reported only as properties of
the loaded model rows; they are not a classifier or mechanism boundary.

Reads:  runs/phi_decompose_results.json
        runs/phi_sweep_shard_jupiter.json
        runs/phi_sweep_shard_saturn.json
        runs/multi_body_sweep_results.json
Writes: runs/jupiter_boundary_results.json
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _ols_slope(x, y):
    """OLS slope, intercept, R², n for y = a + b*x."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), int(len(x))
    A = np.vstack([x, np.ones_like(x)]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    b, a = result[0]
    y_pred = b * x + a
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), float(a), float(r2), int(len(x))


def compute_gamma_at_pref(records, p_ref_target, tol=0.02):
    """Compute γ = slope of ln(Φ) vs E[H] at a specific p_ref."""
    filtered = []
    for r in records:
        p = r.get("p_ref", r.get("p_eff", 0))
        if abs(p - p_ref_target) > tol:
            continue
        phi = r.get("phi_normal", r.get("phi_time", None))
        eh = r.get("E_H", None)
        st = r.get("S_T", 0)
        if phi is not None and phi > 0 and eh is not None and eh > 0 and st > 0.01:
            filtered.append((eh, math.log(phi)))

    if len(filtered) < 3:
        return {"gamma": float("nan"), "R2": float("nan"), "n": len(filtered)}

    x = [f[0] for f in filtered]
    y = [f[1] for f in filtered]
    slope, intercept, r2, n = _ols_slope(x, y)
    return {"gamma": slope, "intercept": intercept, "R2": r2, "n": n}


def analyze_body_by_distance_and_pref(records, n_dist_bins=5):
    """Compute γ in a 2D grid of (distance_bin, p_ref)."""
    # Get distance range
    dists = [r.get("dist_au", 0) for r in records if r.get("dist_au", 0) > 0]
    if not dists:
        return {}

    d_arr = np.array(dists)
    bin_edges = np.percentile(d_arr, np.linspace(0, 100, n_dist_bins + 1))

    # Get unique p_refs
    p_refs = sorted(set(round(r.get("p_ref", 0), 4) for r in records if r.get("p_ref", 0) > 0))

    grid = {}
    for i in range(n_dist_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_records = [
            r for r in records if lo <= r.get("dist_au", 0) <= hi and r.get("dist_au", 0) > 0
        ]
        if not bin_records:
            continue

        mean_dist = np.mean([r["dist_au"] for r in bin_records])
        bin_key = f"d_{mean_dist:.2f}"

        grid[bin_key] = {
            "mean_dist_au": float(mean_dist),
            "dist_range": [float(lo), float(hi)],
            "n_records": len(bin_records),
            "gamma_by_pref": {},
        }

        for p_ref in p_refs:
            g = compute_gamma_at_pref(bin_records, p_ref)
            if g["n"] >= 3:
                grid[bin_key]["gamma_by_pref"][str(round(p_ref, 4))] = g

    return grid


def analyze_boundary_crossing(grid):
    """Find descriptive gamma zero crossings in the loaded binned rows."""
    crossings = []
    for bin_key, info in sorted(grid.items()):
        for p_ref_str, g in sorted(info.get("gamma_by_pref", {}).items()):
            gamma = g["gamma"]
            if not math.isnan(gamma) and g["n"] >= 5:
                crossings.append(
                    {
                        "dist_au": info["mean_dist_au"],
                        "p_ref": float(p_ref_str),
                        "gamma": gamma,
                        "R2": g["R2"],
                        "n": g["n"],
                    }
                )

    if not crossings:
        return {"n_crossings": 0}

    # Find sign changes in γ
    sign_changes = []
    by_pref = defaultdict(list)
    for c in crossings:
        by_pref[c["p_ref"]].append(c)

    for p_ref, entries in sorted(by_pref.items()):
        entries.sort(key=lambda x: x["dist_au"])
        for i in range(len(entries) - 1):
            g1 = entries[i]["gamma"]
            g2 = entries[i + 1]["gamma"]
            if g1 * g2 < 0:  # sign change
                # Linear interpolation for crossing distance
                d1, d2 = entries[i]["dist_au"], entries[i + 1]["dist_au"]
                d_cross = d1 + (d2 - d1) * abs(g1) / (abs(g1) + abs(g2))
                sign_changes.append(
                    {
                        "p_ref": p_ref,
                        "d_cross_au": float(d_cross),
                        "gamma_near": g1,
                        "gamma_far": g2,
                    }
                )

    return {
        "n_total_points": len(crossings),
        "sign_changes": sign_changes,
        "n_sign_changes": len(sign_changes),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("HISTORICAL JUPITER/SATURN GAMMA SIGN DIAGNOSTIC")
    print("CLASSIFIER AND PHASE-BOUNDARY INTERPRETATIONS RETIRED")
    print("=" * 70)

    # --- Load phi_decompose for Jupiter, Saturn, and comparison bodies ---
    phi_path = _HERE / "phi_decompose_results.json"
    print(f"\nLoading {phi_path.name} ...")
    phi_data = _load_json(phi_path)

    # Split by target
    by_target = defaultdict(list)
    for r in phi_data["results"]:
        by_target[r["target"]].append(r)

    # --- Per-target aggregate gamma at multiple p_ref values ---
    print("\n" + "=" * 70)
    print("γ AT MULTIPLE p_ref VALUES")
    print("=" * 70)

    p_ref_targets = [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    target_gamma_table = {}

    for target in sorted(by_target.keys()):
        records = by_target[target]
        gamma_row = {}
        for p_ref in p_ref_targets:
            g = compute_gamma_at_pref(records, p_ref, tol=0.015)
            gamma_row[str(round(p_ref, 2))] = g

        target_gamma_table[target] = gamma_row

        # Print summary
        vals = [
            f"{g['gamma']:+.2f}" if not math.isnan(g["gamma"]) else "  nan"
            for g in gamma_row.values()
        ]
        print(
            f"  {target:10s}: " + " | ".join(f"p={p:.2f}:{v}" for p, v in zip(p_ref_targets, vals))
        )

    # --- Jupiter + Saturn: distance × p_ref grid ---
    print("\n" + "=" * 70)
    print("JUPITER: γ IN (DISTANCE, p_ref) SPACE")
    print("=" * 70)

    jupiter_grid = analyze_body_by_distance_and_pref(by_target["jupiter"])
    for bin_key, info in sorted(jupiter_grid.items()):
        print(f"\n  {bin_key} ({info['mean_dist_au']:.2f} AU, n={info['n_records']}):")
        for p_str, g in sorted(info.get("gamma_by_pref", {}).items()):
            sign = "+" if g["gamma"] > 0 else ""
            r2_str = f"R²={g['R2']:.2f}" if not math.isnan(g["R2"]) else "R²=nan"
            print(f"    p={p_str}: γ = {sign}{g['gamma']:.3f}  ({r2_str}, n={g['n']})")

    # --- Jupiter descriptive zero crossings ---
    jupiter_crossings = analyze_boundary_crossing(jupiter_grid)
    print(f"\n  Descriptive zero-sign changes found: {jupiter_crossings['n_sign_changes']}")
    for sc in jupiter_crossings.get("sign_changes", []):
        print(
            f"    p_ref={sc['p_ref']:.3f}: crossing at d ≈ {sc['d_cross_au']:.2f} AU "
            f"(γ: {sc['gamma_near']:+.3f} → {sc['gamma_far']:+.3f})"
        )

    # --- Saturn grid ---
    print("\n" + "=" * 70)
    print("SATURN: γ IN (DISTANCE, p_ref) SPACE")
    print("=" * 70)

    saturn_grid = analyze_body_by_distance_and_pref(by_target["saturn"])
    for bin_key, info in sorted(saturn_grid.items()):
        print(f"\n  {bin_key} ({info['mean_dist_au']:.2f} AU, n={info['n_records']}):")
        for p_str, g in sorted(info.get("gamma_by_pref", {}).items()):
            sign = "+" if g["gamma"] > 0 else ""
            r2_str = f"R²={g['R2']:.2f}" if not math.isnan(g["R2"]) else "R²=nan"
            print(f"    p={p_str}: γ = {sign}{g['gamma']:.3f}  ({r2_str}, n={g['n']})")

    saturn_crossings = analyze_boundary_crossing(saturn_grid)
    print(f"\n  Descriptive zero-sign changes found: {saturn_crossings['n_sign_changes']}")
    for sc in saturn_crossings.get("sign_changes", []):
        print(
            f"    p_ref={sc['p_ref']:.3f}: crossing at d ≈ {sc['d_cross_au']:.2f} AU "
            f"(γ: {sc['gamma_near']:+.3f} → {sc['gamma_far']:+.3f})"
        )

    # --- Comparison: all 8 bodies, composite gamma near p=0.1 ---
    print("\n" + "=" * 70)
    print("ALL BODIES: HISTORICAL COMPOSITE γ NEAR p_ref = 0.1")
    print("=" * 70)

    body_ranking = []
    for target in sorted(by_target.keys()):
        g = compute_gamma_at_pref(by_target[target], 0.1184, tol=0.02)
        body_ranking.append((target, g["gamma"], g["R2"], g["n"]))
    body_ranking.sort(key=lambda x: x[1] if not math.isnan(x[1]) else 0)

    for target, gamma, r2, n in body_ranking:
        marker = " <<< near zero" if abs(gamma) < 0.15 and not math.isnan(gamma) else ""
        print(f"  {target:10s}: γ = {gamma:+.3f}  (R² = {r2:.3f}, n = {n}){marker}")

    print("\n  'near zero' marks the fitted-slope reference only; it is not a class boundary.")

    # --- Φ evolution across Jupiter synodic cycle ---
    print("\n" + "=" * 70)
    print("JUPITER: Φ EVOLUTION BY DISTANCE (AT p_ref ≈ 0.1)")
    print("=" * 70)

    jup_records = [
        r
        for r in by_target["jupiter"]
        if abs(r.get("p_ref", 0) - 0.1184) < 0.02 and r["S_T"] > 0.01
    ]
    if jup_records:
        dists = np.array([r["dist_au"] for r in jup_records])
        phis = np.array([r.get("phi_normal", 1.0) for r in jup_records])
        ehs = np.array([r["E_H"] for r in jup_records])
        etas = np.array([r.get("eta_normal", r.get("eta_sim", 0)) for r in jup_records])
        sts = np.array([r["S_T"] for r in jup_records])

        # Bin by distance quintiles
        edges = np.percentile(dists, [0, 20, 40, 60, 80, 100])
        for i in range(5):
            mask = (dists >= edges[i]) & (dists <= edges[i + 1])
            if np.sum(mask) < 3:
                continue
            phi_valid = phis[mask]
            phi_valid = phi_valid[phi_valid > 0]
            print(
                f"  d ∈ [{edges[i]:.2f}, {edges[i + 1]:.2f}] AU: "
                f"Φ̄ = {np.mean(phi_valid):.3f}, "
                f"Φ>1: {100 * np.mean(phi_valid > 1):.0f}%, "
                f"E[H] = {np.mean(ehs[mask]):.1f}, "
                f"S̄_T = {np.mean(sts[mask]):.3f}, "
                f"n = {np.sum(mask)}"
            )

    # --- Save ---
    elapsed = time.time() - t0
    output = {
        "experiment": "jupiter_boundary",
        "claim_status": (
            "historical modeled diagnostic; classifier and phase-boundary interpretations retired"
        ),
        "description": "Archived gamma-by-distance and p_ref summaries for Jupiter and Saturn",
        "elapsed_s": round(elapsed, 1),
        "gamma_by_target_and_pref": target_gamma_table,
        "jupiter_distance_pref_grid": jupiter_grid,
        "jupiter_boundary_crossings": jupiter_crossings,
        "saturn_distance_pref_grid": saturn_grid,
        "saturn_boundary_crossings": saturn_crossings,
        "body_ranking_p01": [
            {"target": t, "gamma": g, "R2": r, "n": n} for t, g, r, n in body_ranking
        ],
    }

    out_path = _HERE / "jupiter_boundary_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
