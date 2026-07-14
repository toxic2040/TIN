"""Historical Venus epoch and distance-bin gamma diagnostic.

This script reproduces an archived comparison of Venus model outputs across
synodic-angle and distance bins.

Bins Venus synodic sweep data by Sun-Earth-Probe angle (SEP):
  - Opposition:   SEP ≥ 120°
  - Quadrature:   20° ≤ SEP < 120°
  - Conjunction:  SEP < 20°

Computes γ = ∂ln(Φ)/∂E[H] for each bin independently.

The former trap/cluster and phase-transition interpretations are retired.
Fitted-slope sign changes are descriptive properties of the loaded rows, not a
classifier, mechanism change, or mission-design result.

Also extends the analysis to all 8 orbital targets for comparison.

Reads:  runs/phi_decompose_results.json (per-seed, per-epoch data)
        runs/synodic_sweep_results.json (Venus synodic time series)
        runs/multi_body_sweep_results.json (multi-body synodic time series)
Writes: runs/venus_epoch_decomposition_results.json
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
    """Simple OLS slope and R² for y = a + b*x."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan"), int(len(x))
    A = np.vstack([x, np.ones_like(x)]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    b, a = result[0]
    y_pred = b * x + a
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), float(r2), int(len(x))


def compute_gamma_from_records(records, p_eff_ref=0.1):
    """Compute γ from a set of records at a reference p_eff.

    γ = slope of ln(Φ) vs E[H] at fixed p_eff.
    Records must have 'E_H', 'phi_normal', 'p_ref' (or 'p_eff').
    """
    # Filter to reference p_eff
    filtered = []
    for r in records:
        p = r.get("p_ref", r.get("p_eff", None))
        if p is None:
            continue
        if abs(p - p_eff_ref) > 0.02:
            continue
        phi = r.get("phi_normal", None)
        eh = r.get("E_H", None)
        if phi is not None and phi > 0 and eh is not None and eh > 0:
            filtered.append((eh, math.log(phi)))

    if len(filtered) < 3:
        return {"gamma": float("nan"), "R2": float("nan"), "n_pts": len(filtered)}

    x = [f[0] for f in filtered]
    y = [f[1] for f in filtered]
    slope, r2, n = _ols_slope(x, y)
    return {"gamma": slope, "R2": r2, "n_pts": n}


def analyze_phi_decompose_by_epoch(phi_data):
    """Analyze phi_decompose data, computing per-target, per-epoch-bin gamma.

    Uses epoch_day as a proxy for synodic phase. Maps epoch_day to SEP
    approximately using the synodic period geometry.
    """
    results = phi_data["results"]

    # Group records by target
    by_target = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue
        by_target[r["target"]].append(r)

    target_analysis = {}
    for target, records in sorted(by_target.items()):
        # Compute aggregate gamma
        agg = compute_gamma_from_records(records)

        # Group by distance bins as a proxy for synodic phase
        # (distance correlates with SEP for heliocentric targets)
        dists = [r.get("dist_au", 0) for r in records if r.get("dist_au", 0) > 0]
        if not dists:
            target_analysis[target] = {"aggregate": agg, "by_phase": {}}
            continue

        d_arr = np.array(dists)
        d_terciles = np.percentile(d_arr, [33.3, 66.7])

        phase_bins = {"near": [], "mid": [], "far": []}
        for r in records:
            d = r.get("dist_au", 0)
            if d <= 0:
                continue
            if d <= d_terciles[0]:
                phase_bins["near"].append(r)
            elif d <= d_terciles[1]:
                phase_bins["mid"].append(r)
            else:
                phase_bins["far"].append(r)

        by_phase = {}
        for phase, precs in phase_bins.items():
            g = compute_gamma_from_records(precs)
            by_phase[phase] = {
                **g,
                "n_records": len(precs),
                "mean_dist_au": float(np.mean([r.get("dist_au", 0) for r in precs]))
                if precs
                else 0,
            }

        target_analysis[target] = {
            "aggregate": agg,
            "n_total_records": len(records),
            "dist_terciles_au": [float(d_terciles[0]), float(d_terciles[1])],
            "by_phase": by_phase,
        }

    return target_analysis


def analyze_synodic_time_series(synodic_data):
    """Analyze synodic sweep time series with SEP-based binning."""
    ts = synodic_data.get("time_series", [])
    if not ts:
        return {}

    # Bin by SEP
    sep_bins = {
        "conjunction": {"sep_range": "SEP < 20°", "records": []},
        "quadrature": {"sep_range": "20° ≤ SEP < 120°", "records": []},
        "opposition": {"sep_range": "SEP ≥ 120°", "records": []},
    }

    for r in ts:
        sep = r.get("sep_deg", 90)
        if sep < 20:
            sep_bins["conjunction"]["records"].append(r)
        elif sep < 120:
            sep_bins["quadrature"]["records"].append(r)
        else:
            sep_bins["opposition"]["records"].append(r)

    analysis = {}
    for phase, info in sep_bins.items():
        recs = info["records"]
        if not recs:
            analysis[phase] = {"n_epochs": 0}
            continue

        drs = [r["DR"] for r in recs if r.get("DR", 0) > 0]
        etas = [r["eta"] for r in recs if r.get("eta", 0) > 0]
        sts = [r["S_T"] for r in recs]
        dists = [r["dist_au"] for r in recs if r.get("dist_au", 0) > 0]

        analysis[phase] = {
            "sep_range": info["sep_range"],
            "n_epochs": len(recs),
            "mean_DR": float(np.mean(drs)) if drs else 0,
            "std_DR": float(np.std(drs)) if drs else 0,
            "mean_eta": float(np.mean(etas)) if etas else 0,
            "mean_S_T": float(np.mean(sts)) if sts else 0,
            "min_S_T": float(np.min(sts)) if sts else 0,
            "mean_dist_au": float(np.mean(dists)) if dists else 0,
            "n_blackout": sum(1 for r in recs if r.get("S_T", 0) == 0),
        }

    return analysis


def analyze_multi_body_by_sep(mb_data):
    """Analyze multi-body sweep, binning each target by distance terciles."""
    ts = mb_data.get("time_series", [])
    if not ts:
        return {}

    by_target = defaultdict(list)
    for r in ts:
        by_target[r["target"]].append(r)

    analysis = {}
    for target, records in sorted(by_target.items()):
        dists = [r["dist_au"] for r in records if r.get("dist_au", 0) > 0]
        if len(dists) < 10:
            continue

        d_arr = np.array(dists)
        d_terc = np.percentile(d_arr, [33.3, 66.7])

        phase_stats = {}
        for phase, lo, hi in [
            ("near", 0, d_terc[0]),
            ("mid", d_terc[0], d_terc[1]),
            ("far", d_terc[1], 100),
        ]:
            precs = [r for r in records if lo < r.get("dist_au", 0) <= hi]
            if phase == "near":
                precs = [r for r in records if 0 < r.get("dist_au", 0) <= hi]
            if not precs:
                continue
            drs = [r["DR"] for r in precs if r.get("DR", 0) > 0]
            etas = [r["eta"] for r in precs if r.get("eta", 0) > 0]
            sts = [r["S_T"] for r in precs]

            phase_stats[phase] = {
                "n_epochs": len(precs),
                "mean_DR": float(np.mean(drs)) if drs else 0,
                "mean_eta": float(np.mean(etas)) if etas else 0,
                "mean_S_T": float(np.mean(sts)) if sts else 0,
                "n_blackout": sum(1 for r in precs if r.get("S_T", 0) == 0),
                "mean_dist_au": float(np.mean([r["dist_au"] for r in precs])),
            }

        analysis[target] = {
            "dist_terciles_au": [float(d_terc[0]), float(d_terc[1])],
            "by_phase": phase_stats,
        }

    return analysis


def main():
    t0 = time.time()
    print("=" * 70)
    print("HISTORICAL VENUS EPOCH/DISTANCE-BIN GAMMA DIAGNOSTIC")
    print("CLASSIFICATION-BOUNDARY INTERPRETATION RETIRED")
    print("=" * 70)

    # --- phi_decompose analysis (all 8 targets, distance-binned gamma) ---
    phi_path = _HERE / "phi_decompose_results.json"
    print(f"\nLoading {phi_path.name} ...")
    phi_data = _load_json(phi_path)
    phi_analysis = analyze_phi_decompose_by_epoch(phi_data)

    print("\n--- γ by synodic phase (distance terciles, p_ref ≈ 0.1) ---")
    for target, info in sorted(phi_analysis.items()):
        agg = info["aggregate"]
        print(
            f"\n  {target:10s} [aggregate γ = {agg['gamma']:+.3f}, "
            f"R² = {agg['R2']:.3f}, n = {agg['n_pts']}]"
        )
        for phase, pinfo in sorted(info.get("by_phase", {}).items()):
            print(
                f"    {phase:6s}: γ = {pinfo['gamma']:+.3f}, "
                f"R² = {pinfo['R2']:.3f}, n = {pinfo['n_pts']}, "
                f"d̄ = {pinfo['mean_dist_au']:.2f} AU"
            )

    # --- Venus-specific: synodic time series with SEP bins ---
    venus_synodic = {}
    synodic_path = _HERE / "synodic_sweep_results.json"
    if synodic_path.exists():
        print(f"\nLoading {synodic_path.name} ...")
        synodic_data = _load_json(synodic_path)
        venus_synodic = analyze_synodic_time_series(synodic_data)

        print("\n--- Venus synodic by SEP ---")
        for phase in ["opposition", "quadrature", "conjunction"]:
            info = venus_synodic.get(phase, {})
            if info.get("n_epochs", 0) > 0:
                print(
                    f"  {phase:12s}: {info['n_epochs']:3d} epochs, "
                    f"DR̄ = {info['mean_DR']:.3f}, η̄ = {info['mean_eta']:.3f}, "
                    f"S̄_T = {info['mean_S_T']:.3f}, "
                    f"blackouts = {info['n_blackout']}"
                )

    # --- Multi-body: distance-binned DR/eta for Venus, Mars, Jupiter ---
    multi_body_analysis = {}
    mb_path = _HERE / "multi_body_sweep_results.json"
    if mb_path.exists():
        print(f"\nLoading {mb_path.name} ...")
        mb_data = _load_json(mb_path)
        multi_body_analysis = analyze_multi_body_by_sep(mb_data)

        print("\n--- Multi-body distance-binned DR ---")
        for target, info in sorted(multi_body_analysis.items()):
            print(f"\n  {target}:")
            for phase in ["near", "mid", "far"]:
                pinfo = info.get("by_phase", {}).get(phase, {})
                if pinfo:
                    print(
                        f"    {phase:5s}: DR̄ = {pinfo['mean_DR']:.3f}, "
                        f"η̄ = {pinfo['mean_eta']:.3f}, "
                        f"S̄_T = {pinfo['mean_S_T']:.3f}, "
                        f"d̄ = {pinfo['mean_dist_au']:.2f} AU, "
                        f"blackout = {pinfo['n_blackout']}"
                    )

    # --- Descriptive zero-sign check in the loaded distance bins ---
    print("\n" + "=" * 70)
    print("HISTORICAL VENUS ZERO-SIGN CHECK — NOT A CLASSIFIER")
    print("=" * 70)

    venus_phi = phi_analysis.get("venus", {})
    near = venus_phi.get("by_phase", {}).get("near", {})
    far = venus_phi.get("by_phase", {}).get("far", {})

    if near.get("gamma") is not None and far.get("gamma") is not None:
        g_near = near["gamma"]
        g_far = far["gamma"]
        print(f"\n  Venus near-distance tercile: γ = {g_near:+.3f}")
        print(f"  Venus far-distance tercile:  γ = {g_far:+.3f}")
        if not math.isnan(g_near) and not math.isnan(g_far):
            if g_near > 0 and g_far < 0:
                print("\n  Descriptive fitted-slope sign change in the tested bins.")
                print("  This does not establish a classification or phase boundary.")
            elif g_near * g_far > 0:
                print(
                    f"\n  Both fitted slopes are {'positive' if g_near > 0 else 'negative'} "
                    "in the tested bins."
                )
            else:
                print(f"\n  Loaded-bin slopes: near γ = {g_near:+.3f}, far γ = {g_far:+.3f}")
    else:
        print("\n  Insufficient data for Venus epoch decomposition.")

    # --- Save ---
    elapsed = time.time() - t0
    output = {
        "experiment": "venus_epoch_decomposition",
        "claim_status": (
            "historical modeled diagnostic; trap/cluster and phase-boundary interpretations retired"
        ),
        "description": "Archived gamma summaries by distance and synodic-angle bin",
        "elapsed_s": round(elapsed, 1),
        "phi_decompose_analysis": phi_analysis,
        "venus_synodic_by_sep": venus_synodic,
        "multi_body_by_distance": multi_body_analysis,
    }

    out_path = _HERE / "venus_epoch_decomposition_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
