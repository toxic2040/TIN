"""run_pair_gamma_mosaic.py — Per-pair gamma mosaic analysis.

Tests whether the trap/cluster classification is a global invariant
or decomposes into a local mosaic at the (source, dest) pair level.

For each pair, computes:
    gamma_pair = ln(phi_pair) / (E_H_pair * (-lambda_pair))

Then analyzes P(gamma_pair) for bimodality, gap structure, and sign census.

Reads:
    crawdad_contacts.Exp{1,2,3,6}_results.json
    domain_transfer_results.json

Writes:
    pair_gamma_mosaic_results.json
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

CRAWDAD_TRACES = {
    "Exp1": {"file": "crawdad_contacts.Exp1_results.json", "n_nodes": 9},
    "Exp2": {"file": "crawdad_contacts.Exp2_results.json", "n_nodes": 12},
    "Exp3": {"file": "crawdad_contacts.Exp3_results.json", "n_nodes": 41},
    "Exp6": {"file": "crawdad_contacts.Exp6_results.json", "n_nodes": 98},
}

DOMAIN_TRACES = ["cambridge", "leo"]

MIN_SEEDS = 3
MIN_PAIRS = 4


def _is_active(r):
    return (
        r.get("S_T", 0) > 0
        and r.get("eta_normal", 0) > 0
        and r.get("eta_lyap", 0) > 0
        and r.get("E_H", 0) > 0
    )


def _bimodality_coefficient(vals):
    """BC = (skewness^2 + 1) / kurtosis.  BC > 5/9 ≈ 0.556 suggests bimodality."""
    n = len(vals)
    if n < 4:
        return float("nan")
    m = np.mean(vals)
    s = np.std(vals, ddof=1)
    if s == 0:
        return float("nan")
    skew = float(np.mean(((vals - m) / s) ** 3))
    kurt = float(np.mean(((vals - m) / s) ** 4))
    if kurt == 0:
        return float("nan")
    return (skew**2 + 1) / kurt


def _max_gap(vals):
    """Largest gap in sorted values and its location."""
    sv = np.sort(vals)
    if len(sv) < 2:
        return 0.0, float("nan")
    diffs = np.diff(sv)
    idx = int(np.argmax(diffs))
    gap = float(diffs[idx])
    location = float((sv[idx] + sv[idx + 1]) / 2)
    return gap, location


def _load_crawdad(trace_name, meta):
    path = _HERE / meta["file"]
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def _load_domain_transfer():
    path = _HERE / "domain_transfer_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _compute_pair_gammas(rows, trace_label):
    """Compute gamma_pair for each (source, dest) at each p_eff.

    Returns list of dicts with pair info and gamma values.
    """
    active = [r for r in rows if _is_active(r)]
    if not active:
        return []

    # Group by (source, dest, p_eff)
    groups = defaultdict(list)
    for r in active:
        key = (r["source"], r["dest"], r["p_eff"])
        groups[key].append(r)

    pair_gammas = []
    for (src, dst, p_eff), seed_rows in groups.items():
        if len(seed_rows) < MIN_SEEDS:
            continue

        eta_normals = np.array([r["eta_normal"] for r in seed_rows])
        eta_lyaps = np.array([r["eta_lyap"] for r in seed_rows])
        ehs = np.array([r["E_H"] for r in seed_rows])

        eta_mean = float(np.mean(eta_normals))
        eta_lyap_mean = float(np.mean(eta_lyaps))
        eh_mean = float(np.mean(ehs))

        if eta_lyap_mean <= 0 or eh_mean <= 0:
            continue

        phi_pair = eta_mean / eta_lyap_mean

        # lambda from each seed
        lam_vals = []
        for r in seed_rows:
            if r["eta_lyap"] > 0 and r["E_H"] > 0:
                lam_vals.append(np.log(r["eta_lyap"]) / r["E_H"])
        if not lam_vals:
            continue
        lam_mean = float(np.mean(lam_vals))
        if lam_mean >= 0:
            continue

        if phi_pair <= 0:
            continue

        gamma_pair = np.log(phi_pair) / (eh_mean * (-lam_mean))

        pair_gammas.append(
            {
                "trace": trace_label,
                "source": src,
                "dest": dst,
                "p_eff": p_eff,
                "gamma_pair": float(gamma_pair),
                "phi_pair": float(phi_pair),
                "E_H": eh_mean,
                "lambda": lam_mean,
                "eta_mean": eta_mean,
                "eta_lyap_mean": eta_lyap_mean,
                "n_seeds": len(seed_rows),
            }
        )

    return pair_gammas


def _analyze_distribution(gamma_vals, label):
    """Analyze a set of gamma_pair values for bimodality and gap structure."""
    arr = np.array(gamma_vals)
    n = len(arr)
    if n < MIN_PAIRS:
        return None

    mean_g = float(np.mean(arr))
    std_g = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median_g = float(np.median(arr))
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q75 - q25

    n_pos = int(np.sum(arr > 0.1))
    n_neg = int(np.sum(arr < -0.1))
    n_near_zero = int(np.sum(np.abs(arr) <= 0.1))

    bc = _bimodality_coefficient(arr)
    gap, gap_loc = _max_gap(arr)

    # Gap relative to range
    rng = float(np.max(arr) - np.min(arr))
    gap_ratio = gap / rng if rng > 0 else 0.0

    # Does the gap straddle zero?
    gap_straddles_zero = False
    sv = np.sort(arr)
    diffs = np.diff(sv)
    if len(diffs) > 0:
        idx = int(np.argmax(diffs))
        gap_straddles_zero = bool(sv[idx] < 0 < sv[idx + 1])

    # Verdict
    if bc > 5 / 9 and gap_straddles_zero:
        verdict = "BIMODAL"
    elif std_g > 0 and abs(mean_g) / std_g > 2 and n_neg <= 1 and n_pos >= n - 2:
        verdict = "UNIMODAL_CLUSTER"
    elif std_g > 0 and abs(mean_g) / std_g > 2 and n_pos <= 1 and n_neg >= n - 2:
        verdict = "UNIMODAL_TRAP"
    elif n_neg > 0 and n_pos > 0 and not gap_straddles_zero:
        verdict = "MIXED_NO_GAP"
    else:
        verdict = "AMBIGUOUS"

    return {
        "label": label,
        "n_pairs": n,
        "mean": mean_g,
        "std": std_g,
        "median": median_g,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_near_zero": n_near_zero,
        "bimodality_coeff": bc,
        "max_gap": gap,
        "max_gap_location": gap_loc,
        "max_gap_ratio": gap_ratio,
        "gap_straddles_zero": gap_straddles_zero,
        "verdict": verdict,
    }


def main():
    print()
    print("  Pair-Level Gamma Mosaic Analysis")
    print("  " + "=" * 55)
    print()

    all_pair_gammas = []

    # --- CRAWDAD traces ---
    for trace_name, meta in CRAWDAD_TRACES.items():
        try:
            rows = _load_crawdad(trace_name, meta)
        except FileNotFoundError as e:
            print(f"  SKIP {trace_name}: {e}")
            continue
        pgs = _compute_pair_gammas(rows, trace_name)
        all_pair_gammas.extend(pgs)
        print(f"  Loaded {trace_name}: {len(pgs)} pair-p_eff combos")

    # --- Domain transfer traces (Cambridge, LEO) ---
    dt = _load_domain_transfer()
    for domain in DOMAIN_TRACES:
        if domain not in dt:
            print(f"  SKIP {domain}: not in domain_transfer_results.json")
            continue
        rows = dt[domain]["rows"]
        pgs = _compute_pair_gammas(rows, domain)
        all_pair_gammas.extend(pgs)
        print(f"  Loaded {domain}: {len(pgs)} pair-p_eff combos")

    print(f"\n  Total pair-gamma entries: {len(all_pair_gammas)}")

    if not all_pair_gammas:
        print("  ERROR: no data. Exiting.")
        return

    # --- Per-trace, per-p_eff analysis ---
    print()
    print("  Per-Trace Distribution Analysis")
    print("  " + "-" * 55)
    hdr = f"  {'Trace':<12s} {'p_eff':>5s} {'n':>4s} {'mean':>7s} {'std':>6s} "
    hdr += f"{'n+':>3s} {'n-':>3s} {'gap':>6s} {'BC':>5s} {'verdict':<18s}"
    print(hdr)

    trace_results = defaultdict(dict)
    for pg in all_pair_gammas:
        key = (pg["trace"], pg["p_eff"])
        trace_results[key] = trace_results.get(key, [])
    # Regroup
    trace_p_groups = defaultdict(list)
    for pg in all_pair_gammas:
        trace_p_groups[(pg["trace"], pg["p_eff"])].append(pg["gamma_pair"])

    per_trace_stats = {}
    for (trace, p_eff), gvals in sorted(trace_p_groups.items()):
        label = f"{trace}_p{p_eff}"
        stats = _analyze_distribution(gvals, label)
        if stats is None:
            continue
        per_trace_stats[label] = stats

        row = f"  {trace:<12s} {p_eff:>5.2f} {stats['n_pairs']:>4d} "
        row += f"{stats['mean']:>+7.3f} {stats['std']:>6.3f} "
        row += f"{stats['n_positive']:>3d} {stats['n_negative']:>3d} "
        row += f"{stats['max_gap']:>6.3f} {stats['bimodality_coeff']:>5.3f} "
        row += f"{stats['verdict']:<18s}"
        print(row)

    # --- Per-trace pooled across p_eff ---
    print()
    print("  Per-Trace Pooled (all p_eff)")
    print("  " + "-" * 55)

    traces = sorted(set(pg["trace"] for pg in all_pair_gammas))
    pooled_results = {}
    for trace in traces:
        # Average gamma_pair per unique pair across p_eff values
        pair_means = defaultdict(list)
        for pg in all_pair_gammas:
            if pg["trace"] == trace:
                pair_means[(pg["source"], pg["dest"])].append(pg["gamma_pair"])
        avg_gammas = [float(np.mean(vs)) for vs in pair_means.values()]
        stats = _analyze_distribution(avg_gammas, trace)
        if stats is None:
            continue
        pooled_results[trace] = stats

        row = f"  {trace:<12s} n={stats['n_pairs']:<3d} "
        row += f"mean={stats['mean']:>+.3f}  std={stats['std']:.3f}  "
        row += f"[{stats['min']:+.2f}, {stats['max']:+.2f}]  "
        row += f"n+={stats['n_positive']}  n-={stats['n_negative']}  "
        row += f"gap={stats['max_gap']:.3f}  BC={stats['bimodality_coeff']:.3f}  "
        row += f"{stats['verdict']}"
        print(row)

    # --- Cross-trace pooled: all cluster-class traces ---
    print()
    print("  Cross-Trace Pooled (cluster class: Exp1-3, Exp6, Cambridge, LEO)")
    print("  " + "-" * 55)

    cluster_gammas = []
    for pg in all_pair_gammas:
        # All loaded traces are cluster-class
        cluster_gammas.append(pg["gamma_pair"])

    if cluster_gammas:
        stats = _analyze_distribution(cluster_gammas, "all_cluster")
        print(f"  n={stats['n_pairs']}  mean={stats['mean']:+.3f}  std={stats['std']:.3f}")
        print(f"  range=[{stats['min']:+.3f}, {stats['max']:+.3f}]")
        print(f"  n+={stats['n_positive']}  n-={stats['n_negative']}  n~0={stats['n_near_zero']}")
        print(
            f"  BC={stats['bimodality_coeff']:.3f}  max_gap={stats['max_gap']:.3f} at {stats['max_gap_location']:+.3f}"
        )
        print(f"  gap_straddles_zero={stats['gap_straddles_zero']}")
        print(f"  VERDICT: {stats['verdict']}")
    else:
        stats = None

    # --- Consistency check: recover config-level gamma ---
    print()
    print("  Consistency: config-level gamma recovery")
    print("  " + "-" * 55)
    for trace in traces:
        for p_target in [0.05, 0.1, 0.3]:
            entries = [
                pg
                for pg in all_pair_gammas
                if pg["trace"] == trace and abs(pg["p_eff"] - p_target) < 0.001
            ]
            if len(entries) < 5:
                continue
            # Weighted mean (weight by E_H as proxy for leverage)
            gammas = np.array([e["gamma_pair"] for e in entries])
            ehs = np.array([e["E_H"] for e in entries])
            weights = ehs / np.sum(ehs)
            gamma_weighted = float(np.sum(gammas * weights))
            gamma_simple = float(np.mean(gammas))
            print(
                f"  {trace:<12s} p={p_target:.2f}: "
                f"gamma_simple={gamma_simple:+.3f}  "
                f"gamma_weighted={gamma_weighted:+.3f}  "
                f"(n={len(entries)} pairs)"
            )

    # --- Mosaic frontier: fraction of pairs with sign opposite to config mean ---
    print()
    print("  Mosaic Frontier: pairs with opposite sign")
    print("  " + "-" * 55)
    for trace in traces:
        pair_means = defaultdict(list)
        for pg in all_pair_gammas:
            if pg["trace"] == trace:
                pair_means[(pg["source"], pg["dest"])].append(pg["gamma_pair"])
        avg_gammas = {k: float(np.mean(vs)) for k, vs in pair_means.items()}
        if not avg_gammas:
            continue
        config_mean = float(np.mean(list(avg_gammas.values())))
        config_sign = 1 if config_mean > 0 else -1
        n_opposite = sum(1 for g in avg_gammas.values() if g * config_sign < 0)
        n_total = len(avg_gammas)
        frac = n_opposite / n_total if n_total > 0 else 0
        print(
            f"  {trace:<12s}: config_gamma={config_mean:+.3f}  "
            f"opposite_sign={n_opposite}/{n_total} ({frac:.1%})"
        )
        if n_opposite > 0:
            opposites = [(k, v) for k, v in avg_gammas.items() if v * config_sign < 0]
            for (s, d), g in sorted(opposites, key=lambda x: x[1]):
                print(f"    {s}->{d}: gamma={g:+.3f}")

    # --- Save ---
    output = {
        "per_trace_per_p": per_trace_stats,
        "per_trace_pooled": pooled_results,
        "cross_trace_cluster": stats,
        "pair_gammas": all_pair_gammas,
    }
    out_path = _HERE / "pair_gamma_mosaic_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, allow_nan=True)
    print(f"\n  Saved -> {out_path.name} ({os.path.getsize(out_path) / 1024:.1f} KB)")
    print("  DONE.")


if __name__ == "__main__":
    main()
