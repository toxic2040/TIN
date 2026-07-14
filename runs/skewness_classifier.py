"""Historical skewness comparison for two assigned source-domain groups.

The original Phase 2 analysis assigned orbital rows to a "trap" group and
Bluetooth rows to a "cluster" group, then optimized a threshold on those same
labels. Those labels are not independent of the retired gamma-classification
program, so the calculation is retained only as an in-sample historical
comparison.

This script reproduces the archived calculation:
1. Compare skewness across the assigned orbital and Bluetooth groups
2. Measure in-sample threshold agreement with those assigned labels
3. Record the historical threshold without treating it as a classifier
4. How does skewness vary with p_eff and E[H]?
5. Bootstrap CI on the mean skewness per assigned group

No independent classifier, cross-domain replacement, or current decision rule
is claimed.

Reads:  runs/phi_decompose_results.json
        runs/crawdad_contacts.Exp{1,2,3,6}_results.json
        runs/vehicular_gamma_results.json
Writes: runs/skewness_classifier_results.json
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


def _skewness(arr):
    """Sample skewness (Fisher)."""
    n = len(arr)
    if n < 8:
        return float("nan")
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s < 1e-15:
        return float("nan")
    return float(np.mean(((arr - m) / s) ** 3) * n * n / ((n - 1) * (n - 2)))


def extract_per_config_skewness(phi_data, min_seeds=15):
    """Extract skewness for each (target, n_orb, p_ref) config."""
    results = phi_data["results"]

    groups = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue
        eta = r["eta_normal"]
        if eta is not None and eta > 0:
            key = (r["target"], r["n_orb"], r["p_ref"], r.get("alpha", 1))
            groups[key].append(math.log(eta))

    configs = []
    for key, ln_etas in groups.items():
        if len(ln_etas) < min_seeds:
            continue
        arr = np.array(ln_etas)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_seeds:
            continue

        target, n_orb, p_ref, alpha = key
        configs.append(
            {
                "target": target,
                "n_orb": n_orb,
                "p_ref": p_ref,
                "alpha": alpha,
                "skewness": _skewness(arr),
                "mean_ln_eta": float(np.mean(arr)),
                "var_ln_eta": float(np.var(arr, ddof=1)),
                "n_seeds": len(arr),
                "assigned_group": "orbital",
            }
        )

    return configs


def extract_crawdad_skewness(cdata, trace_name, min_seeds=5):
    """Extract skewness for each CRAWDAD config."""
    results = cdata["results"]

    groups = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue
        eta = r["eta_normal"]
        if eta is not None and eta > 0:
            key = (r["source"], r["dest"], r["ttl"], r["p_eff"])
            groups[key].append(math.log(eta))

    configs = []
    for key, ln_etas in groups.items():
        if len(ln_etas) < min_seeds:
            continue
        arr = np.array(ln_etas)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_seeds:
            continue

        source, dest, ttl, p_eff = key
        configs.append(
            {
                "trace": trace_name,
                "p_eff": p_eff,
                "skewness": _skewness(arr),
                "mean_ln_eta": float(np.mean(arr)),
                "var_ln_eta": float(np.var(arr, ddof=1)),
                "n_seeds": len(arr),
                "assigned_group": "bluetooth",
            }
        )

    return configs


def find_optimal_threshold(trap_skews, cluster_skews):
    """Find the historical split maximizing in-sample assigned-group agreement."""
    all_vals = sorted(set(trap_skews + cluster_skews))
    if len(all_vals) < 2:
        return float("nan"), 0.0

    best_thresh = float("nan")
    best_acc = 0.0

    trap_arr = np.array(trap_skews)
    clust_arr = np.array(cluster_skews)

    for i in range(len(all_vals) - 1):
        thresh = (all_vals[i] + all_vals[i + 1]) / 2.0

        # Preserve the original in-sample split convention. The group labels are
        # assigned from source domain; this is not an independent classifier.
        trap_correct = np.sum(trap_arr > thresh)
        clust_correct = np.sum(clust_arr <= thresh)
        acc = (trap_correct + clust_correct) / (len(trap_arr) + len(clust_arr))

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

        # Also try other direction
        trap_correct2 = np.sum(trap_arr < thresh)
        clust_correct2 = np.sum(clust_arr >= thresh)
        acc2 = (trap_correct2 + clust_correct2) / (len(trap_arr) + len(clust_arr))

        if acc2 > best_acc:
            best_acc = acc2
            best_thresh = thresh

    return float(best_thresh), float(best_acc)


def bootstrap_mean_ci(values, n_boot=1000, ci=0.95):
    """Bootstrap CI for the mean."""
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_means, 100 * (1 + ci) / 2)
    return float(lo), float(hi)


def main():
    t0 = time.time()
    print("=" * 70)
    print("HISTORICAL SKEWNESS GROUP COMPARISON — CLASSIFIER RETIRED")
    print("=" * 70)

    # --- Assigned orbital group (historically called trap) ---
    phi_path = _HERE / "phi_decompose_results.json"
    print(f"\nLoading {phi_path.name} ...")
    phi_data = _load_json(phi_path)
    trap_configs = extract_per_config_skewness(phi_data, min_seeds=15)
    print(f"  {len(trap_configs):,} assigned orbital rows with ≥15 seeds")

    # --- Assigned Bluetooth group (historically called cluster) ---
    cluster_configs = []
    for trace in ["Exp1", "Exp2", "Exp3", "Exp6"]:
        cpath = _HERE / f"crawdad_contacts.{trace}_results.json"
        if cpath.exists():
            print(f"Loading {cpath.name} ...")
            cdata = _load_json(cpath)
            cfgs = extract_crawdad_skewness(cdata, trace, min_seeds=5)
            cluster_configs.extend(cfgs)
            print(f"  {len(cfgs):,} {trace} configs")

    # --- Filter to finite skewness ---
    trap_skews = [c["skewness"] for c in trap_configs if np.isfinite(c["skewness"])]
    cluster_skews = [c["skewness"] for c in cluster_configs if np.isfinite(c["skewness"])]

    print(
        f"\n  Total: {len(trap_skews)} assigned orbital, "
        f"{len(cluster_skews)} assigned Bluetooth rows (finite skewness)"
    )

    # --- Distribution comparison ---
    print("\n" + "=" * 70)
    print("HISTORICAL SKEWNESS DISTRIBUTION BY ASSIGNED GROUP")
    print("=" * 70)

    trap_arr = np.array(trap_skews)
    clust_arr = np.array(cluster_skews)

    trap_lo, trap_hi = bootstrap_mean_ci(trap_skews)
    clust_lo, clust_hi = bootstrap_mean_ci(cluster_skews)

    print(
        f"\n  ORBITAL ASSIGNED GROUP: mean = {np.mean(trap_arr):+.3f}  "
        f"95% CI [{trap_lo:+.3f}, {trap_hi:+.3f}]"
    )
    print(f"           median = {np.median(trap_arr):+.3f}, std = {np.std(trap_arr):.3f}")
    print(f"           {100 * np.mean(trap_arr < 0):.1f}% negative")

    print(
        f"\n  BLUETOOTH ASSIGNED GROUP: mean = {np.mean(clust_arr):+.3f}  "
        f"95% CI [{clust_lo:+.3f}, {clust_hi:+.3f}]"
    )
    print(f"           median = {np.median(clust_arr):+.3f}, std = {np.std(clust_arr):.3f}")
    print(f"           {100 * np.mean(clust_arr < 0):.1f}% negative")

    # --- CIs overlap? ---
    ci_overlap = trap_hi > clust_lo and clust_hi > trap_lo
    print(f"\n  95% CIs overlap: {ci_overlap}")

    # --- Historical in-sample assigned-group separation ---
    print("\n--- Historical in-sample assigned-group separation (descriptive only) ---")
    gap = np.mean(trap_arr) - np.mean(clust_arr)
    print(f"  Mean gap: {gap:+.3f} (assigned orbital minus assigned Bluetooth)")

    # --- Optimal threshold ---
    thresh, acc = find_optimal_threshold(trap_skews, cluster_skews)
    print(f"\n  Historical in-sample split: skewness = {thresh:+.3f}")
    print(f"  Assigned-group agreement at split: {100 * acc:.1f}%")

    # --- Per-target skewness within the assigned orbital group ---
    print("\n" + "=" * 70)
    print("PER-TARGET SKEWNESS")
    print("=" * 70)

    by_target = defaultdict(list)
    for c in trap_configs:
        if np.isfinite(c["skewness"]):
            by_target[c["target"]].append(c["skewness"])

    for target in sorted(by_target.keys()):
        vals = np.array(by_target[target])
        lo, hi = bootstrap_mean_ci(list(vals)) if len(vals) > 10 else (float("nan"), float("nan"))
        print(
            f"  {target:10s}: mean = {np.mean(vals):+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
            f"neg = {100 * np.mean(vals < 0):.0f}%  n = {len(vals)}"
        )

    by_trace = defaultdict(list)
    for c in cluster_configs:
        if np.isfinite(c["skewness"]):
            by_trace[c.get("trace", "unknown")].append(c["skewness"])

    for trace in sorted(by_trace.keys()):
        vals = np.array(by_trace[trace])
        lo, hi = bootstrap_mean_ci(list(vals)) if len(vals) > 10 else (float("nan"), float("nan"))
        print(
            f"  {trace:10s}: mean = {np.mean(vals):+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
            f"neg = {100 * np.mean(vals < 0):.0f}%  n = {len(vals)}"
        )

    # --- p_eff dependence ---
    print("\n" + "=" * 70)
    print("HISTORICAL SKEWNESS vs p_eff BY ASSIGNED GROUP")
    print("=" * 70)

    by_pref = defaultdict(lambda: {"trap": [], "cluster": []})
    for c in trap_configs:
        if np.isfinite(c["skewness"]):
            p = round(c["p_ref"], 2)
            by_pref[p]["trap"].append(c["skewness"])
    for c in cluster_configs:
        if np.isfinite(c["skewness"]):
            p = round(c["p_eff"], 2)
            by_pref[p]["cluster"].append(c["skewness"])

    for p in sorted(by_pref.keys()):
        t_vals = by_pref[p]["trap"]
        c_vals = by_pref[p]["cluster"]
        t_mean = f"{np.mean(t_vals):+.3f}" if t_vals else "  ---"
        c_mean = f"{np.mean(c_vals):+.3f}" if c_vals else "  ---"
        print(
            f"  p = {p:.2f}:  orbital = {t_mean} (n={len(t_vals):4d})  "
            f"Bluetooth = {c_mean} (n={len(c_vals):4d})"
        )

    # --- Historical comparison with gamma-assigned groups ---
    print("\n" + "=" * 70)
    print("HISTORICAL GAMMA-ASSIGNED GROUP COMPARISON — NOT INDEPENDENT")
    print("=" * 70)

    # For each orbital target, does skewness sign agree with γ sign?
    # Preserve the original inner/outer orbital subset comparison without
    # interpreting either subset as a class.
    inner = ["mars", "mercury", "venus", "ceres"]
    outer = ["jupiter", "saturn", "europa", "titan"]

    inner_skews = [s for t in inner for s in by_target.get(t, [])]
    outer_skews = [s for t in outer for s in by_target.get(t, [])]

    print(
        f"\n  Inner orbital subset:       mean skew = {np.mean(inner_skews):+.3f} "
        f"(n={len(inner_skews)})"
    )
    print(
        f"  Outer orbital subset:      mean skew = {np.mean(outer_skews):+.3f} "
        f"(n={len(outer_skews)})"
    )
    print(
        f"  Bluetooth assigned group:  mean skew = {np.mean(clust_arr):+.3f} (n={len(clust_arr)})"
    )

    if inner_skews and outer_skews:
        print(f"\n  Inner vs outer difference: {np.mean(inner_skews) - np.mean(outer_skews):+.3f}")
        print(
            f"  Outer orbital vs Bluetooth difference: "
            f"{np.mean(outer_skews) - np.mean(clust_arr):+.3f}"
        )

    # --- Save ---
    elapsed = time.time() - t0
    output = {
        "experiment": "skewness_classifier",
        "description": (
            "Historical in-sample skewness comparison for source-domain-assigned groups; "
            "the classifier interpretation is retired"
        ),
        "claim_status": "historical diagnostic; not an independent classifier or decision rule",
        "label_semantics": {
            "trap_stats": "assigned orbital group",
            "cluster_stats": "assigned Bluetooth group",
        },
        "elapsed_s": round(elapsed, 1),
        "trap_stats": {
            "mean": float(np.mean(trap_arr)),
            "median": float(np.median(trap_arr)),
            "ci_95": [trap_lo, trap_hi],
            "frac_negative": float(np.mean(trap_arr < 0)),
            "n": len(trap_skews),
        },
        "cluster_stats": {
            "mean": float(np.mean(clust_arr)),
            "median": float(np.median(clust_arr)),
            "ci_95": [clust_lo, clust_hi],
            "frac_negative": float(np.mean(clust_arr < 0)),
            "n": len(cluster_skews),
        },
        "optimal_threshold": thresh,
        "accuracy_at_threshold": acc,
        "per_target": {
            target: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
            for target, vals in by_target.items()
        },
        "per_trace": {
            trace: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
            for trace, vals in by_trace.items()
        },
    }

    out_path = _HERE / "skewness_classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
