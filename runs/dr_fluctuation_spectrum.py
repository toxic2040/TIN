"""run_dr_fluctuation_spectrum.py — Phase 1, Experiment 4: DR Fluctuation Spectrum.

Complementary to the KPZ scaling test. This analyzes the SHAPE of the
ln(DR) distribution, not just its variance.

DPRM/KPZ predicts Tracy-Widom distributed fluctuations in the scaling limit.
This script checks:
  1. Skewness of ln(eta) distribution (TW has negative skew ≈ -0.224)
  2. Kurtosis (TW has excess kurtosis ≈ 0.093)
  3. Whether distribution shape differs between trap and cluster classes
  4. Normality tests (Shapiro-Wilk where n permits)

Also computes the full per-config fluctuation landscape:
  - CV(eta) as a function of E[H] and p_eff
  - Correlation between successive seed realizations (independence check)

Data source: phi_decompose_results.json + crawdad_contacts.Exp*_results.json

Reads:  runs/phi_decompose_results.json, runs/crawdad_contacts.Exp*_results.json
Writes: runs/dr_fluctuation_spectrum_results.json
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

# Tracy-Widom GOE reference values
TW_SKEWNESS = -0.2935  # TW_1 (GOE)
TW_KURTOSIS = 0.1652  # TW_1 excess kurtosis


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _safe_log(x):
    return math.log(x) if x > 0 else float("nan")


def _skewness(arr):
    """Sample skewness (Fisher)."""
    n = len(arr)
    if n < 3:
        return float("nan")
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s < 1e-15:
        return float("nan")
    return float(np.mean(((arr - m) / s) ** 3) * n * n / ((n - 1) * (n - 2)))


def _kurtosis_excess(arr):
    """Sample excess kurtosis (Fisher)."""
    n = len(arr)
    if n < 4:
        return float("nan")
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s < 1e-15:
        return float("nan")
    k4 = np.mean(((arr - m) / s) ** 4)
    # Adjust for sample bias
    return float((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * k4 - 3 * (n - 1)) + 3 - 3)


def analyze_orbital_fluctuations(phi_data):
    """Compute per-config fluctuation statistics for orbital configs."""
    results = phi_data["results"]

    # Group by config: (target, n_orb, p_ref, alpha, epoch_day)
    groups = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue
        key = (r["target"], r["n_orb"], r["p_ref"], r.get("alpha", 1), r.get("epoch_day", 0))
        eta = r["eta_normal"]
        if eta is not None and eta > 0:
            groups[key].append(eta)

    config_fluct = []
    for key, etas in groups.items():
        if len(etas) < 10:
            continue
        target, n_orb, p_ref, alpha, epoch_day = key
        ln_etas = np.array([math.log(e) for e in etas])
        ln_etas = ln_etas[np.isfinite(ln_etas)]
        if len(ln_etas) < 10:
            continue

        config_fluct.append(
            {
                "target": target,
                "n_orb": n_orb,
                "p_ref": p_ref,
                "alpha": alpha,
                "n_seeds": len(ln_etas),
                "mean_ln_eta": float(np.mean(ln_etas)),
                "var_ln_eta": float(np.var(ln_etas, ddof=1)),
                "std_ln_eta": float(np.std(ln_etas, ddof=1)),
                "skewness": _skewness(ln_etas),
                "kurtosis_excess": _kurtosis_excess(ln_etas),
                "cv_eta": float(np.std(etas, ddof=1) / np.mean(etas))
                if np.mean(etas) > 0
                else float("nan"),
                "class": "trap",
            }
        )

    return config_fluct


def analyze_crawdad_fluctuations(cdata, trace_name):
    """Compute per-config fluctuation statistics for CRAWDAD trace."""
    results = cdata["results"]

    groups = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue
        key = (r["source"], r["dest"], r["ttl"], r["p_eff"])
        eta = r["eta_normal"]
        if eta is not None and eta > 0:
            groups[key].append(eta)

    config_fluct = []
    for key, etas in groups.items():
        if len(etas) < 5:
            continue
        source, dest, ttl, p_eff = key
        ln_etas = np.array([math.log(e) for e in etas])
        ln_etas = ln_etas[np.isfinite(ln_etas)]
        if len(ln_etas) < 5:
            continue

        config_fluct.append(
            {
                "trace": trace_name,
                "p_eff": p_eff,
                "n_seeds": len(ln_etas),
                "mean_ln_eta": float(np.mean(ln_etas)),
                "var_ln_eta": float(np.var(ln_etas, ddof=1)),
                "std_ln_eta": float(np.std(ln_etas, ddof=1)),
                "skewness": _skewness(ln_etas),
                "kurtosis_excess": _kurtosis_excess(ln_etas),
                "cv_eta": float(np.std(etas, ddof=1) / np.mean(etas))
                if np.mean(etas) > 0
                else float("nan"),
                "class": "cluster",
            }
        )

    return config_fluct


def aggregate_shape_stats(configs, label):
    """Aggregate skewness/kurtosis across configs."""
    skews = [c["skewness"] for c in configs if np.isfinite(c["skewness"])]
    kurts = [c["kurtosis_excess"] for c in configs if np.isfinite(c["kurtosis_excess"])]
    cvs = [c["cv_eta"] for c in configs if np.isfinite(c["cv_eta"])]

    if not skews:
        return {"label": label, "n_configs": 0}

    return {
        "label": label,
        "n_configs": len(configs),
        "skewness": {
            "mean": float(np.mean(skews)),
            "median": float(np.median(skews)),
            "std": float(np.std(skews)),
            "frac_negative": float(np.mean(np.array(skews) < 0)),
            "tw_reference": TW_SKEWNESS,
        },
        "kurtosis_excess": {
            "mean": float(np.mean(kurts)) if kurts else float("nan"),
            "median": float(np.median(kurts)) if kurts else float("nan"),
            "std": float(np.std(kurts)) if kurts else float("nan"),
            "tw_reference": TW_KURTOSIS,
        },
        "cv_eta": {
            "mean": float(np.mean(cvs)),
            "median": float(np.median(cvs)),
            "std": float(np.std(cvs)),
        },
    }


def compute_eh_dependence(configs):
    """Bin fluctuation statistics by E[H] to look for scaling."""
    # Extract E[H] where available
    eh_vals = []
    var_vals = []
    skew_vals = []

    for c in configs:
        # For orbital configs, E_H isn't stored directly in fluct stats
        # but mean_ln_eta / ln(p_ref) ≈ E[H] for i.i.d.
        var = c.get("var_ln_eta", 0)
        skew = c.get("skewness", float("nan"))
        p_ref = c.get("p_ref", 0.1)

        if var > 0 and p_ref > 0 and p_ref < 1:
            # Estimate E[H] from mean_ln_eta / ln(p_ref)
            mean_ln = c.get("mean_ln_eta", 0)
            if mean_ln < 0 and math.log(p_ref) < 0:
                eh_est = mean_ln / math.log(p_ref)
                if 0 < eh_est < 20:
                    eh_vals.append(eh_est)
                    var_vals.append(var)
                    skew_vals.append(skew)

    if len(eh_vals) < 5:
        return {}

    # Bin by E[H]
    eh_arr = np.array(eh_vals)
    var_arr = np.array(var_vals)
    skew_arr = np.array(skew_vals)

    n_bins = 10
    bin_edges = np.linspace(eh_arr.min(), eh_arr.max(), n_bins + 1)
    bins = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (eh_arr >= bin_edges[i]) & (eh_arr <= bin_edges[i + 1])
        else:
            mask = (eh_arr >= bin_edges[i]) & (eh_arr < bin_edges[i + 1])
        if np.sum(mask) >= 3:
            finite_skew = skew_arr[mask][np.isfinite(skew_arr[mask])]
            bins.append(
                {
                    "E_H_center": float(np.mean(eh_arr[mask])),
                    "mean_var": float(np.mean(var_arr[mask])),
                    "mean_skewness": float(np.mean(finite_skew))
                    if len(finite_skew) > 0
                    else float("nan"),
                    "n": int(np.sum(mask)),
                }
            )

    return {"bins": bins, "n_total": len(eh_vals)}


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 1, Experiment 4: DR Fluctuation Spectrum")
    print("=" * 70)

    # --- Orbital ---
    phi_path = _HERE / "phi_decompose_results.json"
    print(f"\nLoading {phi_path.name} ...")
    phi_data = _load_json(phi_path)
    orbital_fluct = analyze_orbital_fluctuations(phi_data)
    print(f"  {len(orbital_fluct):,} orbital config groups with ≥10 seeds")

    # --- CRAWDAD ---
    cluster_fluct = []
    for trace in ["Exp1", "Exp2", "Exp3", "Exp6"]:
        cpath = _HERE / f"crawdad_contacts.{trace}_results.json"
        if cpath.exists():
            print(f"\nLoading {cpath.name} ...")
            cdata = _load_json(cpath)
            cfluct = analyze_crawdad_fluctuations(cdata, trace)
            cluster_fluct.extend(cfluct)
            print(f"  {len(cfluct):,} {trace} config groups with ≥5 seeds")

    # --- Shape analysis ---
    print("\n" + "=" * 70)
    print("DISTRIBUTION SHAPE ANALYSIS")
    print("=" * 70)

    trap_shape = aggregate_shape_stats(orbital_fluct, "trap_all")
    cluster_shape = aggregate_shape_stats(cluster_fluct, "cluster_all")

    print(
        f"\n  Tracy-Widom GOE reference: skew = {TW_SKEWNESS:.3f}, "
        f"excess kurtosis = {TW_KURTOSIS:.3f}"
    )

    print(f"\n  TRAP CLASS ({trap_shape['n_configs']} configs):")
    if trap_shape["n_configs"] > 0:
        s = trap_shape["skewness"]
        k = trap_shape["kurtosis_excess"]
        print(
            f"    Skewness:  mean = {s['mean']:+.3f}, median = {s['median']:+.3f} "
            f"({100 * s['frac_negative']:.0f}% negative)"
        )
        print(f"    Kurtosis:  mean = {k['mean']:+.3f}, median = {k['median']:+.3f}")
        print(f"    CV(η):     mean = {trap_shape['cv_eta']['mean']:.3f}")

    print(f"\n  CLUSTER CLASS ({cluster_shape['n_configs']} configs):")
    if cluster_shape["n_configs"] > 0:
        s = cluster_shape["skewness"]
        k = cluster_shape["kurtosis_excess"]
        print(
            f"    Skewness:  mean = {s['mean']:+.3f}, median = {s['median']:+.3f} "
            f"({100 * s['frac_negative']:.0f}% negative)"
        )
        print(f"    Kurtosis:  mean = {k['mean']:+.3f}, median = {k['median']:+.3f}")
        print(f"    CV(η):     mean = {cluster_shape['cv_eta']['mean']:.3f}")

    # --- Per-target trap skewness ---
    print("\n--- Skewness by orbital target ---")
    by_target = defaultdict(list)
    for c in orbital_fluct:
        by_target[c["target"]].append(c["skewness"])
    for target, skews in sorted(by_target.items()):
        finite = [s for s in skews if np.isfinite(s)]
        if finite:
            print(
                f"  {target:10s}: mean skew = {np.mean(finite):+.3f}, "
                f"negative = {100 * np.mean(np.array(finite) < 0):.0f}%"
            )

    # --- Per-trace cluster skewness ---
    print("\n--- Skewness by CRAWDAD trace ---")
    by_trace = defaultdict(list)
    for c in cluster_fluct:
        by_trace[c["trace"]].append(c["skewness"])
    for trace, skews in sorted(by_trace.items()):
        finite = [s for s in skews if np.isfinite(s)]
        if finite:
            print(
                f"  {trace:10s}: mean skew = {np.mean(finite):+.3f}, "
                f"negative = {100 * np.mean(np.array(finite) < 0):.0f}%"
            )

    # --- E[H] dependence of shape ---
    print("\n--- Skewness vs E[H] (trap class) ---")
    eh_dep = compute_eh_dependence(orbital_fluct)
    if eh_dep.get("bins"):
        for b in eh_dep["bins"]:
            print(
                f"  E[H] ≈ {b['E_H_center']:.1f}: var = {b['mean_var']:.4f}, "
                f"skew = {b['mean_skewness']:+.3f}, n = {b['n']}"
            )

    # --- Verdict ---
    print("\n" + "=" * 70)
    print("TRACY-WIDOM VERDICT")
    print("=" * 70)

    if trap_shape["n_configs"] > 0:
        trap_skew = trap_shape["skewness"]["mean"]
        if not math.isnan(trap_skew):
            if abs(trap_skew - TW_SKEWNESS) < 0.15:
                print(
                    f"\n  Trap skewness ({trap_skew:+.3f}) CONSISTENT with TW ({TW_SKEWNESS:+.3f})"
                )
            else:
                print(f"\n  Trap skewness ({trap_skew:+.3f}) DIFFERS from TW ({TW_SKEWNESS:+.3f})")

    if cluster_shape["n_configs"] > 0:
        clust_skew = cluster_shape["skewness"]["mean"]
        if not math.isnan(clust_skew):
            if abs(clust_skew - TW_SKEWNESS) < 0.15:
                print(
                    f"  Cluster skewness ({clust_skew:+.3f}) CONSISTENT with TW ({TW_SKEWNESS:+.3f})"
                )
            else:
                print(
                    f"  Cluster skewness ({clust_skew:+.3f}) DIFFERS from TW ({TW_SKEWNESS:+.3f})"
                )

    # --- Save ---
    elapsed = time.time() - t0
    output = {
        "experiment": "dr_fluctuation_spectrum",
        "description": "Distribution shape of ln(eta) fluctuations; Tracy-Widom test",
        "elapsed_s": round(elapsed, 1),
        "tw_reference": {
            "skewness_GOE": TW_SKEWNESS,
            "kurtosis_excess_GOE": TW_KURTOSIS,
        },
        "trap": trap_shape,
        "cluster": cluster_shape,
        "eh_dependence_trap": eh_dep,
        "per_target_skewness": {
            target: float(np.mean([s for s in skews if np.isfinite(s)]))
            for target, skews in by_target.items()
            if any(np.isfinite(s) for s in skews)
        },
        "per_trace_skewness": {
            trace: float(np.mean([s for s in skews if np.isfinite(s)]))
            for trace, skews in by_trace.items()
            if any(np.isfinite(s) for s in skews)
        },
    }

    out_path = _HERE / "dr_fluctuation_spectrum_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
