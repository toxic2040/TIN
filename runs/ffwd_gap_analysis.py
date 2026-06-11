"""run_ffwd_gap_analysis.py — Phase 1, Experiment 3: f_fwd Distribution & Gap Analysis.

Computes the forwarding ratio f_fwd for all configurations across the full
dataset and analyzes the gap structure at the f_fwd = 1 critical boundary.

The forwarding ratio (Definition 6 in the paper):
  f_fwd = E[m_fwd / (m + 1)]
where m = retry contacts, m_fwd = onward contacts at each decision point.

For configurations where f_fwd is not directly stored, we estimate it from
the relationship between phi and the chain law:
  - In trap class: Φ < 1 implies f_fwd < 1
  - In cluster class: Φ > 1 implies f_fwd > 1
  - The myopic component phi_myopic encodes the dead-end structure

From topology_sweep_results.json we have actual dead-end fractions and
branching ratios that can reconstruct f_fwd.

Reads:  runs/topology_sweep_results.json
        runs/phi_decompose_results.json
        runs/tail_ratio_survey_results.json
Writes: runs/ffwd_gap_analysis_results.json
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


def analyze_topology_sweep(topo_data):
    """Extract branching and dead-end structure from topology sweep.

    The topology records contain:
      - mean_temporal_dead_end_frac: fraction of decision points that are dead ends
      - mean_branching: mean number of next-hop contacts
      - mean_temporal_branching: branching accounting for time ordering

    f_fwd is estimated as: (1 - dead_end_frac) * branching_ratio
    where branching_ratio = forward_contacts / total_contacts
    """
    topo = topo_data.get("topology", [])
    joined = topo_data.get("joined_results", [])

    # Topology-level analysis: dead-end structure per contact plan
    topo_stats = []
    for t in topo:
        topo_stats.append(
            {
                "target": t["target"],
                "n_orb": t["n_orb"],
                "epoch_day": t.get("epoch_day", 0),
                "dead_end_frac": t.get(
                    "mean_temporal_dead_end_frac", t.get("static_dead_end_frac", 0)
                ),
                "mean_branching": t.get("mean_temporal_branching", t.get("mean_branching", 0)),
                "contact_count": t.get("contact_count", 0),
                "node_count": t.get("node_count", 0),
            }
        )

    # Joined analysis: phi decomposition + topology
    phi_with_topo = []
    for r in joined:
        if r.get("S_T", 0) < 0.01:
            continue
        phi_n = r.get("phi_normal", r.get("phi_time", None))
        if phi_n is None or phi_n <= 0:
            continue
        dead_end = r.get("mean_temporal_dead_end_frac", r.get("static_dead_end_frac", 0))
        branching = r.get("mean_temporal_branching", r.get("mean_branching", 0))

        # Estimate f_fwd from topology
        # f_fwd ≈ (onward fraction) = 1 - dead_end_frac (at the decision point level)
        # More precisely: f_fwd = E[m_fwd/(m+1)]
        # With m = branching - 1 (total contacts minus current), m_fwd = non-dead-end contacts
        # Approximate: f_fwd ≈ (1 - dead_end) if branching >> 1,
        #              f_fwd ≈ branching * (1 - dead_end) / (branching + 1) otherwise
        if branching > 0:
            f_fwd_est = branching * (1.0 - dead_end) / (branching + 1.0)
        else:
            f_fwd_est = 0.0

        phi_with_topo.append(
            {
                "target": r.get("target"),
                "n_orb": r.get("n_orb"),
                "p_ref": r.get("p_ref"),
                "alpha": r.get("alpha", 1),
                "phi_normal": float(phi_n),
                "ln_phi": float(math.log(phi_n)),
                "dead_end_frac": float(dead_end),
                "mean_branching": float(branching),
                "f_fwd_est": float(f_fwd_est),
                "E_H": r.get("E_H", 0),
                "S_T": r.get("S_T", 0),
            }
        )

    return topo_stats, phi_with_topo


def analyze_phi_decompose_proxy(phi_data):
    """Estimate f_fwd indirectly from phi_myopic and phi_retry decomposition.

    From the paper: Φ = Φ_m × Φ_r
      - Φ_m < 1 → dead-end dominated (f_fwd < 1 implied)
      - Φ_m > 1 → forward-channel dominated (f_fwd > 1 implied)

    The connection: f_fwd ≈ Φ_m^(1/E[H]) when phi_myopic captures pure topology.
    """
    results = phi_data["results"]

    # Group by (target, n_orb, p_ref) and average phi_myopic across seeds/epochs
    groups = defaultdict(list)
    for r in results:
        if r["S_T"] < 0.01:
            continue
        phi_m = r.get("phi_myopic")
        if phi_m is not None and phi_m > 0 and r.get("E_H", 0) > 0:
            key = (r["target"], r["n_orb"], r["p_ref"])
            groups[key].append(
                {
                    "phi_myopic": phi_m,
                    "phi_normal": r.get("phi_normal", 1.0),
                    "phi_retry": r.get("phi_retry", 1.0),
                    "E_H": r["E_H"],
                }
            )

    proxy_stats = []
    for key, records in groups.items():
        target, n_orb, p_ref = key
        mean_phi_m = np.mean([r["phi_myopic"] for r in records])
        mean_phi_n = np.mean([r["phi_normal"] for r in records])
        mean_phi_r = np.mean([r["phi_retry"] for r in records])
        mean_eh = np.mean([r["E_H"] for r in records])

        # f_fwd proxy: if Φ_m = (f_fwd)^E[H] approximately, then
        # f_fwd ≈ Φ_m^(1/E[H])
        if mean_phi_m > 0 and mean_eh > 0:
            f_fwd_proxy = mean_phi_m ** (1.0 / mean_eh) if mean_phi_m > 0 else 0
        else:
            f_fwd_proxy = float("nan")

        proxy_stats.append(
            {
                "target": target,
                "n_orb": n_orb,
                "p_ref": p_ref,
                "mean_phi_myopic": float(mean_phi_m),
                "mean_phi_normal": float(mean_phi_n),
                "mean_phi_retry": float(mean_phi_r),
                "mean_E_H": float(mean_eh),
                "f_fwd_proxy": float(f_fwd_proxy),
                "n_records": len(records),
                "class": "trap",  # all orbital
            }
        )

    return proxy_stats


def compute_gap_statistics(f_fwd_values, classes):
    """Compute gap statistics for the f_fwd distribution."""
    trap_vals = [f for f, c in zip(f_fwd_values, classes) if c == "trap" and np.isfinite(f)]
    cluster_vals = [f for f, c in zip(f_fwd_values, classes) if c == "cluster" and np.isfinite(f)]

    if not trap_vals or not cluster_vals:
        return {"gap": float("nan")}

    trap_max = max(trap_vals)
    cluster_min = min(cluster_vals)
    gap = cluster_min - trap_max

    return {
        "trap_max_ffwd": float(trap_max),
        "trap_mean_ffwd": float(np.mean(trap_vals)),
        "trap_std_ffwd": float(np.std(trap_vals)),
        "cluster_min_ffwd": float(cluster_min),
        "cluster_mean_ffwd": float(np.mean(cluster_vals)),
        "cluster_std_ffwd": float(np.std(cluster_vals)),
        "gap": float(gap),
        "gap_exists": gap > 0,
        "n_trap": len(trap_vals),
        "n_cluster": len(cluster_vals),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 1, Experiment 3: f_fwd Gap Analysis")
    print("=" * 70)

    # --- Topology sweep (direct dead-end / branching data) ---
    topo_path = _HERE / "topology_sweep_results.json"
    topo_stats = []
    phi_topo = []
    if topo_path.exists():
        print(f"\nLoading {topo_path.name} ...")
        topo_data = _load_json(topo_path)
        topo_stats, phi_topo = analyze_topology_sweep(topo_data)
        print(f"  {len(topo_stats)} topology records, {len(phi_topo)} joined records")

        # Dead-end summary by target
        print("\n--- Dead-end fractions by target ---")
        by_target = defaultdict(list)
        for t in topo_stats:
            by_target[t["target"]].append(t["dead_end_frac"])
        for target, fracs in sorted(by_target.items()):
            print(
                f"  {target:10s}: mean dead-end = {np.mean(fracs):.3f} "
                f"(range [{np.min(fracs):.3f}, {np.max(fracs):.3f}])"
            )

    # --- phi_decompose proxy (f_fwd from phi_myopic decomposition) ---
    phi_path = _HERE / "phi_decompose_results.json"
    print(f"\nLoading {phi_path.name} ...")
    phi_data = _load_json(phi_path)
    proxy_stats = analyze_phi_decompose_proxy(phi_data)
    print(f"  {len(proxy_stats)} config groups with f_fwd proxy")

    # --- f_fwd proxy distribution ---
    print("\n--- f_fwd proxy distribution (Φ_m^{1/E[H]}) by target ---")
    by_target_proxy = defaultdict(list)
    for s in proxy_stats:
        if np.isfinite(s["f_fwd_proxy"]):
            by_target_proxy[s["target"]].append(s["f_fwd_proxy"])
    for target, vals in sorted(by_target_proxy.items()):
        arr = np.array(vals)
        print(
            f"  {target:10s}: f̂_fwd = {np.mean(arr):.3f} ± {np.std(arr):.3f} "
            f"(range [{np.min(arr):.3f}, {np.max(arr):.3f}], n={len(arr)})"
        )

    # --- Tail ratio survey (has true class labels) ---
    tail_path = _HERE / "tail_ratio_survey_results.json"
    if tail_path.exists():
        print(f"\nLoading {tail_path.name} ...")
        tail_data = _load_json(tail_path)
        tail_records = tail_data.get("records", [])

        print("\n--- Tail ratio survey (R = p90/p10) vs class ---")
        for r in sorted(tail_records, key=lambda x: x.get("R", 0)):
            print(
                f"  {r['label']:20s}: R = {r.get('R', 0):7.2f}, "
                f"class = {r['true_class']}, γ = {r.get('gamma_p01', 0):+.3f}"
            )

    # --- Φ distribution analysis at f_fwd = 1 boundary ---
    print("\n" + "=" * 70)
    print("Φ DISTRIBUTION AT THE CLASSIFICATION BOUNDARY")
    print("=" * 70)

    # Collect all phi_normal values at p_ref ≈ 0.1
    phi_at_ref = defaultdict(list)
    for r in phi_data["results"]:
        if r["S_T"] < 0.01:
            continue
        if abs(r["p_ref"] - 0.1) > 0.02:
            continue
        phi = r.get("phi_normal")
        if phi is not None and phi > 0:
            phi_at_ref[r["target"]].append(phi)

    print("\n  Target      mean(Φ)   Φ<1(%)  Φ>1(%)  median(Φ)")
    for target in sorted(phi_at_ref.keys()):
        vals = np.array(phi_at_ref[target])
        below = 100.0 * np.mean(vals < 1.0)
        above = 100.0 * np.mean(vals > 1.0)
        print(
            f"  {target:10s}  {np.mean(vals):7.3f}  {below:6.1f}  {above:6.1f}  {np.median(vals):.3f}"
        )

    # --- Gap analysis using proxy ---
    print("\n" + "=" * 70)
    print("f_fwd GAP ANALYSIS")
    print("=" * 70)

    # All orbital targets are trap class
    all_ffwd = []
    all_class = []
    for s in proxy_stats:
        if np.isfinite(s["f_fwd_proxy"]):
            all_ffwd.append(s["f_fwd_proxy"])
            all_class.append("trap")

    # Note: cluster f_fwd would need CRAWDAD data with the same decomposition
    # For now, report trap-side gap boundary
    trap_fwd = [f for f, c in zip(all_ffwd, all_class) if c == "trap"]
    if trap_fwd:
        arr = np.array(trap_fwd)
        print("\n  Trap-class f̂_fwd:")
        print(f"    max  = {np.max(arr):.4f}  (closest to boundary)")
        print(f"    mean = {np.mean(arr):.4f}")
        print(f"    min  = {np.min(arr):.4f}")
        print(f"    n    = {len(arr)}")
        print("\n  Paper reports f_fwd gap: [0.97, 1.31]")
        print(f"  Proxy max (trap side): {np.max(arr):.4f}")

    # --- Phi < 1 vs Phi > 1 as surrogate for f_fwd < 1 vs > 1 ---
    print("\n--- Φ as f_fwd surrogate (Φ < 1 ↔ trap, Φ > 1 ↔ cluster) ---")
    all_phis = []
    for r in phi_data["results"]:
        if r["S_T"] < 0.01:
            continue
        phi = r.get("phi_normal")
        if phi is not None and phi > 0 and np.isfinite(phi):
            all_phis.append(float(phi))

    if all_phis:
        arr = np.array(all_phis)
        # Find the gap around Φ = 1
        near_1 = arr[(arr > 0.5) & (arr < 2.0)]
        print(f"  Total configs: {len(arr):,}")
        print(f"  Φ < 1: {np.sum(arr < 1.0):,} ({100 * np.mean(arr < 1.0):.1f}%)")
        print(f"  Φ > 1: {np.sum(arr > 1.0):,} ({100 * np.mean(arr > 1.0):.1f}%)")
        print(f"  Φ ∈ [0.95, 1.05]: {np.sum((arr > 0.95) & (arr < 1.05)):,}")
        print(f"  Φ ∈ [0.9, 1.1]:   {np.sum((arr > 0.9) & (arr < 1.1)):,}")

    # --- Save ---
    elapsed = time.time() - t0
    output = {
        "experiment": "ffwd_gap_analysis",
        "description": "Forwarding ratio distribution and gap structure analysis",
        "elapsed_s": round(elapsed, 1),
        "topology_stats": topo_stats[:20],  # sample
        "proxy_stats_by_target": {
            target: {
                "mean_ffwd": float(np.mean(vals)),
                "std_ffwd": float(np.std(vals)),
                "min_ffwd": float(np.min(vals)),
                "max_ffwd": float(np.max(vals)),
                "n": len(vals),
            }
            for target, vals in by_target_proxy.items()
        },
        "phi_distribution_at_p01": {
            target: {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "frac_below_1": float(np.mean(np.array(vals) < 1.0)),
                "frac_above_1": float(np.mean(np.array(vals) > 1.0)),
                "n": len(vals),
            }
            for target, vals in phi_at_ref.items()
        },
    }

    out_path = _HERE / "ffwd_gap_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
