"""run_phi_sweep_merge.py — Merge 8 Φ-sweep shards and analyze.

Usage: python runs/run_phi_sweep_merge.py

Reads:  runs/phi_sweep_shard_{target}.json  (8 files)
Writes: runs/phi_sweep_results.json

Analysis:
  1. Φ_time and Φ_rel distributions per target
  2. Does Φ_rel → 1?  (reliability oracle hypothesis)
  3. Partial correlations: Φ vs (α, d) controlling for n
  4. Braess boundary in Φ-space: where does Φ cross 1?
  5. σ-algebra independence test
"""

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
TARGETS = ["mercury", "venus", "mars", "ceres", "europa", "jupiter", "saturn", "titan"]


def _sanitize(obj):
    if isinstance(obj, float):
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if np.isnan(obj):
            return "nan"
    return obj


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    return _sanitize(d)


def _phi_summary(results, label):
    """Compute Φ distribution stats for a set of results."""
    phis_time = []
    phis_rel = []
    for r in results:
        if r.get("n_feasible", 0) == 0:
            continue
        pt = r.get("phi_time", 0)
        pr = r.get("phi_rel", 0)
        if isinstance(pt, str):
            continue
        if isinstance(pr, str):
            continue
        if np.isfinite(pt) and pt > 0:
            phis_time.append(pt)
        if np.isfinite(pr) and pr > 0:
            phis_rel.append(pr)

    summary = {"label": label, "n_active": len(phis_time)}
    if phis_time:
        pt_arr = np.array(phis_time)
        summary["phi_time"] = {
            "mean": float(pt_arr.mean()),
            "std": float(pt_arr.std()),
            "median": float(np.median(pt_arr)),
            "min": float(pt_arr.min()),
            "max": float(pt_arr.max()),
            "frac_gt_1": float(np.mean(pt_arr > 1.0)),
        }
    if phis_rel:
        pr_arr = np.array(phis_rel)
        summary["phi_rel"] = {
            "mean": float(pr_arr.mean()),
            "std": float(pr_arr.std()),
            "median": float(np.median(pr_arr)),
            "min": float(pr_arr.min()),
            "max": float(pr_arr.max()),
            "frac_gt_1": float(np.mean(pr_arr > 1.0)),
        }
    return summary


def _partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    if len(x) < 5:
        return float("nan")
    x, y, z = np.array(x), np.array(y), np.array(z)
    # Residualize x and y on z
    A = np.vstack([z, np.ones(len(z))]).T
    bx, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    by, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    rx = x - A @ bx
    ry = y - A @ by
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom < 1e-15:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def main():
    all_results = []
    shards_loaded = []

    for target in TARGETS:
        shard_path = _HERE / f"phi_sweep_shard_{target}.json"
        if not shard_path.exists():
            print(f"  WARNING: missing shard {shard_path.name}")
            continue
        with open(shard_path) as f:
            shard = json.load(f)
        all_results.extend(shard["results"])
        shards_loaded.append(target)
        print(f"  Loaded {target}: {len(shard['results'])} configs")

    print(f"\n  Total: {len(all_results)} configs from {len(shards_loaded)} shards")

    # Global summary
    global_summary = _phi_summary(all_results, "global")

    # Per-target summaries
    target_summaries = {}
    print("\n  === PHI DISTRIBUTIONS (per target) ===")
    print(
        f"  {'Target':>8s}  {'N_act':>5s}  "
        f"{'Phi_t_mean':>10s}  {'Phi_t_std':>9s}  {'Phi_t>1':>7s}  "
        f"{'Phi_r_mean':>10s}  {'Phi_r_std':>9s}  {'Phi_r>1':>7s}"
    )
    for target in TARGETS:
        t_results = [r for r in all_results if r["target"] == target]
        if not t_results:
            continue
        s = _phi_summary(t_results, target)
        target_summaries[target] = s
        pt = s.get("phi_time", {})
        pr = s.get("phi_rel", {})
        print(
            f"  {target:>8s}  {s['n_active']:5d}  "
            f"{pt.get('mean', 0):10.4f}  {pt.get('std', 0):9.4f}  {pt.get('frac_gt_1', 0):6.1%}  "
            f"{pr.get('mean', 0):10.4f}  {pr.get('std', 0):9.4f}  {pr.get('frac_gt_1', 0):6.1%}"
        )

    # Reliability oracle hypothesis: Φ_rel closer to 1 than Φ_time?
    print("\n  === RELIABILITY ORACLE HYPOTHESIS ===")
    print("  |Φ-1| comparison: is |Φ_rel - 1| < |Φ_time - 1|?")
    for target in TARGETS:
        t_results = [r for r in all_results if r["target"] == target]
        active = [
            r
            for r in t_results
            if r.get("n_feasible", 0) > 0
            and isinstance(r.get("phi_time"), (int, float))
            and isinstance(r.get("phi_rel"), (int, float))
            and np.isfinite(r["phi_time"])
            and r["phi_time"] > 0
            and np.isfinite(r["phi_rel"])
            and r["phi_rel"] > 0
        ]
        if not active:
            continue
        dev_time = np.mean([abs(r["phi_time"] - 1.0) for r in active])
        dev_rel = np.mean([abs(r["phi_rel"] - 1.0) for r in active])
        closer = "YES" if dev_rel < dev_time else "no"
        print(f"  {target:>8s}: |Φ_time-1|={dev_time:.4f}, |Φ_rel-1|={dev_rel:.4f}  -> {closer}")

    # Partial correlations: Φ_time vs (alpha, dist_au) controlling for n_orb
    print("\n  === PARTIAL CORRELATIONS (Φ_time) ===")
    active_all = [
        r
        for r in all_results
        if r.get("n_feasible", 0) > 0
        and isinstance(r.get("phi_time"), (int, float))
        and np.isfinite(r["phi_time"])
        and r["phi_time"] > 0
    ]
    if len(active_all) > 10:
        phi_t = [r["phi_time"] for r in active_all]
        alphas = [r["alpha"] for r in active_all]
        dists = [r["dist_au"] for r in active_all]
        n_orbs = [r["n_orb"] for r in active_all]
        p_effs = [r["p_eff"] for r in active_all]

        rho_alpha = _partial_corr(phi_t, alphas, n_orbs)
        rho_dist = _partial_corr(phi_t, dists, n_orbs)
        rho_peff = _partial_corr(phi_t, p_effs, n_orbs)
        rho_n = float(np.corrcoef(phi_t, n_orbs)[0, 1])

        print(f"  rho(Φ_time, alpha | n) = {rho_alpha:.4f}")
        print(f"  rho(Φ_time, dist  | n) = {rho_dist:.4f}")
        print(f"  rho(Φ_time, p_eff | n) = {rho_peff:.4f}")
        print(f"  rho(Φ_time, n_orb)     = {rho_n:.4f}")

        # σ-algebra independence: Φ depends on topology (n), not physics (α, d)?
        indep = abs(rho_alpha) < 0.15 and abs(rho_dist) < 0.15
        print(f"\n  σ-algebra separation: {'CONFIRMED' if indep else 'REJECTED'}")
        print("    (threshold: |rho| < 0.15 for alpha, dist)")

    # Braess boundary: per n_orb, what fraction of configs have Φ > 1?
    print("\n  === BRAESS BOUNDARY IN Φ-SPACE ===")
    print(f"  {'n_orb':>5s}  {'N':>5s}  {'Φ>1':>7s}  {'mean_Φ':>8s}")
    for n in sorted(set(r["n_orb"] for r in all_results)):
        sub = [r for r in active_all if r["n_orb"] == n]
        if not sub:
            continue
        phis = [r["phi_time"] for r in sub]
        print(
            f"  {n:5d}  {len(sub):5d}  {np.mean(np.array(phis) > 1.0):6.1%}  {np.mean(phis):8.4f}"
        )

    # Per n_orb × target
    print(f"\n  {'Target':>8s}  {'n=3':>8s}  {'n=6':>8s}  {'n=12':>8s}  {'n=24':>8s}")
    for target in TARGETS:
        vals = []
        for n in [3, 6, 12, 24]:
            sub = [r for r in active_all if r["target"] == target and r["n_orb"] == n]
            if sub:
                vals.append(f"{np.mean([r['phi_time'] for r in sub]):8.4f}")
            else:
                vals.append(f"{'---':>8s}")
        print(f"  {target:>8s}  {'  '.join(vals)}")

    # Save
    output = {
        "n_shards": len(shards_loaded),
        "shards_loaded": shards_loaded,
        "n_total": len(all_results),
        "global_summary": _clean(global_summary),
        "target_summaries": _clean(target_summaries),
        "results": _clean(all_results),
    }

    out_path = _HERE / "phi_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Merged -> {out_path.name}")
    print("  Done.")


if __name__ == "__main__":
    main()
