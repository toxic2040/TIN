#!/usr/bin/env python3
"""run_beta_eff_production.py — Production-scale beta_eff survey.

Reanalyses ALL production + campaign EPYC data (~99K configs, ~73K with var_H > 0)
to compute beta_eff (effective Boltzmann temperature) and correlate with gamma.

Uses the same 2-family reconstruction from run_beta_eff_survey.py but on the full
production dataset instead of just per_path_cov (154 configs).

Outputs: runs/beta_eff_production_results.json
"""

import glob
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).parent


def _reconstruct_two_family(E_H, var_H, n_paths):
    """Reconstruct a 2-family hop distribution from moments.

    Given E[H] and Var[H], find integer hop counts H_1 < H_2 and
    fraction f_1 such that:
        E[H] = f_1 * H_1 + f_2 * H_2
        Var[H] = f_1 * (H_1 - E_H)^2 + f_2 * (H_2 - E_H)^2

    Returns (H_1, H_2, n_1, n_2) or None if no valid decomposition found.
    """
    if var_H <= 0:
        return None

    best = None
    best_err = float("inf")

    for h1 in range(1, 20):
        for h2 in range(h1 + 1, 20):
            delta = h2 - h1
            f1 = (h2 - E_H) / delta
            if f1 < 0.001 or f1 > 0.999:
                continue
            f2 = 1.0 - f1
            var_pred = f1 * (h1 - E_H) ** 2 + f2 * (h2 - E_H) ** 2
            err = abs(var_pred - var_H) / max(var_H, 1e-12)
            if err < best_err:
                best_err = err
                n1 = round(f1 * n_paths)
                n2 = n_paths - n1
                if n1 > 0 and n2 > 0:
                    best = (h1, h2, n1, n2, err)

    if best is None or best[4] > 0.05:
        return None
    return best[:4]


def load_all_production():
    """Load all production + campaign JSONs, return flat list of config dicts."""
    configs = []
    sources = []

    # Production
    for path in sorted(
        glob.glob(str(_HERE / "epyc_results/production_2026_03_11/production_*.json"))
    ):
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict) and "results" in raw:
            data = raw["results"]
        else:
            data = [v for v in raw.values() if isinstance(v, dict)]
        for d in data:
            if isinstance(d, dict):
                d["_source"] = Path(path).name
                configs.append(d)
        sources.append((Path(path).name, len(data)))

    # Campaign
    for path in sorted(glob.glob(str(_HERE / "epyc_results/campaign_2026_03_11/*.json"))):
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict) and "results" in raw:
            data = raw["results"]
        else:
            continue
        for d in data:
            if isinstance(d, dict):
                d["_source"] = Path(path).name
                configs.append(d)
        sources.append((Path(path).name, len(data)))

    return configs, sources


def main():
    print("=" * 72)
    print("PRODUCTION-SCALE BETA_EFF SURVEY")
    print("=" * 72)

    t0 = time.time()
    configs, sources = load_all_production()
    t_load = time.time() - t0
    print(f"\n  Loaded {len(configs)} configs from {len(sources)} files in {t_load:.1f}s")

    # Filter to var_H > 0
    candidates = [d for d in configs if d.get("var_H", 0) > 0 and d.get("n_paths", 0) > 2]
    print(f"  With var_H > 0 and n_paths > 2: {len(candidates)}")

    # Determine which lyapunov key is used
    lyap_key = None
    for k in ("lyapunov_exp", "lyapunov"):
        if candidates and k in candidates[0]:
            lyap_key = k
            break
    if lyap_key is None:
        # Check a few
        for c in candidates[:10]:
            for k in c:
                if "lyap" in k.lower():
                    lyap_key = k
                    break
            if lyap_key:
                break
    print(f"  Lyapunov key: {lyap_key}")

    # Reconstruct
    t1 = time.time()
    results = []
    skipped_reconstruct = 0
    skipped_dU = 0

    for cfg in candidates:
        E_H = cfg.get("E_H", 0)
        var_H = cfg.get("var_H", 0)
        n_paths = cfg.get("n_paths", 0)
        lyap = cfg.get(lyap_key, 0)
        body = cfg.get("body", "?")

        if abs(lyap) < 1e-15 or n_paths < 3:
            skipped_reconstruct += 1
            continue

        decomp = _reconstruct_two_family(E_H, var_H, n_paths)
        if decomp is None:
            skipped_reconstruct += 1
            continue

        H_1, H_2, n_1, n_2 = decomp
        f_1 = n_1 / n_paths
        f_2 = n_2 / n_paths

        U_1 = H_1 * lyap
        U_2 = H_2 * lyap
        dU = U_1 - U_2

        if abs(dU) < 1e-15:
            skipped_dU += 1
            continue

        if f_1 <= 0 or f_2 <= 0:
            continue

        beta_eff = math.log(f_1 / f_2) / dU

        # Also grab gamma if available
        gamma = cfg.get("gamma", cfg.get("gamma_normal", float("nan")))
        var_log_p = cfg.get("var_log_p", float("nan"))
        s_t = cfg.get("S_T", cfg.get("s_t", float("nan")))
        eta = cfg.get("eta", float("nan"))
        dr = cfg.get("DR", cfg.get("dr", float("nan")))
        p_eff = cfg.get("p_eff", cfg.get("p_success", None))

        results.append(
            {
                "body": body,
                "H_1": H_1,
                "H_2": H_2,
                "f_1": round(f_1, 6),
                "f_2": round(f_2, 6),
                "dU": round(dU, 6),
                "beta_eff": round(beta_eff, 6),
                "sign": "positive" if beta_eff > 0 else "negative",
                "E_H": round(E_H, 4),
                "var_H": round(var_H, 6),
                "lyapunov": round(lyap, 6),
                "gamma": round(gamma, 6) if not math.isnan(gamma) else None,
                "var_log_p": round(var_log_p, 6) if not math.isnan(var_log_p) else None,
                "S_T": round(s_t, 6) if not math.isnan(s_t) else None,
                "eta": round(eta, 6) if not math.isnan(eta) else None,
                "DR": round(dr, 6) if not math.isnan(dr) else None,
                "p_eff": p_eff,
            }
        )

    t_compute = time.time() - t1
    print(f"  Reconstructed: {len(results)} configs in {t_compute:.2f}s")
    print(f"  Skipped (no valid decomposition): {skipped_reconstruct}")
    print(f"  Skipped (dU ~ 0): {skipped_dU}")

    # ── Statistics ──
    beta_arr = np.array([r["beta_eff"] for r in results])
    n_neg = int(np.sum(beta_arr < 0))
    n_pos = int(np.sum(beta_arr > 0))

    print(f"\n{'=' * 72}")
    print("GLOBAL STATISTICS")
    print(f"{'=' * 72}")
    print(f"  Total: {len(results)}")
    print(f"  Negative temperature: {n_neg} ({100 * n_neg / len(results):.1f}%)")
    print(f"  Positive temperature: {n_pos} ({100 * n_pos / len(results):.1f}%)")
    print(f"  beta_eff range: [{beta_arr.min():.2f}, {beta_arr.max():.2f}]")
    print(f"  beta_eff mean: {beta_arr.mean():.4f}")
    print(f"  beta_eff median: {float(np.median(beta_arr)):.4f}")
    print(f"  beta_eff std: {beta_arr.std():.4f}")

    # ── Per-body ──
    print(f"\n{'=' * 72}")
    print("PER-BODY SUMMARY")
    print(f"{'=' * 72}")
    by_body = defaultdict(list)
    for r in results:
        by_body[r["body"]].append(r)

    print(
        f"  {'body':>10s} | {'count':>6s} | {'mean_beta':>10s} | "
        f"{'med_beta':>10s} | {'std_beta':>10s} | {'%neg':>5s} | "
        f"{'mean_gamma':>10s} | {'mean_vlp':>10s}"
    )
    print(f"  {'-' * 90}")

    body_summary = {}
    for body in sorted(by_body.keys()):
        recs = by_body[body]
        betas = np.array([r["beta_eff"] for r in recs])
        gammas = [r["gamma"] for r in recs if r["gamma"] is not None]
        vlps = [r["var_log_p"] for r in recs if r["var_log_p"] is not None]
        nn = int(np.sum(betas < 0))
        mg = np.mean(gammas) if gammas else float("nan")
        mv = np.mean(vlps) if vlps else float("nan")

        print(
            f"  {body:>10s} | {len(recs):>6d} | {betas.mean():>+10.3f} | "
            f"{float(np.median(betas)):>+10.3f} | {betas.std():>10.3f} | "
            f"{100 * nn / len(recs):>5.1f} | "
            f"{mg:>+10.4f} | {mv:>10.6f}"
        )

        body_summary[body] = {
            "count": len(recs),
            "mean_beta": float(betas.mean()),
            "median_beta": float(np.median(betas)),
            "std_beta": float(betas.std()),
            "pct_negative": round(100 * nn / len(recs), 1),
            "mean_gamma": round(mg, 6) if not math.isnan(mg) else None,
            "mean_var_log_p": round(mv, 6) if not math.isnan(mv) else None,
        }

    # ── Correlation: beta_eff vs gamma (per-config) ──
    print(f"\n{'=' * 72}")
    print("CORRELATION ANALYSIS — per-config beta_eff vs gamma")
    print(f"{'=' * 72}")

    # Global
    pairs = [
        (r["beta_eff"], r["gamma"])
        for r in results
        if r["gamma"] is not None and not math.isnan(r["gamma"])
    ]
    if len(pairs) > 10:
        b_arr = np.array([p[0] for p in pairs])
        g_arr = np.array([p[1] for p in pairs])

        # Clip extreme beta_eff for robust correlation
        mask = np.abs(b_arr) < np.percentile(np.abs(b_arr), 99)
        b_clip = b_arr[mask]
        g_clip = g_arr[mask]

        rho_s, p_s = stats.spearmanr(b_arr, g_arr)
        rho_p, p_p = stats.pearsonr(b_arr, g_arr)
        rho_sc, p_sc = stats.spearmanr(b_clip, g_clip)

        print(f"  All configs (n={len(pairs)}):")
        print(f"    Spearman rho = {rho_s:+.4f}, p = {p_s:.2e}")
        print(f"    Pearson  r   = {rho_p:+.4f}, p = {p_p:.2e}")
        print(f"  Clipped to 99th pct |beta| (n={mask.sum()}):")
        print(f"    Spearman rho = {rho_sc:+.4f}, p = {p_sc:.2e}")

    # Per-body correlations
    print("\n  Per-body:")
    print(
        f"  {'body':>10s} | {'n':>6s} | {'Spearman':>10s} | {'p-value':>10s} | "
        f"{'Pearson':>10s} | {'p-value':>10s}"
    )
    print(f"  {'-' * 70}")

    per_body_corr = {}
    for body in sorted(by_body.keys()):
        recs = by_body[body]
        pairs_b = [
            (r["beta_eff"], r["gamma"])
            for r in recs
            if r["gamma"] is not None and not math.isnan(r["gamma"])
        ]
        if len(pairs_b) < 5:
            print(f"  {body:>10s} | {len(pairs_b):>6d} | {'(too few)':>10s}")
            continue
        bb = np.array([p[0] for p in pairs_b])
        gg = np.array([p[1] for p in pairs_b])
        rs, ps = stats.spearmanr(bb, gg)
        rp, pp = stats.pearsonr(bb, gg)
        print(
            f"  {body:>10s} | {len(pairs_b):>6d} | {rs:>+10.4f} | {ps:>10.2e} | "
            f"{rp:>+10.4f} | {pp:>10.2e}"
        )
        per_body_corr[body] = {
            "n": len(pairs_b),
            "spearman_rho": round(rs, 4),
            "spearman_p": float(ps),
            "pearson_r": round(rp, 4),
            "pearson_p": float(pp),
        }

    # ── Correlation: beta_eff vs var_log_p ──
    pairs_vlp = [
        (r["beta_eff"], r["var_log_p"])
        for r in results
        if r["var_log_p"] is not None and not math.isnan(r["var_log_p"])
    ]
    if len(pairs_vlp) > 10:
        bv = np.array([p[0] for p in pairs_vlp])
        vv = np.array([p[1] for p in pairs_vlp])
        rs, ps = stats.spearmanr(bv, vv)
        print(f"\n  beta_eff vs var_log_p (n={len(pairs_vlp)}):")
        print(f"    Spearman rho = {rs:+.4f}, p = {ps:.2e}")

    # ── Save ──
    output = {
        "description": "Production-scale beta_eff (effective Boltzmann temperature)",
        "sources": "runs/epyc_results/production_2026_03_11/ + campaign_2026_03_11/",
        "total_configs_loaded": len(configs),
        "total_with_var_H": len(candidates),
        "total_reconstructed": len(results),
        "n_negative_temperature": n_neg,
        "n_positive_temperature": n_pos,
        "beta_eff_stats": {
            "mean": round(float(beta_arr.mean()), 4),
            "median": round(float(np.median(beta_arr)), 4),
            "std": round(float(beta_arr.std()), 4),
            "min": round(float(beta_arr.min()), 4),
            "max": round(float(beta_arr.max()), 4),
        },
        "per_body": body_summary,
        "correlations": {
            "global_spearman_rho": round(rho_s, 4) if "rho_s" in dir() else None,
            "global_spearman_p": float(p_s) if "p_s" in dir() else None,
            "global_pearson_r": round(rho_p, 4) if "rho_p" in dir() else None,
            "per_body": per_body_corr,
        },
        "wall_time_s": round(time.time() - t0, 1),
    }

    # Don't save all 73K individual results — just the summary + per-body
    # Save a sample for spot-checking
    output["sample_results"] = results[:50]

    outpath = _HERE / "beta_eff_production_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    size_kb = outpath.stat().st_size / 1024
    print(f"\n  Saved -> {outpath.name} ({size_kb:.0f} KB)")
    print(f"  Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
