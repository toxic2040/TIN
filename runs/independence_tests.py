#!/usr/bin/env python3
"""Independence tests for S_T and eta — pooled, stratified, and heteroscedastic.

Three analyses in one pass:
  1. Mutual information I(S_T; eta) — pooled across all configs
  2. Conditional MI I(S_T; eta | body) — stratified within each body
  3. Var(eta) vs S_T — heteroscedasticity check (self-averaging failure)

Addresses Simpson's paradox risk (pooled MI near zero hiding within-body
dependence) and finite-size trap (Var(eta) inflating at low S_T where
self-averaging is weakest).

Sources: runs/epyc_results/production_2026_03_11/ + campaign_2026_03_11/
Output:  runs/independence_test_results.json
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


def mutual_information_binned(x, y, n_bins=30):
    """Compute mutual information I(X; Y) using binned estimator.

    Returns I in nats, plus H(X), H(Y), H(X,Y) for diagnostics.
    Uses equal-frequency binning (quantile-based) to handle skewed margins.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n < 100:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Equal-frequency binning
    x_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    y_edges = np.percentile(y, np.linspace(0, 100, n_bins + 1))

    # Make edges strictly increasing
    x_edges = np.unique(x_edges)
    y_edges = np.unique(y_edges)

    if len(x_edges) < 3 or len(y_edges) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    # Marginal histograms
    hist_x = hist_xy.sum(axis=1)
    hist_y = hist_xy.sum(axis=0)

    # Normalize
    p_xy = hist_xy / n
    p_x = hist_x / n
    p_y = hist_y / n

    # Entropies (avoid log(0))
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.ravel())

    mi = h_x + h_y - h_xy

    return float(mi), float(h_x), float(h_y), float(h_xy)


def load_production_data():
    """Load all production + campaign data with S_T and eta."""
    configs = []
    for pattern in [
        str(_HERE / "epyc_results/production_2026_03_11/production_*.json"),
        str(_HERE / "epyc_results/campaign_2026_03_11/*.json"),
    ]:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                raw = json.load(f)
            data = raw if isinstance(raw, list) else raw.get("results", [])
            for d in data:
                if isinstance(d, dict):
                    configs.append(d)
    return configs


def main():
    t0 = time.time()

    print("=" * 72)
    print("INDEPENDENCE TESTS: S_T vs eta")
    print("=" * 72)

    configs = load_production_data()
    print(f"  Loaded {len(configs)} configs")

    # Filter to configs with valid S_T and eta
    valid = []
    for d in configs:
        s_t = d.get("S_T", float("nan"))
        eta = d.get("eta", float("nan"))
        body = d.get("body", "?")
        if not math.isnan(s_t) and not math.isnan(eta) and s_t > 0 and eta > 0:
            valid.append({"S_T": s_t, "eta": eta, "body": body})

    print(f"  Valid (S_T > 0, eta > 0): {len(valid)}")

    st_arr = np.array([v["S_T"] for v in valid])
    eta_arr = np.array([v["eta"] for v in valid])
    bodies = [v["body"] for v in valid]

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: MUTUAL INFORMATION — POOLED
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("TEST 2a: MUTUAL INFORMATION — POOLED")
    print(f"{'=' * 72}")

    mi, h_st, h_eta, h_joint = mutual_information_binned(st_arr, eta_arr)
    # Normalized MI: fraction of marginal entropy explained
    nmi_st = mi / h_st if h_st > 0 else 0
    nmi_eta = mi / h_eta if h_eta > 0 else 0

    print(f"  n = {len(valid)}")
    print(f"  H(S_T)     = {h_st:.4f} nats")
    print(f"  H(eta)     = {h_eta:.4f} nats")
    print(f"  H(S_T,eta) = {h_joint:.4f} nats")
    print(f"  I(S_T;eta) = {mi:.4f} nats")
    print(f"  I / H(S_T) = {nmi_st:.4f}  (fraction of S_T entropy explained by eta)")
    print(f"  I / H(eta) = {nmi_eta:.4f}  (fraction of eta entropy explained by S_T)")

    # For comparison: what would random pairing give?
    # Permutation test
    mi_null = []
    rng = np.random.default_rng(42)
    for _ in range(100):
        eta_perm = rng.permutation(eta_arr)
        mi_p, _, _, _ = mutual_information_binned(st_arr, eta_perm)
        if not math.isnan(mi_p):
            mi_null.append(mi_p)
    mi_null = np.array(mi_null)
    print(f"  Null (permutation, 100 reps): mean={mi_null.mean():.4f}, std={mi_null.std():.4f}")
    print(f"  Observed / null mean = {mi / mi_null.mean():.2f}x")
    z_score = (mi - mi_null.mean()) / mi_null.std() if mi_null.std() > 0 else 0
    print(f"  Z-score vs null: {z_score:.1f}")

    # Also compute Spearman for reference
    rho_s, p_s = stats.spearmanr(st_arr, eta_arr)
    print(f"  Spearman rho = {rho_s:+.4f}, p = {p_s:.2e}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2b: MUTUAL INFORMATION — STRATIFIED BY BODY (Simpson check)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("TEST 2b: MUTUAL INFORMATION — STRATIFIED BY BODY")
    print(f"{'=' * 72}")

    by_body = defaultdict(lambda: {"st": [], "eta": []})
    for v in valid:
        by_body[v["body"]]["st"].append(v["S_T"])
        by_body[v["body"]]["eta"].append(v["eta"])

    print(
        f"\n  {'body':>10s} | {'n':>6s} | {'I(S_T;eta)':>10s} | "
        f"{'I/H(S_T)':>8s} | {'I/H(eta)':>8s} | "
        f"{'Spearman':>10s} | {'p':>10s} | {'S_T range':>12s} | {'eta range':>12s}"
    )
    print(f"  {'-' * 110}")

    body_results = {}
    total_weighted_mi = 0
    total_n = 0

    for body in sorted(by_body.keys()):
        st_b = np.array(by_body[body]["st"])
        eta_b = np.array(by_body[body]["eta"])
        n_b = len(st_b)

        mi_b, h_st_b, h_eta_b, _ = mutual_information_binned(
            st_b, eta_b, n_bins=min(20, int(np.sqrt(n_b)))
        )
        nmi_st_b = mi_b / h_st_b if (h_st_b > 0 and not math.isnan(mi_b)) else float("nan")
        nmi_eta_b = mi_b / h_eta_b if (h_eta_b > 0 and not math.isnan(mi_b)) else float("nan")

        if n_b >= 10:
            rho_b, p_b = stats.spearmanr(st_b, eta_b)
        else:
            rho_b, p_b = float("nan"), float("nan")

        st_range = f"[{st_b.min():.2f},{st_b.max():.2f}]"
        eta_range = f"[{eta_b.min():.2f},{eta_b.max():.2f}]"

        mi_str = f"{mi_b:.4f}" if not math.isnan(mi_b) else "N/A"
        nmi_st_str = f"{nmi_st_b:.4f}" if not math.isnan(nmi_st_b) else "N/A"
        nmi_eta_str = f"{nmi_eta_b:.4f}" if not math.isnan(nmi_eta_b) else "N/A"

        print(
            f"  {body:>10s} | {n_b:>6d} | {mi_str:>10s} | "
            f"{nmi_st_str:>8s} | {nmi_eta_str:>8s} | "
            f"{rho_b:>+10.4f} | {p_b:>10.2e} | {st_range:>12s} | {eta_range:>12s}"
        )

        if not math.isnan(mi_b):
            total_weighted_mi += mi_b * n_b
            total_n += n_b

        body_results[body] = {
            "n": n_b,
            "mi": round(mi_b, 6) if not math.isnan(mi_b) else None,
            "nmi_st": round(nmi_st_b, 6) if not math.isnan(nmi_st_b) else None,
            "spearman_rho": round(rho_b, 4) if not math.isnan(rho_b) else None,
            "spearman_p": float(p_b) if not math.isnan(p_b) else None,
        }

    avg_cond_mi = total_weighted_mi / total_n if total_n > 0 else 0
    print(f"\n  Weighted average conditional MI: {avg_cond_mi:.4f} nats")
    print(f"  Pooled MI:                       {mi:.4f} nats")

    if avg_cond_mi > mi * 1.5 and avg_cond_mi > 0.01:
        print("  WARNING: Simpson's paradox detected — conditional MI >> pooled MI")
    elif avg_cond_mi > mi * 1.2:
        print("  CAUTION: Some within-body dependence masked by pooling")
    else:
        print("  OK: No Simpson's paradox — pooled and conditional MI are comparable")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2c: HETEROSCEDASTICITY — Var(eta) vs S_T
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("TEST 2c: HETEROSCEDASTICITY — Var(eta) vs S_T")
    print(f"{'=' * 72}")

    # Bin S_T into deciles and compute Var(eta) in each bin
    n_bins_het = 10
    st_edges = np.percentile(st_arr, np.linspace(0, 100, n_bins_het + 1))
    st_edges = np.unique(st_edges)

    print(
        f"\n  {'S_T bin':>20s} | {'n':>6s} | {'mean(eta)':>10s} | "
        f"{'std(eta)':>10s} | {'CV(eta)':>8s} | {'Var(eta)':>12s}"
    )
    print(f"  {'-' * 80}")

    het_bins = []
    for i in range(len(st_edges) - 1):
        lo, hi = st_edges[i], st_edges[i + 1]
        if i == len(st_edges) - 2:
            mask = (st_arr >= lo) & (st_arr <= hi)
        else:
            mask = (st_arr >= lo) & (st_arr < hi)
        eta_bin = eta_arr[mask]
        st_bin = st_arr[mask]

        if len(eta_bin) < 5:
            continue

        mean_eta = float(np.mean(eta_bin))
        std_eta = float(np.std(eta_bin))
        var_eta = float(np.var(eta_bin))
        cv_eta = std_eta / mean_eta if mean_eta > 0 else float("nan")
        mean_st = float(np.mean(st_bin))

        print(
            f"  [{lo:.3f}, {hi:.3f}){']' if i == len(st_edges) - 2 else ')'}"
            f" | {len(eta_bin):>6d} | {mean_eta:>10.4f} | "
            f"{std_eta:>10.4f} | {cv_eta:>8.4f} | {var_eta:>12.6f}"
        )

        het_bins.append(
            {
                "st_lo": round(lo, 4),
                "st_hi": round(hi, 4),
                "n": len(eta_bin),
                "mean_st": round(mean_st, 4),
                "mean_eta": round(mean_eta, 4),
                "std_eta": round(std_eta, 4),
                "var_eta": round(var_eta, 6),
                "cv_eta": round(cv_eta, 4) if not math.isnan(cv_eta) else None,
            }
        )

    # Test for trend: Spearman of (mean_st_bin, var_eta_bin)
    if len(het_bins) >= 4:
        bin_sts = np.array([b["mean_st"] for b in het_bins])
        bin_vars = np.array([b["var_eta"] for b in het_bins])
        bin_cvs = np.array([b["cv_eta"] for b in het_bins if b["cv_eta"] is not None])
        bin_sts_cv = np.array([b["mean_st"] for b in het_bins if b["cv_eta"] is not None])

        rho_het, p_het = stats.spearmanr(bin_sts, bin_vars)
        print(f"\n  Trend: Spearman(mean_S_T, Var(eta)) = {rho_het:+.4f}, p = {p_het:.4f}")

        if len(bin_cvs) >= 4:
            rho_cv, p_cv = stats.spearmanr(bin_sts_cv, bin_cvs)
            print(f"  Trend: Spearman(mean_S_T, CV(eta))  = {rho_cv:+.4f}, p = {p_cv:.4f}")

        # Check if low S_T has inflated variance
        low_st = [b for b in het_bins if b["mean_st"] < 0.3]
        high_st = [b for b in het_bins if b["mean_st"] > 0.7]

        if low_st and high_st:
            low_var = np.mean([b["var_eta"] for b in low_st])
            high_var = np.mean([b["var_eta"] for b in high_st])
            ratio = low_var / high_var if high_var > 0 else float("inf")
            print(f"\n  Low S_T (<0.3) mean Var(eta): {low_var:.6f}")
            print(f"  High S_T (>0.7) mean Var(eta): {high_var:.6f}")
            print(f"  Ratio: {ratio:.2f}x")

            if ratio > 3:
                print(f"  HETEROSCEDASTIC: Var(eta) inflates {ratio:.1f}x at low S_T")
                print("  -> Self-averaging fails in catastrophe regime")
                print("  -> Factored theory valid for S_T > 0.3, unreliable below")
            elif ratio > 1.5:
                print(f"  MILD heteroscedasticity ({ratio:.1f}x) — flag but not fatal")
            else:
                print(f"  HOMOSCEDASTIC: Var(eta) stable across S_T range ({ratio:.1f}x)")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SYNTHESIS")
    print(f"{'=' * 72}")

    print(f"""
  Pooled MI:           I(S_T; eta) = {mi:.4f} nats
  Conditional MI:      I(S_T; eta | body) ~ {avg_cond_mi:.4f} nats (weighted avg)
  Null (permutation):  {mi_null.mean():.4f} +/- {mi_null.std():.4f} nats
  Spearman (pooled):   rho = {rho_s:+.4f}
    """)

    # Save
    output = {
        "description": "Independence tests: S_T vs eta — pooled MI, stratified MI, heteroscedasticity",
        "n_valid": len(valid),
        "pooled": {
            "mi_nats": round(mi, 6),
            "h_st": round(h_st, 6),
            "h_eta": round(h_eta, 6),
            "nmi_st": round(nmi_st, 6),
            "nmi_eta": round(nmi_eta, 6),
            "null_mean": round(float(mi_null.mean()), 6),
            "null_std": round(float(mi_null.std()), 6),
            "z_score": round(z_score, 1),
            "spearman_rho": round(rho_s, 4),
        },
        "stratified": body_results,
        "conditional_mi_weighted": round(avg_cond_mi, 6),
        "heteroscedasticity": het_bins,
        "wall_time_s": round(time.time() - t0, 1),
    }

    outpath = _HERE / "independence_test_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved -> {outpath.name}")
    print(f"  Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
