#!/usr/bin/env python3
"""Functional form derivation v2: two-branch fitting strategy.

Strategy: fit β_myopic and β_retry SEPARATELY, then combine.

Branch 1 (E[H] < 4.5): Retry is negligible. β_n ≈ -β_myopic.
Branch 2 (E[H] > 5.0): Both active. Use the DIFFERENCE between
  branches to extract β_retry.

Then combine into a unified model and test on all data.
"""

import json
import os

import numpy as np
from scipy.optimize import brentq, minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_combined():
    records = []
    for fname in ["phi_surface_results.json", "phi_surface_outer_results.json"]:
        path = os.path.join(SCRIPT_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data["results"]:
            if r["eta_sim"] > 0 and r["eta_lyap"] > 0 and r["E_H"] > 0:
                records.append(r)
    return records


def extract_slope_grid(records, p_bins, h_bins):
    ln_phi = np.array([np.log(r["phi_time"]) for r in records])
    log_n = np.array([np.log(r["n_orb"]) for r in records])
    p_eff = np.array([r["p_eff"] for r in records])
    E_H = np.array([r["E_H"] for r in records])

    cells = []
    for i in range(len(p_bins) - 1):
        for j in range(len(h_bins) - 1):
            mask = (
                (p_eff >= p_bins[i])
                & (p_eff < p_bins[i + 1])
                & (E_H >= h_bins[j])
                & (E_H < h_bins[j + 1])
            )
            n = mask.sum()
            if n < 30:
                continue
            y = ln_phi[mask]
            X = np.column_stack([np.ones(n), log_n[mask], p_eff[mask]])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            ss_res = np.sum((y - X @ beta) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            cells.append(
                {
                    "p_mean": float(p_eff[mask].mean()),
                    "h_mean": float(E_H[mask].mean()),
                    "beta_n": float(beta[1]),
                    "r2": float(r2),
                    "n": int(n),
                    "ln_p_mean": float(np.log(np.clip(p_eff[mask], 1e-12, 1)).mean()),
                    "chain_surv_mean": float(
                        (E_H[mask] * np.log(np.clip(p_eff[mask], 1e-12, 1))).mean()
                    ),
                }
            )
    return cells


def main():
    print("Loading data...")
    records = load_combined()
    print(f"  {len(records):,} active raw results\n")

    # Fine grid
    p_bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    h_bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 20]
    cells = extract_slope_grid(records, p_bins, h_bins)

    # Split into branches
    branch1 = [c for c in cells if c["h_mean"] < 4.5]  # myopic-only
    branch2 = [c for c in cells if c["h_mean"] > 5.0]  # both active

    print(f"Branch 1 (E[H] < 4.5): {len(branch1)} cells")
    print(f"Branch 2 (E[H] > 5.0): {len(branch2)} cells")
    print()

    # ==================================================================
    # BRANCH 1: Fit β_myopic alone (β_n ≈ -β_myopic)
    # ==================================================================
    print("=" * 70)
    print("BRANCH 1: β_myopic (E[H] < 4.5, retry negligible)")
    print("=" * 70)
    print()

    p1 = np.array([c["p_mean"] for c in branch1])
    h1 = np.array([c["h_mean"] for c in branch1])
    beta1 = np.array([c["beta_n"] for c in branch1])
    # β_n ≈ -β_myopic, so β_myopic ≈ -β_n
    bm1 = -beta1

    # Candidate forms for β_myopic(p, h):
    # A: δ·h·p^γ
    # B: δ·h·ln(1+p/p₀) / ln(1+1/p₀)  (log-saturating)
    # C: (a + b·ln(p))·h  (log-linear per hop)
    # D: δ·h·(1 - exp(-p/p₀))  (exponential saturation)

    def myopic_A(params, p, h):
        delta, gamma = params
        return delta * h * np.power(np.clip(p, 1e-12, 1), gamma)

    def myopic_B(params, p, h):
        delta, p0 = params
        return delta * h * np.log(1 + p / p0) / np.log(1 + 1 / p0)

    def myopic_C(params, p, h):
        a, b = params
        return (a + b * np.log(np.clip(p, 1e-12, 1))) * h

    def myopic_D(params, p, h):
        delta, p0 = params
        return delta * h * (1 - np.exp(-p / p0))

    def fit_myopic(name, fn, p0, bounds=None):
        def cost(params):
            pred = fn(params, p1, h1)
            return np.sum((bm1 - pred) ** 2)

        result = minimize(cost, p0, bounds=bounds, method="L-BFGS-B" if bounds else "Nelder-Mead")
        pred = fn(result.x, p1, h1)
        ss_res = np.sum((bm1 - pred) ** 2)
        ss_tot = np.sum((bm1 - np.mean(bm1)) ** 2)
        r2 = 1 - ss_res / ss_tot
        mae = np.mean(np.abs(bm1 - pred))
        return result.x, r2, mae, pred

    models_myopic = [
        ("A: δ·h·p^γ", myopic_A, [0.3, 0.2], [(0, 5), (0.01, 1)]),
        ("B: δ·h·log-sat(p/p₀)", myopic_B, [0.3, 0.01], [(0, 5), (1e-6, 1)]),
        ("C: (a+b·ln p)·h", myopic_C, [0.3, 0.05], None),
        ("D: δ·h·(1-exp(-p/p₀))", myopic_D, [0.3, 0.05], [(0, 5), (1e-4, 10)]),
    ]

    for name, fn, p0_init, bounds in models_myopic:
        params, r2, mae, pred = fit_myopic(name, fn, p0_init, bounds)
        print(f"  {name}")
        print(f"    R² = {r2:.4f},  MAE = {mae:.4f}")
        print(f"    params = {', '.join(f'{p:.5f}' for p in params)}")

        # Show predictions for each cell
        for i, c in enumerate(sorted(branch1, key=lambda x: (x["h_mean"], x["p_mean"]))):
            idx = branch1.index(c)
            err = bm1[idx] - pred[idx]
            if abs(err) > 0.15:
                flag = " <-- !"
            else:
                flag = ""
            # Only print a subset for brevity
        print()

    # Detailed dive on best myopic model
    print("-" * 70)
    print("BEST MYOPIC FIT: trying multi-start for each model")
    print("-" * 70)
    print()

    best_overall_r2 = -999
    best_overall_name = ""
    best_overall_params = None
    best_overall_fn = None

    for name, fn, p0_base, bounds in models_myopic:
        best_r2 = -999
        best_params = None
        # Multi-start
        for scale in [0.5, 1.0, 2.0, 5.0]:
            try:
                p0 = [x * scale for x in p0_base]
                if bounds:
                    p0 = [max(b[0], min(b[1], x)) for x, b in zip(p0, bounds)]
                params, r2, _, _ = fit_myopic(name, fn, p0, bounds)
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = params
            except Exception:
                pass
        if best_r2 > best_overall_r2:
            best_overall_r2 = best_r2
            best_overall_name = name
            best_overall_params = best_params
            best_overall_fn = fn
        print(f"  {name}: R² = {best_r2:.4f}, params = {best_params}")

    print(f"\n  WINNER: {best_overall_name} (R² = {best_overall_r2:.4f})")
    print()

    # Show cell-by-cell for winner
    pred_best = best_overall_fn(best_overall_params, p1, h1)
    print(f"  {'p_eff':>8s} {'E[H]':>6s} {'β_myopic':>10s} {'predicted':>10s} {'error':>8s}")
    print("  " + "-" * 48)
    for i, c in enumerate(sorted(branch1, key=lambda x: (x["h_mean"], x["p_mean"]))):
        idx = branch1.index(c)
        print(
            f"  {p1[idx]:>8.4f} {h1[idx]:>6.2f} {bm1[idx]:>+10.4f} {pred_best[idx]:>+10.4f} "
            f"{bm1[idx] - pred_best[idx]:>+8.4f}"
        )

    # ==================================================================
    # BRANCH 2: Extract β_retry from E[H] > 5 data
    # ==================================================================
    print()
    print("=" * 70)
    print("BRANCH 2: β_retry (E[H] > 5.0)")
    print("=" * 70)
    print()

    p2 = np.array([c["p_mean"] for c in branch2])
    h2 = np.array([c["h_mean"] for c in branch2])
    beta2 = np.array([c["beta_n"] for c in branch2])

    # β_n = β_retry - β_myopic
    # We know β_myopic from Branch 1. Extrapolate it to E[H] > 5:
    bm2_predicted = best_overall_fn(best_overall_params, p2, h2)
    # So β_retry = β_n + β_myopic
    br2 = beta2 + bm2_predicted

    print("  Extracted β_retry at E[H] > 5:")
    print(f"  {'p_eff':>8s} {'E[H]':>6s} {'β_n':>8s} {'β_myopic':>10s} {'β_retry':>10s}")
    print("  " + "-" * 48)
    for i, c in enumerate(sorted(branch2, key=lambda x: (x["h_mean"], x["p_mean"]))):
        idx = branch2.index(c)
        print(
            f"  {p2[idx]:>8.4f} {h2[idx]:>6.2f} {beta2[idx]:>+8.4f} {bm2_predicted[idx]:>+10.4f} "
            f"{br2[idx]:>+10.4f}"
        )

    print(f"\n  Mean β_retry: {br2.mean():.4f}")
    print(f"  Std  β_retry: {br2.std():.4f}")
    print(f"  Min  β_retry: {br2.min():.4f}")
    print(f"  Max  β_retry: {br2.max():.4f}")

    # Does β_retry depend on p_eff? on E[H]?
    if len(br2) > 5:
        rho_p = np.corrcoef(p2, br2)[0, 1]
        rho_h = np.corrcoef(h2, br2)[0, 1]
        rho_cs = np.corrcoef(h2 * np.log(np.clip(p2, 1e-12, 1)), br2)[0, 1]
        print("\n  Correlations of β_retry:")
        print(f"    rho(β_retry, p_eff)       = {rho_p:+.4f}")
        print(f"    rho(β_retry, E[H])        = {rho_h:+.4f}")
        print(f"    rho(β_retry, chain_surv)  = {rho_cs:+.4f}")

    # Candidate forms for β_retry(p, h):
    # R1: constant r₀
    # R2: r₀ · (1 - p^h)  [chain failure probability]
    # R3: r₀ · h / (h + h₀)  [saturating in hops]
    # R4: r₀ · h  [linear in hops]

    print()
    print("  Fitting β_retry models:")
    for rname, rfn in [
        ("R1: constant r₀", lambda params, p, h: np.full_like(p, params[0])),
        (
            "R2: r₀·(1-p^h)",
            lambda params, p, h: params[0] * (1.0 - np.power(np.clip(p, 1e-12, 1), h)),
        ),
        (
            "R3: r₀·h/(h+h₀)",
            lambda params, p, h: (
                params[0] * h / (h + params[1]) if len(params) > 1 else params[0] * h / (h + 3)
            ),
        ),
    ]:
        if "R3" in rname:
            p0_r = [2.0, 3.0]
            bounds_r = [(0, 10), (0.1, 50)]
        else:
            p0_r = [np.mean(br2)]
            bounds_r = [(0, 10)]

        def cost_r(params, fn=rfn):
            pred = fn(params, p2, h2)
            return np.sum((br2 - pred) ** 2)

        result_r = minimize(cost_r, p0_r, bounds=bounds_r, method="L-BFGS-B")
        pred_r = rfn(result_r.x, p2, h2)
        ss_res = np.sum((br2 - pred_r) ** 2)
        ss_tot = np.sum((br2 - np.mean(br2)) ** 2)
        r2_r = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"    {rname}: R² = {r2_r:.4f}, params = {result_r.x}")

    # ==================================================================
    # UNIFIED MODEL: combine best myopic + retry
    # ==================================================================
    print()
    print("=" * 70)
    print("UNIFIED MODEL: fit to ALL cells simultaneously")
    print("=" * 70)
    print()

    # Using the insights from separate fits:
    # β_myopic = δ·h·p^γ  (from Branch 1)
    # β_retry = r₀·(1 - p^h)  (candidate from Branch 2)
    # β_n = β_retry - β_myopic = r₀·(1-p^h) - δ·h·p^γ
    #
    # KEY INSIGHT: Force r₀ > 0 by initializing from Branch 2 estimate.
    # Use unweighted cell-level fit.

    p_all = np.array([c["p_mean"] for c in cells])
    h_all = np.array([c["h_mean"] for c in cells])
    beta_all = np.array([c["beta_n"] for c in cells])

    def unified_model(params, p, h):
        r, delta, gamma = params
        retry = r * (1.0 - np.power(np.clip(p, 1e-12, 1), h))
        myopic = delta * h * np.power(np.clip(p, 1e-12, 1), gamma)
        return retry - myopic

    # Multi-start with r pinned near the mean extracted β_retry
    mean_br = float(br2.mean()) if len(br2) > 0 else 1.0
    best_uni_r2 = -999
    best_uni_params = None

    for r_init in [mean_br * 0.5, mean_br, mean_br * 1.5, mean_br * 2.0]:
        for d_init in [0.1, 0.2, 0.3, 0.5]:
            for g_init in [0.05, 0.1, 0.2, 0.3, 0.5]:
                try:

                    def cost_uni(params):
                        pred = unified_model(params, p_all, h_all)
                        return np.sum((beta_all - pred) ** 2)

                    result_uni = minimize(
                        cost_uni,
                        [r_init, d_init, g_init],
                        bounds=[(0.01, 5), (0.01, 5), (0.01, 1)],
                        method="L-BFGS-B",
                    )
                    pred = unified_model(result_uni.x, p_all, h_all)
                    ss_res = np.sum((beta_all - pred) ** 2)
                    ss_tot = np.sum((beta_all - np.mean(beta_all)) ** 2)
                    r2 = 1 - ss_res / ss_tot
                    if r2 > best_uni_r2:
                        best_uni_r2 = r2
                        best_uni_params = result_uni.x
                except Exception:
                    pass

    if best_uni_params is not None:
        r, delta, gamma = best_uni_params
        print(f"  UNIFIED: β_n = {r:.4f}·(1−p^h) − {delta:.4f}·h·p^{gamma:.4f}")
        print(f"  R² = {best_uni_r2:.4f}")
        print()

        # Decomposition table
        pred = unified_model(best_uni_params, p_all, h_all)
        retry_vals = r * (1.0 - np.power(np.clip(p_all, 1e-12, 1), h_all))
        myopic_vals = delta * h_all * np.power(np.clip(p_all, 1e-12, 1), gamma)

        print(
            f"  {'p_eff':>8s} {'E[H]':>6s} {'β_data':>8s} {'β_pred':>8s} {'retry':>8s} {'myopic':>8s} {'err':>8s}"
        )
        print("  " + "-" * 58)
        sorted_idx = sorted(
            range(len(cells)), key=lambda i: (cells[i]["h_mean"], cells[i]["p_mean"])
        )
        for idx in sorted_idx:
            err = beta_all[idx] - pred[idx]
            flag = " !" if abs(err) > 0.25 else ""
            print(
                f"  {p_all[idx]:>8.4f} {h_all[idx]:>6.2f} {beta_all[idx]:>+8.4f} {pred[idx]:>+8.4f} "
                f"{retry_vals[idx]:>+8.4f} {myopic_vals[idx]:>+8.4f} {err:>+8.4f}{flag}"
            )

        # Braess boundary
        print()
        print("  BRAESS BOUNDARY (β_n = 0):")
        for h_test in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
            try:

                def f_braess(p):
                    return r * (1 - p**h_test) - delta * h_test * p**gamma

                # Check endpoints
                f_lo = f_braess(1e-8)
                f_hi = f_braess(0.999)
                if f_lo > 0 and f_hi > 0:
                    print(f"    E[H]={h_test:.1f}: β_n > 0 at all p  (retry always wins)")
                elif f_lo < 0 and f_hi < 0:
                    print(f"    E[H]={h_test:.1f}: β_n < 0 at all p  (myopic always wins)")
                else:
                    p_zero = brentq(f_braess, 1e-8, 0.999)
                    n_star = np.exp(0)  # β_n=0 means Φ is independent of n
                    cs = p_zero**h_test
                    print(
                        f"    E[H]={h_test:.1f}: p_eff* = {p_zero:.4f}  "
                        f"(chain_surv = {cs:.2e}, "
                        f"retry={r * (1 - cs):.3f}, myopic={delta * h_test * p_zero**gamma:.3f})"
                    )
            except Exception as e:
                print(f"    E[H]={h_test:.1f}: error ({e})")

        # Physical interpretation
        print()
        print("  PHYSICAL INTERPRETATION:")
        print(f"    r₀ = {r:.4f} — the retry benefit coefficient")
        print("       (1 − p_eff^E[H]) is the probability that the oracle chain fails")
        print(f"       When it fails, the greedy's retry gives ~{r:.2f} units of log(n) advantage")
        print()
        print(f"    δ₀ = {delta:.4f} — the per-hop myopic penalty coefficient")
        print(f"    γ  = {gamma:.4f} — the exposure exponent")
        print("       p_eff^γ is the exposure function (how much wrong choices commit)")
        print("       γ < 1 means sublinear exposure: even at low p, some penalty exists")
        print()
        print("    The balance point: β_n = 0 when")
        print("       r₀·(1-p^h) = δ₀·h·p^γ")
        print("       oracle failure benefit = cumulative myopic penalty")

    # ==================================================================
    # Also try: M3-style (log-linear) as unified with regime awareness
    # ==================================================================
    print()
    print("=" * 70)
    print("ALTERNATIVE: log-linear with E[H] threshold")
    print("=" * 70)
    print()

    # β_n = a + b·h + c·h·ln(p) + d·max(0, h-h₀)
    def threshold_model(params, p, h):
        a, b, c, d, h0 = params
        base = a + b * h + c * h * np.log(np.clip(p, 1e-12, 1))
        boost = d * np.maximum(0, h - h0)
        return base + boost

    best_th_r2 = -999
    best_th_params = None
    for a_init in [-0.5, 0, 0.5]:
        for b_init in [-0.2, -0.1, 0]:
            for c_init in [-0.1, -0.05, 0]:
                for d_init in [0.1, 0.3, 0.5]:
                    for h0_init in [3.0, 4.0, 5.0]:
                        try:

                            def cost_th(params):
                                pred = threshold_model(params, p_all, h_all)
                                return np.sum((beta_all - pred) ** 2)

                            result_th = minimize(
                                cost_th,
                                [a_init, b_init, c_init, d_init, h0_init],
                                bounds=[(-3, 3), (-1, 1), (-0.5, 0.5), (0, 3), (1, 8)],
                                method="L-BFGS-B",
                            )
                            pred_th = threshold_model(result_th.x, p_all, h_all)
                            ss_res = np.sum((beta_all - pred_th) ** 2)
                            ss_tot = np.sum((beta_all - np.mean(beta_all)) ** 2)
                            r2_th = 1 - ss_res / ss_tot
                            if r2_th > best_th_r2:
                                best_th_r2 = r2_th
                                best_th_params = result_th.x
                        except Exception:
                            pass

    if best_th_params is not None:
        a, b, c, d, h0 = best_th_params
        print(f"  β_n = {a:.3f} + {b:+.3f}·h + {c:+.4f}·h·ln(p) + {d:+.3f}·max(0, h−{h0:.1f})")
        print(f"  R² = {best_th_r2:.4f}")
        print()

        pred_th = threshold_model(best_th_params, p_all, h_all)
        print(f"  {'p_eff':>8s} {'E[H]':>6s} {'β_data':>8s} {'β_pred':>8s} {'err':>8s}")
        print("  " + "-" * 44)
        for idx in sorted_idx:
            err = beta_all[idx] - pred_th[idx]
            flag = " !" if abs(err) > 0.25 else ""
            print(
                f"  {p_all[idx]:>8.4f} {h_all[idx]:>6.2f} {beta_all[idx]:>+8.4f} {pred_th[idx]:>+8.4f} "
                f"{err:>+8.4f}{flag}"
            )

        # Physical interpretation
        print()
        print("  The log-linear terms: β_base = a + b·h + c·h·ln(p)")
        print(f"    = {a:.3f} + ({b:+.3f} + {c:+.4f}·ln p)·h")
        print(f"    At p=1.0: slope in h = {b:.3f}")
        print(f"    At p=0.01: slope in h = {b + c * np.log(0.01):.3f}")
        print(f"    At p=0.001: slope in h = {b + c * np.log(0.001):.3f}")
        print()
        print(f"  The threshold term: {d:+.3f}·max(0, h−{h0:.1f})")
        print(f"    Kicks in at E[H] > {h0:.1f}")
        print(f"    Adds {d:.3f} per unit E[H] above threshold")

    # Save
    results = {
        "best_myopic": {
            "name": best_overall_name,
            "params": [float(x) for x in best_overall_params]
            if best_overall_params is not None
            else None,
            "r2": float(best_overall_r2),
        },
        "unified_M6": {
            "params": [float(x) for x in best_uni_params] if best_uni_params is not None else None,
            "r2": float(best_uni_r2),
        },
        "threshold_model": {
            "params": [float(x) for x in best_th_params] if best_th_params is not None else None,
            "r2": float(best_th_r2),
        },
    }
    out_path = os.path.join(SCRIPT_DIR, "functional_forms_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
