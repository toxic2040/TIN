#!/usr/bin/env python3
"""Φ decomposition analysis — processes Experiment 1+3 results.

Reads: runs/phi_decompose_results.json
Analyses:
  1. Factorization test: Φ ≈ Φ_myopic × Φ_retry
  2. Direct b_myopic and b_retry from separate log(n) slopes
  3. Test β_myopic formula: 0.287·h·(1−exp(−p/0.019)) against direct measurement
  4. Fit β_retry functional form with direct data
  5. Braess critical surface: b_net = 0
  6. Retry counter: N_attempts(n, p_eff) scaling law
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    # Try full results first, fall back to lean
    for fname in ["phi_decompose_results.json", "phi_decompose_lean_results.json"]:
        path = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(path):
            print(f"  Loading {fname}...")
            with open(path) as f:
                data = json.load(f)
            return data
    print("ERROR: no decomposition results found (tried full and lean)")
    sys.exit(1)
    return data


def active_records(records):
    """Keep only records where all three phi values are finite and positive."""
    active = []
    for r in records:
        try:
            pn = float(r["phi_normal"])
            pm = float(r["phi_myopic"])
            pr = float(r["phi_retry"])
            if (
                np.isfinite(pn)
                and np.isfinite(pm)
                and np.isfinite(pr)
                and pn > 0
                and pm > 0
                and pr > 0
                and r["eta_lyap"] > 0
                and r["E_H"] > 0
            ):
                active.append(r)
        except (ValueError, TypeError):
            continue
    return active


def fit_slope(ln_phi, log_n, p_eff):
    """Fit ln(Phi) = a + b*log(n) + c*p_eff, return b (log(n) slope)."""
    if len(ln_phi) < 10:
        return np.nan, np.nan, 0
    X = np.column_stack([np.ones(len(ln_phi)), log_n, p_eff])
    beta = np.linalg.lstsq(X, ln_phi, rcond=None)[0]
    ss_res = np.sum((ln_phi - X @ beta) ** 2)
    ss_tot = np.sum((ln_phi - np.mean(ln_phi)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta[1], r2, len(ln_phi)


def main():
    data = load_data()
    raw = data["results"]
    params = data.get("parameters", {})
    print("\n  Φ Decomposition Analysis")
    print(f"  {'=' * 60}")
    print(f"  Total rows: {len(raw):,}")

    active = active_records(raw)
    print(f"  Active rows (all phis finite & positive): {len(active):,}")
    print()

    # Extract arrays
    phi_n = np.array([r["phi_normal"] for r in active])
    phi_m = np.array([r["phi_myopic"] for r in active])
    phi_r = np.array([r["phi_retry"] for r in active])
    phi_prod = phi_m * phi_r
    log_n = np.array([np.log(r["n_orb"]) for r in active])
    p_eff = np.array([r["p_eff"] for r in active])
    E_H = np.array([r["E_H"] for r in active])
    targets = np.array([r["target"] for r in active])
    n_orbs = np.array([r["n_orb"] for r in active])
    ln_peff = np.log(np.clip(p_eff, 1e-12, 1.0))
    chain_surv = E_H * ln_peff

    # Retry counter data
    attempts = np.array([r.get("mean_attempts_per_bundle", 0) for r in active])
    failed_contacts = np.array([r.get("mean_failed_per_bundle", 0) for r in active])

    # ==================================================================
    # SECTION 1: FACTORIZATION TEST — Φ ≈ Φ_myopic × Φ_retry
    # ==================================================================
    print("=" * 70)
    print("SECTION 1: FACTORIZATION TEST  Φ ≈ Φ_myopic × Φ_retry")
    print("=" * 70)
    print()

    fact_error = np.abs(phi_n - phi_prod) / np.maximum(np.abs(phi_n), 1e-12)
    print("  Factorization error statistics:")
    print(f"    Mean:   {fact_error.mean():.4f} ({fact_error.mean() * 100:.2f}%)")
    print(f"    Median: {np.median(fact_error):.4f} ({np.median(fact_error) * 100:.2f}%)")
    print(f"    P95:    {np.percentile(fact_error, 95):.4f}")
    print(f"    P99:    {np.percentile(fact_error, 99):.4f}")
    print(f"    Max:    {fact_error.max():.4f}")
    print(f"    Fraction < 5%:  {(fact_error < 0.05).mean() * 100:.1f}%")
    print(f"    Fraction < 10%: {(fact_error < 0.10).mean() * 100:.1f}%")
    print()

    # Per-target factorization
    target_order = ["mercury", "venus", "mars", "ceres", "europa", "jupiter", "saturn", "titan"]
    print("  Per-target factorization error:")
    print(
        f"  {'target':>10s} {'N':>6s} {'mean_err':>9s} {'P95':>9s} {'Φ_mean':>8s} {'Φm_mean':>8s} {'Φr_mean':>8s}"
    )
    print("  " + "-" * 60)
    for t in target_order:
        mask = targets == t
        if mask.sum() == 0:
            continue
        fe = fact_error[mask]
        print(
            f"  {t:>10s} {mask.sum():>6d} {fe.mean():>9.4f} {np.percentile(fe, 95):>9.4f} "
            f"{phi_n[mask].mean():>8.3f} {phi_m[mask].mean():>8.3f} {phi_r[mask].mean():>8.3f}"
        )
    print()

    # Log-domain check: ln(Φ) ≈ ln(Φ_m) + ln(Φ_r)
    ln_phi_n = np.log(phi_n)
    ln_phi_m = np.log(phi_m)
    ln_phi_r = np.log(phi_r)
    log_error = np.abs(ln_phi_n - (ln_phi_m + ln_phi_r))
    print("  Log-domain: |ln(Φ) − ln(Φ_m) − ln(Φ_r)|")
    print(f"    Mean:   {log_error.mean():.4f}")
    print(f"    Max:    {log_error.max():.4f}")
    print()

    # ==================================================================
    # SECTION 2: DIRECT b_myopic AND b_retry MEASUREMENT
    # ==================================================================
    print("=" * 70)
    print("SECTION 2: DIRECT b_myopic AND b_retry FROM log(n) SLOPES")
    print("=" * 70)
    print()

    p_bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    h_bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 20]

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

            # b_net from normal Φ
            b_net, r2_n, _ = fit_slope(ln_phi_n[mask], log_n[mask], p_eff[mask])
            # b_myopic from Φ_myopic (should be NEGATIVE — myopic makes Φ_m decrease with n)
            b_myopic_raw, r2_m, _ = fit_slope(ln_phi_m[mask], log_n[mask], p_eff[mask])
            # b_retry from Φ_retry (should be POSITIVE — retry makes Φ_r increase with n)
            b_retry_raw, r2_r, _ = fit_slope(ln_phi_r[mask], log_n[mask], p_eff[mask])

            cells.append(
                {
                    "p_lo": p_bins[i],
                    "p_hi": p_bins[i + 1],
                    "h_lo": h_bins[j],
                    "h_hi": h_bins[j + 1],
                    "p_mean": float(p_eff[mask].mean()),
                    "h_mean": float(E_H[mask].mean()),
                    "n": int(n),
                    "b_net": float(b_net),
                    "b_myopic": float(b_myopic_raw),  # slope of ln(Φ_m) in log(n)
                    "b_retry": float(b_retry_raw),  # slope of ln(Φ_r) in log(n)
                    "b_sum": float(b_myopic_raw + b_retry_raw),  # should ≈ b_net
                    "r2_net": float(r2_n),
                    "r2_myopic": float(r2_m),
                    "r2_retry": float(r2_r),
                    "chain_surv_mean": float((E_H[mask] * ln_peff[mask]).mean()),
                    "mean_attempts": float(attempts[mask].mean()) if attempts.any() else 0,
                }
            )

    print(f"  {len(cells)} cells with N >= 30")
    print()
    print(
        f"  {'p_eff':>8s} {'E[H]':>6s} {'b_net':>7s} {'b_myop':>7s} {'b_retr':>7s} {'sum':>7s} {'err':>7s} {'N':>5s}"
    )
    print("  " + "-" * 60)
    for c in sorted(cells, key=lambda x: (x["h_mean"], x["p_mean"])):
        err = c["b_net"] - c["b_sum"]
        flag = " !" if abs(err) > 0.10 else ""
        print(
            f"  {c['p_mean']:>8.4f} {c['h_mean']:>6.2f} {c['b_net']:>+7.3f} {c['b_myopic']:>+7.3f} "
            f"{c['b_retry']:>+7.3f} {c['b_sum']:>+7.3f} {err:>+7.3f} {c['n']:>5d}{flag}"
        )

    # Additive decomposition test
    b_nets = np.array([c["b_net"] for c in cells])
    b_sums = np.array([c["b_sum"] for c in cells])
    decomp_err = np.abs(b_nets - b_sums)
    print("\n  Additive decomposition: b_net ≈ b_myopic + b_retry")
    print(f"    Mean |error|: {decomp_err.mean():.4f}")
    print(f"    Max  |error|: {decomp_err.max():.4f}")
    print(f"    Correlation:  {np.corrcoef(b_nets, b_sums)[0, 1]:.4f}")
    print()

    # ==================================================================
    # SECTION 3: TEST β_myopic FORMULA FROM BRANCH 1
    # ==================================================================
    print("=" * 70)
    print("SECTION 3: β_myopic FORMULA TEST — 0.287·h·(1−exp(−p/0.019))")
    print("=" * 70)
    print()

    # The formula from functional_forms_v2: β_myopic = δ·h·(1−exp(−p/p₀))
    # with δ=0.287, p₀=0.019
    # Note: β_myopic_formula is the negative of b_myopic (slope of ln(Φ_m))
    # since Φ_m decreases with n, b_myopic < 0, and β_myopic = -b_myopic

    p_arr = np.array([c["p_mean"] for c in cells])
    h_arr = np.array([c["h_mean"] for c in cells])
    bm_direct = np.array([-c["b_myopic"] for c in cells])  # β_myopic = -b_myopic

    # Test old formula
    bm_old_pred = 0.287 * h_arr * (1 - np.exp(-p_arr / 0.019))
    old_err = bm_direct - bm_old_pred
    old_ss_res = np.sum(old_err**2)
    old_ss_tot = np.sum((bm_direct - bm_direct.mean()) ** 2)
    old_r2 = 1 - old_ss_res / old_ss_tot

    print("  OLD formula (from surface sweeps): β_myopic = 0.287·h·(1−exp(−p/0.019))")
    print(f"    R² on direct data: {old_r2:.4f}")
    print(f"    MAE: {np.mean(np.abs(old_err)):.4f}")
    print()

    # Refit with direct measurements
    def myopic_D(params, p, h):
        delta, p0 = params
        return delta * h * (1 - np.exp(-p / p0))

    best_myopic_r2 = -999
    best_myopic_params = None
    for delta_init in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for p0_init in [0.005, 0.01, 0.02, 0.05, 0.1]:
            try:

                def cost(params):
                    pred = myopic_D(params, p_arr, h_arr)
                    return np.sum((bm_direct - pred) ** 2)

                result = minimize(
                    cost, [delta_init, p0_init], bounds=[(0.01, 2), (1e-4, 1)], method="L-BFGS-B"
                )
                pred = myopic_D(result.x, p_arr, h_arr)
                ss_res = np.sum((bm_direct - pred) ** 2)
                r2 = 1 - ss_res / old_ss_tot
                if r2 > best_myopic_r2:
                    best_myopic_r2 = r2
                    best_myopic_params = result.x
            except Exception:
                pass

    if best_myopic_params is not None:
        delta_new, p0_new = best_myopic_params
        print(f"  REFIT formula: β_myopic = {delta_new:.4f}·h·(1−exp(−p/{p0_new:.4f}))")
        print(f"    R² on direct data: {best_myopic_r2:.4f}")
        bm_new_pred = myopic_D(best_myopic_params, p_arr, h_arr)
        print(f"    MAE: {np.mean(np.abs(bm_direct - bm_new_pred)):.4f}")
        print()

        # Cell-by-cell comparison
        print(f"  {'p_eff':>8s} {'E[H]':>6s} {'β_m_data':>9s} {'old_pred':>9s} {'new_pred':>9s}")
        print("  " + "-" * 48)
        for i, c in enumerate(sorted(cells, key=lambda x: (x["h_mean"], x["p_mean"]))):
            idx = cells.index(c)
            print(
                f"  {p_arr[idx]:>8.4f} {h_arr[idx]:>6.2f} {bm_direct[idx]:>+9.4f} "
                f"{bm_old_pred[idx]:>+9.4f} {bm_new_pred[idx]:>+9.4f}"
            )

    # Also test other candidate forms
    print()
    print("  Other candidate forms for β_myopic:")
    candidates = [
        (
            "A: δ·h·p^γ",
            lambda p, h, d, g: d * h * np.power(np.clip(p, 1e-12, 1), g),
            [0.3, 0.2],
            [(0, 2), (0.01, 1)],
        ),
        (
            "C: (a+b·ln p)·h",
            lambda p, h, a, b: (a + b * np.log(np.clip(p, 1e-12, 1))) * h,
            [0.3, 0.05],
            [(0, 2), (-1, 1)],
        ),
    ]
    for name, fn, p0_init, bounds in candidates:
        try:

            def cost_c(params, fn=fn):
                pred = fn(p_arr, h_arr, *params)
                return np.sum((bm_direct - pred) ** 2)

            result_c = minimize(cost_c, p0_init, bounds=bounds, method="L-BFGS-B")
            pred_c = fn(p_arr, h_arr, *result_c.x)
            ss_res = np.sum((bm_direct - pred_c) ** 2)
            r2_c = 1 - ss_res / old_ss_tot
            print(f"    {name}: R² = {r2_c:.4f}, params = {result_c.x}")
        except Exception as e:
            print(f"    {name}: FAILED ({e})")

    print()

    # ==================================================================
    # SECTION 4: DIRECT β_retry FUNCTIONAL FORM
    # ==================================================================
    print("=" * 70)
    print("SECTION 4: DIRECT β_retry FUNCTIONAL FORM")
    print("=" * 70)
    print()

    br_direct = np.array([c["b_retry"] for c in cells])  # b_retry = slope of ln(Φ_r)

    print("  β_retry statistics (direct measurement):")
    print(f"    Mean:   {br_direct.mean():.4f}")
    print(f"    Std:    {br_direct.std():.4f}")
    print(f"    Min:    {br_direct.min():.4f}")
    print(f"    Max:    {br_direct.max():.4f}")
    print(f"    CV:     {br_direct.std() / abs(br_direct.mean()) * 100:.1f}%")
    print()

    # Correlations
    rho_p = np.corrcoef(p_arr, br_direct)[0, 1]
    rho_h = np.corrcoef(h_arr, br_direct)[0, 1]
    cs_arr = np.array([c["chain_surv_mean"] for c in cells])
    rho_cs = np.corrcoef(cs_arr, br_direct)[0, 1]
    rho_lnp = np.corrcoef(np.log(np.clip(p_arr, 1e-12, 1)), br_direct)[0, 1]

    print("  Correlations with β_retry:")
    print(f"    rho(β_retry, p_eff)       = {rho_p:+.4f}")
    print(f"    rho(β_retry, ln(p_eff))   = {rho_lnp:+.4f}")
    print(f"    rho(β_retry, E[H])        = {rho_h:+.4f}")
    print(f"    rho(β_retry, chain_surv)  = {rho_cs:+.4f}")
    print()

    # Candidate functional forms for β_retry
    print("  Candidate forms for β_retry:")

    retry_models = [
        ("R1: constant r₀", lambda params, p, h: np.full_like(p, params[0]), [0.5], [(0, 5)]),
        (
            "R2: r₀·(1 − p^h)  [chain failure prob]",
            lambda params, p, h: params[0] * (1.0 - np.power(np.clip(p, 1e-12, 1), h)),
            [1.0],
            [(0, 10)],
        ),
        (
            "R3: r₀·h/(h + h₀)  [saturating in hops]",
            lambda params, p, h: params[0] * h / (h + params[1]),
            [1.0, 3.0],
            [(0, 10), (0.1, 50)],
        ),
        (
            "R4: r₀ + r₁·h  [linear in hops]",
            lambda params, p, h: params[0] + params[1] * h,
            [0, 0.1],
            [(-5, 5), (-1, 1)],
        ),
        (
            "R5: r₀·max(0, h−h₀)  [threshold]",
            lambda params, p, h: params[0] * np.maximum(0, h - params[1]),
            [0.3, 3.0],
            [(0, 5), (1, 8)],
        ),
        (
            "R6: r₀·(1−exp(−h/h₀))  [exponential onset]",
            lambda params, p, h: params[0] * (1 - np.exp(-h / params[1])),
            [0.5, 3.0],
            [(0, 10), (0.1, 20)],
        ),
        (
            "R7: r₀·h·(1−p^h)  [chain failure × hops]",
            lambda params, p, h: params[0] * h * (1.0 - np.power(np.clip(p, 1e-12, 1), h)),
            [0.1],
            [(0, 5)],
        ),
    ]

    br_ss_tot = np.sum((br_direct - br_direct.mean()) ** 2)
    best_retry_r2 = -999
    best_retry_name = ""
    best_retry_params = None
    best_retry_fn = None

    for name, fn, p0_init, bounds in retry_models:
        best_r2_this = -999
        best_params_this = None
        for scale in [0.5, 1.0, 2.0]:
            try:
                p0 = [x * scale for x in p0_init]
                p0 = [max(b[0], min(b[1], x)) for x, b in zip(p0, bounds)]

                def cost_r(params, fn=fn):
                    pred = fn(params, p_arr, h_arr)
                    return np.sum((br_direct - pred) ** 2)

                result_r = minimize(cost_r, p0, bounds=bounds, method="L-BFGS-B")
                pred_r = fn(result_r.x, p_arr, h_arr)
                ss_res = np.sum((br_direct - pred_r) ** 2)
                r2 = 1 - ss_res / br_ss_tot if br_ss_tot > 0 else 0
                if r2 > best_r2_this:
                    best_r2_this = r2
                    best_params_this = result_r.x
            except Exception:
                pass
        print(f"    {name}")
        print(f"      R² = {best_r2_this:.4f}, params = {best_params_this}")
        if best_r2_this > best_retry_r2:
            best_retry_r2 = best_r2_this
            best_retry_name = name
            best_retry_params = best_params_this
            best_retry_fn = fn

    print(f"\n  BEST RETRY MODEL: {best_retry_name}")
    print(f"    R² = {best_retry_r2:.4f}, params = {best_retry_params}")
    print()

    # Cell-by-cell for best retry model
    if best_retry_fn is not None and best_retry_params is not None:
        br_pred = best_retry_fn(best_retry_params, p_arr, h_arr)
        print(f"  {'p_eff':>8s} {'E[H]':>6s} {'β_r_data':>9s} {'predicted':>9s} {'error':>8s}")
        print("  " + "-" * 48)
        for i, c in enumerate(sorted(cells, key=lambda x: (x["h_mean"], x["p_mean"]))):
            idx = cells.index(c)
            err = br_direct[idx] - br_pred[idx]
            print(
                f"  {p_arr[idx]:>8.4f} {h_arr[idx]:>6.2f} {br_direct[idx]:>+9.4f} "
                f"{br_pred[idx]:>+9.4f} {err:>+8.4f}"
            )

    print()

    # ==================================================================
    # SECTION 5: BRAESS CRITICAL SURFACE
    # ==================================================================
    print("=" * 70)
    print("SECTION 5: BRAESS CRITICAL SURFACE (b_net = 0)")
    print("=" * 70)
    print()

    # Cross-table: b_net by (p_eff × E[H])
    p_bins_cross = [0, 0.01, 0.05, 0.2, 1.01]
    h_bins_cross = [0, 2.5, 3.5, 5.0, 20]

    print("  b_net cross-table (direct from decomposition data):")
    header = f"  {'':>18s}"
    for j in range(len(h_bins_cross) - 1):
        header += f" E[H]=[{h_bins_cross[j]:.1f},{h_bins_cross[j + 1]:.1f})"
    print(header)
    print("  " + "-" * (18 + 20 * len(h_bins_cross)))
    for i in range(len(p_bins_cross) - 1):
        row = f"  p=[{p_bins_cross[i]:.2f},{p_bins_cross[i + 1]:.2f})"
        row = f"{row:>18s}"
        for j in range(len(h_bins_cross) - 1):
            mask = (
                (p_eff >= p_bins_cross[i])
                & (p_eff < p_bins_cross[i + 1])
                & (E_H >= h_bins_cross[j])
                & (E_H < h_bins_cross[j + 1])
            )
            if mask.sum() < 20:
                row += f"{'---':>20s}"
            else:
                b, _, n = fit_slope(ln_phi_n[mask], log_n[mask], p_eff[mask])
                row += f"  {b:+.3f} (n={n:>5d})"
        print(row)
    print()

    # Also show b_myopic and b_retry cross-tables
    for label, arr in [("b_myopic", ln_phi_m), ("b_retry", ln_phi_r)]:
        print(f"  {label} cross-table:")
        header = f"  {'':>18s}"
        for j in range(len(h_bins_cross) - 1):
            header += f" E[H]=[{h_bins_cross[j]:.1f},{h_bins_cross[j + 1]:.1f})"
        print(header)
        for i in range(len(p_bins_cross) - 1):
            row = f"  p=[{p_bins_cross[i]:.02f},{p_bins_cross[i + 1]:.02f})"
            row = f"{row:>18s}"
            for j in range(len(h_bins_cross) - 1):
                mask = (
                    (p_eff >= p_bins_cross[i])
                    & (p_eff < p_bins_cross[i + 1])
                    & (E_H >= h_bins_cross[j])
                    & (E_H < h_bins_cross[j + 1])
                )
                if mask.sum() < 20:
                    row += f"{'---':>20s}"
                else:
                    b, _, n = fit_slope(arr[mask], log_n[mask], p_eff[mask])
                    row += f"  {b:+.3f} (n={n:>5d})"
            print(row)
        print()

    # ==================================================================
    # SECTION 6: RETRY COUNTER SCALING
    # ==================================================================
    print("=" * 70)
    print("SECTION 6: RETRY COUNTER — N_attempts(n, p_eff)")
    print("=" * 70)
    print()

    if attempts.any():
        # Mean attempts by n_orb
        print("  Mean attempts per bundle by n_orb:")
        for n_val in [3, 6, 12, 24]:
            mask = n_orbs == n_val
            if mask.sum() > 0:
                print(
                    f"    n={n_val:>2d}: {attempts[mask].mean():.2f} attempts, "
                    f"{failed_contacts[mask].mean():.2f} failures"
                )
        print()

        # Mean attempts by p_eff bin
        print("  Mean attempts per bundle by p_eff bin:")
        peff_bins_att = [0, 0.01, 0.05, 0.1, 0.3, 1.01]
        for i in range(len(peff_bins_att) - 1):
            lo, hi = peff_bins_att[i], peff_bins_att[i + 1]
            mask = (p_eff >= lo) & (p_eff < hi)
            if mask.sum() > 0:
                print(
                    f"    p=[{lo:.02f},{hi:.02f}): {attempts[mask].mean():.2f} attempts, "
                    f"{failed_contacts[mask].mean():.2f} failures"
                )
        print()

        # Cross-table: attempts by (n_orb × p_eff)
        print("  Attempts cross-table (n_orb × p_eff):")
        header = f"  {'':>10s}"
        for n_val in [3, 6, 12, 24]:
            header += f"  n={n_val:>2d}"
        print(header)
        print("  " + "-" * 50)
        for i in range(len(peff_bins_att) - 1):
            lo, hi = peff_bins_att[i], peff_bins_att[i + 1]
            row = f"  p=[{lo:.02f},{hi:.02f})"
            row = f"{row:>10s}"
            for n_val in [3, 6, 12, 24]:
                mask = (n_orbs == n_val) & (p_eff >= lo) & (p_eff < hi)
                if mask.sum() > 0:
                    row += f"  {attempts[mask].mean():>5.2f}"
                else:
                    row += f"  {'---':>5s}"
            print(row)
        print()

        # Scaling: does N_attempts scale as n, log(n), or saturate?
        print("  Scaling test: N_attempts vs n_orb")
        for pbin_lo, pbin_hi in [(0, 0.05), (0.05, 0.2), (0.2, 1.01)]:
            pmask = (p_eff >= pbin_lo) & (p_eff < pbin_hi)
            att_by_n = {}
            for n_val in [3, 6, 12, 24]:
                mask = pmask & (n_orbs == n_val)
                if mask.sum() > 0:
                    att_by_n[n_val] = attempts[mask].mean()
            if len(att_by_n) >= 3:
                ns = np.array(list(att_by_n.keys()), dtype=float)
                atts = np.array(list(att_by_n.values()))
                # Fit: attempts = a + b*n, a + b*log(n), a + b*sqrt(n)
                for label, x_fn in [("linear", ns), ("log", np.log(ns)), ("sqrt", np.sqrt(ns))]:
                    X = np.column_stack([np.ones(len(ns)), x_fn])
                    beta = np.linalg.lstsq(X, atts, rcond=None)[0]
                    pred = X @ beta
                    ss_res = np.sum((atts - pred) ** 2)
                    ss_tot = np.sum((atts - atts.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    # skip if not enough variation
                print(
                    f"    p=[{pbin_lo},{pbin_hi}): "
                    f"n=3→{att_by_n.get(3, 0):.2f}, n=6→{att_by_n.get(6, 0):.2f}, "
                    f"n=12→{att_by_n.get(12, 0):.2f}, n=24→{att_by_n.get(24, 0):.2f}"
                )

        # Correlation: attempts as predictor of ln(Φ)
        print()
        valid = attempts > 0
        if valid.sum() > 10:
            rho_att_phi = np.corrcoef(attempts[valid], ln_phi_n[valid])[0, 1]
            rho_att_logn = np.corrcoef(attempts[valid], log_n[valid])[0, 1]
            rho_att_peff = np.corrcoef(attempts[valid], p_eff[valid])[0, 1]
            print(f"  Attempt correlations (N={valid.sum()}):")
            print(f"    rho(attempts, ln Φ)   = {rho_att_phi:+.4f}")
            print(f"    rho(attempts, log n)  = {rho_att_logn:+.4f}")
            print(f"    rho(attempts, p_eff)  = {rho_att_peff:+.4f}")
    else:
        print("  No attempt data available.")

    print()

    # ==================================================================
    # SECTION 7: SUMMARY TABLE
    # ==================================================================
    print("=" * 70)
    print("SECTION 7: PER-TARGET SUMMARY")
    print("=" * 70)
    print()

    print(
        f"  {'target':>10s} {'N':>6s} {'mean_Φ':>7s} {'mean_Φm':>8s} {'mean_Φr':>8s} "
        f"{'b_net':>7s} {'b_myop':>7s} {'b_retr':>7s} {'fact_err':>9s}"
    )
    print("  " + "-" * 80)
    for t in target_order:
        mask = targets == t
        if mask.sum() == 0:
            continue
        b_n, _, _ = fit_slope(ln_phi_n[mask], log_n[mask], p_eff[mask])
        b_m, _, _ = fit_slope(ln_phi_m[mask], log_n[mask], p_eff[mask])
        b_r, _, _ = fit_slope(ln_phi_r[mask], log_n[mask], p_eff[mask])
        fe = fact_error[mask]
        print(
            f"  {t:>10s} {mask.sum():>6d} {phi_n[mask].mean():>7.3f} {phi_m[mask].mean():>8.3f} "
            f"{phi_r[mask].mean():>8.3f} {b_n:>+7.3f} {b_m:>+7.3f} {b_r:>+7.3f} {fe.mean():>9.4f}"
        )

    print()
    print("  DONE.")
    print()

    # Save
    results = {
        "factorization": {
            "mean_error": float(fact_error.mean()),
            "median_error": float(np.median(fact_error)),
            "p95_error": float(np.percentile(fact_error, 95)),
            "max_error": float(fact_error.max()),
            "frac_under_5pct": float((fact_error < 0.05).mean()),
        },
        "myopic_formula": {
            "old_params": [0.287, 0.019],
            "old_r2": float(old_r2),
            "new_params": [float(x) for x in best_myopic_params]
            if best_myopic_params is not None
            else None,
            "new_r2": float(best_myopic_r2),
        },
        "retry_model": {
            "name": best_retry_name,
            "params": [float(x) for x in best_retry_params]
            if best_retry_params is not None
            else None,
            "r2": float(best_retry_r2),
        },
        "cells": cells,
    }

    out_path = os.path.join(SCRIPT_DIR, "decompose_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
