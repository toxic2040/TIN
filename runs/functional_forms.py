#!/usr/bin/env python3
"""Derive functional forms for b_myopic and b_retry from first principles.

Uses the 129,466-point dataset to extract the log(n) slope (β_n) at fine
(p_eff, E[H]) resolution, then fits candidate decompositions:

    β_n = β_retry(p, h) − β_myopic(p, h)

where β_n is the coefficient of log(n) in: ln(Φ) = a + β_n·log(n) + c·p_eff.

Candidate models tested:
  M1: β_n = α − δ·h·p^γ                    (power exposure, constant retry)
  M2: β_n = r·(1−p^h) − δ·h·p/(p+p₀)      (chain-survival retry, Michaelis-Menten myopic)
  M3: β_n = a + b·h + c·h·ln(p)            (log-linear in chain survival)
  M4: β_n = r·h/(h+h₀) − δ·h·p/(p+p₀)    (saturating retry, MM myopic)
  M5: β_n = r·(1−p^h) − δ·h·(1−(1−p)^κ)   (chain-survival retry, saturable exposure)
"""

import json
import os

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_combined():
    """Load inner + outer surface sweeps, keep active raw results."""
    records = []
    for fname in ["phi_surface_results.json", "phi_surface_outer_results.json"]:
        path = os.path.join(SCRIPT_DIR, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data["results"]:
            if r["eta_sim"] > 0 and r["eta_lyap"] > 0 and r["E_H"] > 0:
                records.append(r)
    return records


def extract_slope_grid(records, p_bins, h_bins):
    """Extract β_n (log(n) slope) in each (p_eff, E[H]) bin.

    Returns list of dicts with: p_mid, h_mid, beta_n, r2, n, p_mean, h_mean.
    """
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
                    "p_lo": p_bins[i],
                    "p_hi": p_bins[i + 1],
                    "h_lo": h_bins[j],
                    "h_hi": h_bins[j + 1],
                    "p_mean": float(p_eff[mask].mean()),
                    "h_mean": float(E_H[mask].mean()),
                    "beta_n": float(beta[1]),
                    "r2": float(r2),
                    "n": int(n),
                }
            )
    return cells


def fit_model(cells, model_fn, p0, bounds=None):
    """Fit a model to the extracted slope cells.

    model_fn(params, p, h) → predicted β_n
    Returns: params, r2, residuals
    """
    p_arr = np.array([c["p_mean"] for c in cells])
    h_arr = np.array([c["h_mean"] for c in cells])
    beta_arr = np.array([c["beta_n"] for c in cells])
    weights = np.array([c["n"] for c in cells], dtype=float)
    weights /= weights.sum()

    def cost(params):
        pred = model_fn(params, p_arr, h_arr)
        return np.sum(weights * (beta_arr - pred) ** 2)

    result = minimize(cost, p0, bounds=bounds, method="L-BFGS-B" if bounds else "Nelder-Mead")
    params = result.x
    pred = model_fn(params, p_arr, h_arr)
    ss_res = np.sum((beta_arr - pred) ** 2)
    ss_tot = np.sum((beta_arr - np.mean(beta_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot
    residuals = beta_arr - pred
    return params, r2, residuals, pred


# ===================================================================
# Model definitions
# ===================================================================


def model_M1(params, p, h):
    """M1: β_n = α − δ·h·p^γ  (power exposure, constant retry)"""
    alpha, delta, gamma = params
    return alpha - delta * h * np.power(np.clip(p, 1e-12, 1), gamma)


def model_M2(params, p, h):
    """M2: β_n = r·(1−p^h) − δ·h·p/(p+p₀)  (chain-survival, Michaelis-Menten)"""
    r, delta, p0 = params
    retry = r * (1.0 - np.power(np.clip(p, 1e-12, 1), h))
    myopic = delta * h * p / (p + p0)
    return retry - myopic


def model_M3(params, p, h):
    """M3: β_n = a + b·h + c·h·ln(p)  (log-linear in chain survival)"""
    a, b, c = params
    return a + b * h + c * h * np.log(np.clip(p, 1e-12, 1))


def model_M4(params, p, h):
    """M4: β_n = r·h/(h+h₀) − δ·h·p/(p+p₀)  (saturating retry, MM myopic)"""
    r, h0, delta, p0 = params
    retry = r * h / (h + h0)
    myopic = delta * h * p / (p + p0)
    return retry - myopic


def model_M5(params, p, h):
    """M5: β_n = r·(1−p^h) − δ·h·(1−(1−p)^κ)  (chain-survival retry, saturable exposure)"""
    r, delta, kappa = params
    retry = r * (1.0 - np.power(np.clip(p, 1e-12, 1), h))
    exposure = 1.0 - np.power(np.clip(1.0 - p, 1e-12, 1), kappa)
    myopic = delta * h * exposure
    return retry - myopic


def model_M6(params, p, h):
    """M6: β_n = r·(1−p^h) − δ·h·p^γ  (chain-survival retry, power exposure)

    The physics: retry activates when oracle chain fails (1-p^h).
    Myopic penalty scales with hops × exposure (p^γ where γ < 1 for sublinear).
    """
    r, delta, gamma = params
    retry = r * (1.0 - np.power(np.clip(p, 1e-12, 1), h))
    myopic = delta * h * np.power(np.clip(p, 1e-12, 1), gamma)
    return retry - myopic


def model_M7(params, p, h):
    """M7: β_n = r·(1−p^h) − δ₁·h·p^γ − δ₂·h  (M6 + baseline myopic)

    Adds a p-independent myopic baseline: even at p_eff→0, wrong next-hop
    choice incurs a structural penalty (choosing nodes with fewer onward contacts).
    """
    r, delta1, gamma, delta2 = params
    retry = r * (1.0 - np.power(np.clip(p, 1e-12, 1), h))
    myopic_exposure = delta1 * h * np.power(np.clip(p, 1e-12, 1), gamma)
    myopic_structural = delta2 * h
    return retry - myopic_exposure - myopic_structural


def main():
    print("Loading data...")
    records = load_combined()
    print(f"  {len(records):,} active raw results")
    print()

    # ===================================================================
    # Extract slope grid at two resolutions
    # ===================================================================

    # Coarse grid (matches the cross-table from Experiment 2)
    p_bins_coarse = [0, 0.01, 0.05, 0.20, 1.01]
    h_bins_coarse = [0, 2.5, 3.5, 5.0, 20]
    cells_coarse = extract_slope_grid(records, p_bins_coarse, h_bins_coarse)

    # Fine grid
    p_bins_fine = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    h_bins_fine = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 20]
    cells_fine = extract_slope_grid(records, p_bins_fine, h_bins_fine)

    print(f"Coarse grid: {len(cells_coarse)} cells")
    print(f"Fine grid: {len(cells_fine)} cells")
    print()

    # ===================================================================
    # Print fine grid
    # ===================================================================
    print("=" * 80)
    print("FINE GRID: β_n by (p_eff, E[H])")
    print("=" * 80)
    print()
    print(f"{'p_eff':>12s} {'E[H]':>8s} {'β_n':>8s} {'R²':>8s} {'N':>8s}")
    print("-" * 48)
    for c in sorted(cells_fine, key=lambda x: (x["h_mean"], x["p_mean"])):
        print(
            f"  {c['p_mean']:>10.4f} {c['h_mean']:>8.2f} {c['beta_n']:>+8.4f} {c['r2']:>8.4f} {c['n']:>8d}"
        )
    print()

    # ===================================================================
    # Fit models to fine grid
    # ===================================================================
    print("=" * 80)
    print("MODEL FITS (fine grid)")
    print("=" * 80)
    print()

    models = [
        ("M1: α − δ·h·p^γ", model_M1, [0.4, 0.3, 0.15], [(-2, 2), (0, 5), (0.01, 1.0)]),
        (
            "M2: r·(1−p^h) − δ·h·p/(p+p₀)",
            model_M2,
            [0.4, 0.45, 0.01],
            [(0, 2), (0, 5), (1e-6, 1.0)],
        ),
        ("M3: a + b·h + c·h·ln(p)", model_M3, [0.5, 0.1, 0.05], None),
        (
            "M4: r·h/(h+h₀) − δ·h·p/(p+p₀)",
            model_M4,
            [1.0, 3.0, 0.45, 0.01],
            [(0, 5), (0.1, 20), (0, 5), (1e-6, 1.0)],
        ),
        ("M5: r·(1−p^h) − δ·h·(1−(1−p)^κ)", model_M5, [0.4, 0.3, 2.0], [(0, 2), (0, 5), (0.1, 20)]),
        ("M6: r·(1−p^h) − δ·h·p^γ", model_M6, [0.5, 0.15, 0.3], [(0, 2), (0, 5), (0.01, 1.0)]),
        (
            "M7: r·(1−p^h) − δ₁·h·p^γ − δ₂·h",
            model_M7,
            [0.6, 0.15, 0.3, 0.05],
            [(0, 3), (0, 5), (0.01, 1.0), (0, 1)],
        ),
    ]

    best_r2 = -1
    best_name = ""
    best_params = None
    all_results = {}

    for name, model_fn, p0, bounds in models:
        try:
            params, r2, residuals, pred = fit_model(cells_fine, model_fn, p0, bounds)
            mae = np.mean(np.abs(residuals))
            max_err = np.max(np.abs(residuals))
            print(f"  {name}")
            print(f"    R² = {r2:.4f},  MAE = {mae:.4f},  max|err| = {max_err:.4f}")
            print(f"    params = {', '.join(f'{p:.4f}' for p in params)}")
            print()

            all_results[name] = {
                "r2": float(r2),
                "mae": float(mae),
                "max_err": float(max_err),
                "params": [float(p) for p in params],
            }

            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_params = params
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            print()

    print(f"BEST MODEL: {best_name} (R² = {best_r2:.4f})")
    print(f"  params = {', '.join(f'{p:.4f}' for p in best_params)}")
    print()

    # ===================================================================
    # Deep dive on best model + M6 (the physics model)
    # ===================================================================
    print("=" * 80)
    print("DEEP DIVE: Model M6 — the physics decomposition")
    print("=" * 80)
    print()

    # Refit M6 with multiple starting points to ensure global optimum
    best_m6_r2 = -1
    best_m6_params = None
    for r_init in [0.3, 0.5, 0.8, 1.0]:
        for d_init in [0.1, 0.2, 0.4]:
            for g_init in [0.1, 0.3, 0.5, 0.8]:
                try:
                    params, r2, _, _ = fit_model(
                        cells_fine,
                        model_M6,
                        [r_init, d_init, g_init],
                        [(0, 3), (0, 5), (0.01, 1.0)],
                    )
                    if r2 > best_m6_r2:
                        best_m6_r2 = r2
                        best_m6_params = params
                except Exception:
                    pass

    if best_m6_params is not None:
        r, delta, gamma = best_m6_params
        print(f"  M6: β_n = {r:.3f}·(1−p^h) − {delta:.3f}·h·p^{gamma:.3f}")
        print(f"  R² = {best_m6_r2:.4f}")
        print()

        # Decompose: show β_retry and β_myopic separately
        print("  Physical interpretation:")
        print(f"    β_retry  = {r:.3f} × (1 − p_eff^E[H])   [oracle chain failure probability]")
        print(f"    β_myopic = {delta:.3f} × E[H] × p_eff^{gamma:.3f}  [exposure × chain length]")
        print("    β_net    = β_retry − β_myopic")
        print()

        # Print decomposition for each cell
        p_arr = np.array([c["p_mean"] for c in cells_fine])
        h_arr = np.array([c["h_mean"] for c in cells_fine])
        beta_arr = np.array([c["beta_n"] for c in cells_fine])

        retry_vals = r * (1.0 - np.power(np.clip(p_arr, 1e-12, 1), h_arr))
        myopic_vals = delta * h_arr * np.power(np.clip(p_arr, 1e-12, 1), gamma)
        pred_vals = retry_vals - myopic_vals

        print(
            f"  {'p_eff':>8s} {'E[H]':>6s} {'β_data':>8s} {'β_pred':>8s} {'β_retry':>8s} {'β_myopic':>9s} {'err':>8s}"
        )
        print("  " + "-" * 62)
        for i in range(len(cells_fine)):
            c = sorted(cells_fine, key=lambda x: (x["h_mean"], x["p_mean"]))[i]
            idx = cells_fine.index(c)
            print(
                f"  {p_arr[idx]:>8.4f} {h_arr[idx]:>6.2f} {beta_arr[idx]:>+8.4f} {pred_vals[idx]:>+8.4f} "
                f"{retry_vals[idx]:>+8.4f} {myopic_vals[idx]:>+9.4f} {beta_arr[idx] - pred_vals[idx]:>+8.4f}"
            )
        print()

        # Key predictions
        print("  Key predictions from M6:")
        test_cases = [
            ("Inner typical (p=0.3, h=3)", 0.3, 3.0),
            ("Mercury (p=0.35, h=2.9)", 0.35, 2.9),
            ("Saturn (p=0.005, h=5.5)", 0.005, 5.5),
            ("Jupiter (p=0.04, h=3.6)", 0.04, 3.6),
            ("Titan (p=0.03, h=2.9)", 0.03, 2.9),
            ("Braess zero (solve for p at h=3)", None, 3.0),
            ("Braess zero (solve for p at h=5)", None, 5.0),
        ]
        for label, p_test, h_test in test_cases:
            if p_test is not None:
                ret = r * (1 - p_test**h_test)
                myp = delta * h_test * p_test**gamma
                bn = ret - myp
                print(f"    {label}: β_retry={ret:.3f}, β_myopic={myp:.3f}, β_net={bn:+.3f}")
            else:
                # Solve β_n = 0 for p
                from scipy.optimize import brentq

                try:
                    p_zero = brentq(
                        lambda p: r * (1 - p**h_test) - delta * h_test * p**gamma, 1e-6, 0.999
                    )
                    print(f"    β_net=0 at h={h_test:.0f}: p_eff* = {p_zero:.4f}")
                except ValueError:
                    print(f"    β_net=0 at h={h_test:.0f}: no crossing in [0,1]")
        print()

    # ===================================================================
    # Also fit M7 with multi-start
    # ===================================================================
    print("=" * 80)
    print("DEEP DIVE: Model M7 — with structural baseline")
    print("=" * 80)
    print()

    best_m7_r2 = -1
    best_m7_params = None
    for r_init in [0.3, 0.6, 1.0, 1.5]:
        for d1_init in [0.1, 0.3]:
            for g_init in [0.1, 0.3, 0.5]:
                for d2_init in [0.01, 0.05, 0.15]:
                    try:
                        params, r2, _, _ = fit_model(
                            cells_fine,
                            model_M7,
                            [r_init, d1_init, g_init, d2_init],
                            [(0, 3), (0, 5), (0.01, 1.0), (0, 1)],
                        )
                        if r2 > best_m7_r2:
                            best_m7_r2 = r2
                            best_m7_params = params
                    except Exception:
                        pass

    if best_m7_params is not None:
        r, d1, gamma, d2 = best_m7_params
        print(f"  M7: β_n = {r:.3f}·(1−p^h) − {d1:.3f}·h·p^{gamma:.3f} − {d2:.3f}·h")
        print(f"  R² = {best_m7_r2:.4f}")
        print()
        print("  Physical interpretation:")
        print(f"    β_retry     = {r:.3f} × (1 − p_eff^E[H])  [chain failure activates retry]")
        print(
            f"    β_myopic_exp = {d1:.3f} × E[H] × p_eff^{gamma:.3f}  [exposure-dependent penalty]"
        )
        print(
            f"    β_myopic_str = {d2:.3f} × E[H]  [structural penalty: wrong-node independent of p]"
        )
        print("    β_net = β_retry − β_myopic_exp − β_myopic_str")
        print()

        # Decomposition table
        p_arr = np.array([c["p_mean"] for c in cells_fine])
        h_arr = np.array([c["h_mean"] for c in cells_fine])
        beta_arr = np.array([c["beta_n"] for c in cells_fine])

        retry_vals = r * (1.0 - np.power(np.clip(p_arr, 1e-12, 1), h_arr))
        myopic_exp = d1 * h_arr * np.power(np.clip(p_arr, 1e-12, 1), gamma)
        myopic_str = d2 * h_arr
        pred_vals = retry_vals - myopic_exp - myopic_str

        print(
            f"  {'p_eff':>8s} {'E[H]':>6s} {'β_data':>8s} {'β_pred':>8s} {'retry':>8s} {'myop_exp':>9s} {'myop_str':>9s} {'err':>8s}"
        )
        print("  " + "-" * 74)
        sorted_cells = sorted(
            range(len(cells_fine)), key=lambda i: (cells_fine[i]["h_mean"], cells_fine[i]["p_mean"])
        )
        for idx in sorted_cells:
            print(
                f"  {p_arr[idx]:>8.4f} {h_arr[idx]:>6.2f} {beta_arr[idx]:>+8.4f} {pred_vals[idx]:>+8.4f} "
                f"{retry_vals[idx]:>+8.4f} {myopic_exp[idx]:>+9.4f} {myopic_str[idx]:>+9.4f} "
                f"{beta_arr[idx] - pred_vals[idx]:>+8.4f}"
            )
        print()

        # Braess boundary
        print("  Braess boundary (β_net = 0):")
        from scipy.optimize import brentq

        for h_test in [2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
            try:
                p_zero = brentq(
                    lambda p: r * (1 - p**h_test) - d1 * h_test * p**gamma - d2 * h_test,
                    1e-8,
                    0.999,
                )
                print(
                    f"    h={h_test:.1f}: p_eff* = {p_zero:.4f}  "
                    f"(chain_surv = {p_zero**h_test:.2e})"
                )
            except ValueError:
                # Check if β_net is always positive or always negative
                bn_lo = r * (1 - 1e-8**h_test) - d1 * h_test * 1e-8**gamma - d2 * h_test
                bn_hi = r * (1 - 0.999**h_test) - d1 * h_test * 0.999**gamma - d2 * h_test
                if bn_lo > 0 and bn_hi > 0:
                    print(f"    h={h_test:.1f}: β_net > 0 everywhere (retry always wins)")
                elif bn_lo < 0 and bn_hi < 0:
                    print(f"    h={h_test:.1f}: β_net < 0 everywhere (myopic always wins)")
                else:
                    print(f"    h={h_test:.1f}: no clean crossing found")

    # ===================================================================
    # Save results
    # ===================================================================
    out = {
        "cells_fine": cells_fine,
        "cells_coarse": cells_coarse,
        "model_fits": all_results,
    }
    if best_m6_params is not None:
        out["M6_best"] = {
            "r": float(best_m6_params[0]),
            "delta": float(best_m6_params[1]),
            "gamma": float(best_m6_params[2]),
            "r2": float(best_m6_r2),
        }
    if best_m7_params is not None:
        out["M7_best"] = {
            "r": float(best_m7_params[0]),
            "delta1": float(best_m7_params[1]),
            "gamma": float(best_m7_params[2]),
            "delta2": float(best_m7_params[3]),
            "r2": float(best_m7_r2),
        }

    out_path = os.path.join(SCRIPT_DIR, "functional_forms_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
