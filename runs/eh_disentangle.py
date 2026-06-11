#!/usr/bin/env python3
"""Experiment 2: E[H] Disentanglement Analysis.

Tests whether the p_eff-dependent slope in ln(Phi) is an E[H] effect in disguise.
Pure analysis of existing phi_surface data — no new simulation.

Key questions:
1. Does log(n) still predict Phi after controlling for both p_eff AND E[H]?
2. Does E[H] have independent predictive power beyond p_eff?
3. Is b_net better predicted by p_eff alone or by p_eff^E[H]?
4. Does b_net collapse as a function of oracle chain log-survival?
"""

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_active_raw(path):
    """Load raw results, keep only active (eta_sim > 0, eta_lyap > 0)."""
    with open(path) as f:
        data = json.load(f)
    raw = data["results"]
    active = [r for r in raw if r["eta_sim"] > 0 and r["eta_lyap"] > 0 and r["E_H"] > 0]
    return active


def partial_corr(x, y, controls):
    """Partial correlation of x and y controlling for columns in controls.

    Uses residual method: regress x and y on controls, correlate residuals.
    """
    X = np.column_stack(controls)
    # Add intercept
    X = np.column_stack([np.ones(len(x)), X])
    # Residuals of x on controls
    beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
    res_x = x - X @ beta_x
    # Residuals of y on controls
    beta_y = np.linalg.lstsq(X, y, rcond=None)[0]
    res_y = y - X @ beta_y
    # Correlation of residuals
    return np.corrcoef(res_x, res_y)[0, 1]


def extract_arrays(records):
    """Extract numpy arrays from list of dicts."""
    ln_phi = np.array([np.log(r["phi_time"]) for r in records])
    log_n = np.array([np.log(r["n_orb"]) for r in records])
    p_eff = np.array([r["p_eff"] for r in records])
    E_H = np.array([r["E_H"] for r in records])
    # Oracle chain log-survival: E[H] * ln(p_eff)
    # Guard against p_eff = 0
    ln_peff = np.log(np.clip(p_eff, 1e-12, 1.0))
    chain_log_surv = E_H * ln_peff
    target = np.array([r["target"] for r in records])
    n_orb = np.array([r["n_orb"] for r in records])
    return {
        "ln_phi": ln_phi,
        "log_n": log_n,
        "p_eff": p_eff,
        "E_H": E_H,
        "ln_peff": ln_peff,
        "chain_log_surv": chain_log_surv,
        "target": target,
        "n_orb": n_orb,
    }


def fit_slope_in_bin(ln_phi, log_n, p_eff):
    """Fit ln(Phi) = a + b*log(n) + c*p_eff, return b (the log(n) slope)."""
    if len(ln_phi) < 10:
        return np.nan, np.nan, 0
    X = np.column_stack([np.ones(len(ln_phi)), log_n, p_eff])
    beta, residuals, _, _ = np.linalg.lstsq(X, ln_phi, rcond=None)
    ss_res = np.sum((ln_phi - X @ beta) ** 2)
    ss_tot = np.sum((ln_phi - np.mean(ln_phi)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta[1], r2, len(ln_phi)


def main():
    # Load both sweeps
    inner_path = os.path.join(SCRIPT_DIR, "phi_surface_results.json")
    outer_path = os.path.join(SCRIPT_DIR, "phi_surface_outer_results.json")

    if not os.path.exists(inner_path):
        print(f"ERROR: {inner_path} not found")
        sys.exit(1)
    if not os.path.exists(outer_path):
        print(f"ERROR: {outer_path} not found")
        sys.exit(1)

    print("Loading inner surface sweep...")
    inner = load_active_raw(inner_path)
    print(f"  {len(inner):,} active raw results")

    print("Loading outer surface sweep...")
    outer = load_active_raw(outer_path)
    print(f"  {len(outer):,} active raw results")

    combined = inner + outer
    print(f"  {len(combined):,} total active results")
    print()

    d = extract_arrays(combined)

    # ===================================================================
    # SECTION 1: Partial Correlations
    # ===================================================================
    print("=" * 70)
    print("SECTION 1: PARTIAL CORRELATIONS")
    print("=" * 70)
    print()

    # Raw correlations
    rho_logn = np.corrcoef(d["ln_phi"], d["log_n"])[0, 1]
    rho_peff = np.corrcoef(d["ln_phi"], d["p_eff"])[0, 1]
    rho_EH = np.corrcoef(d["ln_phi"], d["E_H"])[0, 1]
    rho_chain = np.corrcoef(d["ln_phi"], d["chain_log_surv"])[0, 1]

    print("Raw correlations with ln(Phi):")
    print(f"  rho(ln Phi, log n)              = {rho_logn:+.4f}")
    print(f"  rho(ln Phi, p_eff)              = {rho_peff:+.4f}")
    print(f"  rho(ln Phi, E[H])               = {rho_EH:+.4f}")
    print(f"  rho(ln Phi, E[H]*ln(p_eff))     = {rho_chain:+.4f}")
    print()

    # Partial: log(n) controlling for p_eff only
    pc_logn_peff = partial_corr(d["ln_phi"], d["log_n"], [d["p_eff"]])
    # Partial: log(n) controlling for p_eff AND E[H]
    pc_logn_peff_EH = partial_corr(d["ln_phi"], d["log_n"], [d["p_eff"], d["E_H"]])
    # Partial: E[H] controlling for p_eff and log(n)
    pc_EH_peff_logn = partial_corr(d["ln_phi"], d["E_H"], [d["p_eff"], d["log_n"]])
    # Partial: chain_log_surv controlling for log(n)
    pc_chain_logn = partial_corr(d["ln_phi"], d["chain_log_surv"], [d["log_n"]])
    # Partial: p_eff controlling for log(n) and E[H]
    pc_peff_logn_EH = partial_corr(d["ln_phi"], d["p_eff"], [d["log_n"], d["E_H"]])

    print("Partial correlations with ln(Phi):")
    print(f"  rho(ln Phi, log n | p_eff)           = {pc_logn_peff:+.4f}")
    print(f"  rho(ln Phi, log n | p_eff, E[H])     = {pc_logn_peff_EH:+.4f}")
    print(f"  rho(ln Phi, E[H]  | p_eff, log n)    = {pc_EH_peff_logn:+.4f}")
    print(f"  rho(ln Phi, chain_surv | log n)       = {pc_chain_logn:+.4f}")
    print(f"  rho(ln Phi, p_eff | log n, E[H])      = {pc_peff_logn_EH:+.4f}")
    print()

    # ===================================================================
    # SECTION 2: Competing Regression Models
    # ===================================================================
    print("=" * 70)
    print("SECTION 2: COMPETING MODELS (which parameterization wins?)")
    print("=" * 70)
    print()

    ln_phi = d["ln_phi"]
    log_n = d["log_n"]
    p_eff = d["p_eff"]
    E_H = d["E_H"]
    chain_surv = d["chain_log_surv"]

    def fit_and_report(name, X, y):
        X_full = np.column_stack([np.ones(len(y)), X])
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
        y_hat = X_full @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        n = len(y)
        k = X_full.shape[1] - 1
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        print(f"  {name}")
        print(f"    R² = {r2:.4f},  R²_adj = {r2_adj:.4f},  N = {n:,}")
        coef_names = ["intercept"] + [f"beta_{i}" for i in range(k)]
        for cname, b in zip(coef_names, beta):
            print(f"    {cname:>12s} = {b:+.4f}")
        print()
        return r2, r2_adj, beta

    # Model A: inner model (log_n + p_eff)
    r2_A, _, _ = fit_and_report(
        "Model A: ln(Phi) = a + b*log(n) + c*p_eff", np.column_stack([log_n, p_eff]), ln_phi
    )

    # Model B: add E[H] as separate predictor
    r2_B, _, _ = fit_and_report("Model B: + E[H]", np.column_stack([log_n, p_eff, E_H]), ln_phi)

    # Model C: replace p_eff with chain_log_surv
    r2_C, _, _ = fit_and_report(
        "Model C: ln(Phi) = a + b*log(n) + c*E[H]*ln(p_eff)",
        np.column_stack([log_n, chain_surv]),
        ln_phi,
    )

    # Model D: all three (log_n, p_eff, E[H])  — same as B but let's also try interactions
    # Model E: log_n, p_eff, E[H], and interaction log_n * p_eff
    r2_E, _, _ = fit_and_report(
        "Model E: + log(n)*p_eff interaction",
        np.column_stack([log_n, p_eff, E_H, log_n * p_eff]),
        ln_phi,
    )

    # Model F: log_n, chain_surv, and interaction log_n * chain_surv
    r2_F, _, _ = fit_and_report(
        "Model F: log(n) + chain_surv + log(n)*chain_surv",
        np.column_stack([log_n, chain_surv, log_n * chain_surv]),
        ln_phi,
    )

    # Model G: the full competition model — log_n, p_eff, E_H, log_n*p_eff, log_n*E_H
    r2_G, _, _ = fit_and_report(
        "Model G: full interaction (log_n, p_eff, E_H, log_n*p_eff, log_n*E_H)",
        np.column_stack([log_n, p_eff, E_H, log_n * p_eff, log_n * E_H]),
        ln_phi,
    )

    print("  Model comparison summary:")
    print(f"    A (inner model):          R² = {r2_A:.4f}")
    print(f"    B (+ E[H]):               R² = {r2_B:.4f}  delta = {r2_B - r2_A:+.4f}")
    print(f"    C (chain_surv):           R² = {r2_C:.4f}  delta = {r2_C - r2_A:+.4f}")
    print(f"    E (+ interaction):        R² = {r2_E:.4f}  delta = {r2_E - r2_A:+.4f}")
    print(f"    F (chain + interaction):  R² = {r2_F:.4f}  delta = {r2_F - r2_A:+.4f}")
    print(f"    G (full):                 R² = {r2_G:.4f}  delta = {r2_G - r2_A:+.4f}")
    print()

    # ===================================================================
    # SECTION 3: b_net by p_eff bin
    # ===================================================================
    print("=" * 70)
    print("SECTION 3: b_net(log n slope) BY p_eff BIN")
    print("=" * 70)
    print()

    peff_bins = [0, 0.005, 0.02, 0.1, 0.5, 1.01]
    print(
        f"{'p_eff range':>20s} {'b_net':>8s} {'R²':>8s} {'N':>8s} {'mean E[H]':>10s} {'mean chain_surv':>16s}"
    )
    print("-" * 76)
    for i in range(len(peff_bins) - 1):
        lo, hi = peff_bins[i], peff_bins[i + 1]
        mask = (p_eff >= lo) & (p_eff < hi)
        if mask.sum() < 10:
            continue
        b, r2, n = fit_slope_in_bin(ln_phi[mask], log_n[mask], p_eff[mask])
        mean_eh = E_H[mask].mean()
        mean_cs = chain_surv[mask].mean()
        print(
            f"  [{lo:.3f}, {hi:.3f}){' ':>6s} {b:+.4f}   {r2:.4f}   {n:>6d}   {mean_eh:>8.2f}   {mean_cs:>14.2f}"
        )
    print()

    # ===================================================================
    # SECTION 4: b_net by chain_log_surv bin
    # ===================================================================
    print("=" * 70)
    print("SECTION 4: b_net BY CHAIN LOG-SURVIVAL BIN (E[H]*ln(p_eff))")
    print("=" * 70)
    print()

    # Chain log survival is always negative (or zero). More negative = more fragile.
    cs_percentiles = np.percentile(chain_surv, [0, 10, 25, 50, 75, 90, 100])
    cs_bins = sorted(set(np.round(cs_percentiles, 1)))
    if len(cs_bins) < 3:
        cs_bins = np.linspace(chain_surv.min(), chain_surv.max(), 7)

    print(
        f"{'chain_surv range':>24s} {'b_net':>8s} {'R²':>8s} {'N':>8s} {'mean p_eff':>11s} {'mean E[H]':>10s}"
    )
    print("-" * 76)
    for i in range(len(cs_bins) - 1):
        lo, hi = cs_bins[i], cs_bins[i + 1]
        mask = (chain_surv >= lo) & (chain_surv < hi)
        if mask.sum() < 10:
            continue
        b, r2, n = fit_slope_in_bin(ln_phi[mask], log_n[mask], p_eff[mask])
        mean_pe = p_eff[mask].mean()
        mean_eh = E_H[mask].mean()
        print(
            f"  [{lo:+.1f}, {hi:+.1f}){' ':>6s} {b:+.4f}   {r2:.4f}   {n:>6d}   {mean_pe:>9.4f}   {mean_eh:>8.2f}"
        )
    print()

    # ===================================================================
    # SECTION 5: b_net by E[H] bin (controlling for p_eff)
    # ===================================================================
    print("=" * 70)
    print("SECTION 5: b_net BY E[H] BIN")
    print("=" * 70)
    print()

    eh_bins = [0, 2, 3, 4, 5, 7, 20]
    print(f"{'E[H] range':>16s} {'b_net':>8s} {'R²':>8s} {'N':>8s} {'mean p_eff':>11s}")
    print("-" * 52)
    for i in range(len(eh_bins) - 1):
        lo, hi = eh_bins[i], eh_bins[i + 1]
        mask = (E_H >= lo) & (E_H < hi)
        if mask.sum() < 10:
            continue
        b, r2, n = fit_slope_in_bin(ln_phi[mask], log_n[mask], p_eff[mask])
        mean_pe = p_eff[mask].mean()
        print(f"  [{lo:.0f}, {hi:.0f}){' ':>6s} {b:+.4f}   {r2:.4f}   {n:>6d}   {mean_pe:>9.4f}")
    print()

    # ===================================================================
    # SECTION 6: Per-target b_net and E[H] profile
    # ===================================================================
    print("=" * 70)
    print("SECTION 6: PER-TARGET PROFILE")
    print("=" * 70)
    print()

    targets_ordered = ["mercury", "venus", "mars", "ceres", "europa", "jupiter", "saturn", "titan"]
    print(
        f"{'target':>10s} {'b_net':>8s} {'R²':>8s} {'N':>8s} {'mean p_eff':>11s} {'mean E[H]':>10s} {'mean chain':>11s}"
    )
    print("-" * 76)
    for t in targets_ordered:
        mask = d["target"] == t
        if mask.sum() < 10:
            continue
        b, r2, n = fit_slope_in_bin(ln_phi[mask], log_n[mask], p_eff[mask])
        mean_pe = p_eff[mask].mean()
        mean_eh = E_H[mask].mean()
        mean_cs = chain_surv[mask].mean()
        print(
            f"  {t:>10s} {b:+.4f}   {r2:.4f}   {n:>6d}   {mean_pe:>9.4f}   {mean_eh:>8.2f}   {mean_cs:>9.2f}"
        )
    print()

    # ===================================================================
    # SECTION 7: Cross-tabulation — b_net by (p_eff bin, E[H] bin)
    # ===================================================================
    print("=" * 70)
    print("SECTION 7: b_net CROSS-TABLE (p_eff bin x E[H] bin)")
    print("=" * 70)
    print()

    peff_bins2 = [0, 0.01, 0.05, 0.2, 1.01]
    eh_bins2 = [0, 2.5, 3.5, 5.0, 20]

    header = f"{'':>18s}"
    for j in range(len(eh_bins2) - 1):
        header += f" E[H]=[{eh_bins2[j]:.1f},{eh_bins2[j + 1]:.1f})"
    print(header)
    print("-" * len(header))

    for i in range(len(peff_bins2) - 1):
        row = f"  p=[{peff_bins2[i]:.2f},{peff_bins2[i + 1]:.2f})"
        row = f"{row:>18s}"
        for j in range(len(eh_bins2) - 1):
            mask = (
                (p_eff >= peff_bins2[i])
                & (p_eff < peff_bins2[i + 1])
                & (E_H >= eh_bins2[j])
                & (E_H < eh_bins2[j + 1])
            )
            if mask.sum() < 20:
                row += f"{'---':>18s}"
            else:
                b, _, n = fit_slope_in_bin(ln_phi[mask], log_n[mask], p_eff[mask])
                row += f"  {b:+.3f} (n={n:>5d})"
        print(row)
    print()

    # ===================================================================
    # SECTION 8: The decisive test — does p_eff^E[H] collapse b_net?
    # ===================================================================
    print("=" * 70)
    print("SECTION 8: DECISIVE TEST — COLLAPSE VARIABLE")
    print("=" * 70)
    print()

    # For each (target, n_orb) pair, compute the local b_net
    # Then correlate b_net with mean p_eff vs mean p_eff^E[H]
    print("Computing per-(target, p_eff_bin) slopes to test collapse...")
    print()

    target_peff_slopes = []
    for t in targets_ordered:
        tmask = d["target"] == t
        for i in range(len(peff_bins) - 1):
            lo, hi = peff_bins[i], peff_bins[i + 1]
            mask = tmask & (p_eff >= lo) & (p_eff < hi)
            if mask.sum() < 30:
                continue
            b, r2, n = fit_slope_in_bin(ln_phi[mask], log_n[mask], p_eff[mask])
            if np.isnan(b):
                continue
            mean_pe = p_eff[mask].mean()
            mean_eh = E_H[mask].mean()
            mean_cs = chain_surv[mask].mean()
            # p_eff^E[H] — geometric mean
            peff_eh = np.exp(chain_surv[mask].mean())
            target_peff_slopes.append(
                {
                    "target": t,
                    "p_eff_bin": f"[{lo:.3f},{hi:.3f})",
                    "b_net": b,
                    "mean_p_eff": mean_pe,
                    "mean_E_H": mean_eh,
                    "mean_chain_surv": mean_cs,
                    "p_eff_to_EH": peff_eh,
                    "n": n,
                }
            )

    if target_peff_slopes:
        b_vals = np.array([s["b_net"] for s in target_peff_slopes])
        pe_vals = np.array([s["mean_p_eff"] for s in target_peff_slopes])
        eh_vals = np.array([s["mean_E_H"] for s in target_peff_slopes])
        cs_vals = np.array([s["mean_chain_surv"] for s in target_peff_slopes])
        peh_vals = np.array([s["p_eff_to_EH"] for s in target_peff_slopes])

        rho_b_pe = np.corrcoef(b_vals, pe_vals)[0, 1]
        rho_b_eh = np.corrcoef(b_vals, eh_vals)[0, 1]
        rho_b_cs = np.corrcoef(b_vals, cs_vals)[0, 1]
        # Use log(p_eff) as alternative
        rho_b_lnpe = np.corrcoef(b_vals, np.log(np.clip(pe_vals, 1e-12, 1)))[0, 1]

        print(
            f"  Correlations with b_net across {len(target_peff_slopes)} (target, p_eff_bin) cells:"
        )
        print(f"    rho(b_net, mean p_eff)          = {rho_b_pe:+.4f}")
        print(f"    rho(b_net, ln(p_eff))           = {rho_b_lnpe:+.4f}")
        print(f"    rho(b_net, mean E[H])           = {rho_b_eh:+.4f}")
        print(f"    rho(b_net, E[H]*ln(p_eff))      = {rho_b_cs:+.4f}  <-- chain survival")
        print()
        print(f"  VERDICT: {'chain_surv WINS' if abs(rho_b_cs) > abs(rho_b_pe) else 'p_eff WINS'}")
        print(f"  (|rho_chain| = {abs(rho_b_cs):.4f} vs |rho_peff| = {abs(rho_b_pe):.4f})")
        print()

        # Print the individual cells
        print(
            f"  {'target':>10s} {'p_eff bin':>16s} {'b_net':>8s} {'p_eff':>8s} {'E[H]':>6s} {'chain_surv':>11s} {'N':>6s}"
        )
        print("  " + "-" * 70)
        for s in sorted(target_peff_slopes, key=lambda x: x["mean_chain_surv"]):
            print(
                f"  {s['target']:>10s} {s['p_eff_bin']:>16s} {s['b_net']:+.4f} {s['mean_p_eff']:>8.4f} {s['mean_E_H']:>6.2f} {s['mean_chain_surv']:>+11.2f} {s['n']:>6d}"
            )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Save results
    results = {
        "raw_correlations": {
            "log_n": float(rho_logn),
            "p_eff": float(rho_peff),
            "E_H": float(rho_EH),
            "chain_log_surv": float(rho_chain),
        },
        "partial_correlations": {
            "log_n_given_peff": float(pc_logn_peff),
            "log_n_given_peff_EH": float(pc_logn_peff_EH),
            "EH_given_peff_logn": float(pc_EH_peff_logn),
            "chain_given_logn": float(pc_chain_logn),
            "peff_given_logn_EH": float(pc_peff_logn_EH),
        },
        "model_r2": {
            "A_inner": float(r2_A),
            "B_plus_EH": float(r2_B),
            "C_chain_surv": float(r2_C),
            "E_interaction": float(r2_E),
            "F_chain_interaction": float(r2_F),
            "G_full": float(r2_G),
        },
        "target_peff_slopes": target_peff_slopes,
    }

    out_path = os.path.join(SCRIPT_DIR, "eh_disentangle_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
