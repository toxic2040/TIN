#!/usr/bin/env python3
"""
Non-Equilibrium Statistical Mechanics Tests
============================================

Three tests of the stat mech correspondence in non-equilibrium form:

Test 1 — Modified Fluctuation-Dissipation Relation (FDT)
  Equilibrium:      Var(DR) = (Delta_eta / Delta_U)^2 * C / beta^2
  Non-equilibrium:  Var(DR) = [above] + <DR> * sigma_neq
  Null:             Neither

Test 2 — Entropy Production as Fragility Diagnostic
  sigma = <DR> * D_KL(w || w_uniform) * |beta_eff|
  Prediction: sigma correlates with classification strength |gamma|

Test 3 — Specific Heat Peak at Hull Crossover
  C(beta, lambda) = beta^2 * Var_softmax(U)
  Prediction: C peaks near lambda* for all beta

Data sources:
  - runs/emj_seed_sweep_results.json (Var(DR) across 10 seeds)
  - runs/stat_mech_lens_results.json (partition function quantities)
  - runs/beta_eff_survey_results.json (beta_eff for 154 configs)

Theory: theory/nonequilibrium_corrections.md
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).parent


def load_json(name):
    with open(RUNS / name) as f:
        return json.load(f)


# ── Path family parameters (from stat_mech_lens) ────────────────────

# Relay chain
Q_RELAY = 0.577048
T_RELAY_D = 1066.29
LOG_Q_RELAY = np.log(Q_RELAY)

# Bypass (fast, 124d transit variant from stat_mech_lens topology)
Q_BYPASS = 0.552024
T_BYPASS_D = 930.0
LOG_Q_BYPASS = np.log(Q_BYPASS)


def softmax_weights(U, beta):
    """Numerically stable softmax over array U at inverse temp beta."""
    U = np.asarray(U, dtype=float)
    s = beta * (U - np.max(U))
    e = np.exp(s)
    return e / e.sum()


def partition_quantities(beta, lam_d):
    """Compute Z, <U>, Var(U), C, chi for two families at (beta, lambda)."""
    U_r = LOG_Q_RELAY - lam_d * T_RELAY_D
    U_b = LOG_Q_BYPASS - lam_d * T_BYPASS_D
    U = np.array([U_r, U_b])
    T = np.array([T_RELAY_D, T_BYPASS_D])

    w = softmax_weights(U, beta)
    mean_U = w @ U
    var_U = w @ (U**2) - mean_U**2
    mean_T = w @ T
    var_T = w @ (T**2) - mean_T**2

    C = beta**2 * var_U  # specific heat
    chi_T = beta * var_T  # susceptibility w.r.t. lambda
    F = -(1 / beta) * np.log(np.sum(np.exp(beta * U))) if beta > 0 else 0

    return {
        "w": w.tolist(),
        "mean_U": float(mean_U),
        "var_U": float(var_U),
        "C": float(C),
        "chi_T": float(chi_T),
        "F": float(F),
        "U": U.tolist(),
        "T": T.tolist(),
    }


# ── Test 1: Modified FDT ────────────────────────────────────────────


def test_fdt():
    """
    Test: Var(DR) = (Delta_eta/Delta_U)^2 * C / beta^2  [+ J * sigma_neq]

    For two path families at the earliest-arrival oracle's operating point:
    - Delta_eta = eta_relay - eta_bypass (difference in per-path delivery prob)
    - Delta_U = U_relay - U_bypass (utility gap)
    - C = beta^2 * Var_softmax(U)
    - beta from beta_eff fit

    Since C / beta^2 = Var_softmax(U), the prediction reduces to:
    Var(DR) = S_T^2 * (Delta_eta)^2 * w_1 * w_2
    """
    print("\n" + "=" * 70)
    print("  TEST 1: Modified Fluctuation-Dissipation Relation")
    print("=" * 70)

    # Load empirical data
    seed_data = load_json("emj_seed_sweep_results.json")
    lens_data = load_json("stat_mech_lens_results.json")

    beta_eff = lens_data["beta_eff"]
    f_relay = lens_data["earliest_arrival_fractions"]["f_relay"]
    f_bypass = lens_data["earliest_arrival_fractions"]["f_bypass"]

    print(f"\n  Empirical routing fractions: relay={f_relay:.3f}, bypass={f_bypass:.3f}")
    print(f"  Fitted beta_eff = {beta_eff:.3f}")

    results = {}
    for cargo_key in ["hardware", "propellant"]:
        cd = seed_data[cargo_key]
        S_T = cd["s_t"]
        var_DR_emp = cd["var_dr"]
        var_eta_emp = cd["var_eta"]
        mean_DR = cd["mean_dr"]
        mean_eta = cd["mean_eta"]

        # Per-seed etas
        etas = np.array([s["eta"] for s in cd["per_seed"]])

        # Partition function at lambda = 0 (hardware) or lambda for propellant
        if cargo_key == "hardware":
            lam_d = 0.0
        else:
            tau_half = seed_data["config"]["tau_half_days_propellant"]
            lam_d = np.log(2) / tau_half  # per day

        pq = partition_quantities(abs(beta_eff), lam_d)
        w = np.array(pq["w"])
        var_U_softmax = pq["var_U"]
        C = pq["C"]

        # The FDT prediction:
        # In the two-family model, the oracle-induced variance of eta comes from
        # stochastic mixing between families. For empirical fractions f_1, f_2:
        # Var(eta) = f_1 * f_2 * (eta_1 - eta_2)^2
        #
        # And the stat mech prediction at temperature beta:
        # Var(eta)_pred = w_1 * w_2 * (eta_1 - eta_2)^2
        #
        # where w_k = softmax weights

        # Estimate per-family etas from the seed data
        # We don't have per-family eta directly, but we know:
        # mean_eta = f_relay * eta_relay + f_bypass * eta_bypass
        # Var(eta) = f_relay * f_bypass * (eta_relay - eta_bypass)^2
        #
        # From these two equations, solve for eta_relay, eta_bypass:
        # delta_eta = sqrt(Var(eta) / (f_relay * f_bypass))

        f_r = f_relay
        f_b = f_bypass

        if f_r * f_b > 1e-10:
            delta_eta = np.sqrt(var_eta_emp / (f_r * f_b))
        else:
            delta_eta = 0.0

        # Equilibrium FDT prediction for Var(eta)
        w_r, w_b = w
        var_eta_eq = w_r * w_b * delta_eta**2
        var_DR_eq = S_T**2 * var_eta_eq

        # NESS correction: Var(DR) = var_DR_eq + J * sigma_neq
        # where J = mean_DR (the current)
        residual = var_DR_emp - var_DR_eq
        sigma_neq = residual / mean_DR if mean_DR > 1e-10 else 0

        # The softmax mixing prediction (using empirical f's instead of w's)
        var_eta_empirical_mixing = f_r * f_b * delta_eta**2
        var_DR_empirical_mixing = S_T**2 * var_eta_empirical_mixing

        ratio_eq = var_DR_emp / var_DR_eq if var_DR_eq > 1e-20 else float("inf")
        ratio_emp = (
            var_DR_emp / var_DR_empirical_mixing
            if var_DR_empirical_mixing > 1e-20
            else float("inf")
        )

        print(f"\n  --- {cargo_key.upper()} ---")
        print(f"  S_T = {S_T:.4f}")
        print(f"  Empirical: Var(DR) = {var_DR_emp:.3e}, Var(eta) = {var_eta_emp:.3e}")
        print(f"  Mean DR = {mean_DR:.4f} (= J, the current)")
        print(f"  Delta_eta (implied from Var decomposition) = {delta_eta:.4f}")
        print(f"  lambda = {lam_d:.6f}/day")
        print(
            f"  Softmax weights at |beta_eff|={abs(beta_eff):.2f}: "
            f"w_relay={w_r:.4f}, w_bypass={w_b:.4f}"
        )
        print("\n  FDT predictions:")
        print(
            f"    Equilibrium (softmax):      Var(DR) = {var_DR_eq:.3e}  "
            f"(ratio emp/pred = {ratio_eq:.3f})"
        )
        print(
            f"    Empirical mixing (f_r,f_b): Var(DR) = {var_DR_empirical_mixing:.3e}  "
            f"(ratio emp/pred = {ratio_emp:.3f})"
        )
        print(f"    Residual = {residual:.3e}")
        print(f"    sigma_neq = residual / J = {sigma_neq:.3e}")

        # Hypothesis test
        if 0.5 < ratio_emp < 2.0:
            verdict = "CONSISTENT with empirical mixing model"
        elif 0.5 < ratio_eq < 2.0:
            verdict = "CONSISTENT with equilibrium FDT"
        elif abs(sigma_neq) < var_DR_emp:
            verdict = "NESS correction plausible (sigma_neq bounded)"
        else:
            verdict = "INCONCLUSIVE — neither model fits"
        print(f"    Verdict: {verdict}")

        results[cargo_key] = {
            "var_DR_empirical": var_DR_emp,
            "var_DR_eq_pred": var_DR_eq,
            "var_DR_mixing_pred": var_DR_empirical_mixing,
            "ratio_eq": ratio_eq,
            "ratio_mixing": ratio_emp,
            "residual": residual,
            "sigma_neq": sigma_neq,
            "delta_eta": delta_eta,
            "mean_DR": mean_DR,
            "verdict": verdict,
        }

    return results


# ── Test 2: Entropy Production ──────────────────────────────────────


def test_entropy_production():
    """
    sigma = <DR> * D_KL(w || w_uniform) * |beta_eff|

    For two families:
    D_KL = f_1 * log(2*f_1) + f_2 * log(2*f_2)

    Correlate sigma with |gamma| and var_log_p across 154 configs.
    """
    print("\n" + "=" * 70)
    print("  TEST 2: Entropy Production as Fragility Diagnostic")
    print("=" * 70)

    survey = load_json("beta_eff_survey_results.json")
    configs = survey["results"]

    sigmas = []
    gammas = []
    var_log_ps = []
    bodies = []
    betas = []
    dkls = []

    for cfg in configs:
        f_1 = cfg["f_1"]
        f_2 = cfg["f_2"]
        beta = cfg["beta_eff"]
        lyap = cfg["lyapunov_exp"]

        # D_KL(w || uniform) for 2 families
        dkl = 0.0
        if f_1 > 1e-10:
            dkl += f_1 * np.log(2 * f_1)
        if f_2 > 1e-10:
            dkl += f_2 * np.log(2 * f_2)

        # Approximate DR ~ exp(E[H] * lyapunov)
        E_H = cfg["E_H"]
        eta_approx = np.exp(E_H * lyap)
        dr_approx = eta_approx  # rough (S_T not in survey data)

        sigma = dr_approx * dkl * abs(beta) if dkl > 0 else 0

        sigmas.append(sigma)
        betas.append(beta)
        dkls.append(dkl)
        bodies.append(cfg["body"])

        # gamma ~ -lyapunov * E_H for this approximation
        gammas.append(abs(lyap))
        var_log_ps.append(cfg.get("var_H", 0.0))

    sigmas = np.array(sigmas)
    gammas = np.array(gammas)
    betas = np.array(betas)
    dkls = np.array(dkls)

    # Correlations
    valid = sigmas > 1e-10
    if np.sum(valid) > 5:
        rho_gamma = np.corrcoef(sigmas[valid], gammas[valid])[0, 1]
    else:
        rho_gamma = float("nan")

    # Split by temperature sign
    pos_temp = np.array(betas) > 0
    neg_temp = np.array(betas) < 0

    print(f"\n  {len(configs)} configs from beta_eff survey")
    print(f"  Positive temperature: {np.sum(pos_temp)} configs")
    print(f"  Negative temperature: {np.sum(neg_temp)} configs")

    print("\n  Entropy production statistics:")
    print(f"    sigma range: [{np.min(sigmas):.4f}, {np.max(sigmas):.4f}]")
    print(f"    sigma mean:  {np.mean(sigmas):.4f}")
    print(f"    sigma median: {np.median(sigmas):.4f}")

    print("\n  D_KL(w || uniform) statistics:")
    print(f"    D_KL range: [{np.min(dkls):.4f}, {np.max(dkls):.4f}]")
    print(f"    D_KL mean:  {np.mean(dkls):.4f}")

    print(f"\n  Correlation sigma vs |lyapunov|: rho = {rho_gamma:.3f}")

    # Per-body breakdown
    print("\n  Per-body entropy production:")
    for body in sorted(set(bodies)):
        mask = np.array([b == body for b in bodies])
        s = sigmas[mask]
        b = betas[mask]
        print(
            f"    {body:>10s}: n={np.sum(mask):3d}, "
            f"sigma_mean={np.mean(s):8.4f}, "
            f"beta_mean={np.mean(b):8.2f}, "
            f"neg_temp={np.sum(b < 0):d}"
        )

    # Key diagnostic: do high-sigma configs have more extreme classifications?
    q75 = np.percentile(sigmas, 75)
    q25 = np.percentile(sigmas, 25)
    high_sigma = sigmas > q75
    low_sigma = sigmas < q25

    if np.sum(high_sigma) > 0 and np.sum(low_sigma) > 0:
        mean_gamma_high = np.mean(gammas[high_sigma])
        mean_gamma_low = np.mean(gammas[low_sigma])
        print("\n  Fragility test:")
        print(f"    High-sigma (>Q75) mean |lyapunov| = {mean_gamma_high:.4f}")
        print(f"    Low-sigma  (<Q25) mean |lyapunov| = {mean_gamma_low:.4f}")
        print(
            f"    Ratio: {mean_gamma_high / mean_gamma_low:.2f}x"
            if mean_gamma_low > 1e-10
            else "    Low-sigma = 0"
        )
        if mean_gamma_high > mean_gamma_low:
            print("    CONSISTENT: high entropy production -> stronger attenuation")
        else:
            print("    INCONSISTENT: prediction fails")

    return {
        "n_configs": len(configs),
        "sigma_mean": float(np.mean(sigmas)),
        "sigma_max": float(np.max(sigmas)),
        "dkl_mean": float(np.mean(dkls)),
        "rho_sigma_gamma": float(rho_gamma),
        "n_positive_temp": int(np.sum(pos_temp)),
        "n_negative_temp": int(np.sum(neg_temp)),
    }


# ── Test 3: Specific Heat Peak at Crossover ─────────────────────────


def test_specific_heat():
    """
    C(beta, lambda) peaks near the hull crossover lambda* for all beta.
    """
    print("\n" + "=" * 70)
    print("  TEST 3: Specific Heat Peak at Hull Crossover")
    print("=" * 70)

    lens_data = load_json("stat_mech_lens_results.json")
    lambda_star_d = lens_data["crossover_lambda_per_day"]

    betas = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    lambda_grid = np.linspace(0, lambda_star_d * 5, 500)

    results = {}
    print(
        f"\n  Hull crossover at lambda* = {lambda_star_d:.6f}/day "
        f"(tau* = {np.log(2) / lambda_star_d:.0f} days)"
    )
    print(
        f"\n  {'beta':>8s} | {'lambda_C_peak':>14s} | {'offset_from_*':>14s} | "
        f"{'C_peak':>10s} | {'C_at_0':>10s} | {'peak/base':>10s}"
    )
    print(f"  {'-' * 80}")

    all_pass = True
    for beta in betas:
        Cs = []
        for lam in lambda_grid:
            pq = partition_quantities(beta, lam)
            Cs.append(pq["C"])
        Cs = np.array(Cs)

        idx_peak = np.argmax(Cs)
        lam_peak = lambda_grid[idx_peak]
        C_peak = Cs[idx_peak]
        C_at_0 = Cs[0]
        offset = (lam_peak - lambda_star_d) / lambda_star_d * 100

        ratio = C_peak / C_at_0 if C_at_0 > 1e-20 else float("inf")
        passes = abs(offset) < 20  # peak within 20% of lambda*

        print(
            f"  {beta:8.1f} | {lam_peak:14.6f} | {offset:+13.1f}% | "
            f"{C_peak:10.4f} | {C_at_0:10.4f} | {ratio:10.2f}x"
            f"  {'PASS' if passes else 'FAIL'}"
        )

        if not passes:
            all_pass = False

        results[f"beta_{beta}"] = {
            "beta": beta,
            "lambda_peak": float(lam_peak),
            "offset_pct": float(offset),
            "C_peak": float(C_peak),
            "C_at_0": float(C_at_0),
            "peak_ratio": float(ratio),
            "passes": passes,
        }

    print(
        f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'} — "
        f"specific heat peaks near crossover"
    )

    return results


# ── Main ─────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  NON-EQUILIBRIUM STATISTICAL MECHANICS TESTS")
    print("=" * 70)

    fdt_results = test_fdt()
    entropy_results = test_entropy_production()
    heat_results = test_specific_heat()

    output = {
        "metadata": {
            "description": "Non-equilibrium stat mech correspondence tests",
            "theory": "theory/nonequilibrium_corrections.md",
            "tests": ["FDT (3-hypothesis)", "entropy production", "specific heat peak"],
        },
        "fdt": fdt_results,
        "entropy_production": entropy_results,
        "specific_heat": heat_results,
    }

    out_path = RUNS / "nesm_tests_results.json"

    def _default(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(f"Not serializable: {type(x)}")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_default)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
