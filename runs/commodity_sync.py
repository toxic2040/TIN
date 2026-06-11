#!/usr/bin/env python3
"""
Commodity Synchronization Analysis
===================================

Computes the synchronization penalty F_sync for multi-commodity
delivery to a hub that requires all binding commodities to function.

Three analysis modes:
1. Closed-form F_sync for the two-commodity case (hardware + propellant)
2. Monte Carlo validation of the closed form
3. Parameter sweep showing when the penalty becomes binding

Theory: theory/commodity_synchronization.md
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# ── Closed-form synchronization factor ──────────────────────────────


def f_sync_analytic(p_h: float, p_p: float, D: float) -> dict:
    """
    Closed-form synchronization factor for two commodities.

    Parameters
    ----------
    p_h : float
        Per-epoch hardware delivery ratio.
    p_p : float
        Per-epoch propellant delivery ratio.
    D : float
        Per-epoch propellant survival fraction = exp(-T_syn / tau).

    Returns
    -------
    dict with keys:
        f_sync: synchronization factor (0 to 1)
        p_hw_first: P(hardware arrives first or simultaneously)
        p_prop_first: P(propellant arrives first)
        e_decay_given_prop_first: E[D^Delta | propellant first]
        penalty: 1 - f_sync
    """
    q = p_h + p_p - p_h * p_p
    if q < 1e-15:
        return {
            "f_sync": 0.0,
            "p_hw_first": 0.0,
            "p_prop_first": 0.0,
            "e_decay_given_prop_first": 0.0,
            "penalty": 1.0,
        }

    p_hw_first = p_h / q
    p_prop_first = p_p * (1 - p_h) / q

    denom = 1 - D * (1 - p_h)
    if denom < 1e-15:
        e_decay = 0.0
    else:
        e_decay = p_h * D / denom

    f = p_hw_first * 1.0 + p_prop_first * e_decay
    return {
        "f_sync": f,
        "p_hw_first": p_hw_first,
        "p_prop_first": p_prop_first,
        "e_decay_given_prop_first": e_decay,
        "penalty": 1 - f,
    }


# ── Monte Carlo validation ──────────────────────────────────────────


def _mc_worker(args):
    """Module-level worker for ProcessPoolExecutor."""
    p_h, p_p, D, n_trials, seed = args
    rng = np.random.default_rng(seed)

    surviving_fractions = []
    for _ in range(n_trials):
        # Simulate geometric first-success times
        t_hw = 1
        while rng.random() > p_h:
            t_hw += 1

        t_prop = 1
        while rng.random() > p_p:
            t_prop += 1

        # Synchronization: propellant arrives at t_prop, hub activates at max
        t_activate = max(t_hw, t_prop)

        if t_prop <= t_hw:
            # Propellant arrived first or simultaneously; decays during wait
            wait = t_hw - t_prop
            surviving = D**wait
        else:
            # Hardware arrived first; propellant arrives fresh
            surviving = 1.0

        surviving_fractions.append(surviving)

    return {
        "f_sync_mc": float(np.mean(surviving_fractions)),
        "f_sync_std": float(np.std(surviving_fractions) / np.sqrt(n_trials)),
        "n_trials": n_trials,
    }


def f_sync_monte_carlo(
    p_h: float,
    p_p: float,
    D: float,
    n_trials: int = 100_000,
    n_seeds: int = 8,
) -> dict:
    """Monte Carlo estimate of F_sync with parallel seed batches."""
    seeds = [42 + i * 7 for i in range(n_seeds)]
    trials_per = n_trials // n_seeds

    tasks = [(p_h, p_p, D, trials_per, s) for s in seeds]
    n_workers = min(os.cpu_count() or 4, n_seeds)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_mc_worker, tasks))

    # Pool results
    all_f = [r["f_sync_mc"] for r in results]
    return {
        "f_sync_mc": float(np.mean(all_f)),
        "f_sync_mc_std": float(np.std(all_f) / np.sqrt(n_seeds)),
        "n_total": trials_per * n_seeds,
    }


# ── Parameter sweep ─────────────────────────────────────────────────


def decay_factor(t_syn_days: float, tau_half_days: float) -> float:
    """Per-epoch survival fraction."""
    tau = tau_half_days / np.log(2)
    return float(np.exp(-t_syn_days / tau))


def main():
    print("=" * 70)
    print("COMMODITY SYNCHRONIZATION ANALYSIS")
    print("=" * 70)

    T_SYN_EM = 780.0  # Earth-Mars synodic (days)
    T_SYN_MJ = 816.0  # Mars-Jupiter synodic (days)

    # ── 1. EMJ Worked Examples ──────────────────────────────────────

    print("\n--- 1. EMJ Worked Examples ---\n")

    cases = [
        ("Mars hub (EMJ relay)", 0.418, 0.027, T_SYN_EM, 180.0),
        ("Jupiter hub (EMJ relay)", 0.040, 0.004, T_SYN_MJ, 180.0),
        ("Symmetric DR (design study)", 0.30, 0.30, T_SYN_EM, 180.0),
        ("Symmetric + ZBO", 0.30, 0.30, T_SYN_EM, 730.0),
        ("Cislunar hub", 0.90, 0.90, 7.0, 180.0),
        ("Mars ISRU-era (improved)", 0.60, 0.35, T_SYN_EM, 300.0),
    ]

    worked_results = {}
    for name, p_h, p_p, t_syn, tau_half in cases:
        D = decay_factor(t_syn, tau_half)
        analytic = f_sync_analytic(p_h, p_p, D)

        print(f"  {name}:")
        print(
            f"    p_h={p_h:.3f}, p_p={p_p:.3f}, "
            f"T_syn={t_syn:.0f}d, tau_half={tau_half:.0f}d, D={D:.4f}"
        )
        print(f"    P(hw first) = {analytic['p_hw_first']:.4f}")
        print(f"    P(prop first) = {analytic['p_prop_first']:.4f}")
        print(f"    E[D^Delta | prop first] = {analytic['e_decay_given_prop_first']:.6f}")
        print(f"    F_sync = {analytic['f_sync']:.4f}")
        print(f"    Penalty = {analytic['penalty']:.4f} ({analytic['penalty'] * 100:.1f}%)")

        worked_results[name] = {
            "p_h": p_h,
            "p_p": p_p,
            "t_syn_days": t_syn,
            "tau_half_days": tau_half,
            "D": D,
            **analytic,
        }
        print()

    # ── 2. Monte Carlo Validation ───────────────────────────────────

    print("\n--- 2. Monte Carlo Validation ---\n")

    mc_cases = [
        ("Symmetric DR", 0.30, 0.30, T_SYN_EM, 180.0),
        ("Symmetric + ZBO", 0.30, 0.30, T_SYN_EM, 730.0),
        ("EMJ Mars", 0.418, 0.027, T_SYN_EM, 180.0),
    ]

    mc_results = {}
    for name, p_h, p_p, t_syn, tau_half in mc_cases:
        D = decay_factor(t_syn, tau_half)
        analytic = f_sync_analytic(p_h, p_p, D)
        mc = f_sync_monte_carlo(p_h, p_p, D, n_trials=200_000, n_seeds=8)

        gap = abs(analytic["f_sync"] - mc["f_sync_mc"])
        print(f"  {name}:")
        print(f"    Analytic: F_sync = {analytic['f_sync']:.6f}")
        print(f"    MC:       F_sync = {mc['f_sync_mc']:.6f} +/- {mc['f_sync_mc_std']:.6f}")
        print(
            f"    Gap:      {gap:.6f} "
            f"({'PASS' if gap < 3 * mc['f_sync_mc_std'] + 1e-4 else 'FAIL'})"
        )
        print()

        mc_results[name] = {
            "analytic": analytic["f_sync"],
            "mc": mc["f_sync_mc"],
            "mc_std": mc["f_sync_mc_std"],
            "gap": gap,
        }

    # ── 3. Phase Diagram: F_sync(p_h, p_p) at fixed D ──────────────

    print("\n--- 3. Synchronization Penalty Phase Diagram ---\n")

    D_severe = decay_factor(T_SYN_EM, 180.0)  # LH2, Mars synodic
    D_moderate = decay_factor(T_SYN_EM, 730.0)  # ZBO

    p_values = np.arange(0.05, 1.0, 0.05)
    phase_severe = np.zeros((len(p_values), len(p_values)))
    phase_moderate = np.zeros((len(p_values), len(p_values)))

    for i, p_h in enumerate(p_values):
        for j, p_p in enumerate(p_values):
            phase_severe[i, j] = f_sync_analytic(p_h, p_p, D_severe)["penalty"]
            phase_moderate[i, j] = f_sync_analytic(p_h, p_p, D_moderate)["penalty"]

    # Report the regime where penalty > 10%
    mask_10 = phase_severe > 0.10
    n_severe_10 = np.sum(mask_10)
    n_total = phase_severe.size
    print(f"  Severe decay (tau_half=180d, D={D_severe:.4f}):")
    print(f"    Penalty > 10%: {n_severe_10}/{n_total} cells ({n_severe_10 / n_total * 100:.1f}%)")
    print(f"    Max penalty: {np.max(phase_severe) * 100:.1f}%")
    print(
        f"    Penalty > 10% requires p_h/p_p ratio < "
        f"{_find_ratio_threshold(p_values, phase_severe, 0.10):.1f}"
    )
    print()

    mask_mod = phase_moderate > 0.10
    n_mod_10 = np.sum(mask_mod)
    print(f"  Moderate decay (tau_half=730d, D={D_moderate:.4f}):")
    print(f"    Penalty > 10%: {n_mod_10}/{n_total} cells ({n_mod_10 / n_total * 100:.1f}%)")
    print(f"    Max penalty: {np.max(phase_moderate) * 100:.1f}%")
    print()

    # ── 4. tau_half sweep at symmetric DR ───────────────────────────

    print("\n--- 4. Tau-Half Sweep (Symmetric DR = 0.30) ---\n")

    tau_halfs = np.arange(60, 2001, 20)
    penalties = []
    for th in tau_halfs:
        D = decay_factor(T_SYN_EM, th)
        r = f_sync_analytic(0.30, 0.30, D)
        penalties.append(r["penalty"])

    penalties = np.array(penalties)
    print(f"  {'tau_half':>10s}  {'Penalty':>10s}  {'F_sync':>10s}")
    print(f"  {'(days)':>10s}  {'(%)':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for th, pen in zip(tau_halfs[::10], penalties[::10]):
        print(f"  {th:10.0f}  {pen * 100:10.1f}  {1 - pen:10.4f}")

    # Find the tau_half where penalty drops below 5%
    idx_5 = np.searchsorted(-penalties, -0.05)
    if idx_5 < len(tau_halfs):
        print(f"\n  Penalty < 5% at tau_half >= {tau_halfs[idx_5]:.0f} days")
    else:
        print("\n  Penalty never drops below 5% in sweep range")

    # ── 5. C5 Verdict Table ─────────────────────────────────────────

    print("\n--- 5. Condition 5 Verdicts ---\n")

    epsilon = 0.01
    verdicts = []
    for name, p_h, p_p, t_syn, tau_half in cases:
        D = decay_factor(t_syn, tau_half)
        analytic = f_sync_analytic(p_h, p_p, D)
        fs = analytic["f_sync"]

        dr_eff = p_p * fs
        if dr_eff > 0 and dr_eff < 1:
            r_c4 = np.log(epsilon) / np.log(1 - p_p)
            r_c5 = np.log(epsilon) / np.log(1 - dr_eff)
            delta_r = r_c5 - r_c4
        else:
            r_c4 = float("inf")
            r_c5 = float("inf")
            delta_r = 0

        verdict = {
            "name": name,
            "dr_prop": p_p,
            "dr_prop_eff": dr_eff,
            "f_sync": fs,
            "R_c4": r_c4,
            "R_c5": r_c5,
            "delta_R": delta_r,
        }
        verdicts.append(verdict)

        print(f"  {name}:")
        print(f"    DR_prop = {p_p:.4f} -> DR_prop_eff = {dr_eff:.4f} (F_sync = {fs:.4f})")
        print(f"    C4 reserve: {r_c4:.1f} epochs")
        print(f"    C5 reserve: {r_c5:.1f} epochs")
        print(f"    Additional epochs from sync: {delta_r:+.1f}")
        c5_binding = "YES" if delta_r > 1.0 else "NO"
        print(f"    C5 changes verdict? {c5_binding}")
        print()

    # ── Save Results ────────────────────────────────────────────────

    output = {
        "metadata": {
            "description": "Commodity synchronization penalty analysis",
            "theory": "theory/commodity_synchronization.md",
            "model": "Independent Bernoulli delivery, geometric first-success",
        },
        "worked_examples": worked_results,
        "mc_validation": mc_results,
        "phase_diagram": {
            "p_values": p_values.tolist(),
            "penalty_severe_LH2": phase_severe.tolist(),
            "penalty_moderate_ZBO": phase_moderate.tolist(),
        },
        "tau_sweep": {
            "tau_halfs": tau_halfs.tolist(),
            "penalties": penalties.tolist(),
            "dr_pair": [0.30, 0.30],
            "t_syn_days": T_SYN_EM,
        },
        "c5_verdicts": verdicts,
    }

    out_path = Path(__file__).parent / "commodity_sync_results.json"
    with open(out_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )
    print(f"\nResults saved to {out_path}")


def _find_ratio_threshold(p_values, penalty_grid, threshold):
    """Find the max p_h/p_p ratio where penalty > threshold."""
    max_ratio = 0
    for i, p_h in enumerate(p_values):
        for j, p_p in enumerate(p_values):
            if penalty_grid[i, j] > threshold and p_p > 0.01:
                ratio = p_h / p_p
                if ratio > max_ratio:
                    max_ratio = ratio
    return max_ratio if max_ratio > 0 else float("inf")


if __name__ == "__main__":
    main()
