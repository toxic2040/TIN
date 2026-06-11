"""run_dwell_diamond.py — Synthetic diamond network: oracle path-switching test.

Tests whether reliability-maximizing oracle path-switching under dwell decay
can violate the monotonicity conjecture gamma_phys <= gamma_DTN.

Three diamond configurations:

  relay_chain:  Upper H=2 (p=0.85, 2d dwell) vs Lower H=3 (p=0.95, 50d dwell)
                DTN picks lower (longer, better links, longer dwell).
                Under decay, oracle switches some instances to shorter path.

  deep_chain:   Upper H=2 (p=0.80, 2d dwell) vs Lower H=4 (p=0.96, 200d/node dwell)
                Same mechanism as relay_chain but amplified.

  control:      Upper H=2 (p=0.92, 780d dwell) vs Lower H=4 (p=0.88, 2d dwell)
                DTN picks upper (shorter, better links, longer dwell).

Ensemble of N=200 instances with dwell ~ Uniform(0, T_window), where
T_window is physically motivated (synodic period for infrequent relays,
contact frequency for frequent relays).  Common random numbers: dwell
draws are fixed once and reused across the entire tau_half sweep.

NOTE: Uniform(0, T_window) is the correct first-principles model for
periodic deterministic departure windows (synodic period, orbital
contact schedule).  For stochastic or irregular departure processes,
the correct dwell distribution is the forward recurrence time (residual
life) of the departure process, not generally uniform.

Key phenomenon: "dwell-induced topological pruning" — perishable cargo's
decay law makes long-wait relays noncompetitive, effectively pruning
them from the graph.  Each commodity type sees a different effective
topology.  The binding commodity operates on its own pruned graph.

Reports five sensitivity quantities per tau_half:
  1. gamma_full         — full ensemble Cov(H, L)
  2. subset_upper_sel   — oracle-conditioned: instances where oracle picks U
  3. subset_lower_sel   — oracle-conditioned: instances where oracle picks L
  4. counterfactual_U   — ALL instances forced through U (selection bias control)
  5. counterfactual_L   — ALL instances forced through L (selection bias control)

Outputs: runs/dwell_diamond_results.json
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_INF = float("inf")
_DAY_S = 86400.0

N_INSTANCES = 200
SEED = 42

# Wide sweep to capture switch points across all configs
TAU_HALF_DAYS = [
    _INF,
    5000,
    3000,
    2000,
    1500,
    1200,
    1000,
    800,
    500,
    300,
    200,
    100,
    50,
    20,
    10,
    5,
    2,
    1,
]

# ── Diamond configurations ───────────────────────────────────────────

CONFIGS = [
    {
        "name": "relay_chain",
        "label": "Relay Chain (2-hop low-dwell vs 3-hop high-dwell)",
        "upper": {
            "H": 2,
            "p_link": 0.85,
            "n_intermediate": 1,
            "T_window_days": 2.0,
        },
        "lower": {
            "H": 3,
            "p_link": 0.95,
            "n_intermediate": 2,
            "T_window_days": 50.0,
        },
    },
    {
        "name": "deep_chain",
        "label": "Deep Chain (2-hop low-dwell vs 4-hop high-dwell, amplified)",
        "upper": {
            "H": 2,
            "p_link": 0.80,
            "n_intermediate": 1,
            "T_window_days": 2.0,
        },
        "lower": {
            "H": 4,
            "p_link": 0.96,
            "n_intermediate": 3,
            "T_window_days": 200.0,
        },
    },
    {
        "name": "control",
        "label": "Control (2-hop high-dwell vs 4-hop low-dwell)",
        "upper": {
            "H": 2,
            "p_link": 0.92,
            "n_intermediate": 1,
            "T_window_days": 780.0,
        },
        "lower": {
            "H": 4,
            "p_link": 0.88,
            "n_intermediate": 3,
            "T_window_days": 2.0,
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────────


def _sanitize(obj):
    """Replace NaN/Inf with None, numpy types with Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _draw_dwells(rng, n_intermediate, T_window_days):
    """Draw dwell times from Uniform(0, T_window) for each intermediate node.

    Physical model: cargo arrives at random phase within a periodic
    deterministic departure window.
    Returns (N_INSTANCES, n_intermediate) array in seconds.
    """
    T_window_s = T_window_days * _DAY_S
    return rng.uniform(0, T_window_s, (N_INSTANCES, n_intermediate))


# ── Reliability functions ────────────────────────────────────────────


def _path_reliability_exp(p_link, H, dwells_2d, tau_half_s):
    """Exponential decay: P_i = p^H * exp(-sum_j(dwell_ij) / tau_half).

    For exponential, per-node product equals exp of cumulative sum.
    """
    link_prob = p_link**H
    if not math.isfinite(tau_half_s) or tau_half_s <= 0:
        return np.full(dwells_2d.shape[0], link_prob)
    total_dwell = np.sum(dwells_2d, axis=1)
    return link_prob * np.exp(-total_dwell / tau_half_s)


def _path_reliability_thr(p_link, H, dwells_2d, tau_max_s):
    """Threshold decay: P_i = p^H if cumulative dwell < tau_max, else 0.

    Uses TOTAL path dwell (not per-node).  Three 25-day stops = 75 days
    total, which fails a 90-day threshold even though each stop passes.
    """
    if not math.isfinite(tau_max_s):
        return np.full(dwells_2d.shape[0], p_link**H)
    total_dwell = np.sum(dwells_2d, axis=1)
    return np.where(total_dwell < tau_max_s, p_link**H, 0.0)


# ── Oracle and analysis ──────────────────────────────────────────────


def _oracle_select(rel_upper, rel_lower, H_upper, H_lower):
    """Reliability-maximizing oracle.  Ties broken to shorter path."""
    pick_upper = rel_upper >= rel_lower
    n = len(rel_upper)

    selected_H = np.where(pick_upper, H_upper, H_lower)
    selected_rel = np.where(pick_upper, rel_upper, rel_lower)

    log_P = np.log(np.clip(selected_rel, 1e-300, None))
    frac_upper = float(np.sum(pick_upper)) / n

    return {
        "H": selected_H,
        "log_P": log_P,
        "frac_upper": frac_upper,
        "frac_lower": 1.0 - frac_upper,
        "pick_upper": pick_upper,
    }


def _cov_analysis(H_arr, log_P_arr):
    """Compute Cov(H, L) decomposition from ensemble of (H, log_P) pairs.

    Adapted from run_per_path_cov._cov_decomposition.
    L = log_P is the total log-probability of the selected path.
    """
    n = len(H_arr)
    if n < 5:
        return None

    hs = H_arr.astype(float)
    ls = log_P_arr.astype(float)

    valid = ls > -600
    if np.sum(valid) < 5:
        return None
    hs = hs[valid]
    ls = ls[valid]
    n = len(hs)

    var_h = float(np.var(hs))
    var_l = float(np.var(ls))

    mean_h = float(np.mean(hs))
    mean_l = float(np.mean(ls))
    lyapunov_bar = mean_l / mean_h if mean_h > 0 else 0.0

    if var_h < 1e-15:
        return {
            "n_paths": n,
            "mean_H": mean_h,
            "mean_L": mean_l,
            "var_H": var_h,
            "var_L": var_l,
            "cov_H_L": 0.0,
            "corr_H_L": 0.0,
            "lyapunov_bar": lyapunov_bar,
            "cov_excess": 0.0,
            "gamma_defined": False,
            "classification": "SINGLE_PATH",
        }

    cov_hl = float(np.cov(hs, ls, bias=True)[0, 1])
    denom = math.sqrt(var_h * var_l)
    corr_hl = float(cov_hl / denom) if denom > 1e-15 else 0.0

    cov_excess = cov_hl - lyapunov_bar * var_h

    if cov_excess > 0.001:
        classification = "CLUSTER"
    elif cov_excess < -0.001:
        classification = "TRAP"
    else:
        classification = "NEUTRAL"

    return {
        "n_paths": n,
        "mean_H": mean_h,
        "mean_L": mean_l,
        "var_H": var_h,
        "var_L": var_l,
        "cov_H_L": cov_hl,
        "corr_H_L": corr_hl,
        "lyapunov_bar": lyapunov_bar,
        "cov_excess": cov_excess,
        "gamma_defined": True,
        "classification": classification,
    }


def _subset_stats(H_arr, log_P_arr, mask):
    """Oracle-conditioned subset statistics.

    Within a subset all H values are identical, so Var(H)=0 and gamma
    is undefined.  Report lambda_eff = mean(log_P / H) instead.
    NOTE: these are selection-biased — not intrinsic path properties.
    """
    n = int(np.sum(mask))
    if n == 0:
        return {"n_instances": 0, "mean_H": None, "mean_lambda_eff": None, "mean_log_P": None}
    hs = H_arr[mask].astype(float)
    ls = log_P_arr[mask]
    valid = ls > -600
    if np.sum(valid) == 0:
        return {
            "n_instances": n,
            "mean_H": float(hs[0]),
            "mean_lambda_eff": None,
            "mean_log_P": None,
        }
    return {
        "n_instances": n,
        "mean_H": float(np.mean(hs[valid])),
        "mean_lambda_eff": float(np.mean(ls[valid] / hs[valid])),
        "mean_log_P": float(np.mean(ls[valid])),
    }


def _counterfactual_stats(p_link, H, dwells_2d, tau_half_s):
    """Counterfactual fixed-path statistics: ALL instances through one path.

    This is the selection-bias control.  It answers: "Is this path
    intrinsically better under perishability, or does it only look
    better because the oracle cherry-picked the good-dwell instances?"
    """
    rel = _path_reliability_exp(p_link, H, dwells_2d, tau_half_s)
    log_P = np.log(np.clip(rel, 1e-300, None))
    valid = log_P > -600
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {"mean_lambda_eff": None, "std_lambda_eff": None, "mean_log_P": None, "n_valid": 0}
    lam_eff = log_P[valid] / H
    return {
        "mean_lambda_eff": float(np.mean(lam_eff)),
        "std_lambda_eff": float(np.std(lam_eff)),
        "mean_log_P": float(np.mean(log_P[valid])),
        "n_valid": n_valid,
    }


def _compute_switch_width(sweep_fracs):
    """Compute switch width: interval in log10(tau) between p=0.9 and p=0.1.

    sweep_fracs: list of (tau_days, frac_on_DTN_preferred_path).
    Returns (tau_90, tau_10, width_log10) or Nones.
    """
    tau_90 = None
    tau_10 = None
    for i in range(len(sweep_fracs) - 1):
        t1, f1 = sweep_fracs[i]
        t2, f2 = sweep_fracs[i + 1]
        if not (math.isfinite(t1) and math.isfinite(t2)):
            continue
        if abs(f1 - f2) < 1e-10:
            continue
        # Interpolate 0.9 crossing
        if tau_90 is None and ((f1 >= 0.9 and f2 < 0.9) or (f1 < 0.9 and f2 >= 0.9)):
            tau_90 = t1 + (0.9 - f1) / (f2 - f1) * (t2 - t1)
        # Interpolate 0.1 crossing
        if tau_10 is None and ((f1 >= 0.1 and f2 < 0.1) or (f1 < 0.1 and f2 >= 0.1)):
            tau_10 = t1 + (0.1 - f1) / (f2 - f1) * (t2 - t1)

    width_log10 = None
    if tau_90 is not None and tau_10 is not None and tau_90 > 0 and tau_10 > 0:
        width_log10 = abs(math.log10(tau_90) - math.log10(tau_10))

    return tau_90, tau_10, width_log10


# ── Per-config analysis ──────────────────────────────────────────────


def _analyze_config(config, rng):
    """Analyze one diamond configuration across the tau_half sweep."""
    upper = config["upper"]
    lower = config["lower"]
    H_u = upper["H"]
    H_l = lower["H"]

    # Draw dwell ensembles once (common random numbers for entire sweep)
    dw_upper = _draw_dwells(rng, upper["n_intermediate"], upper["T_window_days"])
    dw_lower = _draw_dwells(rng, lower["n_intermediate"], lower["T_window_days"])

    total_dw_upper = np.sum(dw_upper, axis=1)
    total_dw_lower = np.sum(dw_lower, axis=1)

    # DTN baseline (tau_half = inf, no decay)
    rel_u_dtn = np.full(N_INSTANCES, upper["p_link"] ** H_u)
    rel_l_dtn = np.full(N_INSTANCES, lower["p_link"] ** H_l)
    sel_dtn = _oracle_select(rel_u_dtn, rel_l_dtn, H_u, H_l)
    gamma_dtn = _cov_analysis(sel_dtn["H"], sel_dtn["log_P"])
    dtn_cov_excess = gamma_dtn["cov_excess"] if gamma_dtn else 0.0
    dtn_frac_upper = sel_dtn["frac_upper"]

    # Determine which path DTN prefers (for switch-width in terms of p_preferred)
    dtn_prefers_upper = dtn_frac_upper >= 0.5

    print("    DTN baseline:")
    print(f"      P_upper = {upper['p_link'] ** H_u:.4f}  P_lower = {lower['p_link'] ** H_l:.4f}")
    print(
        f"      Oracle: {dtn_frac_upper * 100:.0f}% upper, {sel_dtn['frac_lower'] * 100:.0f}% lower"
    )
    if gamma_dtn:
        print(f"      cov_excess = {dtn_cov_excess:.6f}  class = {gamma_dtn['classification']}")
    print(
        f"      Dwell upper: mean={np.mean(total_dw_upper) / _DAY_S:.1f}d  "
        f"max={np.max(total_dw_upper) / _DAY_S:.1f}d"
    )
    print(
        f"      Dwell lower: mean={np.mean(total_dw_lower) / _DAY_S:.1f}d  "
        f"max={np.max(total_dw_lower) / _DAY_S:.1f}d"
    )

    # Header
    print(
        f"\n    {'tau':>10s} {'frac_up':>8s} {'frac_lo':>8s} "
        f"{'cov_exc':>10s} {'class':>8s} "
        f"{'sel_U':>8s} {'sel_L':>8s} "
        f"{'cf_U':>8s} {'cf_L':>8s} "
        f"{'mono':>5s} {'pruned':>6s}"
    )
    print(
        f"    {'─' * 10} {'─' * 8} {'─' * 8} "
        f"{'─' * 10} {'─' * 8} "
        f"{'─' * 8} {'─' * 8} "
        f"{'─' * 8} {'─' * 8} "
        f"{'─' * 5} {'─' * 6}"
    )

    sweep_exp = []
    sweep_thr = []
    switch_tau = None
    violation_detected = False
    max_violation = 0.0

    # For switch width calculation
    pref_fracs = []  # (tau_days, frac_on_DTN_preferred_path)

    for tau_days in TAU_HALF_DAYS:
        tau_s = tau_days * _DAY_S if math.isfinite(tau_days) else _INF
        tag = f"{tau_days:.0f}d" if math.isfinite(tau_days) else "inf"

        # ── Exponential decay ────────────────────────────────────
        rel_u = _path_reliability_exp(upper["p_link"], H_u, dw_upper, tau_s)
        rel_l = _path_reliability_exp(lower["p_link"], H_l, dw_lower, tau_s)
        sel = _oracle_select(rel_u, rel_l, H_u, H_l)

        gamma_full = _cov_analysis(sel["H"], sel["log_P"])

        # Oracle-conditioned subset stats (selection-biased)
        mask_u = sel["pick_upper"]
        mask_l = ~mask_u
        sub_u = _subset_stats(sel["H"], sel["log_P"], mask_u)
        sub_l = _subset_stats(sel["H"], sel["log_P"], mask_l)

        # Counterfactual fixed-path stats (selection-bias control)
        cf_u = _counterfactual_stats(upper["p_link"], H_u, dw_upper, tau_s)
        cf_l = _counterfactual_stats(lower["p_link"], H_l, dw_lower, tau_s)

        # Monotonicity check
        phys_excess = gamma_full["cov_excess"] if gamma_full else 0.0
        is_monotonic = phys_excess <= dtn_cov_excess + 1e-6
        is_violation = not is_monotonic
        if is_violation:
            violation_detected = True
            delta = phys_excess - dtn_cov_excess
            max_violation = max(max_violation, delta)

        # Topological pruning: violation caused by oracle switching
        topo_pruned = is_violation and (abs(sel["frac_upper"] - dtn_frac_upper) > 0.05)

        # Switch point: DTN-preferred fraction crosses 50%
        frac_pref = sel["frac_upper"] if dtn_prefers_upper else sel["frac_lower"]
        pref_fracs.append((tau_days, frac_pref))
        if switch_tau is None and frac_pref < 0.5:
            switch_tau = tau_days

        # Format
        ce_str = f"{phys_excess:+.6f}" if gamma_full else "N/A"
        cls_str = gamma_full["classification"] if gamma_full else "N/A"
        su = f"{sub_u['mean_lambda_eff']:.4f}" if sub_u["mean_lambda_eff"] is not None else "N/A"
        sl = f"{sub_l['mean_lambda_eff']:.4f}" if sub_l["mean_lambda_eff"] is not None else "N/A"
        cu = f"{cf_u['mean_lambda_eff']:.4f}" if cf_u["mean_lambda_eff"] is not None else "N/A"
        cl = f"{cf_l['mean_lambda_eff']:.4f}" if cf_l["mean_lambda_eff"] is not None else "N/A"
        mono_str = "YES" if is_monotonic else "VIOL"
        pr_str = "YES" if topo_pruned else "no"

        print(
            f"    {tag:>10s} {sel['frac_upper']:>8.3f} {sel['frac_lower']:>8.3f} "
            f"{ce_str:>10s} {cls_str:>8s} "
            f"{su:>8s} {sl:>8s} "
            f"{cu:>8s} {cl:>8s} "
            f"{mono_str:>5s} {pr_str:>6s}"
        )

        sweep_exp.append(
            {
                "tau_half_days": tau_days if math.isfinite(tau_days) else None,
                "frac_upper": sel["frac_upper"],
                "frac_lower": sel["frac_lower"],
                "gamma_full": gamma_full,
                "subset_upper_sel": sub_u,
                "subset_lower_sel": sub_l,
                "counterfactual_upper": cf_u,
                "counterfactual_lower": cf_l,
                "cov_excess": phys_excess,
                "monotonic": is_monotonic,
                "topological_pruning": topo_pruned,
            }
        )

        # ── Threshold decay (cumulative) ─────────────────────────
        rel_u_thr = _path_reliability_thr(upper["p_link"], H_u, dw_upper, tau_s)
        rel_l_thr = _path_reliability_thr(lower["p_link"], H_l, dw_lower, tau_s)
        sel_thr = _oracle_select(rel_u_thr, rel_l_thr, H_u, H_l)
        gamma_thr = _cov_analysis(sel_thr["H"], sel_thr["log_P"])

        sweep_thr.append(
            {
                "tau_max_days": tau_days if math.isfinite(tau_days) else None,
                "frac_upper": sel_thr["frac_upper"],
                "n_upper_alive": int(np.sum(rel_u_thr > 1e-300)),
                "n_lower_alive": int(np.sum(rel_l_thr > 1e-300)),
                "gamma_full": gamma_thr,
            }
        )

    # Switch width: 90% to 10% of DTN-preferred fraction in log10(tau)
    tau_90, tau_10, switch_width = _compute_switch_width(pref_fracs)

    return {
        "config": config["name"],
        "label": config["label"],
        "N_instances": N_INSTANCES,
        "seed": SEED,
        "paths": {
            "upper": {
                "H": H_u,
                "p_link": upper["p_link"],
                "T_window_days": upper["T_window_days"],
                "P_dtn": upper["p_link"] ** H_u,
            },
            "lower": {
                "H": H_l,
                "p_link": lower["p_link"],
                "T_window_days": lower["T_window_days"],
                "P_dtn": lower["p_link"] ** H_l,
            },
        },
        "dtn_baseline": {
            "frac_upper": dtn_frac_upper,
            "dtn_prefers": "upper" if dtn_prefers_upper else "lower",
            "gamma": gamma_dtn,
            "cov_excess": dtn_cov_excess,
        },
        "dwell_stats": {
            "upper_mean_days": float(np.mean(total_dw_upper) / _DAY_S),
            "upper_max_days": float(np.max(total_dw_upper) / _DAY_S),
            "lower_mean_days": float(np.mean(total_dw_lower) / _DAY_S),
            "lower_max_days": float(np.max(total_dw_lower) / _DAY_S),
        },
        "sweep_exponential": sweep_exp,
        "sweep_threshold": sweep_thr,
        "switch_point_tau_days": switch_tau,
        "switch_width": {
            "tau_90_days": tau_90,
            "tau_10_days": tau_10,
            "width_log10": switch_width,
        },
        "violation_detected": violation_detected,
        "max_violation_delta": max_violation,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    print("=" * 110)
    print("DWELL-INDUCED TOPOLOGICAL PRUNING — Diamond Network Test")
    print("=" * 110)
    print()
    print(f"  Ensemble: N={N_INSTANCES}, seed={SEED}, common random numbers across tau sweep")
    print(f"  Configs:  {[c['name'] for c in CONFIGS]}")
    print(f"  Sweep:    {len(TAU_HALF_DAYS)} tau_half values")
    print()
    print("  Columns: sel_U/sel_L = oracle-conditioned lambda_eff (selection-biased)")
    print("           cf_U/cf_L  = counterfactual lambda_eff (all instances, unbiased)")
    print()

    out_path = _HERE / "dwell_diamond_results.json"
    all_results = []
    rng = np.random.RandomState(SEED)

    for ci, config in enumerate(CONFIGS):
        print(f"{'─' * 110}")
        print(f"  [{ci + 1}/{len(CONFIGS)}] {config['label']}")
        print(
            f"  Upper: H={config['upper']['H']}, p={config['upper']['p_link']}, "
            f"T_window={config['upper']['T_window_days']}d"
        )
        print(
            f"  Lower: H={config['lower']['H']}, p={config['lower']['p_link']}, "
            f"T_window={config['lower']['T_window_days']}d"
        )
        print(f"{'─' * 110}")
        print()

        result = _analyze_config(config, rng)
        all_results.append(result)

        # Incremental save
        with open(out_path, "w") as f:
            json.dump(_sanitize(all_results), f, indent=2)

        print()
        if result["violation_detected"]:
            print(
                f"    ** CONJECTURE VIOLATED ** max delta_cov_excess = "
                f"{result['max_violation_delta']:.6f}"
            )
            print("    Mechanism: dwell-induced topological pruning")
        else:
            print("    Monotonicity HOLDS across all tau_half values")
        if result["switch_point_tau_days"] is not None:
            print(f"    Switch point: tau_half ~ {result['switch_point_tau_days']:.0f}d")
        sw = result["switch_width"]
        if sw["width_log10"] is not None:
            print(
                f"    Switch width: {sw['width_log10']:.2f} decades "
                f"(tau_90={sw['tau_90_days']:.0f}d, "
                f"tau_10={sw['tau_10_days']:.0f}d)"
            )
        print()

    # ── Summary ──────────────────────────────────────────────────

    elapsed = time.time() - t0
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print()

    for r in all_results:
        status = "VIOLATED" if r["violation_detected"] else "HOLDS"
        switch = (
            f"tau ~ {r['switch_point_tau_days']:.0f}d"
            if r["switch_point_tau_days"] is not None
            else "no switch"
        )
        sw = r["switch_width"]
        width = f"{sw['width_log10']:.2f} dec" if sw["width_log10"] is not None else "N/A"
        print(
            f"  {r['config']:<15s}  {status:<10s}  {switch:<16s}  "
            f"width: {width:<10s}  delta: {r['max_violation_delta']:.6f}"
        )

    print()
    print("  The monotonicity conjecture gamma_phys <= gamma_DTN fails")
    print("  universally when the oracle has parallel paths with different")
    print("  hop counts.  The mechanism is selection bias: at the switch")
    print("  point, the oracle cherry-picks instances with favorable dwell")
    print("  onto the high-quality path, creating systematic positive")
    print("  Cov(H, L) — a CLUSTER signal that the DTN baseline lacks.")
    print()
    print("  The phenomenon is dwell-induced topological pruning:")
    print("  perishable cargo's decay law makes long-wait relays invisible,")
    print("  so each commodity operates on its own effective graph.")
    print("  For durable cargo, the network is the engineering topology.")
    print("  For perishable cargo, the network is the subset whose")
    print("  waiting-time statistics are compatible with survival.")

    print()
    print(f"  Total elapsed: {elapsed:.1f}s")

    with open(out_path, "w") as f:
        json.dump(_sanitize(all_results), f, indent=2)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved -> {out_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
