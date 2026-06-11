"""run_diamond_beta_eff.py — Compute effective Boltzmann temperature beta_eff
on the 3 dwell-diamond configurations.

For a two-path diamond with utility gap delta_U = U_preferred - U_other,
the Boltzmann routing fraction is:

    f_preferred / f_other = exp(beta_eff * delta_U)

so:

    beta_eff = ln(f_preferred / f_other) / delta_U

At tau_half = inf (DTN baseline), utility is purely hardware:
    U_k = log(p_k^H_k) = H_k * log(p_k)

Under decay (finite tau_half), the effective utility includes dwell penalty:
    Q_k(tau) = p_k^H_k * exp(-T_k_eff / tau_s)  where tau_s = tau_half * DAY / ln(2)
    U_k(tau) = log(Q_k(tau)) = H_k * log(p_k) - T_k_eff / tau_s

where T_k_eff is the expected total dwell time on path k.

Sweeps across all tau_half values from the dwell_diamond_results.json,
fitting beta_eff at each point and checking for sign changes.

Outputs: runs/diamond_beta_eff_results.json
"""

import json
import math
import os
import time
from pathlib import Path

_HERE = Path(__file__).parent
DAY = 86400.0
LN2 = math.log(2)

# ── Configuration definitions (from run_dwell_diamond.py) ──────────

CONFIGS = [
    {
        "name": "relay_chain",
        "upper": {"H": 2, "p_link": 0.85, "T_window_days": 2.0, "n_intermediate": 1},
        "lower": {"H": 3, "p_link": 0.95, "T_window_days": 50.0, "n_intermediate": 2},
    },
    {
        "name": "deep_chain",
        "upper": {"H": 2, "p_link": 0.80, "T_window_days": 2.0, "n_intermediate": 1},
        "lower": {"H": 4, "p_link": 0.96, "T_window_days": 200.0, "n_intermediate": 3},
    },
    {
        "name": "control",
        "upper": {"H": 2, "p_link": 0.92, "T_window_days": 780.0, "n_intermediate": 1},
        "lower": {"H": 4, "p_link": 0.88, "T_window_days": 2.0, "n_intermediate": 3},
    },
]


def _sanitize(obj):
    """Replace NaN/Inf with None, numpy types with Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _compute_utility(p_link, H, T_window_days, n_intermediate, tau_half_days):
    """Compute log-reliability utility for a path.

    U = H * log(p_link) - T_eff / tau_s

    T_eff = n_intermediate * T_window_days * DAY / 2  (mean of Uniform(0, T_window))
    tau_s = tau_half_days * DAY / ln(2)

    At tau_half = inf, the dwell term vanishes.
    """
    U_hardware = H * math.log(p_link)

    if tau_half_days is None or not math.isfinite(tau_half_days):
        return U_hardware

    tau_s = tau_half_days * DAY / LN2
    # Expected total dwell: each intermediate node draws Uniform(0, T_window)
    # Mean dwell per node = T_window / 2, n_intermediate nodes
    T_eff = n_intermediate * T_window_days * DAY / 2.0

    return U_hardware - T_eff / tau_s


def _compute_beta_eff(f_pref, f_other, delta_U):
    """Compute beta_eff = ln(f_pref / f_other) / delta_U.

    Returns None if undefined (zero fractions or zero utility gap).
    """
    if f_pref <= 0 or f_other <= 0:
        return None
    if abs(delta_U) < 1e-15:
        return None
    return math.log(f_pref / f_other) / delta_U


def main():
    t0 = time.time()
    print("=" * 100)
    print("DIAMOND BETA_EFF — Effective Boltzmann Temperature on Dwell-Diamond")
    print("=" * 100)
    print()

    # ── Load results ──────────────────────────────────────────────
    results_path = _HERE / "dwell_diamond_results.json"
    if results_path.exists():
        with open(results_path) as f:
            raw_results = json.load(f)
        print(f"  Loaded {results_path.name}: {len(raw_results)} configs")
    else:
        print(f"  WARNING: {results_path.name} not found, computing analytically only")
        raw_results = None
    print()

    # Build lookup: config_name -> results
    results_by_name = {}
    if raw_results:
        for r in raw_results:
            results_by_name[r["config"]] = r

    all_output = []

    for ci, cfg in enumerate(CONFIGS):
        name = cfg["name"]
        upper = cfg["upper"]
        lower = cfg["lower"]

        print(f"{'=' * 100}")
        print(f"  Config: {name}")
        print(
            f"  Upper: H={upper['H']}, p={upper['p_link']}, "
            f"T_window={upper['T_window_days']}d, n_inter={upper['n_intermediate']}"
        )
        print(
            f"  Lower: H={lower['H']}, p={lower['p_link']}, "
            f"T_window={lower['T_window_days']}d, n_inter={lower['n_intermediate']}"
        )
        print()

        # ── DTN baseline (tau_half = inf) ─────────────────────────
        U_upper_dtn = _compute_utility(
            upper["p_link"],
            upper["H"],
            upper["T_window_days"],
            upper["n_intermediate"],
            float("inf"),
        )
        U_lower_dtn = _compute_utility(
            lower["p_link"],
            lower["H"],
            lower["T_window_days"],
            lower["n_intermediate"],
            float("inf"),
        )

        Q_upper = upper["p_link"] ** upper["H"]
        Q_lower = lower["p_link"] ** lower["H"]

        print(f"  Hardware reliability:  Q_upper = {Q_upper:.6f}  Q_lower = {Q_lower:.6f}")
        print(f"  Log-utility (DTN):     U_upper = {U_upper_dtn:.6f}  U_lower = {U_lower_dtn:.6f}")

        # Determine DTN preference
        rdata = results_by_name.get(name)
        if rdata:
            dtn_frac_upper = rdata["dtn_baseline"]["frac_upper"]
            dtn_prefers = rdata["dtn_baseline"]["dtn_prefers"]
        else:
            # Analytical: DTN picks higher reliability
            dtn_prefers = "upper" if Q_upper >= Q_lower else "lower"
            dtn_frac_upper = 1.0 if dtn_prefers == "upper" else 0.0

        print(f"  DTN prefers: {dtn_prefers} (frac_upper={dtn_frac_upper:.3f})")
        print()

        # ── Sweep table ───────────────────────────────────────────
        header = (
            f"  {'tau_half':>10s} | {'f_upper':>8s} {'f_lower':>8s} | "
            f"{'U_upper':>10s} {'U_lower':>10s} | "
            f"{'deltaU':>10s} | {'beta_eff':>12s}"
        )
        sep = f"  {'─' * 10}-+-{'─' * 8}-{'─' * 8}-+-"
        sep += f"{'─' * 10}-{'─' * 10}-+-{'─' * 10}-+-{'─' * 12}"
        print(header)
        print(sep)

        sweep_records = []
        beta_values = []
        sign_changes = 0
        prev_sign = None

        # Get sweep entries from results
        if rdata:
            sweep_entries = rdata["sweep_exponential"]
        else:
            sweep_entries = []

        for entry in sweep_entries:
            tau_days = entry["tau_half_days"]  # None means inf
            f_upper = entry["frac_upper"]
            f_lower = entry["frac_lower"]

            # Compute utilities at this tau_half
            U_upper = _compute_utility(
                upper["p_link"],
                upper["H"],
                upper["T_window_days"],
                upper["n_intermediate"],
                tau_days if tau_days is not None else float("inf"),
            )
            U_lower = _compute_utility(
                lower["p_link"],
                lower["H"],
                lower["T_window_days"],
                lower["n_intermediate"],
                tau_days if tau_days is not None else float("inf"),
            )

            # delta_U = U_preferred - U_other (from DTN baseline perspective)
            if dtn_prefers == "lower":
                delta_U = U_lower - U_upper
                f_pref = f_lower
                f_other = f_upper
            else:
                delta_U = U_upper - U_lower
                f_pref = f_upper
                f_other = f_lower

            beta = _compute_beta_eff(f_pref, f_other, delta_U)

            # Track sign changes
            if beta is not None:
                cur_sign = 1 if beta > 0 else (-1 if beta < 0 else 0)
                if prev_sign is not None and cur_sign != 0 and prev_sign != 0:
                    if cur_sign != prev_sign:
                        sign_changes += 1
                if cur_sign != 0:
                    prev_sign = cur_sign
                beta_values.append(beta)

            # Format tau
            if tau_days is None:
                tau_str = "inf"
            else:
                tau_str = f"{tau_days:.0f}d"

            beta_str = f"{beta:+.4f}" if beta is not None else "undef"

            print(
                f"  {tau_str:>10s} | {f_upper:>8.4f} {f_lower:>8.4f} | "
                f"{U_upper:>10.6f} {U_lower:>10.6f} | "
                f"{delta_U:>+10.6f} | {beta_str:>12s}"
            )

            sweep_records.append(
                {
                    "tau_half_days": tau_days,
                    "f_upper": f_upper,
                    "f_lower": f_lower,
                    "U_upper": U_upper,
                    "U_lower": U_lower,
                    "delta_U": delta_U,
                    "beta_eff": beta,
                }
            )

        print()

        # ── Summary for this config ───────────────────────────────
        finite_betas = [b for b in beta_values if b is not None]
        if finite_betas:
            all_negative = all(b < 0 for b in finite_betas)
            all_positive = all(b > 0 for b in finite_betas)
            min_beta = min(finite_betas)
            max_beta = max(finite_betas)
            mean_beta = sum(finite_betas) / len(finite_betas)

            print(f"  beta_eff range: [{min_beta:+.4f}, {max_beta:+.4f}]")
            print(f"  beta_eff mean:  {mean_beta:+.4f}")
            print(f"  All negative:   {all_negative}")
            print(f"  All positive:   {all_positive}")
            print(f"  Sign changes:   {sign_changes}")

            if sign_changes > 0:
                print("  ** SIGN CHANGE DETECTED ** beta_eff changes sign under decay")
            elif all_negative:
                print("  beta_eff is always negative -> oracle acts as ANTI-Boltzmann")
                print("  (prefers LOWER utility path, i.e., reliability-maximizing)")
            elif all_positive:
                print("  beta_eff is always positive -> oracle acts as Boltzmann")
                print("  (prefers HIGHER utility path)")
        else:
            print("  No finite beta_eff values (single-path routing throughout)")
            all_negative = None
            all_positive = None
            min_beta = None
            max_beta = None
            mean_beta = None

        print()

        config_result = {
            "config": name,
            "dtn_prefers": dtn_prefers,
            "Q_upper": Q_upper,
            "Q_lower": Q_lower,
            "U_upper_dtn": U_upper_dtn,
            "U_lower_dtn": U_lower_dtn,
            "sweep": sweep_records,
            "summary": {
                "n_finite_beta": len(finite_betas),
                "min_beta": min_beta,
                "max_beta": max_beta,
                "mean_beta": mean_beta,
                "all_negative": all_negative,
                "all_positive": all_positive,
                "sign_changes": sign_changes,
            },
        }
        all_output.append(config_result)

    # ── Global summary ────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=" * 100)
    print("GLOBAL SUMMARY")
    print("=" * 100)
    print()
    print(
        f"  {'Config':<15s} | {'DTN pref':>10s} | {'beta range':>22s} | "
        f"{'mean':>8s} | {'sign chg':>8s} | {'verdict':>20s}"
    )
    print(f"  {'─' * 15}-+-{'─' * 10}-+-{'─' * 22}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 20}")

    for r in all_output:
        s = r["summary"]
        if s["min_beta"] is not None:
            rng_str = f"[{s['min_beta']:+.3f}, {s['max_beta']:+.3f}]"
            mean_str = f"{s['mean_beta']:+.3f}"
            chg_str = str(s["sign_changes"])
            if s["sign_changes"] > 0:
                verdict = "SIGN CHANGE"
            elif s["all_negative"]:
                verdict = "always negative"
            elif s["all_positive"]:
                verdict = "always positive"
            else:
                verdict = "mixed"
        else:
            rng_str = "N/A"
            mean_str = "N/A"
            chg_str = "N/A"
            verdict = "single-path only"

        print(
            f"  {r['config']:<15s} | {r['dtn_prefers']:>10s} | {rng_str:>22s} | "
            f"{mean_str:>8s} | {chg_str:>8s} | {verdict:>20s}"
        )

    print()

    # ── Interpretation ────────────────────────────────────────────
    print("  INTERPRETATION:")
    print()

    # Check if beta is always negative for earliest-arrival
    all_configs_negative = all(
        r["summary"]["all_negative"] is True
        for r in all_output
        if r["summary"]["n_finite_beta"] > 0
    )
    any_sign_change = any(r["summary"]["sign_changes"] > 0 for r in all_output)

    if any_sign_change:
        print("  beta_eff CHANGES SIGN under decay in at least one config.")
        print("  This means the oracle's routing preference INVERTS relative")
        print("  to the utility gap as tau_half decreases — the oracle switches")
        print("  from the DTN-preferred path to the alternative when decay makes")
        print("  the long-dwell path uncompetitive.")
        print()
        print("  At the sign-change point, delta_U crosses zero: the dwell penalty")
        print("  on the long-dwell path exactly offsets its hardware advantage.")
        print("  beta_eff diverges near this crossing (Boltzmann analogy breaks down).")
    elif all_configs_negative:
        print("  beta_eff is ALWAYS NEGATIVE for earliest-arrival oracle.")
        print("  The oracle systematically prefers the path with HIGHER reliability,")
        print("  which corresponds to LOWER (more negative) log-utility U.")
        print("  This is anti-Boltzmann behavior: the 'temperature' is negative,")
        print("  meaning the system occupies LOWER-energy (higher-reliability) states.")
    else:
        print("  beta_eff behavior varies across configs.")

    print()
    print(f"  Total elapsed: {elapsed:.2f}s")

    # ── Save ──────────────────────────────────────────────────────
    out_path = _HERE / "diamond_beta_eff_results.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize(all_output), f, indent=2)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved -> {out_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
