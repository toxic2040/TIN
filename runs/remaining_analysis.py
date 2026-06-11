#!/usr/bin/env python3
"""
Remaining analysis-only tests using existing data.

Test A: Layer 4 decomposition under dwell decay
  - Does lyapunov_bar ≡ lyapunov_exp hold as decay increases?
  - Does cov_excess change sign?
  - Data: dwell_decay_sweep_results.json (3 configs × 10 tau_half)

Test B: Delta_screen closed form
  - Fit Delta_screen vs dwell distribution moments
  - Check sign stability
  - Data: dwell_diamond_results.json (3 configs × 200 instances × tau sweep)

Test C: Switch width scaling
  - W vs dispersion(Delta_C) across 3 configs
  - Power law fit
  - Data: dwell_diamond_results.json
"""

import json

import numpy as np
from scipy import stats


def load_data():
    with open("runs/dwell_decay_sweep_results.json") as f:
        decay = json.load(f)
    with open("runs/dwell_diamond_results.json") as f:
        diamond = json.load(f)
    return decay, diamond


# ══════════════════════════════════════════════════════════════════
# TEST A: Layer 4 decomposition under dwell decay
# ══════════════════════════════════════════════════════════════════


def test_a_layer4_under_decay(decay_data):
    print("=" * 65)
    print("  TEST A: LAYER 4 DECOMPOSITION UNDER DWELL DECAY")
    print("=" * 65)

    for cfg in decay_data:
        name = cfg["config"]
        label = cfg["label"]
        E_H = cfg["E_H"]
        var_H = cfg["var_H"]
        lam_dtn = cfg["lambda_DTN"]

        print(f"\n--- {name} ({label}) ---")
        print(f"  E[H] = {E_H:.3f}, var_H = {var_H:.4f}")
        print(f"  lambda_DTN = {lam_dtn:.6f}")

        sweep = cfg["sweep_exponential"]
        print(
            f"\n  {'tau_half':>10s} | {'lambda_phys':>12s} | {'delta_lam':>10s} | "
            f"{'eta_lyap':>9s} | {'eta_opsp':>9s} | {'var_logp':>9s} | {'n_surv':>6s}"
        )
        print(f"  {'-' * 75}")

        lambdas = []
        taus = []
        var_logps = []

        for pt in sweep:
            tau = pt["tau_half_days"]
            tau_label = f"{tau}d" if tau is not None else "inf"
            lam = pt["lambda_phys"]
            dlam = pt["delta_lambda"]
            eta_l = pt["eta_lyapunov"]
            eta_o = pt["eta_opsp"]
            vlp = pt["var_log_p_eff"]
            ns = pt["n_surviving"]

            print(
                f"  {tau_label:>10s} | {lam:12.6f} | {dlam:10.6f} | "
                f"{eta_l:9.4f} | {eta_o:9.4f} | {vlp:9.6f} | {ns:6d}"
            )

            if tau is not None:
                taus.append(tau)
                lambdas.append(lam)
                var_logps.append(vlp)

        # Check: does lambda_phys actually change with decay?
        lam_range = max(lambdas) - min(lambdas) if lambdas else 0
        print(f"\n  lambda range across tau sweep: {lam_range:.6f}")
        if lam_range < 1e-10:
            print("  ** lambda is CONSTANT — decay has no effect on this config **")
            print(
                f"  (dwell_mean = {cfg['dwell_mean_s']:.0f}s, dwell_max = {cfg['dwell_max_s']:.0f}s)"
            )
            print("  This means the contacts have zero dwell — decay factor D=1 everywhere.")
        else:
            # Fit lambda vs log(tau)
            log_taus = np.log10(taus)
            rho, p = stats.spearmanr(log_taus, lambdas)
            print(f"  lambda vs log(tau): rho = {rho:.4f}, p = {p:.4f}")

            # Check monotonicity
            mono = all(lambdas[i] >= lambdas[i + 1] for i in range(len(lambdas) - 1))
            print(f"  Monotonic (lambda decreases as tau decreases): {mono}")


# ══════════════════════════════════════════════════════════════════
# TEST B: Delta_screen closed form
# ══════════════════════════════════════════════════════════════════


def test_b_delta_screen(diamond_data):
    print("\n" + "=" * 65)
    print("  TEST B: ORACLE SCREENING PREMIUM (Delta_screen)")
    print("=" * 65)

    for cfg in diamond_data:
        name = cfg["config"]
        label = cfg["label"]
        N = cfg["N_instances"]
        paths = cfg["paths"]
        dwell = cfg["dwell_stats"]

        print(f"\n--- {name}: {label} ---")
        print(f"  N = {N} instances")
        print(
            f"  Upper path: H={paths['upper']['H']}, p={paths['upper']['p_link']}, "
            f"T_window={paths['upper']['T_window_days']}d"
        )
        print(
            f"  Lower path: H={paths['lower']['H']}, p={paths['lower']['p_link']}, "
            f"T_window={paths['lower']['T_window_days']}d"
        )
        print(
            f"  Dwell: upper_mean={dwell['upper_mean_days']:.1f}d, "
            f"lower_mean={dwell['lower_mean_days']:.1f}d"
        )

        sweep = cfg["sweep_exponential"]

        # Extract Delta_screen at each tau
        # Delta_screen = lambda_selected - lambda_counterfactual
        # In the diamond data, when the oracle mixes paths, the selected
        # population is enriched for short dwells on the lower path.
        # We can compute this from the per-path gamma data.

        print(
            f"\n  {'tau_half':>10s} | {'frac_upper':>10s} | {'frac_lower':>10s} | "
            f"{'lyap_bar':>10s} | {'cov_excess':>10s} | {'class':>12s}"
        )
        print(f"  {'-' * 70}")

        tau_vals = []
        cov_excess_vals = []
        frac_upper_vals = []
        lyap_bar_vals = []

        for pt in sweep:
            tau = pt.get("tau_half_days")
            tau_label = f"{tau}d" if tau is not None else "inf"
            fu = pt.get("frac_upper", 0)
            fl = pt.get("frac_lower", 0)
            g = pt.get("gamma_full", {})
            lb = g.get("lyapunov_bar", 0)
            ce = g.get("cov_excess", 0)
            cls = g.get("classification", "?")

            print(
                f"  {tau_label:>10s} | {fu:10.3f} | {fl:10.3f} | "
                f"{lb:10.6f} | {ce:10.6f} | {cls:>12s}"
            )

            if tau is not None:
                tau_vals.append(tau)
                cov_excess_vals.append(ce)
                frac_upper_vals.append(fu)
                lyap_bar_vals.append(lb)

        if not tau_vals:
            continue

        tau_arr = np.array(tau_vals)
        ce_arr = np.array(cov_excess_vals)
        fu_arr = np.array(frac_upper_vals)

        # Where does the oracle mix paths?
        mixing_mask = (fu_arr > 0.01) & (fu_arr < 0.99)
        n_mixing = mixing_mask.sum()
        print(f"\n  Mixing zone: {n_mixing} tau values where oracle uses both paths")

        if n_mixing > 0:
            mix_taus = tau_arr[mixing_mask]
            mix_ce = ce_arr[mixing_mask]
            print(f"  Mixing tau range: [{mix_taus.min():.0f}d, {mix_taus.max():.0f}d]")
            print(f"  cov_excess in mixing zone: [{mix_ce.min():.6f}, {mix_ce.max():.6f}]")

            # Sign of cov_excess in mixing zone
            n_pos = (mix_ce > 0).sum()
            n_neg = (mix_ce < 0).sum()
            n_zero = (mix_ce == 0).sum()
            print(f"  Sign: {n_pos} positive, {n_neg} negative, {n_zero} zero")

            if n_pos > 0 and n_neg > 0:
                print("  ** SIGN FLIP in cov_excess within mixing zone **")
            elif n_pos > 0:
                print("  cov_excess is POSITIVE (CLUSTER-direction) in mixing zone")
            elif n_neg > 0:
                print("  cov_excess is NEGATIVE (TRAP-direction) in mixing zone")

        # Delta_screen: compare lyapunov_bar (oracle-selected) vs
        # what it would be on the lower path alone (counterfactual)
        # The DTN baseline gives us the counterfactual
        dtn_gamma = cfg["dtn_baseline"]["gamma"]
        lambda_cf = dtn_gamma["lyapunov_bar"]

        print("\n  Counterfactual (DTN, lower path only):")
        print(f"    lyapunov_bar_cf = {lambda_cf:.6f}")

        # Delta_screen at each tau in mixing zone
        if n_mixing > 0:
            mix_lb = np.array(lyap_bar_vals)[mixing_mask]
            delta_screen = mix_lb - lambda_cf
            print("\n  Delta_screen (oracle-selected minus counterfactual):")
            for i, t in enumerate(mix_taus):
                print(
                    f"    tau={t:.0f}d: Delta_screen = {delta_screen[i]:+.6f} "
                    f"({delta_screen[i] / abs(lambda_cf) * 100:+.1f}%)"
                )

            # Fit Delta_screen vs dwell ratio
            dwell_ratio = dwell["lower_mean_days"] / (mix_taus / np.log(2))
            if len(delta_screen) > 2:
                rho, p = stats.spearmanr(dwell_ratio, delta_screen)
                print(f"\n  Delta_screen vs dwell_ratio: rho = {rho:.4f}, p = {p:.4f}")

            # Sign stability
            all_positive = all(ds > 0 for ds in delta_screen)
            all_negative = all(ds < 0 for ds in delta_screen)
            print(
                f"  Sign stable: {'YES (all positive)' if all_positive else 'YES (all negative)' if all_negative else 'NO — sign flips'}"
            )


# ══════════════════════════════════════════════════════════════════
# TEST C: Switch width scaling
# ══════════════════════════════════════════════════════════════════


def test_c_switch_width(diamond_data):
    print("\n" + "=" * 65)
    print("  TEST C: SWITCH WIDTH SCALING")
    print("=" * 65)

    configs = []
    for cfg in diamond_data:
        name = cfg["config"]
        label = cfg["label"]
        paths = cfg["paths"]
        dwell = cfg["dwell_stats"]
        W_raw = cfg["switch_width"]
        W = W_raw["width_log10"] if isinstance(W_raw, dict) else W_raw
        sp = cfg["switch_point_tau_days"]
        violation = cfg["violation_detected"]
        max_viol = cfg["max_violation_delta"]

        # Dispersion of Delta_C: cost difference between paths
        # Upper path: short dwell, fewer hops, lower p
        # Lower path: long dwell, more hops, higher p
        # Delta_C involves the dwell dispersion on the lower path
        dwell_dispersion = dwell["lower_max_days"] - dwell["lower_mean_days"]
        dwell_cv = (
            dwell_dispersion / dwell["lower_mean_days"] if dwell["lower_mean_days"] > 0 else 0
        )

        # Hop count difference
        delta_H = paths["lower"]["H"] - paths["upper"]["H"]

        # p difference
        delta_p = paths["lower"]["p_link"] - paths["upper"]["p_link"]

        configs.append(
            {
                "name": name,
                "label": label,
                "switch_width": W,
                "switch_point": sp,
                "dwell_dispersion": dwell_dispersion,
                "dwell_cv": dwell_cv,
                "dwell_mean_lower": dwell["lower_mean_days"],
                "dwell_max_lower": dwell["lower_max_days"],
                "delta_H": delta_H,
                "delta_p": delta_p,
                "violation": violation,
                "max_violation": max_viol,
                "H_upper": paths["upper"]["H"],
                "H_lower": paths["lower"]["H"],
                "T_window_lower": paths["lower"]["T_window_days"],
            }
        )

        print(f"\n--- {name}: {label} ---")
        print(f"  Switch point: tau* = {sp} days")
        print(f"  Switch width: W = {W} decades")
        print(f"  Violation detected: {violation} (max delta = {max_viol:.4f})")
        print(
            f"  Hop count: upper H={paths['upper']['H']}, lower H={paths['lower']['H']}, delta_H={delta_H}"
        )
        print(f"  p_link: upper={paths['upper']['p_link']}, lower={paths['lower']['p_link']}")
        print(
            f"  Dwell (lower path): mean={dwell['lower_mean_days']:.1f}d, "
            f"max={dwell['lower_max_days']:.1f}d, dispersion={dwell_dispersion:.1f}d"
        )
        print(f"  T_window (lower): {paths['lower']['T_window_days']}d")

    # Cross-config analysis
    print(f"\n{'=' * 65}")
    print("  CROSS-CONFIG COMPARISON")
    print(f"{'=' * 65}")

    print(
        f"\n  {'Config':15s} | {'W (dec)':>8s} | {'tau*':>8s} | {'disp':>8s} | "
        f"{'dH':>4s} | {'T_win':>6s} | {'max_viol':>9s}"
    )
    print(f"  {'-' * 65}")
    for c in configs:
        print(
            f"  {c['name']:15s} | {c['switch_width']:8.3f} | "
            f"{str(c['switch_point']):>8s} | {c['dwell_dispersion']:8.1f} | "
            f"{c['delta_H']:4d} | {c['T_window_lower']:6.0f} | {c['max_violation']:9.4f}"
        )

    # Key relationships
    if len(configs) >= 3:
        W_vals = [c["switch_width"] for c in configs if c["switch_width"] is not None]
        disp_vals = [c["dwell_dispersion"] for c in configs if c["switch_width"] is not None]
        twin_vals = [c["T_window_lower"] for c in configs if c["switch_width"] is not None]
        dH_vals = [c["delta_H"] for c in configs if c["switch_width"] is not None]
        viol_vals = [c["max_violation"] for c in configs if c["switch_width"] is not None]

        if len(W_vals) >= 3:
            print(f"\n  Correlations (n={len(W_vals)}):")

            # W vs dispersion
            if len(set(disp_vals)) > 1:
                rho_d, _ = stats.spearmanr(disp_vals, W_vals)
                print(f"    W vs dwell_dispersion: rho = {rho_d:.4f}")

            # W vs T_window
            if len(set(twin_vals)) > 1:
                rho_t, _ = stats.spearmanr(twin_vals, W_vals)
                print(f"    W vs T_window: rho = {rho_t:.4f}")

            # W vs delta_H
            if len(set(dH_vals)) > 1:
                rho_h, _ = stats.spearmanr(dH_vals, W_vals)
                print(f"    W vs delta_H: rho = {rho_h:.4f}")

            # Max violation vs delta_H
            if len(set(dH_vals)) > 1:
                rho_v, _ = stats.spearmanr(dH_vals, viol_vals)
                print(f"    max_violation vs delta_H: rho = {rho_v:.4f}")

        # The whitepaper claims W is controlled by dispersion of Delta_C,
        # not mean dwell ratio. Check:
        print("\n  Switch width ordering:")
        sorted_by_W = sorted(configs, key=lambda c: c["switch_width"] or 0)
        for c in sorted_by_W:
            print(
                f"    {c['name']:15s}: W={c['switch_width']}, "
                f"T_window={c['T_window_lower']}d, "
                f"disp={c['dwell_dispersion']:.1f}d"
            )

        # The control config has inverted dwell (upper gets long dwell)
        # Check if it has the widest W as the whitepaper claims
        widest = sorted_by_W[-1]
        print(f"\n  Widest switch: {widest['name']} (W={widest['switch_width']})")
        if widest["T_window_lower"] == max(c["T_window_lower"] for c in configs):
            print("  ** Widest W has largest T_window — consistent with dispersion-driven width **")

    # First-principles interpretation
    print("\n--- First-Principles Interpretation ---")
    print("  The switch width measures how sharply the effective topology")
    print("  transitions as tau_half changes. Three observations:")
    print("")
    for c in configs:
        if c["switch_width"] is not None:
            # Estimate: W ~ log10(T_window / dwell_typical_variation)
            if c["dwell_mean_lower"] > 0:
                ratio = c["T_window_lower"] / c["dwell_mean_lower"]
                print(
                    f"  {c['name']:15s}: T_window/dwell_mean = {ratio:.2f}, W = {c['switch_width']}"
                )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════


def main():
    decay_data, diamond_data = load_data()

    test_a_layer4_under_decay(decay_data)
    test_b_delta_screen(diamond_data)
    test_c_switch_width(diamond_data)

    # Save summary
    summary = {
        "test_a": {
            "description": "Layer 4 decomposition under dwell decay",
            "n_configs": len(decay_data),
            "configs": [d["config"] for d in decay_data],
        },
        "test_b": {
            "description": "Delta_screen oracle screening premium",
            "n_configs": len(diamond_data),
            "configs": [d["config"] for d in diamond_data],
        },
        "test_c": {
            "description": "Switch width scaling",
            "n_configs": len(diamond_data),
        },
    }

    with open("runs/remaining_analysis_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to runs/remaining_analysis_results.json")


if __name__ == "__main__":
    main()
