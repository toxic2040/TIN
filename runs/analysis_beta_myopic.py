"""analysis_beta_myopic.py — Test the orbital β_myopic formula on CRAWDAD data.

Orbital formula: β_myopic = δ₀ · E[H] · (1 − exp(−p_eff / p₀))
  with δ₀ = 0.311, p₀ = 0.042, R² = 0.68

Two tests:

A. WITHIN-TRACE: At each (trace, p_eff), measure slope of ln(Φ) vs E[H].
   NOTE: This measures ∂ln(Φ)/∂E[H] at fixed n, which is a DIFFERENT quantity
   than β_myopic = ∂ln(Φ)/∂log(n). The orbital formula predicts:
     ∂ln(Φ)/∂E[H] ≈ δ₀ · (1 − exp(−p/p₀)) · log(n)
   (the log(n) factor is required for a fair comparison).
   This test characterizes how distortion grows with hop count within each class.

B. CROSS-TRACE: At each p_eff, compute mean ln(Φ_myopic) per trace.
   Fit slope vs log(n). Compare to orbital β_myopic prediction.
   NOTE: Only 4 data points (4 traces), so fits have minimal degrees of freedom.
   Confounded: n varies jointly with topology, contact patterns, ρ_pair.

Also tests ln(Φ_normal) (full greedy+retry) for comparison.

Reads: crawdad_contacts.Exp{1,2,3,6}_results.json
Writes: analysis_beta_myopic_results.json
"""

import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

EXPERIMENTS = {
    "Exp1": {"file": "crawdad_contacts.Exp1_results.json", "n_nodes": 9},
    "Exp2": {"file": "crawdad_contacts.Exp2_results.json", "n_nodes": 12},
    "Exp3": {"file": "crawdad_contacts.Exp3_results.json", "n_nodes": 41},
    "Exp6": {"file": "crawdad_contacts.Exp6_results.json", "n_nodes": 98},
}

# Orbital fit parameters
DELTA_0 = 0.311
P_0 = 0.042

P_EFFS = [0.02, 0.05, 0.1, 0.3, 0.5]


def _predicted_beta(p_eff, eh):
    """Orbital β_myopic prediction."""
    return DELTA_0 * eh * (1.0 - np.exp(-p_eff / P_0))


def _load(meta):
    path = _HERE / meta["file"]
    with open(path) as f:
        return json.load(f)


def main():
    print()
    print("  β_myopic Transfer Test: Orbital Formula on CRAWDAD")
    print("  " + "=" * 55)
    print()

    # Load all traces
    traces = {}
    for exp_name, meta in EXPERIMENTS.items():
        try:
            data = _load(meta)
        except FileNotFoundError:
            continue
        active = [
            r
            for r in data["results"]
            if r.get("S_T", 0) > 0
            and r.get("eta_normal", 0) > 0
            and r.get("eta_lyap", 0) > 0
            and r.get("E_H", 0) > 0
        ]
        traces[exp_name] = {
            "n_nodes": meta["n_nodes"],
            "active": active,
        }

    # ══════════════════════════════════════════════════════════════════
    # TEST A: Within-trace slope of ln(Φ) vs E[H] at each p_eff
    # ══════════════════════════════════════════════════════════════════
    print("  TEST A: Within-trace slope of ln(Φ) vs E[H]")
    print("  " + "-" * 55)
    print()

    within_results = {}
    for exp_name, tdata in traces.items():
        active = tdata["active"]
        n = tdata["n_nodes"]
        within_results[exp_name] = {"n_nodes": n, "by_p_eff": {}}

        for p_target in P_EFFS:
            group = [r for r in active if abs(r["p_eff"] - p_target) < 0.001]
            if len(group) < 20:
                continue

            # ln(Φ_normal) vs E[H]
            phi_norm = [(r["E_H"], np.log(r["phi_normal"])) for r in group if r["phi_normal"] > 0]
            # ln(Φ_myopic) vs E[H] (no-retry mode)
            phi_myop = [
                (r["E_H"], np.log(r["phi_myopic"])) for r in group if r.get("phi_myopic", 0) > 0
            ]

            entry = {"p_eff": p_target}

            # Fit normal Φ
            if len(phi_norm) > 10:
                ehs = np.array([x[0] for x in phi_norm])
                lps = np.array([x[1] for x in phi_norm])
                # Remove outliers (>5σ)
                mu, sig = np.mean(lps), np.std(lps)
                if sig > 0:
                    mask = np.abs(lps - mu) < 5 * sig
                    ehs, lps = ehs[mask], lps[mask]
                if len(ehs) > 5 and np.std(ehs) > 0:
                    coeffs = np.polyfit(ehs, lps, 1)
                    pred_ln = np.polyval(coeffs, ehs)
                    ss_res = np.sum((lps - pred_ln) ** 2)
                    ss_tot = np.sum((lps - lps.mean()) ** 2)
                    entry["normal_slope"] = float(coeffs[0])
                    entry["normal_intercept"] = float(coeffs[1])
                    entry["normal_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
                    entry["normal_n"] = int(len(ehs))

            # Fit myopic Φ
            if len(phi_myop) > 10:
                ehs_m = np.array([x[0] for x in phi_myop])
                lps_m = np.array([x[1] for x in phi_myop])
                mu_m, sig_m = np.mean(lps_m), np.std(lps_m)
                if sig_m > 0:
                    mask_m = np.abs(lps_m - mu_m) < 5 * sig_m
                    ehs_m, lps_m = ehs_m[mask_m], lps_m[mask_m]
                if len(ehs_m) > 5 and np.std(ehs_m) > 0:
                    coeffs_m = np.polyfit(ehs_m, lps_m, 1)
                    pred_m = np.polyval(coeffs_m, ehs_m)
                    ss_res_m = np.sum((lps_m - pred_m) ** 2)
                    ss_tot_m = np.sum((lps_m - lps_m.mean()) ** 2)
                    entry["myopic_slope"] = float(coeffs_m[0])
                    entry["myopic_intercept"] = float(coeffs_m[1])
                    entry["myopic_r2"] = float(1 - ss_res_m / ss_tot_m) if ss_tot_m > 0 else 0
                    entry["myopic_n"] = int(len(ehs_m))

            # Predicted slope from orbital formula
            # ∂ln(Φ)/∂E[H] at fixed n ≈ δ₀·(1−exp(−p/p₀))·log(n)
            mean_eh = float(np.mean([r["E_H"] for r in group]))
            log_n = np.log(n)
            entry["predicted_slope"] = float(DELTA_0 * (1.0 - np.exp(-p_target / P_0)) * log_n)
            entry["predicted_slope_without_logn"] = float(DELTA_0 * (1.0 - np.exp(-p_target / P_0)))
            entry["log_n"] = float(log_n)
            entry["mean_eh"] = mean_eh

            within_results[exp_name]["by_p_eff"][str(round(p_target, 4))] = entry

            # Print
            ns = entry.get("normal_slope", float("nan"))
            ms = entry.get("myopic_slope", float("nan"))
            ps = entry["predicted_slope"]
            nr2 = entry.get("normal_r2", float("nan"))
            mr2 = entry.get("myopic_r2", float("nan"))
            ratio = ms / ps if ps != 0 and not np.isnan(ms) else float("nan")
            print(
                f"    {exp_name} p={p_target:.2f}: "
                f"slope_normal={ns:+.3f} (R²={nr2:.2f})  "
                f"slope_myopic={ms:+.3f} (R²={mr2:.2f})  "
                f"predicted={ps:+.3f}  ratio={ratio:.2f}"
            )

        print()

    # ══════════════════════════════════════════════════════════════════
    # TEST B: Cross-trace slope of mean ln(Φ) vs log(n)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("  TEST B: Cross-trace slope of mean ln(Φ) vs log(n)")
    print("  " + "-" * 55)
    print()

    cross_results = {}
    for p_target in P_EFFS:
        p_str = str(round(p_target, 4))
        points_normal = []  # (log_n, mean_ln_phi_normal)
        points_myopic = []  # (log_n, mean_ln_phi_myopic)

        for exp_name, tdata in traces.items():
            active = tdata["active"]
            n = tdata["n_nodes"]
            group = [r for r in active if abs(r["p_eff"] - p_target) < 0.001]
            if not group:
                continue

            # Mean ln(Φ_normal)
            phi_n = [np.log(r["phi_normal"]) for r in group if r["phi_normal"] > 0]
            if phi_n:
                # Remove outliers
                arr = np.array(phi_n)
                mu, sig = np.mean(arr), np.std(arr)
                if sig > 0:
                    arr = arr[np.abs(arr - mu) < 5 * sig]
                points_normal.append((np.log(n), float(np.mean(arr)), exp_name))

            # Mean ln(Φ_myopic)
            phi_m = [np.log(r["phi_myopic"]) for r in group if r.get("phi_myopic", 0) > 0]
            if phi_m:
                arr_m = np.array(phi_m)
                mu_m, sig_m = np.mean(arr_m), np.std(arr_m)
                if sig_m > 0:
                    arr_m = arr_m[np.abs(arr_m - mu_m) < 5 * sig_m]
                points_myopic.append((np.log(n), float(np.mean(arr_m)), exp_name))

        entry = {"p_eff": p_target}

        # Fit cross-trace slope for normal
        if len(points_normal) >= 3:
            ln_ns = np.array([p[0] for p in points_normal])
            ln_phis = np.array([p[1] for p in points_normal])
            coeffs_n = np.polyfit(ln_ns, ln_phis, 1)
            entry["normal_slope_vs_logn"] = float(coeffs_n[0])
            entry["normal_points"] = [
                {"exp": p[2], "log_n": p[0], "mean_ln_phi": p[1]} for p in points_normal
            ]

        # Fit cross-trace slope for myopic
        if len(points_myopic) >= 3:
            ln_ns_m = np.array([p[0] for p in points_myopic])
            ln_phis_m = np.array([p[1] for p in points_myopic])
            coeffs_m = np.polyfit(ln_ns_m, ln_phis_m, 1)
            entry["myopic_slope_vs_logn"] = float(coeffs_m[0])
            entry["myopic_points"] = [
                {"exp": p[2], "log_n": p[0], "mean_ln_phi": p[1]} for p in points_myopic
            ]

        # Predicted β_myopic for each trace
        mean_ehs = {}
        for exp_name, tdata in traces.items():
            group = [r for r in tdata["active"] if abs(r["p_eff"] - p_target) < 0.001]
            if group:
                mean_ehs[exp_name] = float(np.mean([r["E_H"] for r in group]))
        grand_eh = float(np.mean(list(mean_ehs.values()))) if mean_ehs else 0
        entry["predicted_beta_myopic"] = float(_predicted_beta(p_target, grand_eh))
        entry["mean_eh_across_traces"] = grand_eh

        cross_results[p_str] = entry

        # Print
        ns_slope = entry.get("normal_slope_vs_logn", float("nan"))
        ms_slope = entry.get("myopic_slope_vs_logn", float("nan"))
        pred = entry["predicted_beta_myopic"]
        print(
            f"    p={p_target:.2f}: slope_normal={ns_slope:+.3f}  "
            f"slope_myopic={ms_slope:+.3f}  "
            f"predicted_β={pred:+.3f}  "
            f"(E[H]={grand_eh:.2f})"
        )

    # ══════════════════════════════════════════════════════════════════
    # TEST C: p_eff saturation profile — does (1−exp(−p/p₀)) transfer?
    # ══════════════════════════════════════════════════════════════════
    print()
    print("  TEST C: p_eff saturation profile")
    print("  " + "-" * 55)
    print()
    print("  Normalized within-trace slopes vs orbital prediction:")
    print("  (All slopes divided by slope at p=0.5 to compare shape)")
    print()

    for exp_name, tdata in traces.items():
        bp = within_results[exp_name]["by_p_eff"]
        # Normalize to p=0.5
        ref_slope = bp.get("0.5", {}).get("normal_slope", None)
        if ref_slope is None or ref_slope == 0:
            continue

        print(f"    {exp_name} (n={tdata['n_nodes']}):")
        for p_target in P_EFFS:
            p_str = str(round(p_target, 4))
            if p_str not in bp:
                continue
            obs = bp[p_str].get("normal_slope", float("nan"))
            obs_norm = obs / ref_slope if ref_slope != 0 else float("nan")
            pred_norm = (1 - np.exp(-p_target / P_0)) / (1 - np.exp(-0.5 / P_0))
            print(
                f"      p={p_target:.2f}: observed_norm={obs_norm:.3f}  predicted_norm={pred_norm:.3f}"
            )
        print()

    # ══════════════════════════════════════════════════════════════════
    # SIGN TEST: orbital formula predicts β_myopic > 0 (Φ grows with
    # scale). On CRAWDAD (Cluster class, γ > 0), does ln(Φ) grow or
    # shrink with n?
    # ══════════════════════════════════════════════════════════════════
    print()
    print("  SIGN TEST: Does ln(Φ) increase with n on CRAWDAD?")
    print("  (Orbital: yes, because β_myopic > 0 in all configs)")
    print()
    for p_target in P_EFFS:
        p_str = str(round(p_target, 4))
        if p_str in cross_results:
            ns = cross_results[p_str].get("normal_slope_vs_logn", float("nan"))
            sign = "POSITIVE (same as orbital)" if ns > 0 else "NEGATIVE (opposite)"
            print(f"    p={p_target:.2f}: slope={ns:+.3f}  → {sign}")

    # Save
    output = {
        "orbital_params": {"delta_0": DELTA_0, "p_0": P_0},
        "within_trace": within_results,
        "cross_trace": cross_results,
    }

    out_path = _HERE / "analysis_beta_myopic_results.json"
    with open(out_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
            default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x,
        )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  Saved -> {out_path.name} ({size_kb:.1f} KB)")
    print("  DONE.")


if __name__ == "__main__":
    main()
