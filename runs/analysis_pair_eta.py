"""analysis_pair_eta.py — Test the per-pair η ansatz.

Ansatz: η_pair = η_mean · exp(δE[H]_pair · λ · γ)

For each CRAWDAD trace:
  1. Group results by (source, dest, p_eff)
  2. Compute η_pair = mean(eta_normal) per pair, η_mean = global mean
  3. Compute E[H]_pair and δE[H]_pair = E[H]_pair - E[H]_mean
  4. Estimate λ from eta_lyap: λ = mean(log(eta_lyap) / E[H])
  5. Estimate γ from cross-trace analysis
  6. Test: does log(η_pair / η_mean) correlate with δE[H]_pair · λ?
  7. Check if γ sign correctly predicts amplification vs damping

Reads: crawdad_contacts.Exp{1,2,3,6}_results.json
Writes: analysis_pair_eta_results.json
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

EXPERIMENTS = {
    "Exp1": {"file": "crawdad_contacts.Exp1_results.json", "n_nodes": 9},
    "Exp2": {"file": "crawdad_contacts.Exp2_results.json", "n_nodes": 12},
    "Exp3": {"file": "crawdad_contacts.Exp3_results.json", "n_nodes": 41},
    "Exp6": {"file": "crawdad_contacts.Exp6_results.json", "n_nodes": 98},
}


def _load_gamma():
    """Load p_eff-specific γ values from cross-trace analysis JSON."""
    path = _HERE / "crawdad_cross_trace_analysis.json"
    if not path.exists():
        # Fallback to approximate values if JSON not available
        return None
    with open(path) as f:
        cta = json.load(f)
    return cta["gamma_classification"]["gamma_normal"]


def _load(meta):
    path = _HERE / meta["file"]
    with open(path) as f:
        return json.load(f)


def _analyze_trace(exp_name, data, gamma_data=None):
    results = data["results"]
    n_nodes = data["trace_summary"]["n_nodes"]

    # Filter to active entries
    active = [
        r
        for r in results
        if r.get("S_T", 0) > 0
        and r.get("eta_normal", 0) > 0
        and r.get("eta_lyap", 0) > 0
        and r.get("E_H", 0) > 0
    ]

    if not active:
        return None

    p_effs = sorted(set(r["p_eff"] for r in active))
    trace_results = {}

    for p_eff in p_effs:
        group = [r for r in active if abs(r["p_eff"] - p_eff) < 0.001]
        if len(group) < 10:
            continue

        # Global means
        eta_mean = float(np.mean([r["eta_normal"] for r in group]))
        eh_mean = float(np.mean([r["E_H"] for r in group]))

        # Per-pair aggregation
        pairs = defaultdict(list)
        for r in group:
            key = (r["source"], r["dest"])
            pairs[key].append(r)

        # Estimate λ from eta_lyap
        lam_vals = []
        for r in group:
            if r["eta_lyap"] > 0 and r["E_H"] > 0:
                lam_vals.append(np.log(r["eta_lyap"]) / r["E_H"])
        if not lam_vals:
            continue
        lam_mean = float(np.mean(lam_vals))
        if lam_mean >= 0:
            continue

        # Use p_eff-specific γ from cross-trace analysis if available
        gamma = 0.9  # fallback
        if gamma_data and exp_name in gamma_data:
            p_key = str(round(p_eff, 4))
            gbp = gamma_data[exp_name].get("gamma_by_p", {})
            if p_key in gbp:
                gamma = gbp[p_key]
            else:
                # Use mean across available p values
                gvals = list(gbp.values())
                if gvals:
                    gamma = float(np.mean(gvals))

        # Per-pair test
        log_ratio_obs = []  # log(η_pair / η_mean)
        delta_eh = []  # δE[H]_pair
        predicted = []  # δE[H]_pair · λ · γ (the predicted log ratio)
        pair_labels = []
        eta_pair_vals = []
        eh_pair_vals = []

        for (src, dst), rows in pairs.items():
            if len(rows) < 3:
                continue
            eta_p = float(np.mean([r["eta_normal"] for r in rows]))
            eh_p = float(np.mean([r["E_H"] for r in rows]))

            if eta_p <= 0 or eta_mean <= 0:
                continue

            lr = np.log(eta_p / eta_mean)
            deh = eh_p - eh_mean
            pred = deh * lam_mean * gamma

            log_ratio_obs.append(lr)
            delta_eh.append(deh)
            predicted.append(pred)
            pair_labels.append(f"{src}->{dst}")
            eta_pair_vals.append(eta_p)
            eh_pair_vals.append(eh_p)

        if len(log_ratio_obs) < 5:
            continue

        log_ratio_obs = np.array(log_ratio_obs)
        delta_eh = np.array(delta_eh)
        predicted = np.array(predicted)

        # Correlation: observed log ratio vs predicted
        corr_pred = float(np.corrcoef(log_ratio_obs, predicted)[0, 1])

        # Simpler test: correlation of log(η_pair/η_mean) with δE[H]
        corr_deh = float(np.corrcoef(log_ratio_obs, delta_eh)[0, 1])

        # Variance of η_pair across pairs
        eta_pair_arr = np.array(eta_pair_vals)
        cv_pair = float(np.std(eta_pair_arr) / np.mean(eta_pair_arr))

        # Variance of E[H] across pairs
        eh_pair_arr = np.array(eh_pair_vals)
        cv_eh = float(np.std(eh_pair_arr) / np.mean(eh_pair_arr))

        # RMSE of ansatz
        residuals = log_ratio_obs - predicted
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        # R² of ansatz
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((log_ratio_obs - np.mean(log_ratio_obs)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Best linear fit: log(η_pair/η_mean) = a · δE[H] + b
        if len(delta_eh) > 2 and np.std(delta_eh) > 0:
            coeffs = np.polyfit(delta_eh, log_ratio_obs, 1)
            slope_obs = float(coeffs[0])
            slope_pred = float(lam_mean * gamma)
        else:
            slope_obs = float("nan")
            slope_pred = float(lam_mean * gamma)

        trace_results[str(round(p_eff, 4))] = {
            "n_pairs": len(log_ratio_obs),
            "eta_mean": eta_mean,
            "eh_mean": eh_mean,
            "lambda": lam_mean,
            "gamma": gamma,
            "corr_observed_vs_predicted": corr_pred,
            "corr_log_ratio_vs_delta_eh": corr_deh,
            "r2_ansatz": r2,
            "rmse_ansatz": rmse,
            "mae_ansatz": mae,
            "slope_observed": slope_obs,
            "slope_predicted": slope_pred,
            "slope_ratio": slope_obs / slope_pred if slope_pred != 0 else float("nan"),
            "cv_eta_pair": cv_pair,
            "cv_eh_pair": cv_eh,
        }

    return {
        "n_nodes": n_nodes,
        "n_active": len(active),
        "n_total": len(results),
        "by_p_eff": trace_results,
    }


def main():
    print()
    print("  Per-Pair η Ansatz Validation")
    print("  " + "=" * 50)
    print()

    gamma_data = _load_gamma()
    if gamma_data:
        print("  Loaded p_eff-specific γ from crawdad_cross_trace_analysis.json")
    else:
        print("  WARNING: using fallback γ=0.9 (cross-trace JSON not found)")
    print()

    all_results = {}
    for exp_name, meta in EXPERIMENTS.items():
        print(f"  {exp_name} (n={meta['n_nodes']})...")
        try:
            data = _load(meta)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        result = _analyze_trace(exp_name, data, gamma_data)
        if result is None:
            print("    No active data")
            continue

        all_results[exp_name] = result

        # Print summary
        for p_str, stats in result["by_p_eff"].items():
            r2 = stats["r2_ansatz"]
            corr = stats["corr_observed_vs_predicted"]
            slope_r = stats["slope_ratio"]
            cv_eta = stats["cv_eta_pair"]
            cv_eh = stats["cv_eh_pair"]
            print(
                f"    p={p_str:>6s}: R²={r2:+.3f}  ρ(obs,pred)={corr:+.3f}  "
                f"slope_ratio={slope_r:.2f}  CV(η)={cv_eta:.3f}  CV(E[H])={cv_eh:.3f}  "
                f"({stats['n_pairs']} pairs)"
            )
        print()

    # Cross-trace summary: does variance decrease with n (Cluster prediction)?
    print("  CROSS-TRACE: CV(η_pair) vs n")
    for p_target in [0.02, 0.05, 0.1, 0.3, 0.5]:
        p_str = str(round(p_target, 4))
        pts = []
        for exp_name in ["Exp1", "Exp2", "Exp3", "Exp6"]:
            if exp_name in all_results:
                bp = all_results[exp_name]["by_p_eff"]
                if p_str in bp:
                    pts.append((all_results[exp_name]["n_nodes"], bp[p_str]["cv_eta_pair"]))
        if pts:
            pts.sort()
            trend = " -> ".join(f"n={n}: {cv:.3f}" for n, cv in pts)
            # Check monotonicity
            cvs = [cv for _, cv in pts]
            decreasing = all(cvs[i] >= cvs[i + 1] for i in range(len(cvs) - 1))
            print(f"    p={p_str}: {trend}  {'DECREASING' if decreasing else 'non-monotonic'}")

    # Save
    out_path = _HERE / "analysis_pair_eta_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, allow_nan=True)
    print(f"\n  Saved -> {out_path.name} ({os.path.getsize(out_path) / 1024:.1f} KB)")
    print("  DONE.")


if __name__ == "__main__":
    main()
