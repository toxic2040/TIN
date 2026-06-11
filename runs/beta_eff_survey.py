"""run_beta_eff_survey.py — Compute beta_eff (effective Boltzmann temperature)
across all available multi-family configurations.

Reanalyses existing data from:
  - runs/per_path_cov_results.json  (256 configs, 180 with var_H > 0)
  - runs/emj_bypass_results.json    (EMJ bypass routing fractions)

For each config with var_H > 0 AND a reconstructable 2-family hop distribution:
  - Reconstruct the hop count distribution from E_H, var_H, n_paths
  - Compute utility U_k = H_k * lyapunov_exp  (at lambda=0, hardware)
  - Compute routing fraction f_k = n_paths_k / total_paths
  - Fit beta_eff = ln(f_1/f_2) / (U_1 - U_2)

Also processes EMJ bypass as a special case.

Outputs: runs/beta_eff_survey_results.json
"""

import json
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def _reconstruct_two_family(E_H, var_H, n_paths):
    """Reconstruct a 2-family hop distribution from moments.

    Given E[H] and Var[H], find integer hop counts H_1 < H_2 and
    fraction f_1 such that:
        E[H] = f_1 * H_1 + f_2 * H_2
        Var[H] = f_1 * (H_1 - E_H)^2 + f_2 * (H_2 - E_H)^2

    Returns (H_1, H_2, n_1, n_2) or None if no valid decomposition found.
    """
    if var_H <= 0:
        return None

    best = None
    best_err = float("inf")

    # Try all reasonable pairs of integer hop counts
    for h1 in range(1, 20):
        for h2 in range(h1 + 1, 20):
            delta = h2 - h1
            # Solve for f_1: E_H = f_1*H_1 + (1-f_1)*H_2
            f1 = (h2 - E_H) / delta
            if f1 < 0.001 or f1 > 0.999:
                continue

            # Check variance: Var = f1*(h1-E)^2 + f2*(h2-E)^2
            f2 = 1.0 - f1
            var_pred = f1 * (h1 - E_H) ** 2 + f2 * (h2 - E_H) ** 2
            err = abs(var_pred - var_H) / max(var_H, 1e-12)

            if err < best_err:
                best_err = err
                n1 = round(f1 * n_paths)
                n2 = n_paths - n1
                if n1 > 0 and n2 > 0:
                    best = (h1, h2, n1, n2, err)

    if best is None or best[4] > 0.05:  # tolerance: 5% relative error
        return None

    return best[:4]


def main():
    print("=" * 72)
    print("BETA_EFF SURVEY — Effective Boltzmann Temperature")
    print("=" * 72)

    # ── Load per_path_cov data ──────────────────────────────────────────
    cov_path = _HERE / "per_path_cov_results.json"
    with open(cov_path) as f:
        cov_data = json.load(f)

    print(f"\n  Loaded {len(cov_data)} configs from per_path_cov_results.json")

    # Filter: var_H > 0 and feasible
    candidates = [d for d in cov_data if d.get("feasible", False) and d.get("var_H", 0) > 0]
    print(f"  Configs with var_H > 0: {len(candidates)}")

    # ── Process each candidate ──────────────────────────────────────────
    results = []
    skipped_reconstruct = 0
    skipped_dU_zero = 0

    for cfg in candidates:
        E_H = cfg["E_H"]
        var_H = cfg["var_H"]
        n_paths = cfg["n_paths"]
        lyap = cfg["lyapunov_exp"]
        body = cfg["body"]
        constellation = cfg.get("constellation", "?")
        families = cfg.get("families", "?")
        n_sats = cfg.get("n_sats", "?")
        mode = cfg.get("mode", "?")
        p_eff = cfg.get("p_eff")

        # Reconstruct 2-family distribution
        decomp = _reconstruct_two_family(E_H, var_H, n_paths)
        if decomp is None:
            skipped_reconstruct += 1
            continue

        H_1, H_2, n_1, n_2 = decomp
        f_1 = n_1 / n_paths
        f_2 = n_2 / n_paths

        # Utility at lambda=0 (hardware): U_k = H_k * lyapunov_exp
        U_1 = H_1 * lyap
        U_2 = H_2 * lyap

        dU = U_1 - U_2
        if abs(dU) < 1e-15:
            skipped_dU_zero += 1
            continue

        # beta_eff = ln(f_1/f_2) / (U_1 - U_2)
        if f_1 <= 0 or f_2 <= 0:
            continue
        beta_eff = math.log(f_1 / f_2) / dU

        # Determine which family is dominant
        f_dominant = max(f_1, f_2)
        dominant_H = H_1 if f_1 >= f_2 else H_2

        # Q_k = exp(H_k * lyapunov_exp)
        Q_1 = math.exp(H_1 * lyap)
        Q_2 = math.exp(H_2 * lyap)

        row = {
            "body": body,
            "constellation": constellation,
            "families": families,
            "n_sats": n_sats,
            "mode": mode,
            "p_eff": p_eff,
            "H_1": H_1,
            "H_2": H_2,
            "n_1": n_1,
            "n_2": n_2,
            "f_1": round(f_1, 6),
            "f_2": round(f_2, 6),
            "U_1": round(U_1, 6),
            "U_2": round(U_2, 6),
            "Q_1": round(Q_1, 6),
            "Q_2": round(Q_2, 6),
            "dU": round(dU, 6),
            "beta_eff": round(beta_eff, 6),
            "sign": "positive" if beta_eff > 0 else "negative",
            "f_dominant": round(f_dominant, 6),
            "dominant_H": dominant_H,
            "var_H": round(var_H, 6),
            "E_H": round(E_H, 4),
            "lyapunov_exp": round(lyap, 6),
        }
        results.append(row)

    print(f"  Reconstructed 2-family: {len(results)} configs")
    print(f"  Skipped (no valid decomposition): {skipped_reconstruct}")
    print(f"  Skipped (dU ~ 0): {skipped_dU_zero}")

    # ── Process EMJ bypass ──────────────────────────────────────────────
    emj_path = _HERE / "emj_bypass_results.json"
    with open(emj_path) as f:
        emj_data = json.load(f)

    print("\n  Processing EMJ bypass data...")

    # Extract combined_hw from the two_by_two grid
    combined_hw = emj_data["two_by_two"]["combined_hw"]
    pd = combined_hw["path_distribution"]
    rc = combined_hw["route_counts"]
    total = combined_hw["n_paths"]
    lyap_emj = combined_hw["lyapunov"]  # note: stored as "lyapunov" not "lyapunov_exp"

    # bypass = H=4, relay = H=7
    H_bypass = 4
    H_relay = 7
    n_bypass = pd[str(H_bypass)]
    n_relay = pd[str(H_relay)]
    f_bypass = n_bypass / total
    f_relay = n_relay / total

    # Q_k = exp(H_k * lyapunov_exp)
    Q_bypass = math.exp(H_bypass * lyap_emj)
    Q_relay = math.exp(H_relay * lyap_emj)

    # U_k = log(Q_k) = H_k * lyapunov_exp  (at lambda=0)
    U_bypass = H_bypass * lyap_emj
    U_relay = H_relay * lyap_emj

    dU_emj = U_bypass - U_relay
    beta_eff_emj = math.log(f_bypass / f_relay) / dU_emj

    emj_row = {
        "body": "EMJ",
        "constellation": "combined_hw",
        "families": "bypass+relay",
        "n_sats": "N/A",
        "mode": "hardware",
        "p_eff": None,
        "H_1": H_bypass,
        "H_2": H_relay,
        "n_1": n_bypass,
        "n_2": n_relay,
        "f_1": round(f_bypass, 6),
        "f_2": round(f_relay, 6),
        "U_1": round(U_bypass, 6),
        "U_2": round(U_relay, 6),
        "Q_1": round(Q_bypass, 6),
        "Q_2": round(Q_relay, 6),
        "dU": round(dU_emj, 6),
        "beta_eff": round(beta_eff_emj, 6),
        "sign": "positive" if beta_eff_emj > 0 else "negative",
        "f_dominant": round(max(f_bypass, f_relay), 6),
        "dominant_H": H_bypass if f_bypass >= f_relay else H_relay,
        "var_H": round(combined_hw["var_H"], 6),
        "E_H": round(combined_hw["E_H"], 4),
        "lyapunov_exp": round(lyap_emj, 6),
    }
    results.append(emj_row)

    # Also process combined_prop
    combined_prop = emj_data["two_by_two"]["combined_prop"]
    lyap_prop = combined_prop["lyapunov"]
    pd_prop = combined_prop["path_distribution"]
    total_prop = combined_prop["n_paths"]
    n_bypass_p = pd_prop[str(H_bypass)]
    n_relay_p = pd_prop[str(H_relay)]
    f_bypass_p = n_bypass_p / total_prop
    f_relay_p = n_relay_p / total_prop
    Q_bypass_p = math.exp(H_bypass * lyap_prop)
    Q_relay_p = math.exp(H_relay * lyap_prop)
    U_bypass_p = H_bypass * lyap_prop
    U_relay_p = H_relay * lyap_prop
    dU_prop = U_bypass_p - U_relay_p
    beta_eff_prop = math.log(f_bypass_p / f_relay_p) / dU_prop

    emj_prop_row = {
        "body": "EMJ",
        "constellation": "combined_prop",
        "families": "bypass+relay",
        "n_sats": "N/A",
        "mode": "propellant",
        "p_eff": None,
        "H_1": H_bypass,
        "H_2": H_relay,
        "n_1": n_bypass_p,
        "n_2": n_relay_p,
        "f_1": round(f_bypass_p, 6),
        "f_2": round(f_relay_p, 6),
        "U_1": round(U_bypass_p, 6),
        "U_2": round(U_relay_p, 6),
        "Q_1": round(Q_bypass_p, 6),
        "Q_2": round(Q_relay_p, 6),
        "dU": round(dU_prop, 6),
        "beta_eff": round(beta_eff_prop, 6),
        "sign": "positive" if beta_eff_prop > 0 else "negative",
        "f_dominant": round(max(f_bypass_p, f_relay_p), 6),
        "dominant_H": H_bypass if f_bypass_p >= f_relay_p else H_relay,
        "var_H": round(combined_prop["var_H"], 6),
        "E_H": round(combined_prop["E_H"], 4),
        "lyapunov_exp": round(lyap_prop, 6),
    }
    results.append(emj_prop_row)

    # ── Sort by beta_eff ────────────────────────────────────────────────
    results.sort(key=lambda r: r["beta_eff"])

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("SUMMARY TABLE — sorted by beta_eff")
    print("=" * 72)

    hdr = (
        f"  {'body':>10s} | {'constellation':>14s} | {'families':>20s} | "
        f"{'n':>3s} | {'mode':>10s} | {'p_eff':>5s} | "
        f"{'beta_eff':>10s} | {'sign':>8s} | "
        f"{'f_dom':>6s} | {'dom_H':>5s} | {'var_H':>8s} | {'E_H':>6s} | "
        f"{'H1':>2s} {'H2':>2s} | {'f1':>6s} {'f2':>6s} | {'U1':>8s} {'U2':>8s}"
    )
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for r in results:
        n_str = str(r["n_sats"]) if r["n_sats"] != "N/A" else "N/A"
        p_str = f"{r['p_eff']:.2f}" if r["p_eff"] is not None else "phys"
        print(
            f"  {r['body']:>10s} | {r['constellation']:>14s} | {r['families']:>20s} | "
            f"{n_str:>3s} | {r['mode']:>10s} | {p_str:>5s} | "
            f"{r['beta_eff']:>10.4f} | {r['sign']:>8s} | "
            f"{r['f_dominant']:>6.4f} | {r['dominant_H']:>5d} | "
            f"{r['var_H']:>8.4f} | {r['E_H']:>6.2f} | "
            f"{r['H_1']:>2d} {r['H_2']:>2d} | "
            f"{r['f_1']:>6.4f} {r['f_2']:>6.4f} | "
            f"{r['U_1']:>8.4f} {r['U_2']:>8.4f}"
        )

    # ── Statistics ──────────────────────────────────────────────────────
    n_neg = sum(1 for r in results if r["beta_eff"] < 0)
    n_pos = sum(1 for r in results if r["beta_eff"] > 0)
    n_total = len(results)

    beta_vals = [r["beta_eff"] for r in results]
    beta_arr = np.array(beta_vals)

    print(f"\n{'=' * 72}")
    print("STATISTICS")
    print("=" * 72)
    print(f"  Total configs with beta_eff: {n_total}")
    print(f"  Negative temperature (beta < 0): {n_neg} ({100 * n_neg / n_total:.1f}%)")
    print(f"  Positive temperature (beta > 0): {n_pos} ({100 * n_pos / n_total:.1f}%)")
    print(f"  beta_eff range: [{beta_arr.min():.4f}, {beta_arr.max():.4f}]")
    print(f"  beta_eff mean: {beta_arr.mean():.4f}")
    print(f"  beta_eff median: {float(np.median(beta_arr)):.4f}")
    print(f"  beta_eff std: {beta_arr.std():.4f}")

    # ── By body ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("BY BODY (physics mode only)")
    print("=" * 72)

    physics = [r for r in results if r["mode"] == "physics"]
    by_body = {}
    for r in physics:
        by_body.setdefault(r["body"], []).append(r)

    print(
        f"  {'body':>10s} | {'count':>5s} | {'mean_beta':>10s} | "
        f"{'min_beta':>10s} | {'max_beta':>10s} | {'n_neg':>5s} | {'n_pos':>5s}"
    )
    print(f"  {'-' * 70}")

    for body_name in sorted(by_body):
        betas = [r["beta_eff"] for r in by_body[body_name]]
        ba = np.array(betas)
        nn = sum(1 for b in betas if b < 0)
        np_ = sum(1 for b in betas if b > 0)
        print(
            f"  {body_name:>10s} | {len(betas):>5d} | {ba.mean():>10.4f} | "
            f"{ba.min():>10.4f} | {ba.max():>10.4f} | {nn:>5d} | {np_:>5d}"
        )

    # ── By mode ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("BY MODE")
    print("=" * 72)

    by_mode = {}
    for r in results:
        key = r["mode"] if r["p_eff"] is None else f"uniform_p{r['p_eff']}"
        by_mode.setdefault(key, []).append(r)

    print(f"  {'mode':>15s} | {'count':>5s} | {'mean_beta':>10s} | {'n_neg':>5s} | {'n_pos':>5s}")
    print(f"  {'-' * 55}")

    for mode_name in sorted(by_mode):
        betas = [r["beta_eff"] for r in by_mode[mode_name]]
        ba = np.array(betas)
        nn = sum(1 for b in betas if b < 0)
        np_ = sum(1 for b in betas if b > 0)
        print(f"  {mode_name:>15s} | {len(betas):>5d} | {ba.mean():>10.4f} | {nn:>5d} | {np_:>5d}")

    # ── EMJ detail ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("EMJ BYPASS DETAIL")
    print("=" * 72)

    for r in results:
        if r["body"] == "EMJ":
            print(f"  Mode: {r['mode']}")
            print(
                f"    bypass (H={r['H_1']}): n={r['n_1']}, f={r['f_1']:.4f}, "
                f"Q={r['Q_1']:.6f}, U={r['U_1']:.6f}"
            )
            print(
                f"    relay  (H={r['H_2']}): n={r['n_2']}, f={r['f_2']:.4f}, "
                f"Q={r['Q_2']:.6f}, U={r['U_2']:.6f}"
            )
            print(f"    dU = {r['dU']:.6f}")
            print(f"    ln(f_bypass/f_relay) = {math.log(r['f_1'] / r['f_2']):.6f}")
            print(f"    beta_eff = {r['beta_eff']:.6f}")
            print(
                f"    sign: {r['sign']} ({'negative temperature' if r['beta_eff'] < 0 else 'positive temperature'})"
            )
            print()

    # ── Interpretation ──────────────────────────────────────────────────
    print(f"{'=' * 72}")
    print("INTERPRETATION")
    print("=" * 72)
    print("  beta_eff > 0: system prefers lower-utility (shorter, fewer-hop) paths")
    print("               -> Boltzmann-like selection, 'cold' routing")
    print("  beta_eff < 0: system prefers higher-utility (longer, more-hop) paths")
    print("               -> population inversion, 'negative temperature'")
    print("  |beta_eff| large: strong preference (nearly deterministic)")
    print("  |beta_eff| small: weak preference (nearly uniform routing)")

    # ── Save results ────────────────────────────────────────────────────
    out_path = _HERE / "beta_eff_survey_results.json"
    out = {
        "description": "beta_eff (effective Boltzmann temperature) survey across multi-family configs",
        "sources": [
            "runs/per_path_cov_results.json",
            "runs/emj_bypass_results.json",
        ],
        "n_configs": n_total,
        "n_negative_temperature": n_neg,
        "n_positive_temperature": n_pos,
        "beta_eff_range": [float(beta_arr.min()), float(beta_arr.max())],
        "beta_eff_mean": float(beta_arr.mean()),
        "beta_eff_median": float(np.median(beta_arr)),
        "beta_eff_std": float(beta_arr.std()),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved -> {out_path.name} ({size_kb:.1f} KB, {n_total} entries)")


if __name__ == "__main__":
    main()
