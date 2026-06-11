#!/usr/bin/env python3
"""analysis_competing_risk_v2.py — Survival-based competing-risk test for Phi.

Revision: ξ is the *cumulative* rescue probability (survival CDF at t*),
not the instantaneous hazard.  Heavy tails (α < 1) make the hazard
vanish but the CDF substantial — the power-law IS the mechanism.

Three ξ variants tested:
  1. ξ_surv  = F(t*)                empirical CDF
  2. ξ_PL    = 1 − (t_min/t*)^α    power-law fit
  3. ξ_Lap   = Γ(1−α)·(s·t_0)^α    Laplace exponent (s = p_eff·ρ_pair/s)

Formula (sign-corrected):
  ln Φ = −γ · E[H] · λ · (1 − ξ) / (1 + ξ)
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import gamma as gamma_fn
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
TRACES_DIR = REPO / "data" / "traces"
CROSS_TRACE = RUNS / "crawdad_cross_trace_analysis.json"

TRACES = {
    "Exp1": {"dat": TRACES_DIR / "Exp1" / "contacts.Exp1.dat", "max_id": 9},
    "Exp2": {"dat": TRACES_DIR / "Exp2" / "contacts.Exp2.dat", "max_id": 12},
    "Exp3": {"dat": TRACES_DIR / "Exp3" / "contacts.Exp3.dat", "max_id": 41},
    "Exp6": {"dat": TRACES_DIR / "Exp6" / "contacts.Exp6.dat", "max_id": 98},
}

P_EFF = [0.02, 0.05, 0.1, 0.3, 0.5]

# ── helpers ──────────────────────────────────────────────────────────────


def parse_contacts_raw(path: Path, max_id: int):
    """Undirected contacts from Haggle trace, filtered to iMotes."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#%":
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
                ts, te = float(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue
            if a == b or a > max_id or b > max_id or a < 1 or b < 1:
                continue
            if te - ts < 1.0:
                continue
            out.append((str(min(a, b)), str(max(a, b)), ts, te))
    out.sort(key=lambda x: x[2])
    return out


def inter_contact_times(contacts):
    """Pool inter-contact gaps across all undirected pairs."""
    pair_starts = defaultdict(list)
    for a, b, ts, _te in contacts:
        pair_starts[(a, b)].append(ts)
    gaps = []
    for starts in pair_starts.values():
        ss = sorted(starts)
        for i in range(1, len(ss)):
            g = ss[i] - ss[i - 1]
            if g > 0:
                gaps.append(g)
    return np.array(sorted(gaps), dtype=float)


def fit_power_law_tail(gaps: np.ndarray, q_start: float = 0.5):
    """Fit P(gap > t) ~ (t_min/t)^alpha above the q_start quantile.

    Returns (alpha, t_min, R2).
    Uses Clauset-style MLE: alpha_hat = 1 + n / sum(ln(x_i/x_min)).
    """
    t_min = np.quantile(gaps, q_start)
    tail = gaps[gaps >= t_min]
    n = len(tail)
    if n < 20:
        return 0.0, t_min, 0.0
    # Hill MLE
    alpha = 1.0 + n / np.sum(np.log(tail / t_min))
    # R2 of log-log fit
    log_t = np.log(tail)
    rank = np.arange(len(gaps) - n, len(gaps))
    log_S = np.log(1.0 - rank / len(gaps))
    ok = np.isfinite(log_S) & (log_S > -15)
    if np.sum(ok) < 10:
        return alpha, t_min, 0.0
    c = np.polyfit(log_t[ok], log_S[ok], 1)
    ss_res = np.sum((log_S[ok] - np.polyval(c, log_t[ok])) ** 2)
    ss_tot = np.sum((log_S[ok] - np.mean(log_S[ok])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return alpha, float(t_min), r2


def empirical_cdf_at(gaps_sorted: np.ndarray, t: float) -> float:
    """F(t) = fraction of gaps <= t."""
    idx = np.searchsorted(gaps_sorted, t, side="right")
    return idx / len(gaps_sorted)


def diag(tag, pred, obs):
    """Print diagnostics for a prediction vector."""
    if len(pred) < 3:
        print(f"  {tag}: too few points")
        return 0.0, 0.0, 0.0
    rho = float(np.corrcoef(pred, obs)[0, 1])
    ss_r = float(np.sum((pred - obs) ** 2))
    ss_t = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
    mae = float(np.mean(np.abs(pred - obs)))
    sgn = float(np.mean(np.sign(pred) == np.sign(obs)))
    rats = pred[obs != 0] / obs[obs != 0]
    print(f"\n  {tag}:")
    print(f"    Pearson rho:     {rho:+.4f}")
    print(f"    R2 (ln Phi):     {r2:+.4f}")
    print(f"    MAE (ln Phi):     {mae:.3f}")
    print(f"    Sign agree:       {sgn:.0%}")
    if len(rats):
        print(f"    Mean ratio:       {np.mean(rats):.4f}")
        print(f"    Median ratio:     {np.median(rats):.4f}")
    return rho, r2, mae


# ── main ─────────────────────────────────────────────────────────────────


def main():
    with open(CROSS_TRACE) as f:
        ct = json.load(f)

    W = 80
    print("=" * W)
    print("COMPETING-RISK v2: SURVIVAL-BASED xi")
    print("ln Phi = -gamma * E[H] * lam * (1-xi)/(1+xi)")
    print("=" * W)

    # ── 1. Inter-contact distributions + power-law fits ──────────────────
    print("\n[1] Inter-contact distributions & power-law fits\n")

    trace_gaps = {}
    pl_fits = {}  # {name: (alpha, t_min, R2)}

    for name, info in TRACES.items():
        raw = parse_contacts_raw(info["dat"], info["max_id"])
        gaps = inter_contact_times(raw)
        trace_gaps[name] = gaps

        alpha, t_min, r2 = fit_power_law_tail(gaps, q_start=0.5)
        pl_fits[name] = (alpha, t_min, r2)

        rho = ct["traces"][name]["rho_pair"]
        dur = ct["traces"][name]["trace_summary"]["duration_total_s"]
        pcts = np.percentile(gaps, [25, 50, 75, 90, 95])
        print(
            f"  {name} (n={info['max_id']:>2}): {len(gaps)} gaps, "
            f"rho={rho:.3f}/h, dur={dur / 3600:.0f}h"
        )
        print(f"    alpha={alpha:.3f}, t_min={t_min:.0f}s, R2={r2:.3f}")
        print(
            f"    p25={pcts[0]:.0f}s  p50={pcts[1]:.0f}s  "
            f"p75={pcts[2]:.0f}s  p90={pcts[3]:.0f}s  p95={pcts[4]:.0f}s"
        )

    # ── 2. Three xi variants ─────────────────────────────────────────────
    print("\n[2] Three xi variants at each (trace, p_eff)\n")

    xi_surv = {}
    xi_pl = {}
    xi_lap = {}

    hdr = (
        f"{'Trace':>6} {'p':>5} {'t*(h)':>7} "
        f"{'xi_surv':>8} {'xi_PL':>8} {'xi_Lap':>8} "
        f"{'bal_surv':>9} {'bal_PL':>9} {'bal_Lap':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name in TRACES:
        gaps = trace_gaps[name]
        rho_h = ct["traces"][name]["rho_pair"]
        rho_s = rho_h / 3600.0
        dur = ct["traces"][name]["trace_summary"]["duration_total_s"]
        alpha, t_min, _ = pl_fits[name]

        xi_surv[name] = {}
        xi_pl[name] = {}
        xi_lap[name] = {}

        for p in P_EFF:
            ps = str(p)
            tstar = 1.0 / (p * rho_s) if rho_s > 0 else float("inf")
            flag = " !!" if tstar > dur else ""

            # Variant 1: empirical CDF
            xs = empirical_cdf_at(gaps, tstar)

            # Variant 2: power-law
            if alpha > 0 and t_min > 0 and tstar > t_min:
                xp = 1.0 - (t_min / tstar) ** alpha
            else:
                xp = xs  # fallback

            # Variant 3: Laplace exponent Gamma(1-alpha) * (s * t_min)^alpha
            s = p * rho_s  # commitment rate
            if 0 < alpha < 1 and t_min > 0 and s > 0:
                xl = gamma_fn(1 - alpha) * (s * t_min) ** alpha
                xl = min(xl, 1.0)  # clip
            else:
                xl = xs

            xi_surv[name][ps] = xs
            xi_pl[name][ps] = xp
            xi_lap[name][ps] = xl

            def bal(x):
                return (1.0 - x) / (1.0 + x)

            print(
                f"{name:>6} {p:>5.2f} {tstar / 3600:>7.1f} "
                f"{xs:>8.4f} {xp:>8.4f} {xl:>8.4f} "
                f"{bal(xs):>9.4f} {bal(xp):>9.4f} {bal(xl):>9.4f}{flag}"
            )

    # ── 3. Load simulation aggregates ────────────────────────────────────
    print("\n[3] Loading simulation aggregates...")

    cells = {}
    for name in TRACES:
        rpath = RUNS / f"crawdad_contacts.{name}_results.json"
        with open(rpath) as f:
            data = json.load(f)
        by_p = defaultdict(list)
        for r in data["results"]:
            by_p[r["p_eff"]].append(r)

        cells[name] = {}
        for p in P_EFF:
            ps = str(p)
            active = [
                r for r in by_p.get(p, []) if r["eta_lyap"] > 0 and r.get("phi_normal", 0) > 0
            ]
            if not active:
                cells[name][ps] = {"n": 0}
                continue
            EH = np.array([r["E_H"] for r in active])
            ely = np.array([r["eta_lyap"] for r in active])
            phi = np.array([r["phi_normal"] for r in active])
            lam = np.log(ely) / EH
            cells[name][ps] = {
                "n": len(active),
                "mean_EH": float(np.mean(EH)),
                "mean_lam": float(np.mean(lam)),
                "mean_ln_phi": float(np.mean(np.log(phi))),
                "median_ln_phi": float(np.median(np.log(phi))),
            }

    # ── 4. Predictions: all three xi variants, sign-corrected ────────────
    print("\n[4] Predictions (sign-corrected: ln Phi = -gamma*EH*lam*(1-xi)/(1+xi))\n")

    # Storage for each variant
    variants = {
        "surv": {"pred": [], "obs": [], "lbl": []},
        "PL": {"pred": [], "obs": [], "lbl": []},
        "Lap": {"pred": [], "obs": [], "lbl": []},
    }
    xi_maps = {"surv": xi_surv, "PL": xi_pl, "Lap": xi_lap}

    hdr4 = (
        f"{'Trace':>6} {'p':>5} {'gamma':>6} "
        f"{'xi_s':>6} {'xi_P':>6} {'xi_L':>6} "
        f"{'pred_s':>7} {'pred_P':>7} {'pred_L':>7} "
        f"{'obs':>7} {'r_s':>6} {'r_P':>6} {'r_L':>6}"
    )
    print(hdr4)
    print("-" * len(hdr4))

    for name in TRACES:
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue

            gamma = ct["traces"][name]["gamma_normal_by_p"][ps]
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            preds = {}
            ratios = {}
            for vk, xi_map in xi_maps.items():
                xi = xi_map[name][ps]
                bal = (1.0 - xi) / (1.0 + xi)
                # SIGN-CORRECTED formula
                lp = -gamma * EH * lam * bal
                preds[vk] = lp
                ratios[vk] = lp / obs_val if abs(obs_val) > 1e-12 else float("inf")
                variants[vk]["pred"].append(lp)
                variants[vk]["obs"].append(obs_val)
                variants[vk]["lbl"].append(f"{name}@{p}")

            print(
                f"{name:>6} {p:>5.2f} {gamma:>6.3f} "
                f"{xi_surv[name][ps]:>6.3f} {xi_pl[name][ps]:>6.3f} {xi_lap[name][ps]:>6.3f} "
                f"{preds['surv']:>7.2f} {preds['PL']:>7.2f} {preds['Lap']:>7.2f} "
                f"{obs_val:>7.2f} "
                f"{ratios['surv']:>6.3f} {ratios['PL']:>6.3f} {ratios['Lap']:>6.3f}"
            )

    # ── 5. Also test with gamma_myopic ───────────────────────────────────
    print("\n[5] Predictions with gamma_myopic\n")

    myopic_variants = {
        "surv_m": {"pred": [], "obs": [], "lbl": []},
        "PL_m": {"pred": [], "obs": [], "lbl": []},
    }

    for name in TRACES:
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gm = ct["traces"][name]["gamma_myopic_by_p"][ps]
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            for vk, xi_map in [("surv_m", xi_surv), ("PL_m", xi_pl)]:
                xi = xi_map[name][ps]
                bal = (1.0 - xi) / (1.0 + xi)
                lp = -gm * EH * lam * bal
                myopic_variants[vk]["pred"].append(lp)
                myopic_variants[vk]["obs"].append(obs_val)

    # ── 6. Fixed-gamma test ──────────────────────────────────────────────
    print("[6] Fixed-gamma test: gamma(p=0.3) for all p_eff\n")

    fix_variants = {
        "fix_surv": {"pred": [], "obs": []},
        "fix_PL": {"pred": [], "obs": []},
    }

    hdr6 = (
        f"{'Trace':>6} {'p':>5} {'gamma_fix':>9} "
        f"{'xi_s':>6} {'xi_P':>6} "
        f"{'pred_s':>8} {'pred_P':>8} {'obs':>8} "
        f"{'r_s':>7} {'r_P':>7}"
    )
    print(hdr6)
    print("-" * len(hdr6))

    for name in TRACES:
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            for vk, xi_map in [("fix_surv", xi_surv), ("fix_PL", xi_pl)]:
                xi = xi_map[name][ps]
                bal = (1.0 - xi) / (1.0 + xi)
                lp = -gfix * EH * lam * bal
                fix_variants[vk]["pred"].append(lp)
                fix_variants[vk]["obs"].append(obs_val)

            xs = xi_surv[name][ps]
            xp = xi_pl[name][ps]
            bs = (1 - xs) / (1 + xs)
            bp = (1 - xp) / (1 + xp)
            ps_ = -gfix * EH * lam * bs
            pp_ = -gfix * EH * lam * bp
            rs = ps_ / obs_val if abs(obs_val) > 1e-12 else 0
            rp = pp_ / obs_val if abs(obs_val) > 1e-12 else 0
            print(
                f"{name:>6} {p:>5.2f} {gfix:>9.4f} "
                f"{xs:>6.3f} {xp:>6.3f} "
                f"{ps_:>8.3f} {pp_:>8.3f} {obs_val:>8.3f} "
                f"{rs:>7.3f} {rp:>7.3f}"
            )

    # ── 7. Back-solve: what gamma_eff does each cell need? ───────────────
    print("\n[7] Back-solved gamma_eff (what gamma makes the formula exact?)\n")

    print(f"{'Trace':>6} {'p':>5} {'gamma_obs':>9} {'geff_surv':>9} {'geff_PL':>9} {'geff_Lap':>9}")
    print("-" * 58)

    for name in TRACES:
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gobs = ct["traces"][name]["gamma_normal_by_p"][ps]
            obs_val = c["mean_ln_phi"]
            EH = c["mean_EH"]
            lam = c["mean_lam"]

            geffs = {}
            for vk, xi_map in xi_maps.items():
                xi = xi_map[name][ps]
                bal = (1.0 - xi) / (1.0 + xi)
                denom = -EH * lam * bal
                geffs[vk] = obs_val / denom if abs(denom) > 1e-12 else float("inf")

            print(
                f"{name:>6} {p:>5.2f} {gobs:>9.4f} "
                f"{geffs['surv']:>9.4f} {geffs['PL']:>9.4f} {geffs['Lap']:>9.4f}"
            )

    # ── 8. Diagnostics ───────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("DIAGNOSTICS")
    print("=" * W)

    results_diag = {}

    for tag, v in variants.items():
        r, r2, m = diag(f"gamma_normal + xi_{tag}", np.array(v["pred"]), np.array(v["obs"]))
        results_diag[f"normal_{tag}"] = {"rho": r, "R2": r2, "MAE": m}

    for tag, v in myopic_variants.items():
        r, r2, m = diag(f"gamma_myopic + xi_{tag}", np.array(v["pred"]), np.array(v["obs"]))
        results_diag[f"myopic_{tag}"] = {"rho": r, "R2": r2, "MAE": m}

    for tag, v in fix_variants.items():
        r, r2, m = diag(
            f"gamma_fixed(0.3) + xi_{tag.split('_')[1]}", np.array(v["pred"]), np.array(v["obs"])
        )
        results_diag[f"fixed_{tag}"] = {"rho": r, "R2": r2, "MAE": m}

    # Baseline: bare formula ln Phi = -gamma * EH * lam (no xi correction)
    bare_pred, bare_obs = [], []
    for name in TRACES:
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            g = ct["traces"][name]["gamma_normal_by_p"][ps]
            lp = -g * c["mean_EH"] * c["mean_lam"]
            bare_pred.append(lp)
            bare_obs.append(c["mean_ln_phi"])

    r, r2, m = diag("BASELINE: -gamma*EH*lam (no xi)", np.array(bare_pred), np.array(bare_obs))
    results_diag["baseline_no_xi"] = {"rho": r, "R2": r2, "MAE": m}

    # Fixed-gamma bare baseline
    bare_fix_pred, bare_fix_obs = [], []
    for name in TRACES:
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            lp = -gfix * c["mean_EH"] * c["mean_lam"]
            bare_fix_pred.append(lp)
            bare_fix_obs.append(c["mean_ln_phi"])

    r, r2, m = diag(
        "BASELINE FIXED: -gamma(0.3)*EH*lam (no xi)",
        np.array(bare_fix_pred),
        np.array(bare_fix_obs),
    )
    results_diag["baseline_fixed_no_xi"] = {"rho": r, "R2": r2, "MAE": m}

    # ── 9. xi dynamic range summary ──────────────────────────────────────
    print("\n" + "=" * W)
    print("xi DYNAMIC RANGE")
    print("=" * W)

    for vk, xi_map in [("surv", xi_surv), ("PL", xi_pl), ("Lap", xi_lap)]:
        vals = [xi_map[n][str(p)] for n in TRACES for p in P_EFF]
        bals = [(1 - x) / (1 + x) for x in vals]
        print(f"\n  xi_{vk}:")
        print(f"    xi range:             [{min(vals):.4f}, {max(vals):.4f}]")
        print(f"    (1-xi)/(1+xi) range:  [{min(bals):.4f}, {max(bals):.4f}]")

    # ── 10. Save ─────────────────────────────────────────────────────────
    results = {
        "title": "Competing-risk v2: survival-based xi",
        "power_law_fits": {
            n: {"alpha": a, "t_min": t, "R2": r} for n, (a, t, r) in pl_fits.items()
        },
        "xi_surv": {n: {str(p): xi_surv[n][str(p)] for p in P_EFF} for n in TRACES},
        "xi_PL": {n: {str(p): xi_pl[n][str(p)] for p in P_EFF} for n in TRACES},
        "xi_Lap": {n: {str(p): xi_lap[n][str(p)] for p in P_EFF} for n in TRACES},
        "predictions": {},
        "diagnostics": results_diag,
    }

    for i, lbl in enumerate(variants["surv"]["lbl"]):
        results["predictions"][lbl] = {
            "ln_phi_obs": variants["surv"]["obs"][i],
            "ln_phi_pred_surv": variants["surv"]["pred"][i],
            "ln_phi_pred_PL": variants["PL"]["pred"][i],
            "ln_phi_pred_Lap": variants["Lap"]["pred"][i],
        }

    out = RUNS / "competing_risk_v2_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("VERDICT")
    print("=" * W)

    best_tag = max(results_diag, key=lambda k: results_diag[k]["R2"])
    best = results_diag[best_tag]
    base = results_diag.get("baseline_no_xi", {})
    base_fix = results_diag.get("baseline_fixed_no_xi", {})

    print(f"\n  Best variant:   {best_tag}")
    print(f"    R2 = {best['R2']:+.4f},  rho = {best['rho']:+.4f},  MAE = {best['MAE']:.3f}")
    print("\n  Baseline (per-p gamma, no xi):")
    print(f"    R2 = {base.get('R2', 0):+.4f},  rho = {base.get('rho', 0):+.4f}")
    print("\n  Baseline (fixed gamma, no xi):")
    print(f"    R2 = {base_fix.get('R2', 0):+.4f},  rho = {base_fix.get('rho', 0):+.4f}")

    # Does xi improve over baseline?
    for fix_tag in ["fixed_fix_surv", "fixed_fix_PL"]:
        if fix_tag in results_diag:
            d = results_diag[fix_tag]
            delta = d["R2"] - base_fix.get("R2", 0)
            print(f"\n  {fix_tag} vs fixed baseline: delta R2 = {delta:+.4f}")

    print()


if __name__ == "__main__":
    main()
