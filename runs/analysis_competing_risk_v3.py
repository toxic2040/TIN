#!/usr/bin/env python3
"""analysis_competing_risk_v3.py — Pareto-model competing-risk test.

Key insight: for P(gap > t) ~ t^{-alpha} with alpha < 1,
  h(t) = alpha/t   =>   xi = h(t*)/rho_pair = alpha * p_eff

The T-dependence cancels exactly (infinite mean, finite-time rho_pair).
Formula:  ln Phi = -gamma * E[H] * lambda * (1 - alpha*p) / (1 + alpha*p)

Tests whether the far-tail exponent alpha closes the 20% baseline gap.
Also tests: can (alpha, rho_pair) PREDICT gamma directly?
"""

from __future__ import annotations

import json
from collections import defaultdict
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


def parse_contacts_raw(path, max_id):
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


def fit_tail_alpha(gaps, q_lo=0.50, q_hi=0.99):
    """Log-log slope of survival function between quantiles.

    Returns alpha such that P(gap > t) ~ t^{-alpha} in [q_lo, q_hi].
    """
    n = len(gaps)
    if n < 50:
        return 0.0, 0.0

    # Survival function at each quantile boundary
    t_lo = np.quantile(gaps, q_lo)
    t_hi = np.quantile(gaps, q_hi)
    S_lo = 1 - q_lo
    S_hi = 1 - q_hi

    if t_hi <= t_lo or S_hi <= 0 or S_lo <= 0:
        return 0.0, 0.0

    alpha = -np.log(S_hi / S_lo) / np.log(t_hi / t_lo)
    return float(alpha), float(t_lo)


def fit_tail_alpha_regression(gaps, q_start=0.50):
    """Log-log regression of survival function above q_start quantile.

    More robust than two-point estimate.
    """
    n = len(gaps)
    t_min = np.quantile(gaps, q_start)
    tail = gaps[gaps >= t_min]
    nt = len(tail)
    if nt < 30:
        return 0.0, 0.0, 0.0

    log_t = np.log(tail)
    # Ranks in the full dataset
    start_rank = n - nt
    ranks = np.arange(start_rank, n)
    log_S = np.log(1.0 - ranks / n)
    ok = np.isfinite(log_S) & (log_S > -12) & np.isfinite(log_t)
    if np.sum(ok) < 20:
        return 0.0, 0.0, 0.0

    c = np.polyfit(log_t[ok], log_S[ok], 1)
    alpha = -c[0]
    ss_res = np.sum((log_S[ok] - np.polyval(c, log_t[ok])) ** 2)
    ss_tot = np.sum((log_S[ok] - np.mean(log_S[ok])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return float(alpha), float(t_min), float(r2)


def fit_multi_scale_alpha(gaps):
    """Fit alpha at multiple scale ranges to show scale dependence."""
    ranges = [
        ("p50-p90", 0.50, 0.90),
        ("p50-p95", 0.50, 0.95),
        ("p50-p99", 0.50, 0.99),
        ("p75-p95", 0.75, 0.95),
        ("p75-p99", 0.75, 0.99),
        ("p90-p99", 0.90, 0.99),
    ]
    results = {}
    for label, qlo, qhi in ranges:
        a, _ = fit_tail_alpha(gaps, qlo, qhi)
        results[label] = a
    return results


def diag(tag, pred, obs):
    if len(pred) < 3:
        return 0.0, 0.0, 0.0
    rho = float(np.corrcoef(pred, obs)[0, 1])
    ss_r = float(np.sum((pred - obs) ** 2))
    ss_t = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
    mae = float(np.mean(np.abs(pred - obs)))
    sgn = float(np.mean(np.sign(pred) == np.sign(obs)))
    rats = pred[obs != 0] / obs[obs != 0]
    print(f"\n  {tag}:")
    print(f"    rho  = {rho:+.4f}")
    print(f"    R2   = {r2:+.4f}")
    print(f"    MAE  = {mae:.3f}")
    print(f"    sign = {sgn:.0%}")
    if len(rats):
        print(f"    mean ratio  = {np.mean(rats):.4f}")
        print(f"    med  ratio  = {np.median(rats):.4f}")
    return rho, r2, mae


# ── main ─────────────────────────────────────────────────────────────────


def main():
    with open(CROSS_TRACE) as f:
        ct = json.load(f)

    W = 80
    print("=" * W)
    print("COMPETING-RISK v3: PARETO MODEL  xi = alpha * p_eff")
    print("ln Phi = -gamma * E[H] * lam * (1 - alpha*p) / (1 + alpha*p)")
    print("=" * W)

    # ── 1. Multi-scale alpha estimates ───────────────────────────────────
    print("\n[1] Power-law tail exponent alpha at multiple scales\n")

    trace_gaps = {}
    trace_alpha = {}

    for name, info in TRACES.items():
        raw = parse_contacts_raw(info["dat"], info["max_id"])
        gaps = inter_contact_times(raw)
        trace_gaps[name] = gaps

        multi = fit_multi_scale_alpha(gaps)
        alpha_reg, t_min, r2 = fit_tail_alpha_regression(gaps, q_start=0.5)

        print(f"  {name} (n={info['max_id']:>2}, {len(gaps)} gaps):")
        for label, a in multi.items():
            print(f"    {label:>10}: alpha = {a:.3f}")
        print(f"    regression: alpha = {alpha_reg:.3f} (R2={r2:.3f})")

        # Use the p50-p99 estimate as the "far tail" alpha
        alpha_far = multi.get("p50-p99", alpha_reg)
        trace_alpha[name] = alpha_far
        print(f"    >>> using alpha_far = {alpha_far:.3f}")

    # ── 2. xi = alpha * p_eff ────────────────────────────────────────────
    print("\n[2] xi = alpha * p_eff\n")

    xi_pareto = {}

    hdr = f"{'Trace':>6} {'alpha':>6} {'p':>5} {'xi':>7} {'(1-xi)/(1+xi)':>14}"
    print(hdr)
    print("-" * len(hdr))

    for name in TRACES:
        alpha = trace_alpha[name]
        xi_pareto[name] = {}
        for p in P_EFF:
            xi = alpha * p
            bal = (1 - xi) / (1 + xi)
            xi_pareto[name][str(p)] = xi
            print(f"{name:>6} {alpha:>6.3f} {p:>5.2f} {xi:>7.4f} {bal:>14.4f}")

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
            }

    # ── 4A. Per-p gamma with Pareto xi ───────────────────────────────────
    print("\n[4A] Predictions: per-p gamma + xi = alpha*p (sign-corrected)\n")

    pred_pp, obs_pp, lbl_pp = [], [], []

    hdr4 = (
        f"{'Trace':>6} {'p':>5} {'gamma':>6} {'alpha':>6} "
        f"{'xi':>6} {'bal':>7} "
        f"{'pred':>8} {'obs':>8} {'ratio':>7}"
    )
    print(hdr4)
    print("-" * len(hdr4))

    for name in TRACES:
        alpha = trace_alpha[name]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gamma = ct["traces"][name]["gamma_normal_by_p"][ps]
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            xi = alpha * p
            bal = (1 - xi) / (1 + xi)
            pred_val = -gamma * EH * lam * bal
            ratio = pred_val / obs_val if abs(obs_val) > 1e-12 else 0

            pred_pp.append(pred_val)
            obs_pp.append(obs_val)
            lbl_pp.append(f"{name}@{p}")

            print(
                f"{name:>6} {p:>5.2f} {gamma:>6.3f} {alpha:>6.3f} "
                f"{xi:>6.3f} {bal:>7.4f} "
                f"{pred_val:>8.3f} {obs_val:>8.3f} {ratio:>7.3f}"
            )

    # ── 4B. Fixed gamma + Pareto xi ──────────────────────────────────────
    print("\n[4B] Fixed gamma(p=0.3) + xi = alpha*p\n")

    pred_fix, obs_fix = [], []

    hdr4b = f"{'Trace':>6} {'p':>5} {'gfix':>6} {'xi':>6} {'pred':>8} {'obs':>8} {'ratio':>7}"
    print(hdr4b)
    print("-" * len(hdr4b))

    for name in TRACES:
        alpha = trace_alpha[name]
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            xi = alpha * p
            bal = (1 - xi) / (1 + xi)
            pred_val = -gfix * EH * lam * bal
            ratio = pred_val / obs_val if abs(obs_val) > 1e-12 else 0

            pred_fix.append(pred_val)
            obs_fix.append(obs_val)

            print(
                f"{name:>6} {p:>5.2f} {gfix:>6.3f} {xi:>6.3f} "
                f"{pred_val:>8.3f} {obs_val:>8.3f} {ratio:>7.3f}"
            )

    # ── 4C. Gamma-myopic + Pareto xi ─────────────────────────────────────
    pred_my, obs_my = [], []
    for name in TRACES:
        alpha = trace_alpha[name]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gm = ct["traces"][name]["gamma_myopic_by_p"][ps]
            xi = alpha * p
            bal = (1 - xi) / (1 + xi)
            pred_val = -gm * c["mean_EH"] * c["mean_lam"] * bal
            pred_my.append(pred_val)
            obs_my.append(c["mean_ln_phi"])

    # ── 5. Baselines ─────────────────────────────────────────────────────
    bare_pp, bare_obs = [], []
    bare_fix_pp, bare_fix_obs = [], []

    for name in TRACES:
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gamma = ct["traces"][name]["gamma_normal_by_p"][ps]
            lp_bare = -gamma * c["mean_EH"] * c["mean_lam"]
            bare_pp.append(lp_bare)
            bare_obs.append(c["mean_ln_phi"])

            lp_bare_fix = -gfix * c["mean_EH"] * c["mean_lam"]
            bare_fix_pp.append(lp_bare_fix)
            bare_fix_obs.append(c["mean_ln_phi"])

    # ── 6. Can alpha predict gamma? ──────────────────────────────────────
    print("\n[5] Can (alpha, rho_pair) predict gamma?\n")

    alphas = []
    rhos = []
    gammas_mean = []

    for name in TRACES:
        alpha = trace_alpha[name]
        rho = ct["traces"][name]["rho_pair"]
        g_vals = list(ct["traces"][name]["gamma_normal_by_p"].values())
        g_mean = np.mean(g_vals)

        alphas.append(alpha)
        rhos.append(rho)
        gammas_mean.append(g_mean)

        print(f"  {name}: alpha={alpha:.3f}, rho_pair={rho:.3f}, <gamma>={g_mean:.3f}")

    alphas = np.array(alphas)
    rhos = np.array(rhos)
    gammas_mean = np.array(gammas_mean)

    # Correlations
    print("\n  Correlations with mean gamma_normal:")
    rho_alpha = np.corrcoef(alphas, gammas_mean)[0, 1]
    rho_rho = np.corrcoef(rhos, gammas_mean)[0, 1]
    rho_prod = np.corrcoef(alphas * rhos, gammas_mean)[0, 1]
    rho_inv = np.corrcoef(1.0 / alphas, gammas_mean)[0, 1]
    rho_rho_T = np.corrcoef(
        [ct["traces"][n]["rho_pair"] * ct["traces"][n]["duration_hours"] for n in TRACES],
        gammas_mean,
    )[0, 1]

    print(f"    rho(alpha, gamma)        = {rho_alpha:+.3f}")
    print(f"    rho(rho_pair, gamma)     = {rho_rho:+.3f}")
    print(f"    rho(alpha*rho, gamma)    = {rho_prod:+.3f}")
    print(f"    rho(1/alpha, gamma)      = {rho_inv:+.3f}")
    print(f"    rho(rho*T, gamma)        = {rho_rho_T:+.3f}")

    # Contacts per pair (total)
    cpp = np.array(
        [ct["traces"][n]["rho_pair"] * ct["traces"][n]["duration_hours"] for n in TRACES]
    )
    rho_cpp = np.corrcoef(cpp, gammas_mean)[0, 1]
    print(f"    rho(contacts/pair, gamma)= {rho_cpp:+.3f}")

    # Log-contacts per pair
    rho_lcpp = np.corrcoef(np.log(cpp), gammas_mean)[0, 1]
    print(f"    rho(ln(cpp), gamma)      = {rho_lcpp:+.3f}")

    # What about alpha alone predicting gamma_mean?
    # Linear fit: gamma = a + b*alpha
    if len(alphas) > 2:
        c_ag = np.polyfit(alphas, gammas_mean, 1)
        pred_g = np.polyval(c_ag, alphas)
        ss_r = np.sum((gammas_mean - pred_g) ** 2)
        ss_t = np.sum((gammas_mean - np.mean(gammas_mean)) ** 2)
        r2_ag = 1 - ss_r / ss_t if ss_t > 0 else 0
        print(f"\n  Linear fit: gamma = {c_ag[1]:.3f} + {c_ag[0]:+.3f} * alpha")
        print(f"    R2 = {r2_ag:.3f} (n=4, treat with caution)")

    # ── 7. Back-solved effective alpha ───────────────────────────────────
    print("\n[6] What alpha_eff makes the formula exact at each (trace, p)?")
    print("    Solve: obs = -gamma * EH * lam * (1 - alpha_eff*p) / (1 + alpha_eff*p)\n")

    print(f"{'Trace':>6} {'p':>5} {'alpha_fit':>9} {'alpha_obs':>9}")
    print("-" * 38)

    aeff_all = []
    for name in TRACES:
        alpha_obs = trace_alpha[name]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gamma = ct["traces"][name]["gamma_normal_by_p"][ps]
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            # obs = -gamma*EH*lam * (1-a*p)/(1+a*p)
            # let B = -gamma*EH*lam (bare prediction)
            B = -gamma * EH * lam
            if abs(B) < 1e-12 or p == 0:
                continue
            # obs/B = (1-a*p)/(1+a*p)
            # let r = obs/B
            r = obs_val / B
            # r*(1+a*p) = 1-a*p  =>  r + r*a*p = 1 - a*p  =>  a*p*(r+1) = 1-r
            if abs(r + 1) < 1e-12:
                continue
            a_eff = (1 - r) / (p * (r + 1))
            aeff_all.append(a_eff)

            print(f"{name:>6} {p:>5.2f} {a_eff:>9.3f} {alpha_obs:>9.3f}")

    aeff_arr = np.array(aeff_all)
    print(
        f"\n  alpha_eff:  mean={np.mean(aeff_arr):.3f}, "
        f"median={np.median(aeff_arr):.3f}, "
        f"std={np.std(aeff_arr):.3f}"
    )
    print(f"  alpha_obs:  {', '.join(f'{trace_alpha[n]:.3f}' for n in TRACES)}")

    # ── 8. Test with alpha_eff = median back-solved value ────────────────
    alpha_univ = np.median(aeff_arr)
    print(f"\n[7] Universal alpha_eff = {alpha_univ:.3f} test\n")

    pred_univ, obs_univ = [], []
    pred_univ_fix, obs_univ_fix = [], []

    hdr7 = f"{'Trace':>6} {'p':>5} {'xi':>6} {'pred':>8} {'obs':>8} {'ratio':>7}"
    print(hdr7)
    print("-" * len(hdr7))

    for name in TRACES:
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gamma = ct["traces"][name]["gamma_normal_by_p"][ps]
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            xi = alpha_univ * p
            bal = (1 - xi) / (1 + xi)

            # Per-p gamma
            lp = -gamma * EH * lam * bal
            pred_univ.append(lp)
            obs_univ.append(obs_val)

            # Fixed gamma
            lp_fix = -gfix * EH * lam * bal
            pred_univ_fix.append(lp_fix)
            obs_univ_fix.append(obs_val)

            ratio = lp / obs_val if abs(obs_val) > 1e-12 else 0
            print(f"{name:>6} {p:>5.2f} {xi:>6.3f} {lp:>8.3f} {obs_val:>8.3f} {ratio:>7.3f}")

    # ── DIAGNOSTICS ──────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("DIAGNOSTICS")
    print("=" * W)

    all_diags = {}

    r, r2, m = diag(
        "Per-p gamma + xi=alpha*p (per-trace alpha)", np.array(pred_pp), np.array(obs_pp)
    )
    all_diags["perp_alpha_pertrace"] = {"rho": r, "R2": r2, "MAE": m}

    r, r2, m = diag(
        "Fixed gamma + xi=alpha*p (per-trace alpha)", np.array(pred_fix), np.array(obs_fix)
    )
    all_diags["fix_alpha_pertrace"] = {"rho": r, "R2": r2, "MAE": m}

    r, r2, m = diag(
        "Myopic gamma + xi=alpha*p (per-trace alpha)", np.array(pred_my), np.array(obs_my)
    )
    all_diags["myopic_alpha_pertrace"] = {"rho": r, "R2": r2, "MAE": m}

    r, r2, m = diag(
        f"Per-p gamma + xi=alpha_univ*p (alpha={alpha_univ:.3f})",
        np.array(pred_univ),
        np.array(obs_univ),
    )
    all_diags["perp_alpha_universal"] = {"rho": r, "R2": r2, "MAE": m}

    r, r2, m = diag(
        f"Fixed gamma + xi=alpha_univ*p (alpha={alpha_univ:.3f})",
        np.array(pred_univ_fix),
        np.array(obs_univ_fix),
    )
    all_diags["fix_alpha_universal"] = {"rho": r, "R2": r2, "MAE": m}

    r, r2, m = diag(
        "BASELINE: -gamma*EH*lam (per-p gamma, no xi)", np.array(bare_pp), np.array(bare_obs)
    )
    all_diags["baseline_perp"] = {"rho": r, "R2": r2, "MAE": m}

    r, r2, m = diag(
        "BASELINE FIXED: -gamma(0.3)*EH*lam (no xi)", np.array(bare_fix_pp), np.array(bare_fix_obs)
    )
    all_diags["baseline_fix"] = {"rho": r, "R2": r2, "MAE": m}

    # ── xi RANGE ─────────────────────────────────────────────────────────
    print("\n  xi = alpha*p range:")
    xi_all = [trace_alpha[n] * p for n in TRACES for p in P_EFF]
    bal_all = [(1 - x) / (1 + x) for x in xi_all]
    print(f"    xi:              [{min(xi_all):.4f}, {max(xi_all):.4f}]")
    print(f"    (1-xi)/(1+xi):   [{min(bal_all):.4f}, {max(bal_all):.4f}]")

    # ── SAVE ─────────────────────────────────────────────────────────────
    results = {
        "title": "Competing-risk v3: Pareto xi = alpha * p_eff",
        "trace_alpha": trace_alpha,
        "alpha_eff_universal": float(alpha_univ),
        "alpha_eff_all": [float(x) for x in aeff_all],
        "xi_pareto": xi_pareto,
        "diagnostics": all_diags,
    }
    out = RUNS / "competing_risk_v3_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("VERDICT")
    print("=" * W)

    best = max(all_diags.items(), key=lambda x: x[1]["R2"])
    base_pp = all_diags.get("baseline_perp", {})
    base_fix = all_diags.get("baseline_fix", {})

    print(f"\n  Best variant: {best[0]}")
    print(f"    R2={best[1]['R2']:+.4f}  rho={best[1]['rho']:+.4f}  MAE={best[1]['MAE']:.3f}")

    for tag in ["perp_alpha_pertrace", "fix_alpha_pertrace", "perp_alpha_universal"]:
        d = all_diags.get(tag, {})
        delta_pp = d.get("R2", 0) - base_pp.get("R2", 0)
        delta_fix = d.get("R2", 0) - base_fix.get("R2", 0)
        print(f"\n  {tag}:")
        print(f"    vs per-p baseline:  dR2 = {delta_pp:+.4f}")
        print(f"    vs fixed baseline:  dR2 = {delta_fix:+.4f}")

    print(f"\n  alpha values: {', '.join(f'{n}={trace_alpha[n]:.3f}' for n in TRACES)}")
    print(f"  alpha_eff (back-solved universal): {alpha_univ:.3f}")
    print()


if __name__ == "__main__":
    main()
