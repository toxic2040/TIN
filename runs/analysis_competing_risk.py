#!/usr/bin/env python3
"""analysis_competing_risk.py — Test the competing-risk formula for Phi.

Theory: The myopic router faces a competing-risk race at each decision:
  - Commit rate ~ p_eff * rho_pair
  - Rescue rate  = h(t)   (hazard of the inter-contact process)

At the commitment horizon  t* = 1 / (p_eff * rho_pair):
  xi(p_eff, G) = h(t*) / rho_pair

Predicted:  Phi = exp[ gamma * E[H] * lambda * (1 - xi) / (1 + xi) ]

Tests against observed Phi from 4 CRAWDAD traces x 5 p_eff (20 cells).
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


def parse_contacts_raw(path: Path, max_id: int):
    """Parse Haggle trace into undirected (pairA, pairB, start_s, end_s).

    No symmetrisation — each physical encounter counted once.
    Filters to experiment iMotes only (id <= max_id).
    """
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
            lo, hi = (str(min(a, b)), str(max(a, b)))
            out.append((lo, hi, ts, te))
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
    return np.array(gaps, dtype=float)


def hazard_at(gaps: np.ndarray, t: float, bw_frac: float = 0.10):
    """Life-table hazard h(t) with adaptive bandwidth.

    bw = max(bw_frac * t, 60 s).
    h = events_in_window / (at_risk * bw).
    """
    if len(gaps) == 0 or t <= 0:
        return 0.0
    bw = max(bw_frac * t, 60.0)
    lo, hi = t - bw / 2, t + bw / 2
    at_risk = int(np.sum(gaps >= lo))
    events = int(np.sum((gaps >= lo) & (gaps < hi)))
    if at_risk == 0:
        return 0.0
    return events / (at_risk * bw)


def survival_percentiles(gaps: np.ndarray, pcts=(10, 25, 50, 75, 90, 95, 99)):
    return {f"p{p}": float(np.percentile(gaps, p)) for p in pcts}


# ── main ─────────────────────────────────────────────────────────────────


def main():
    with open(CROSS_TRACE) as f:
        ct = json.load(f)

    W = 80
    print("=" * W)
    print("COMPETING-RISK FORMULA TEST")
    print("Phi = exp[ gamma * E[H] * lambda * (1-xi)/(1+xi) ]")
    print("=" * W)

    # ── 1. Inter-contact time distributions ──────────────────────────────
    print("\n[1] Inter-contact time distributions\n")
    trace_gaps = {}
    gap_stats = {}

    for name, info in TRACES.items():
        raw = parse_contacts_raw(info["dat"], info["max_id"])
        gaps = inter_contact_times(raw)
        trace_gaps[name] = gaps

        n_pairs = len({(a, b) for a, b, _, _ in raw})
        n_max = info["max_id"] * (info["max_id"] - 1) // 2
        rho = ct["traces"][name]["rho_pair"]

        stats = {
            "n_contacts": len(raw),
            "n_pairs_active": n_pairs,
            "n_pairs_possible": n_max,
            "pair_coverage": n_pairs / n_max,
            "n_gaps": len(gaps),
            "median_gap_s": float(np.median(gaps)) if len(gaps) else 0,
            "mean_gap_s": float(np.mean(gaps)) if len(gaps) else 0,
            "percentiles": survival_percentiles(gaps) if len(gaps) else {},
        }
        gap_stats[name] = stats

        print(
            f"  {name:5s} (n={info['max_id']:>3}):  "
            f"{len(gaps):>6} gaps, "
            f"median {stats['median_gap_s']:.0f}s ({stats['median_gap_s'] / 3600:.1f}h), "
            f"rho_pair={rho:.3f}/h, "
            f"coverage={stats['pair_coverage']:.0%}"
        )

    # ── 2. xi table ──────────────────────────────────────────────────────
    print("\n[2] xi = h(t*) / rho_pair   at   t* = 1/(p_eff * rho_pair)\n")

    xi_tbl = {}  # xi_tbl[name][p_str]
    tstar_tbl = {}
    h_tbl = {}

    hdr = f"{'Trace':>6} {'p_eff':>6} {'t*(h)':>8} {'h(t*)/s':>11} {'xi':>8} {'(1-xi)/(1+xi)':>14}"
    print(hdr)
    print("-" * len(hdr))

    for name in TRACES:
        gaps = trace_gaps[name]
        rho_h = ct["traces"][name]["rho_pair"]  # per hour
        rho_s = rho_h / 3600.0  # per second
        dur = ct["traces"][name]["trace_summary"]["duration_total_s"]

        xi_tbl[name] = {}
        tstar_tbl[name] = {}
        h_tbl[name] = {}

        for p in P_EFF:
            ps = str(p)
            tstar = 1.0 / (p * rho_s) if rho_s > 0 else float("inf")
            h = hazard_at(gaps, tstar)
            xi = h / rho_s if rho_s > 0 else 0.0
            bal = (1.0 - xi) / (1.0 + xi)
            flag = " !!" if tstar > dur else ""

            xi_tbl[name][ps] = xi
            tstar_tbl[name][ps] = tstar
            h_tbl[name][ps] = h

            print(
                f"{name:>6} {p:>6.2f} {tstar / 3600:>8.1f} {h:>11.3e} {xi:>8.4f} {bal:>14.4f}{flag}"
            )

    # ── 3. Simulation aggregates ─────────────────────────────────────────
    print("\n[3] Simulation aggregates per (trace, p_eff)\n")

    cells = {}  # cells[name][ps] = {mean_EH, mean_lam, mean_ln_phi, ...}

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
            recs = by_p.get(p, [])
            active = [r for r in recs if r["eta_lyap"] > 0 and r.get("phi_normal", 0) > 0]
            if not active:
                cells[name][ps] = {"n": 0}
                continue
            EH = np.array([r["E_H"] for r in active])
            ely = np.array([r["eta_lyap"] for r in active])
            phi = np.array([r["phi_normal"] for r in active])
            lam = np.log(ely) / EH  # per-config lambda

            cells[name][ps] = {
                "n": len(active),
                "mean_EH": float(np.mean(EH)),
                "mean_lam": float(np.mean(lam)),
                "median_lam": float(np.median(lam)),
                "mean_ln_phi": float(np.mean(np.log(phi))),
                "median_ln_phi": float(np.median(np.log(phi))),
                "mean_phi": float(np.mean(phi)),
                "median_phi": float(np.median(phi)),
            }

    hdr2 = (
        f"{'Trace':>6} {'p':>5} {'n':>6} {'mean lam':>10} "
        f"{'E[H]':>7} {'<ln Phi>':>9} {'med ln Phi':>11}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for name in TRACES:
        for p in P_EFF:
            c = cells[name].get(str(p), {})
            if c.get("n", 0) == 0:
                continue
            print(
                f"{name:>6} {p:>5.2f} {c['n']:>6} {c['mean_lam']:>10.4f} "
                f"{c['mean_EH']:>7.2f} {c['mean_ln_phi']:>9.3f} {c['median_ln_phi']:>11.3f}"
            )

    # ── 4. Prediction vs observation (gamma_normal) ──────────────────────
    print("\n[4] Prediction vs observation  (gamma = gamma_normal)\n")
    print("    ln(Phi_pred) = gamma * E[H] * lam * (1-xi)/(1+xi)\n")

    pred_ln, obs_ln, labels = [], [], []

    hdr3 = (
        f"{'Trace':>6} {'p':>5} {'gamma':>7} {'xi':>7} "
        f"{'ln_pred':>9} {'ln_obs':>9} {'ratio':>7} "
        f"{'Phi_pred':>10} {'Phi_obs':>10}"
    )
    print(hdr3)
    print("-" * len(hdr3))

    for name in TRACES:
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue

            gamma = ct["traces"][name]["gamma_normal_by_p"][ps]
            xi = xi_tbl[name][ps]
            bal = (1.0 - xi) / (1.0 + xi)
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            lp_obs = c["mean_ln_phi"]

            lp_pred = gamma * EH * lam * bal
            r = lp_pred / lp_obs if abs(lp_obs) > 1e-12 else float("inf")
            pphi = np.exp(lp_pred)
            ophi = np.exp(lp_obs)

            pred_ln.append(lp_pred)
            obs_ln.append(lp_obs)
            labels.append(f"{name}@{p}")

            print(
                f"{name:>6} {p:>5.2f} {gamma:>7.3f} {xi:>7.4f} "
                f"{lp_pred:>9.3f} {lp_obs:>9.3f} {r:>7.3f} "
                f"{pphi:>10.1f} {ophi:>10.1f}"
            )

    pred_a = np.array(pred_ln)
    obs_a = np.array(obs_ln)

    # ── 5. Also test with gamma_myopic ───────────────────────────────────
    print("\n[5] Prediction vs observation  (gamma = gamma_myopic)\n")

    pred_m, obs_m = [], []

    hdr4 = (
        f"{'Trace':>6} {'p':>5} {'gamma_m':>8} {'xi':>7} {'ln_pred':>9} {'ln_obs':>9} {'ratio':>7}"
    )
    print(hdr4)
    print("-" * len(hdr4))

    for name in TRACES:
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue

            gamma_m = ct["traces"][name]["gamma_myopic_by_p"][ps]
            xi = xi_tbl[name][ps]
            bal = (1.0 - xi) / (1.0 + xi)
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            lp_obs = c["mean_ln_phi"]

            lp_pred = gamma_m * EH * lam * bal
            r = lp_pred / lp_obs if abs(lp_obs) > 1e-12 else float("inf")

            pred_m.append(lp_pred)
            obs_m.append(lp_obs)

            print(
                f"{name:>6} {p:>5.2f} {gamma_m:>8.3f} {xi:>7.4f} "
                f"{lp_pred:>9.3f} {lp_obs:>9.3f} {r:>7.3f}"
            )

    pred_ma = np.array(pred_m)
    obs_ma = np.array(obs_m)

    # ── 6. Fixed-gamma test: can xi(p) alone explain p-dependence? ───────
    print("\n[6] Fixed-gamma test: use gamma(p=0.3) for ALL p_eff values\n")

    pred_fix, obs_fix = [], []
    for name in TRACES:
        gamma_fix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            xi = xi_tbl[name][ps]
            bal = (1.0 - xi) / (1.0 + xi)
            lp_pred = gamma_fix * c["mean_EH"] * c["mean_lam"] * bal
            pred_fix.append(lp_pred)
            obs_fix.append(c["mean_ln_phi"])

    pred_fa = np.array(pred_fix)
    obs_fa = np.array(obs_fix)

    # ── 7. Back-solve gamma_eff ──────────────────────────────────────────
    print(f"{'Trace':>6} {'p':>5} {'gamma_obs':>9} {'gamma_eff':>9} {'eff/obs':>8}")
    print("-" * 48)

    for name in TRACES:
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            gamma_obs = ct["traces"][name]["gamma_normal_by_p"][ps]
            xi = xi_tbl[name][ps]
            bal = (1.0 - xi) / (1.0 + xi)
            denom = c["mean_EH"] * c["mean_lam"] * bal
            geff = c["mean_ln_phi"] / denom if abs(denom) > 1e-12 else float("inf")
            r = geff / gamma_obs if abs(gamma_obs) > 1e-6 else float("inf")
            print(f"{name:>6} {p:>5.2f} {gamma_obs:>9.4f} {geff:>9.4f} {r:>8.3f}")

    # ── 8. Tail exponents ────────────────────────────────────────────────
    print("\n[7] Inter-contact tail exponents (P(gap>t) ~ t^-alpha)\n")

    for name in TRACES:
        gaps = trace_gaps[name]
        if len(gaps) < 50:
            continue
        sg = np.sort(gaps)
        n = len(sg)
        med = np.median(sg)
        tail_idx = np.searchsorted(sg, med)
        tail = sg[tail_idx:]
        log_t = np.log(tail)
        rank = np.arange(tail_idx, n)
        log_S = np.log(1.0 - rank / n)
        ok = np.isfinite(log_S) & np.isfinite(log_t) & (log_S > -15)
        if np.sum(ok) > 20:
            c = np.polyfit(log_t[ok], log_S[ok], 1)
            alpha = -c[0]
            pcts = survival_percentiles(gaps)
            print(
                f"  {name}: alpha = {alpha:.2f}  "
                f"(med={pcts['p50']:.0f}s, p90={pcts['p90']:.0f}s, p99={pcts['p99']:.0f}s)"
            )
        else:
            print(f"  {name}: insufficient tail data")

    # ── 9. Global diagnostics ────────────────────────────────────────────
    def diag(tag, pr, ob):
        rho = np.corrcoef(pr, ob)[0, 1] if len(pr) > 2 else 0
        ss_r = np.sum((pr - ob) ** 2)
        ss_t = np.sum((ob - np.mean(ob)) ** 2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        mae = np.mean(np.abs(pr - ob))
        sign = np.mean(np.sign(pr) == np.sign(ob))
        rats = pr[ob != 0] / ob[ob != 0]
        print(f"\n  {tag}:")
        print(f"    Pearson rho:    {rho:+.4f}")
        print(f"    R2 (ln Phi):    {r2:+.4f}")
        print(f"    MAE (ln Phi):    {mae:.3f}")
        print(f"    Sign agree:      {sign:.0%}")
        if len(rats) > 0:
            print(f"    Mean ratio:      {np.mean(rats):.3f}")
            print(f"    Median ratio:    {np.median(rats):.3f}")
        return rho, r2, mae

    print("\n" + "=" * W)
    print("DIAGNOSTICS")
    print("=" * W)

    rho_n, r2_n, mae_n = diag("gamma_normal (per-p)", pred_a, obs_a)
    rho_m, r2_m, mae_m = diag("gamma_myopic (per-p)", pred_ma, obs_ma)
    rho_f, r2_f, mae_f = diag("gamma_normal FIXED (p=0.3 for all)", pred_fa, obs_fa)

    # ── 10. Save ─────────────────────────────────────────────────────────
    results = {
        "title": "Competing-risk test: Phi = exp[gamma*EH*lam*(1-xi)/(1+xi)]",
        "gap_stats": gap_stats,
        "xi": xi_tbl,
        "t_star": tstar_tbl,
        "h_at_tstar": h_tbl,
        "predictions_normal": {
            l: {"pred": float(p), "obs": float(o)} for l, p, o in zip(labels, pred_ln, obs_ln)
        },
        "diagnostics": {
            "gamma_normal_perp": {"rho": float(rho_n), "R2": float(r2_n), "MAE": float(mae_n)},
            "gamma_myopic_perp": {"rho": float(rho_m), "R2": float(r2_m), "MAE": float(mae_m)},
            "gamma_normal_fixed": {"rho": float(rho_f), "R2": float(r2_f), "MAE": float(mae_f)},
        },
    }
    out = RUNS / "competing_risk_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("VERDICT")
    print("=" * W)

    if r2_n > 0.80:
        grade = "STRONG"
    elif r2_n > 0.40:
        grade = "MODERATE"
    elif r2_n > 0.0:
        grade = "WEAK"
    else:
        grade = "NEGATIVE R2 — structurally wrong"

    print(f"\n  gamma_normal:  R2 = {r2_n:+.3f}  rho = {rho_n:+.3f}  -> {grade}")
    print(f"  gamma_myopic:  R2 = {r2_m:+.3f}  rho = {rho_m:+.3f}")
    print(f"  fixed-gamma:   R2 = {r2_f:+.3f}  rho = {rho_f:+.3f}")

    xi_vals = [xi_tbl[n][str(p)] for n in TRACES for p in P_EFF]
    bal_vals = [(1 - x) / (1 + x) for x in xi_vals]
    print(f"\n  xi range:             [{min(xi_vals):.4f}, {max(xi_vals):.4f}]")
    print(f"  (1-xi)/(1+xi) range:  [{min(bal_vals):.4f}, {max(bal_vals):.4f}]")

    print()


if __name__ == "__main__":
    main()
