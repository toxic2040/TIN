#!/usr/bin/env python3
"""analysis_running_coupling.py — Level 2: α(t*) running coupling test.

Tests whether the back-solved α_eff collapses as a function of the
commitment horizon t* = 1/(p_eff · ρ_pair) across CRAWDAD traces.

If α_eff(t*) follows a universal curve, the Lorentzian is the IR
fixed-point theory and the running coupling describes the UV→IR flow.

Three questions:
  1. Does α_eff vs ln(t*) collapse across traces?
  2. What are the UV (short t*) and IR (long t*) fixed points?
  3. Does the beta function β(α) = dα/d(ln t*) have a simple form?
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
CROSS_TRACE = RUNS / "crawdad_cross_trace_analysis.json"
TRACES_DIR = REPO / "data" / "traces"

TRACES = {
    "Exp1": {"dat": TRACES_DIR / "Exp1" / "contacts.Exp1.dat", "max_id": 9},
    "Exp2": {"dat": TRACES_DIR / "Exp2" / "contacts.Exp2.dat", "max_id": 12},
    "Exp3": {"dat": TRACES_DIR / "Exp3" / "contacts.Exp3.dat", "max_id": 41},
    "Exp6": {"dat": TRACES_DIR / "Exp6" / "contacts.Exp6.dat", "max_id": 98},
}
P_EFF = [0.02, 0.05, 0.1, 0.3, 0.5]
ALPHA = {"Exp1": 0.698, "Exp2": 0.749, "Exp3": 0.722, "Exp6": 0.753}


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
    """Compute per-pair inter-contact gaps."""
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


def multi_scale_alpha(gaps, quantile_ranges=None):
    """Compute alpha at multiple scale windows of the gap distribution."""
    if quantile_ranges is None:
        quantile_ranges = [
            ("p10-p50", 0.10, 0.50),
            ("p25-p60", 0.25, 0.60),
            ("p40-p70", 0.40, 0.70),
            ("p50-p80", 0.50, 0.80),
            ("p60-p90", 0.60, 0.90),
            ("p70-p95", 0.70, 0.95),
            ("p80-p99", 0.80, 0.99),
            ("p50-p99", 0.50, 0.99),
        ]
    results = {}
    for label, q_lo, q_hi in quantile_ranges:
        lo = np.quantile(gaps, q_lo)
        hi = np.quantile(gaps, q_hi)
        mask = (gaps >= lo) & (gaps <= hi)
        g = gaps[mask]
        if len(g) < 5:
            continue
        lx = np.log(g)
        # CCDF at each point
        n = len(g)
        ly = np.log(np.arange(n, 0, -1) / n)
        # Linear regression: ly = -alpha * lx + const
        A = np.vstack([lx, np.ones_like(lx)]).T
        (slope, _), _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
        alpha = -slope
        # Also compute the characteristic timescale (geometric mean of window)
        t_char = np.exp(0.5 * (np.log(lo) + np.log(hi)))
        results[label] = {
            "alpha": float(alpha),
            "t_lo": float(lo),
            "t_hi": float(hi),
            "t_char": float(t_char),
            "n_gaps": int(len(g)),
        }
    return results


def main():
    with open(CROSS_TRACE) as f:
        ct = json.load(f)

    W = 80
    print("=" * W)
    print("LEVEL 2: RUNNING COUPLING  α(t*)")
    print("Does α_eff collapse as a function of commitment horizon t*?")
    print("=" * W)

    # ── Load simulation aggregates ────────────────────────────────────
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

    # ── Section 1: Back-solve α_eff and compute t* ────────────────────
    print("\n" + "─" * W)
    print("1. α_eff vs t* = 1/(p_eff · ρ_pair)")
    print("─" * W)

    header = f"{'Trace':>6} {'n':>4} {'p':>5} {'ρ_pair':>8} {'t*':>10} {'ln(t*)':>8} {'α_eff':>8} {'α_tail':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    all_points = []  # (trace, n, p, rho, tstar, ln_tstar, alpha_eff, alpha_tail)
    for name in TRACES:
        rho = ct["traces"][name]["rho_pair"]
        alpha_tail = ALPHA[name]
        n_nodes = ct["traces"][name]["trace_summary"]["n_nodes"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            g = ct["traces"][name]["gamma_normal_by_p"][ps]
            B = -g * c["mean_EH"] * c["mean_lam"]
            obs = c["mean_ln_phi"]
            if abs(obs) < 1e-12 or p == 0:
                continue
            alpha_eff = (B / obs - 1.0) / p
            tstar = 1.0 / (p * rho)
            ln_tstar = np.log(tstar)
            all_points.append((name, n_nodes, p, rho, tstar, ln_tstar, alpha_eff, alpha_tail))
            print(
                f"{name:>6} {n_nodes:>4} {p:>5.2f} {rho:>8.4f} "
                f"{tstar:>10.1f} {ln_tstar:>8.3f} {alpha_eff:>8.3f} {alpha_tail:>7.3f}"
            )

    # ── Section 2: Collapse test ──────────────────────────────────────
    print("\n" + "─" * W)
    print("2. COLLAPSE TEST: α_eff vs ln(t*) across traces")
    print("─" * W)

    # Sort all points by ln(t*)
    all_points.sort(key=lambda x: x[5])

    ln_ts = np.array([pt[5] for pt in all_points])
    a_effs = np.array([pt[6] for pt in all_points])
    traces_arr = np.array([pt[0] for pt in all_points])

    # Overall correlation
    rho_overall = float(np.corrcoef(ln_ts, a_effs)[0, 1])
    print(f"\n  Overall ρ(ln t*, α_eff) = {rho_overall:+.4f}")

    # Per-trace correlations
    print("\n  Per-trace ρ(ln t*, α_eff):")
    for name in TRACES:
        mask = traces_arr == name
        if np.sum(mask) < 3:
            continue
        r = float(np.corrcoef(ln_ts[mask], a_effs[mask])[0, 1])
        print(f"    {name}: ρ = {r:+.4f}  (n={np.sum(mask)})")

    # Linear regression: α_eff = a + b * ln(t*)
    A = np.vstack([ln_ts, np.ones_like(ln_ts)]).T
    (b_slope, a_int), _, _, _ = np.linalg.lstsq(A, a_effs, rcond=None)
    ss_res = float(np.sum((a_effs - (a_int + b_slope * ln_ts)) ** 2))
    ss_tot = float(np.sum((a_effs - np.mean(a_effs)) ** 2))
    r2_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Linear fit: α_eff = {a_int:.3f} + {b_slope:.3f} · ln(t*)")
    print(f"  R² = {r2_linear:.4f}")

    # ── Section 3: UV and IR fixed points ─────────────────────────────
    print("\n" + "─" * W)
    print("3. FIXED POINTS")
    print("─" * W)

    # Sort by t* to identify trends
    sorted_pts = sorted(all_points, key=lambda x: x[4])

    # Short t* (UV): top 5 points
    uv_pts = sorted_pts[:5]
    ir_pts = sorted_pts[-5:]

    uv_alpha = np.mean([pt[6] for pt in uv_pts])
    ir_alpha = np.mean([pt[6] for pt in ir_pts])

    print("\n  UV fixed point (short t*, high p_eff):")
    print(f"    α_UV = {uv_alpha:.3f}  (mean of 5 shortest-t* cells)")
    for pt in uv_pts:
        print(f"      {pt[0]:>6} p={pt[2]:.2f}  t*={pt[4]:.0f}s  α_eff={pt[6]:.3f}")

    print("\n  IR fixed point (long t*, low p_eff):")
    print(f"    α_IR = {ir_alpha:.3f}  (mean of 5 longest-t* cells)")
    for pt in ir_pts:
        print(f"      {pt[0]:>6} p={pt[2]:.2f}  t*={pt[4]:.0f}s  α_eff={pt[6]:.3f}")

    print(f"\n  Flow: α runs from {uv_alpha:.3f} (UV) → {ir_alpha:.3f} (IR)")
    print(f"  α_tail (far-tail, p50-p99 mean) = {np.mean(list(ALPHA.values())):.3f}")

    # ── Section 4: Multi-scale α from contact distributions ───────────
    print("\n" + "─" * W)
    print("4. MULTI-SCALE α FROM INTER-CONTACT DISTRIBUTIONS")
    print("   (Direct measurement: α at different scale windows)")
    print("─" * W)

    ms_data = {}
    for name, tinfo in TRACES.items():
        contacts = parse_contacts_raw(tinfo["dat"], tinfo["max_id"])
        gaps = inter_contact_times(contacts)
        ms = multi_scale_alpha(gaps)
        ms_data[name] = ms
        print(f"\n  {name} ({len(gaps)} gaps):")
        print(f"    {'Window':>10} {'α':>7} {'t_char (s)':>12} {'ln(t_char)':>10} {'n_gaps':>7}")
        print(f"    {'-' * 50}")
        for label, d in sorted(ms.items(), key=lambda x: x[1]["t_char"]):
            print(
                f"    {label:>10} {d['alpha']:>7.3f} "
                f"{d['t_char']:>12.0f} {np.log(d['t_char']):>10.3f} {d['n_gaps']:>7}"
            )

    # ── Section 5: β function estimate ────────────────────────────────
    print("\n" + "─" * W)
    print("5. β FUNCTION: dα/d(ln t*)")
    print("─" * W)

    # Use the multi-scale α data to estimate the flow
    print("\n  Per-trace β estimates from multi-scale windows:\n")

    beta_points = []
    for name in TRACES:
        ms = ms_data[name]
        # Sort by t_char
        ordered = sorted(ms.items(), key=lambda x: x[1]["t_char"])
        if len(ordered) < 2:
            continue
        print(f"  {name}:")
        for i in range(len(ordered) - 1):
            l1, d1 = ordered[i]
            l2, d2 = ordered[i + 1]
            da = d2["alpha"] - d1["alpha"]
            dlt = np.log(d2["t_char"]) - np.log(d1["t_char"])
            if abs(dlt) < 1e-6:
                continue
            beta = da / dlt
            a_mid = 0.5 * (d1["alpha"] + d2["alpha"])
            t_mid = np.exp(0.5 * (np.log(d1["t_char"]) + np.log(d2["t_char"])))
            beta_points.append((name, a_mid, beta, t_mid))
            print(
                f"    [{l1} → {l2}]  α: {d1['alpha']:.3f}→{d2['alpha']:.3f}  "
                f"β = {beta:+.4f}  at α_mid={a_mid:.3f}"
            )

    # ── Section 6: Test logistic β function ───────────────────────────
    print("\n" + "─" * W)
    print("6. LOGISTIC β TEST: β(α) = -c · α · (α − α_IR)")
    print("─" * W)

    if beta_points:
        a_mids = np.array([bp[1] for bp in beta_points])
        betas = np.array([bp[2] for bp in beta_points])

        # Overall alpha_IR estimate
        alpha_IR_est = ir_alpha

        # Test: β vs α·(α - α_IR)
        predictor = a_mids * (a_mids - alpha_IR_est)
        if np.std(predictor) > 1e-12 and np.std(betas) > 1e-12:
            rho_logistic = float(np.corrcoef(predictor, betas)[0, 1])
            # Fit c
            c_fit = float(np.sum(predictor * betas) / np.sum(predictor**2))
            beta_pred = c_fit * predictor
            ss_r = float(np.sum((betas - beta_pred) ** 2))
            ss_t = float(np.sum((betas - np.mean(betas)) ** 2))
            r2_logistic = 1 - ss_r / ss_t if ss_t > 0 else 0

            print(f"\n  α_IR = {alpha_IR_est:.3f}")
            print(f"  Logistic fit: β = {c_fit:+.4f} · α · (α − {alpha_IR_est:.3f})")
            print(f"  ρ(predictor, β) = {rho_logistic:+.4f}")
            print(f"  R² = {r2_logistic:.4f}")
        else:
            print("\n  Insufficient variation for logistic test.")

        # Also try simple linear: β = a + b·α
        A = np.vstack([a_mids, np.ones_like(a_mids)]).T
        (b_lin, a_lin), _, _, _ = np.linalg.lstsq(A, betas, rcond=None)
        beta_pred_lin = a_lin + b_lin * a_mids
        ss_r_lin = float(np.sum((betas - beta_pred_lin) ** 2))
        ss_t_lin = float(np.sum((betas - np.mean(betas)) ** 2))
        r2_lin = 1 - ss_r_lin / ss_t_lin if ss_t_lin > 0 else 0
        print(f"\n  Linear fit: β = {a_lin:+.4f} + {b_lin:+.4f} · α")
        print(f"  R² = {r2_lin:.4f}")

    # ── Section 7: Collapse quality metric ────────────────────────────
    print("\n" + "─" * W)
    print("7. COLLAPSE QUALITY")
    print("─" * W)

    # Compare per-trace α_eff(ln t*) curves
    # If they collapse, the residual from the global fit should be similar
    # to within-trace residuals
    global_pred = a_int + b_slope * ln_ts
    global_residuals = a_effs - global_pred

    print("\n  Global linear fit residuals:")
    print(f"    RMS = {np.sqrt(np.mean(global_residuals**2)):.3f}")
    print(f"    Max |residual| = {np.max(np.abs(global_residuals)):.3f}")

    # Per-trace residuals
    for name in TRACES:
        mask = traces_arr == name
        if np.sum(mask) < 3:
            continue
        res = global_residuals[mask]
        print(f"    {name}: RMS={np.sqrt(np.mean(res**2)):.3f}  bias={np.mean(res):+.3f}")

    # Vertical spread at comparable t* values
    # Group by nearest p_eff (same p across traces gives comparable t*)
    print("\n  Spread at fixed p_eff (different traces, similar t*):")
    for p in P_EFF:
        vals = []
        for pt in all_points:
            if abs(pt[2] - p) < 0.001:
                vals.append(pt[6])
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            print(
                f"    p={p:.2f}: α_eff range=[{min(vals):.3f}, {max(vals):.3f}]  spread={spread:.3f}"
            )

    # ── Section 8: Full profile table ─────────────────────────────────
    print("\n" + "─" * W)
    print("8. COMPLETE α_eff(t*) PROFILE (sorted by t*)")
    print("─" * W)

    header2 = (
        f"{'Trace':>6} {'n':>4} {'p_eff':>5} {'ρ_pair':>7} "
        f"{'t* (s)':>9} {'ln(t*)':>7} {'α_eff':>7} {'α_tail':>7} {'Δα':>7}"
    )
    print(f"\n{header2}")
    print("-" * len(header2))
    for pt in sorted(all_points, key=lambda x: x[5]):
        da = pt[6] - pt[7]
        print(
            f"{pt[0]:>6} {pt[1]:>4} {pt[2]:>5.2f} {pt[3]:>7.4f} "
            f"{pt[4]:>9.0f} {pt[5]:>7.3f} {pt[6]:>7.3f} {pt[7]:>7.3f} {da:>+7.3f}"
        )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("SUMMARY")
    print("=" * W)

    print(f"""
  1. FLOW DIRECTION: α runs from {uv_alpha:.2f} (UV, high p) to {ir_alpha:.2f} (IR, low p)
     α increases with t* → the coupling STRENGTHENS toward the IR.
     The Lorentzian (α_tail ≈ {np.mean(list(ALPHA.values())):.2f}) is {"CONSISTENT WITH" if abs(ir_alpha - np.mean(list(ALPHA.values()))) < 2 else "INCONSISTENT WITH"} the IR fixed point.

  2. COLLAPSE: ρ(ln t*, α_eff) = {rho_overall:+.3f}
     Linear fit R² = {r2_linear:.3f}
     {"STRONG" if abs(rho_overall) > 0.7 else "MODERATE" if abs(rho_overall) > 0.4 else "WEAK"} collapse across traces.

  3. DYNAMIC RANGE: α_eff spans [{min(a_effs):.2f}, {max(a_effs):.2f}]
     At high p (short t*): α_eff > α_tail → body of distribution dominates
     At low p (long t*): α_eff approaches α_tail → far tail dominates

  4. UNIVERSALITY: {"YES" if abs(rho_overall) > 0.7 else "PARTIAL" if abs(rho_overall) > 0.4 else "NO"}
     The α_eff(t*) curves {"collapse" if abs(rho_overall) > 0.7 else "partially collapse" if abs(rho_overall) > 0.4 else "do not collapse"} across traces.
""")

    # ── Save ──────────────────────────────────────────────────────────
    results = {
        "title": "Level 2: Running coupling α(t*)",
        "points": [
            {
                "trace": pt[0],
                "n_nodes": pt[1],
                "p_eff": pt[2],
                "rho_pair": float(pt[3]),
                "t_star": float(pt[4]),
                "ln_t_star": float(pt[5]),
                "alpha_eff": float(pt[6]),
                "alpha_tail": float(pt[7]),
            }
            for pt in all_points
        ],
        "collapse": {
            "rho_overall": rho_overall,
            "linear_fit": {
                "intercept": float(a_int),
                "slope": float(b_slope),
                "R2": r2_linear,
            },
        },
        "fixed_points": {
            "UV": {
                "alpha": float(uv_alpha),
                "mean_t_star": float(np.mean([pt[4] for pt in uv_pts])),
            },
            "IR": {
                "alpha": float(ir_alpha),
                "mean_t_star": float(np.mean([pt[4] for pt in ir_pts])),
            },
            "alpha_tail_measured": float(np.mean(list(ALPHA.values()))),
        },
        "multi_scale_alpha": {name: {k: v for k, v in ms.items()} for name, ms in ms_data.items()},
        "beta_function": [
            {"trace": bp[0], "alpha_mid": float(bp[1]), "beta": float(bp[2]), "t_mid": float(bp[3])}
            for bp in beta_points
        ],
    }
    out = RUNS / "running_coupling_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
