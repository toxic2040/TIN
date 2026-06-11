#!/usr/bin/env python3
"""analysis_competing_risk_v4.py — Lorentzian balance test.

User's revised formula:
  xi = alpha * p_eff
  f(xi) = 1 / (1 + xi)
  ln Phi = -gamma * E[H] * lambda * f(xi)

Compared against:
  v3 Pade:    f(xi) = (1 - xi) / (1 + xi)
  Baseline:   f = 1  (no correction)
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

# Alpha values from v3 (p50-p99 far-tail estimates)
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


def diag(tag, pred, obs):
    if len(pred) < 3:
        return 0.0, 0.0, 0.0
    rho = float(np.corrcoef(pred, obs)[0, 1])
    ss_r = float(np.sum((pred - obs) ** 2))
    ss_t = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
    mae = float(np.mean(np.abs(pred - obs)))
    rats = pred[obs != 0] / obs[obs != 0]
    print(f"\n  {tag}:")
    print(f"    rho  = {rho:+.4f}")
    print(f"    R2   = {r2:+.4f}")
    print(f"    MAE  = {mae:.3f}")
    if len(rats):
        print(f"    mean ratio = {np.mean(rats):.4f}")
        print(f"    med  ratio = {np.median(rats):.4f}")
    return rho, r2, mae


def main():
    with open(CROSS_TRACE) as f:
        ct = json.load(f)

    W = 80
    print("=" * W)
    print("COMPETING-RISK v4: LORENTZIAN BALANCE  f(xi) = 1/(1+xi)")
    print("ln Phi = -gamma * E[H] * lam / (1 + alpha*p)")
    print("=" * W)

    # ── Load simulation aggregates ───────────────────────────────────────
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

    # ── Balance function comparison table ────────────────────────────────
    # f1 = 1                         (baseline)
    # f2 = 1/(1+xi)                  (Lorentzian, v4)
    # f3 = (1-xi)/(1+xi)            (Pade, v3)

    print("\n[1] Balance functions at xi = alpha * p_eff\n")

    hdr = (
        f"{'Trace':>6} {'p':>5} {'alpha':>6} {'xi':>6} "
        f"{'f=1':>6} {'1/(1+x)':>8} {'(1-x)/(1+x)':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name in TRACES:
        a = ALPHA[name]
        for p in P_EFF:
            xi = a * p
            f1 = 1.0
            f2 = 1.0 / (1.0 + xi)
            f3 = (1.0 - xi) / (1.0 + xi)
            print(f"{name:>6} {p:>5.2f} {a:>6.3f} {xi:>6.3f} {f1:>6.3f} {f2:>8.4f} {f3:>12.4f}")

    # ── Predictions: three balance functions ──────────────────────────────
    # For each: per-p gamma, fixed gamma(0.3), and universal-alpha variants

    results_pred = {}
    all_diags = {}

    configs = [
        # (label, gamma_mode, alpha_mode, balance_fn)
        ("baseline_perp", "per_p", None, lambda xi: 1.0),
        ("baseline_fix", "fix03", None, lambda xi: 1.0),
        ("lorentz_perp", "per_p", "per_trace", lambda xi: 1.0 / (1.0 + xi)),
        ("lorentz_fix", "fix03", "per_trace", lambda xi: 1.0 / (1.0 + xi)),
        ("pade_perp", "per_p", "per_trace", lambda xi: (1.0 - xi) / (1.0 + xi)),
        ("pade_fix", "fix03", "per_trace", lambda xi: (1.0 - xi) / (1.0 + xi)),
    ]

    # Also test universal alpha (mean of per-trace alphas)
    alpha_univ = np.mean(list(ALPHA.values()))
    configs += [
        ("lorentz_fix_auniv", "fix03", "universal", lambda xi: 1.0 / (1.0 + xi)),
        ("pade_fix_auniv", "fix03", "universal", lambda xi: (1.0 - xi) / (1.0 + xi)),
    ]

    for label, gmode, amode, bfn in configs:
        pred, obs = [], []
        for name in TRACES:
            gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
            for p in P_EFF:
                ps = str(p)
                c = cells[name].get(ps, {})
                if c.get("n", 0) == 0:
                    continue

                if gmode == "per_p":
                    g = ct["traces"][name]["gamma_normal_by_p"][ps]
                else:
                    g = gfix

                if amode == "per_trace":
                    xi = ALPHA[name] * p
                elif amode == "universal":
                    xi = alpha_univ * p
                else:
                    xi = 0.0

                f = bfn(xi)
                lp = -g * c["mean_EH"] * c["mean_lam"] * f
                pred.append(lp)
                obs.append(c["mean_ln_phi"])

        results_pred[label] = (np.array(pred), np.array(obs))

    # ── Print the main comparison table ──────────────────────────────────
    print("\n[2] Lorentzian predictions: ln Phi = -gamma*EH*lam / (1 + alpha*p)\n")

    hdr2 = (
        f"{'Trace':>6} {'p':>5} {'gamma':>6} {'xi':>6} "
        f"{'f':>6} {'pred':>8} {'obs':>8} {'ratio':>7} "
        f"{'base':>8} {'b_ratio':>7}"
    )
    print(hdr2)
    print("-" * len(hdr2))

    for name in TRACES:
        a = ALPHA[name]
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            g = ct["traces"][name]["gamma_normal_by_p"][ps]
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            xi = a * p
            f_lor = 1.0 / (1.0 + xi)
            pred_lor = -g * EH * lam * f_lor
            pred_base = -g * EH * lam
            r_lor = pred_lor / obs_val if abs(obs_val) > 1e-12 else 0
            r_base = pred_base / obs_val if abs(obs_val) > 1e-12 else 0

            print(
                f"{name:>6} {p:>5.2f} {g:>6.3f} {xi:>6.3f} "
                f"{f_lor:>6.3f} {pred_lor:>8.3f} {obs_val:>8.3f} {r_lor:>7.3f} "
                f"{pred_base:>8.3f} {r_base:>7.3f}"
            )

    # ── Fixed gamma table ────────────────────────────────────────────────
    print("\n[3] Fixed gamma(p=0.3) + Lorentzian\n")

    hdr3 = f"{'Trace':>6} {'p':>5} {'gfix':>6} {'xi':>6} {'pred':>8} {'obs':>8} {'ratio':>7}"
    print(hdr3)
    print("-" * len(hdr3))

    for name in TRACES:
        a = ALPHA[name]
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            EH = c["mean_EH"]
            lam = c["mean_lam"]
            obs_val = c["mean_ln_phi"]

            xi = a * p
            f_lor = 1.0 / (1.0 + xi)
            pred_lor = -gfix * EH * lam * f_lor
            r = pred_lor / obs_val if abs(obs_val) > 1e-12 else 0

            print(
                f"{name:>6} {p:>5.2f} {gfix:>6.3f} {xi:>6.3f} "
                f"{pred_lor:>8.3f} {obs_val:>8.3f} {r:>7.3f}"
            )

    # ── Back-solve alpha_eff for Lorentzian ──────────────────────────────
    print("\n[4] Back-solved alpha_eff (Lorentzian)")
    print("    obs = -gamma*EH*lam / (1 + alpha_eff*p)\n")

    print(f"{'Trace':>6} {'p':>5} {'a_eff':>8} {'a_obs':>6}")
    print("-" * 32)

    aeff_all = []
    for name in TRACES:
        aobs = ALPHA[name]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            g = ct["traces"][name]["gamma_normal_by_p"][ps]
            B = -g * c["mean_EH"] * c["mean_lam"]  # bare prediction
            obs_val = c["mean_ln_phi"]
            if abs(obs_val) < 1e-12 or p == 0:
                continue
            # obs = B / (1 + a*p)  =>  1 + a*p = B/obs  =>  a = (B/obs - 1)/p
            a_eff = (B / obs_val - 1.0) / p
            aeff_all.append(a_eff)
            print(f"{name:>6} {p:>5.2f} {a_eff:>8.3f} {aobs:>6.3f}")

    aeff_arr = np.array(aeff_all)
    print(
        f"\n  alpha_eff: mean={np.mean(aeff_arr):.3f}, "
        f"median={np.median(aeff_arr):.3f}, std={np.std(aeff_arr):.3f}"
    )

    # ── Fit optimal alpha per trace (least squares) ──────────────────────
    print("\n[5] Least-squares optimal alpha per trace\n")

    for name in TRACES:
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        best_a, best_ss = 0, float("inf")
        for a_try in np.linspace(0.01, 3.0, 3000):
            ss = 0
            for p in P_EFF:
                ps = str(p)
                c = cells[name].get(ps, {})
                if c.get("n", 0) == 0:
                    continue
                g = ct["traces"][name]["gamma_normal_by_p"][ps]
                B = -g * c["mean_EH"] * c["mean_lam"]
                obs_val = c["mean_ln_phi"]
                pred = B / (1 + a_try * p)
                ss += (pred - obs_val) ** 2
            if ss < best_ss:
                best_ss = ss
                best_a = a_try

        # Also try with fixed gamma
        best_a_fix, best_ss_fix = 0, float("inf")
        for a_try in np.linspace(0.01, 3.0, 3000):
            ss = 0
            for p in P_EFF:
                ps = str(p)
                c = cells[name].get(ps, {})
                if c.get("n", 0) == 0:
                    continue
                B = -gfix * c["mean_EH"] * c["mean_lam"]
                obs_val = c["mean_ln_phi"]
                pred = B / (1 + a_try * p)
                ss += (pred - obs_val) ** 2
            if ss < best_ss_fix:
                best_ss_fix = ss
                best_a_fix = a_try

        print(
            f"  {name}: alpha_opt(per-p g)={best_a:.3f}, "
            f"alpha_opt(fix g)={best_a_fix:.3f}, alpha_obs={ALPHA[name]:.3f}"
        )

    # ── Global optimal alpha ─────────────────────────────────────────────
    print("\n[6] Global least-squares alpha (all traces, fixed gamma)\n")

    best_a_global, best_ss_global = 0, float("inf")
    for a_try in np.linspace(0.01, 5.0, 5000):
        ss = 0
        for name in TRACES:
            gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
            for p in P_EFF:
                ps = str(p)
                c = cells[name].get(ps, {})
                if c.get("n", 0) == 0:
                    continue
                B = -gfix * c["mean_EH"] * c["mean_lam"]
                obs_val = c["mean_ln_phi"]
                pred = B / (1 + a_try * p)
                ss += (pred - obs_val) ** 2
        if ss < best_ss_global:
            best_ss_global = ss
            best_a_global = a_try

    print(f"  alpha_opt_global = {best_a_global:.3f}")

    # Evaluate with global optimal alpha
    pred_gopt, obs_gopt = [], []
    for name in TRACES:
        gfix = ct["traces"][name]["gamma_normal_by_p"]["0.3"]
        for p in P_EFF:
            ps = str(p)
            c = cells[name].get(ps, {})
            if c.get("n", 0) == 0:
                continue
            B = -gfix * c["mean_EH"] * c["mean_lam"]
            pred_gopt.append(B / (1 + best_a_global * p))
            obs_gopt.append(c["mean_ln_phi"])

    # ── DIAGNOSTICS ──────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("DIAGNOSTICS")
    print("=" * W)

    for label, (pr, ob) in sorted(results_pred.items()):
        r, r2, m = diag(label, pr, ob)
        all_diags[label] = {"rho": r, "R2": r2, "MAE": m}

    r, r2, m = diag(
        f"lorentz_fix_alpha_opt (alpha={best_a_global:.3f})",
        np.array(pred_gopt),
        np.array(obs_gopt),
    )
    all_diags["lorentz_fix_alpha_opt"] = {"rho": r, "R2": r2, "MAE": m}

    # ── COMPARISON TABLE ─────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("COMPARISON TABLE")
    print("=" * W)

    print(f"\n  {'Variant':<40} {'R2':>8} {'MAE':>7} {'rho':>7}")
    print("  " + "-" * 64)

    for label in [
        "baseline_perp",
        "baseline_fix",
        "pade_perp",
        "pade_fix",
        "pade_fix_auniv",
        "lorentz_perp",
        "lorentz_fix",
        "lorentz_fix_auniv",
        "lorentz_fix_alpha_opt",
    ]:
        d = all_diags.get(label, {})
        if d:
            print(f"  {label:<40} {d['R2']:>+8.4f} {d['MAE']:>7.3f} {d['rho']:>+7.4f}")

    # ── SAVE ─────────────────────────────────────────────────────────────
    results = {
        "title": "Competing-risk v4: Lorentzian f(xi) = 1/(1+xi)",
        "alpha_per_trace": ALPHA,
        "alpha_universal": float(alpha_univ),
        "alpha_opt_global": float(best_a_global),
        "alpha_eff_backsolved": {
            "mean": float(np.mean(aeff_arr)),
            "median": float(np.median(aeff_arr)),
            "std": float(np.std(aeff_arr)),
        },
        "diagnostics": all_diags,
    }
    out = RUNS / "competing_risk_v4_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("VERDICT")
    print("=" * W)

    lor = all_diags.get("lorentz_fix", {})
    pad = all_diags.get("pade_fix", {})
    base = all_diags.get("baseline_fix", {})
    opt = all_diags.get("lorentz_fix_alpha_opt", {})

    print(
        f"\n  Baseline (no xi):      R2 = {base.get('R2', 0):+.4f}  MAE = {base.get('MAE', 0):.3f}"
    )
    print(f"  Pade (1-xi)/(1+xi):    R2 = {pad.get('R2', 0):+.4f}  MAE = {pad.get('MAE', 0):.3f}")
    print(f"  Lorentzian 1/(1+xi):   R2 = {lor.get('R2', 0):+.4f}  MAE = {lor.get('MAE', 0):.3f}")
    print(f"  Lorentz + opt alpha:   R2 = {opt.get('R2', 0):+.4f}  MAE = {opt.get('MAE', 0):.3f}")

    delta_lor = lor.get("R2", 0) - base.get("R2", 0)
    delta_pad = pad.get("R2", 0) - base.get("R2", 0)
    delta_opt = opt.get("R2", 0) - base.get("R2", 0)

    print("\n  dR2 vs baseline:")
    print(f"    Lorentzian: {delta_lor:+.4f}")
    print(f"    Pade:       {delta_pad:+.4f}")
    print(f"    Lorentz+opt:{delta_opt:+.4f}")
    print()


if __name__ == "__main__":
    main()
