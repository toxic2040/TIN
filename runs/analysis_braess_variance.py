"""analysis_braess_variance.py — Braess-epoch variance analysis on Mars 4-tier data.

Mars data is aggregate per (tier, epoch_day). No per-pair breakdown.
We analyze:
  1. Identify Braess epochs: where T2 DR < T1 DR (adding orbiters hurts)
  2. Variance structure of η across epochs, by tier
  3. Conditional statistics: η|(Braess epoch) vs η|(non-Braess epoch)
  4. Does the variance spike during Braess epochs?
  5. Correlation of Braess with geometric variables (dist_au, sep_deg, n_contacts)
  6. Distance-law residuals: does η deviance from distance law correlate with Braess?

Reads: mars_architecture_results.json
Writes: analysis_braess_variance_results.json
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def main():
    print()
    print("  Braess-Epoch Variance Analysis (Mars 4-Tier)")
    print("  " + "=" * 50)
    print()

    with open(_HERE / "mars_architecture_results.json") as f:
        data = json.load(f)

    results = data["results"]

    # Organize by epoch and tier
    by_epoch = defaultdict(dict)  # epoch -> {tier: row}
    by_tier = defaultdict(list)  # tier -> [rows]
    for r in results:
        by_epoch[r["epoch_day"]][r["tier"]] = r
        by_tier[r["tier"]].append(r)

    epochs = sorted(by_epoch.keys())

    # ── 1. Identify Braess epochs (T2 DR < T1 DR) ──
    braess_epochs = []
    non_braess_epochs = []
    braess_details = []
    for ep in epochs:
        tiers = by_epoch[ep]
        if 1 in tiers and 2 in tiers:
            t1_dr = tiers[1]["DR"]
            t2_dr = tiers[2]["DR"]
            is_braess = t2_dr < t1_dr
            if is_braess:
                braess_epochs.append(ep)
            else:
                non_braess_epochs.append(ep)
            braess_details.append(
                {
                    "epoch_day": ep,
                    "is_braess": is_braess,
                    "T1_DR": t1_dr,
                    "T2_DR": t2_dr,
                    "DR_loss": (t1_dr - t2_dr) / t1_dr if t1_dr > 0 else 0.0,
                    "T1_eta": tiers[1]["eta"],
                    "T2_eta": tiers[2]["eta"],
                    "eta_loss": (tiers[1]["eta"] - tiers[2]["eta"]) / tiers[1]["eta"]
                    if tiers[1]["eta"] > 0
                    else 0.0,
                    "T1_S_T": tiers[1]["S_T"],
                    "T2_S_T": tiers[2]["S_T"],
                    "dist_au": tiers[1]["dist_au"],
                    "sep_deg": tiers[1]["sep_deg"],
                }
            )

    n_braess = len(braess_epochs)
    n_total = len(epochs)
    print(f"  Braess epochs (T2 < T1): {n_braess}/{n_total} ({100 * n_braess / n_total:.0f}%)")

    # ── 2. Per-tier variance of η across all epochs ──
    print()
    print("  Per-tier η statistics across all epochs:")
    tier_stats = {}
    for tn in [1, 2, 3, 4]:
        etas = [r["eta"] for r in by_tier[tn]]
        drs = [r["DR"] for r in by_tier[tn]]
        sts = [r["S_T"] for r in by_tier[tn]]
        tier_stats[tn] = {
            "n": len(etas),
            "eta_mean": float(np.mean(etas)),
            "eta_std": float(np.std(etas)),
            "eta_cv": float(np.std(etas) / np.mean(etas)) if np.mean(etas) > 0 else 0,
            "eta_min": float(np.min(etas)),
            "eta_max": float(np.max(etas)),
            "DR_mean": float(np.mean(drs)),
            "S_T_mean": float(np.mean(sts)),
        }
        s = tier_stats[tn]
        print(
            f"    T{tn}: η={s['eta_mean']:.4f} ± {s['eta_std']:.4f} "
            f"(CV={s['eta_cv']:.3f})  "
            f"DR={s['DR_mean']:.4f}  S_T={s['S_T_mean']:.4f}"
        )

    # ── 3. Conditional: η during Braess vs non-Braess ──
    print()
    print("  Conditional η: Braess epochs vs non-Braess")
    conditional = {}
    for tn in [1, 2, 3, 4]:
        eta_braess = [by_epoch[ep][tn]["eta"] for ep in braess_epochs if tn in by_epoch[ep]]
        eta_normal = [by_epoch[ep][tn]["eta"] for ep in non_braess_epochs if tn in by_epoch[ep]]

        cond = {}
        if eta_braess:
            cond["braess_mean"] = float(np.mean(eta_braess))
            cond["braess_std"] = float(np.std(eta_braess))
            cond["braess_n"] = len(eta_braess)
        if eta_normal:
            cond["nonbraess_mean"] = float(np.mean(eta_normal))
            cond["nonbraess_std"] = float(np.std(eta_normal))
            cond["nonbraess_n"] = len(eta_normal)
        if eta_braess and eta_normal:
            cond["eta_ratio"] = float(np.mean(eta_braess) / np.mean(eta_normal))
            if np.var(eta_normal, ddof=1) > 0:
                cond["variance_ratio"] = float(
                    np.var(eta_braess, ddof=1) / np.var(eta_normal, ddof=1)
                )
            else:
                cond["variance_ratio"] = float("nan")

        conditional[tn] = cond

        b_str = f"η_braess={cond.get('braess_mean', 0):.4f}"
        n_str = f"η_other={cond.get('nonbraess_mean', 0):.4f}"
        r_str = f"ratio={cond.get('eta_ratio', 0):.3f}"
        v_str = f"var_ratio={cond.get('variance_ratio', 0):.2f}"
        print(f"    T{tn}: {b_str}  {n_str}  {r_str}  {v_str}")

    # ── 4. Braess severity vs geometric variables ──
    print()
    print("  Braess severity correlations:")
    if braess_details:
        bd_arr = [d for d in braess_details if d["T1_DR"] > 0]
        dr_losses = np.array([d["DR_loss"] for d in bd_arr])
        eta_losses = np.array([d["eta_loss"] for d in bd_arr])
        dists = np.array([d["dist_au"] for d in bd_arr])
        seps = np.array([d["sep_deg"] for d in bd_arr])
        is_braess_arr = np.array([1.0 if d["is_braess"] else 0.0 for d in bd_arr])

        # Correlations with geometric variables
        corr_dist_eta = float(np.corrcoef(dists, eta_losses)[0, 1])
        corr_sep_eta = float(np.corrcoef(seps, eta_losses)[0, 1])
        corr_dist_braess = float(np.corrcoef(dists, is_braess_arr)[0, 1])
        corr_sep_braess = float(np.corrcoef(seps, is_braess_arr)[0, 1])

        print(f"    ρ(dist_au, eta_loss):   {corr_dist_eta:+.3f}")
        print(f"    ρ(sep_deg, eta_loss):   {corr_sep_eta:+.3f}")
        print(f"    ρ(dist_au, is_braess):  {corr_dist_braess:+.3f}")
        print(f"    ρ(sep_deg, is_braess):  {corr_sep_braess:+.3f}")

        # Conjunction zone analysis (sep < 10°)
        conjunction = [d for d in braess_details if d["sep_deg"] < 10]
        opposition = [d for d in braess_details if d["sep_deg"] > 150]
        mid = [d for d in braess_details if 10 <= d["sep_deg"] <= 150]

        print()
        print("  Braess by orbital phase:")
        if conjunction:
            braess_conj = sum(1 for d in conjunction if d["is_braess"])
            print(
                f"    Conjunction (SEP<10°): {braess_conj}/{len(conjunction)} Braess "
                f"({100 * braess_conj / len(conjunction):.0f}%)"
            )
        if mid:
            braess_mid = sum(1 for d in mid if d["is_braess"])
            print(
                f"    Mid-phase (10-150°):   {braess_mid}/{len(mid)} Braess "
                f"({100 * braess_mid / len(mid):.0f}%)"
            )
        if opposition:
            braess_opp = sum(1 for d in opposition if d["is_braess"])
            print(
                f"    Opposition (SEP>150°): {braess_opp}/{len(opposition)} Braess "
                f"({100 * braess_opp / len(opposition):.0f}%)"
            )

    # ── 5. Distance law residuals ──
    print()
    print("  Distance law: ln(η) = a + b·d_AU for relay tiers (T3, T4)")
    relay_results = {}
    for tn in [3, 4]:
        rows = [r for r in by_tier[tn] if r["eta"] > 0]
        dists = np.array([r["dist_au"] for r in rows])
        ln_etas = np.array([np.log(r["eta"]) for r in rows])
        coeffs = np.polyfit(dists, ln_etas, 1)
        pred = np.polyval(coeffs, dists)
        resid = ln_etas - pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((ln_etas - ln_etas.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot

        # Check if residuals correlate with Braess status
        is_braess_tier = np.array([1.0 if r["epoch_day"] in braess_epochs else 0.0 for r in rows])
        corr_resid_braess = float(np.corrcoef(resid, is_braess_tier)[0, 1])

        relay_results[tn] = {
            "intercept": float(coeffs[1]),
            "slope": float(coeffs[0]),
            "r2": float(r2),
            "resid_std": float(np.std(resid)),
            "corr_resid_braess": corr_resid_braess,
        }

        print(
            f"    T{tn}: ln(η) = {coeffs[1]:.3f} {coeffs[0]:+.3f}·d_AU  "
            f"R²={r2:.3f}  ρ(resid, braess)={corr_resid_braess:+.3f}"
        )

    # ── 6. Tier upgrade analysis: where does each tier improve? ──
    print()
    print("  Tier upgrade analysis (mean DR gain per epoch):")
    upgrades = {}
    for src_t, dst_t, label in [(1, 2, "T1→T2"), (2, 3, "T2→T3"), (3, 4, "T3→T4")]:
        gains = []
        for ep in epochs:
            if src_t in by_epoch[ep] and dst_t in by_epoch[ep]:
                dr_src = by_epoch[ep][src_t]["DR"]
                dr_dst = by_epoch[ep][dst_t]["DR"]
                gain = dr_dst - dr_src
                gains.append(gain)
        if gains:
            gains = np.array(gains)
            n_pos = int(np.sum(gains > 0))
            upgrades[label] = {
                "mean_gain": float(np.mean(gains)),
                "std_gain": float(np.std(gains)),
                "n_positive": n_pos,
                "n_total": len(gains),
                "frac_positive": float(n_pos / len(gains)),
            }
            print(
                f"    {label}: mean ΔDR={np.mean(gains):+.4f} ± {np.std(gains):.4f}  "
                f"positive in {n_pos}/{len(gains)} epochs ({100 * n_pos / len(gains):.0f}%)"
            )

    # Save
    output = {
        "braess_summary": {
            "n_braess": n_braess,
            "n_total": n_total,
            "fraction": n_braess / n_total,
            "braess_epochs": braess_epochs,
            "non_braess_epochs": non_braess_epochs,
        },
        "tier_stats": {str(k): v for k, v in tier_stats.items()},
        "conditional": {str(k): v for k, v in conditional.items()},
        "braess_details": braess_details,
        "geometric_correlations": {
            "dist_au_vs_eta_loss": corr_dist_eta,
            "sep_deg_vs_eta_loss": corr_sep_eta,
            "dist_au_vs_is_braess": corr_dist_braess,
            "sep_deg_vs_is_braess": corr_sep_braess,
        },
        "distance_law": {str(k): v for k, v in relay_results.items()},
        "tier_upgrades": upgrades,
    }

    out_path = _HERE / "analysis_braess_variance_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved -> {out_path.name} ({os.path.getsize(out_path) / 1024:.1f} KB)")
    print("  DONE.")


if __name__ == "__main__":
    main()
