"""Historical diagnostics for archived Earth-Mars architecture model rows.

The calculations below preserve the original row comparisons. Printed text is
descriptive only: it does not establish a causal routing mechanism, architecture
optimum, mission benefit, or design rule.

Extracts:
  1. Synodic profiles (DR, S_T, η vs epoch) per tier
  2. T1/T2 DR comparison historically labeled as a Braess effect
  3. Low-separation row comparison and zero-S_T counts
  4. Bookkeeping decomposition diagnostics
  5. Historical γ-proxy diagnostic per tier (not a classifier)
  6. Distance-η regression (ln η vs d_AU)
  7. Tier-to-tier Δη values from L4/L5 at each epoch
"""

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def load():
    with open(_HERE / "mars_architecture_results.json") as f:
        data = json.load(f)
    return data["config"], data["results"]


def by_tier(results, tier):
    return sorted([r for r in results if r["tier"] == tier], key=lambda r: r["epoch_day"])


def main():
    cfg, results = load()
    tiers = [1, 2, 3, 4]
    tier_data = {t: by_tier(results, t) for t in tiers}
    tier_desc = cfg["tiers"]

    print()
    print("=" * 80)
    print("  HISTORICAL MARS ARCHITECTURE MODEL DIAGNOSTICS — DESCRIPTIVE ONLY")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────
    # 1. Per-tier summary statistics
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 1. TIER SUMMARY ──────────────────────────────────────────────────┐")
    print(
        f"  {'Tier':>4} {'DR_mean':>8} {'DR_std':>8} {'DR_min':>8} {'DR_max':>8} "
        f"{'η_mean':>8} {'S_T_mean':>8} {'S_T=0':>8}"
    )
    print(f"  {'─' * 72}")
    for t in tiers:
        td = tier_data[t]
        drs = [r["DR"] for r in td]
        etas = [r["eta"] for r in td]
        sts = [r["S_T"] for r in td]
        bl = sum(1 for r in td if r["S_T"] == 0)
        print(
            f"  {t:>4} {np.mean(drs):>8.4f} {np.std(drs):>8.4f} {min(drs):>8.4f} {max(drs):>8.4f} "
            f"{np.mean(etas):>8.4f} {np.mean(sts):>8.4f} {bl:>5}/{len(td)}"
        )

    # ─────────────────────────────────────────────────────────────────
    # 2. T1/T2 head-to-head comparison
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 2. HISTORICAL T1/T2 DR COMPARISON ───────────────────────────────┐")
    t1_wins = 0
    t2_wins = 0
    braess_deltas = []
    for r1, r2 in zip(tier_data[1], tier_data[2]):
        delta = r2["DR"] - r1["DR"]
        braess_deltas.append(delta)
        if r1["DR"] > r2["DR"] and r1["DR"] > 0:
            t1_wins += 1
        elif r2["DR"] > r1["DR"] and r2["DR"] > 0:
            t2_wins += 1

    n_epochs = len(tier_data[1])
    print(f"  T1 DR > T2 DR: {t1_wins}/{n_epochs} epochs ({100 * t1_wins / n_epochs:.0f}%)")
    print(f"  T2 DR > T1 DR: {t2_wins}/{n_epochs} epochs ({100 * t2_wins / n_epochs:.0f}%)")
    print(f"  Mean ΔDR (T2−T1): {np.mean(braess_deltas):+.4f}")
    print(
        f"  At opposition (day 0): T1={tier_data[1][0]['DR']:.4f}  T2={tier_data[2][0]['DR']:.4f}"
    )

    # S_T comparison
    st_t1 = [r["S_T"] for r in tier_data[1]]
    st_t2 = [r["S_T"] for r in tier_data[2]]
    print("\n  S_T comparison:")
    print(f"    T1 mean S_T = {np.mean(st_t1):.4f}   T2 mean S_T = {np.mean(st_t2):.4f}")
    print("    The loaded T2 rows have higher mean S_T and lower mean DR than T1.")

    eta_t1 = [r["eta"] for r in tier_data[1] if r["eta"] > 0]
    eta_t2 = [r["eta"] for r in tier_data[2] if r["eta"] > 0]
    print(f"    T1 mean η = {np.mean(eta_t1):.4f}   T2 mean η = {np.mean(eta_t2):.4f}")
    if np.mean(eta_t2) < np.mean(eta_t1):
        print(
            f"    Mean η(T2) is {100 * (1 - np.mean(eta_t2) / np.mean(eta_t1)):.1f}% lower than mean η(T1)."
        )
        print("    This bookkeeping comparison does not identify a causal mechanism.")

    # ─────────────────────────────────────────────────────────────────
    # 3. Conjunction analysis
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 3. HISTORICAL LOW-SEPARATION ROWS ───────────────────────────────┐")
    for sep_thresh in [20, 10, 5]:
        conj = {t: [r for r in tier_data[t] if r["sep_deg"] < sep_thresh] for t in tiers}
        print(f"\n  SEP < {sep_thresh}°:")
        for t in tiers:
            if not conj[t]:
                print(f"    T{t}: no epochs in zone")
                continue
            drs = [r["DR"] for r in conj[t]]
            sts = [r["S_T"] for r in conj[t]]
            bl = sum(1 for s in sts if s == 0)
            print(
                f"    T{t}: {len(conj[t])} epochs, DR_mean={np.mean(drs):.4f}, "
                f"S_T_mean={np.mean(sts):.4f}, zero-S_T rows={bl}"
            )

    # Minimum SEP epoch detail
    min_sep_day = min(results, key=lambda r: r["sep_deg"])["epoch_day"]
    print(f"\n  Deepest conjunction (day {min_sep_day}):")
    for t in tiers:
        r = [x for x in tier_data[t] if x["epoch_day"] == min_sep_day][0]
        print(
            f"    T{t}: SEP={r['sep_deg']:.1f}°  d={r['dist_au']:.2f} AU  "
            f"S_T={r['S_T']:.4f}  η={r['eta']:.4f}  DR={r['DR']:.4f}  "
            f"contacts={r['n_contacts']} (relay={r['n_relay']})"
        )

    # ─────────────────────────────────────────────────────────────────
    # 4. Distance-η regression per tier
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 4. DISTANCE-η REGRESSION ─────────────────────────────────────────┐")
    print("  Model: ln(η) = a + b·d_AU")
    print(f"  {'Tier':>4} {'a':>8} {'b':>8} {'R²':>8} {'ρ(d,η)':>8}")
    print(f"  {'─' * 40}")
    for t in tiers:
        td = tier_data[t]
        active = [(r["dist_au"], r["eta"]) for r in td if r["eta"] > 0]
        if len(active) < 3:
            print(f"  {t:>4}  insufficient data ({len(active)} active epochs)")
            continue
        d_arr = np.array([x[0] for x in active])
        ln_eta = np.log([x[1] for x in active])
        A = np.vstack([np.ones_like(d_arr), d_arr]).T
        coef, res, _, _ = np.linalg.lstsq(A, ln_eta, rcond=None)
        pred = A @ coef
        ss_res = np.sum((ln_eta - pred) ** 2)
        ss_tot = np.sum((ln_eta - np.mean(ln_eta)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rho = np.corrcoef(d_arr, ln_eta)[0, 1]
        print(f"  {t:>4} {coef[0]:>8.3f} {coef[1]:>8.3f} {r2:>8.3f} {rho:>8.3f}")

    # ─────────────────────────────────────────────────────────────────
    # 5. Tier-to-tier Δη at each epoch
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 5. HISTORICAL TIER-TO-TIER η DIFFERENCES ────────────────────────┐")
    print("  Δη = η_relay_tier − η_T2 (baseline with same n_sats)")
    print(
        f"\n  {'Day':>5} {'SEP':>5} {'d_AU':>6} {'η_T2':>7} {'η_T3':>7} {'Δη_T3':>7} {'η_T4':>7} {'Δη_T4':>7}"
    )
    print(f"  {'─' * 58}")

    delta_t3_all = []
    delta_t4_all = []
    for i in range(n_epochs):
        r2 = tier_data[2][i]
        r3 = tier_data[3][i]
        r4 = tier_data[4][i]
        d3 = r3["eta"] - r2["eta"]
        d4 = r4["eta"] - r2["eta"]
        delta_t3_all.append(d3)
        delta_t4_all.append(d4)
        # Print every 4th epoch + conjunction zone
        if i % 4 == 0 or r2["sep_deg"] < 10:
            print(
                f"  {r2['epoch_day']:>5} {r2['sep_deg']:>5.1f} {r2['dist_au']:>6.2f} "
                f"{r2['eta']:>7.4f} {r3['eta']:>7.4f} {d3:>+7.4f} "
                f"{r4['eta']:>7.4f} {d4:>+7.4f}"
            )

    print(f"\n  T3−T2 mean Δη: {np.mean(delta_t3_all):+.4f}")
    print(f"  T4−T2 mean Δη: {np.mean(delta_t4_all):+.4f}")
    print(f"  T3−T2 Δη > 0 at {sum(1 for d in delta_t3_all if d > 0)}/{n_epochs} epochs")
    print(f"  T4−T2 Δη > 0 at {sum(1 for d in delta_t4_all if d > 0)}/{n_epochs} epochs")

    # ─────────────────────────────────────────────────────────────────
    # 6. Historical morphology proxy per tier
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 6. HISTORICAL MORPHOLOGY PROXY (NOT A CLASSIFIER) ───────────────┐")
    print("  For T1→T2 (same p, more nodes): Φ = η / exp(E[H]·λ)")
    print("  γ = ∂ln(Φ)/∂E[H] / (−λ)")
    print()

    # Compare T1 vs T2 at each epoch to get effective Φ ratio
    for t in tiers:
        td = tier_data[t]
        active = [r for r in td if r["eta"] > 0 and r["S_T"] > 0]
        if len(active) < 3:
            print(f"  T{t}: insufficient active epochs")
            continue
        # η = S_T * DR / S_T... actually η = DR / S_T
        # Φ requires knowing chain = exp(E[H]·λ), which we don't have per-epoch
        # But we can look at how η scales with distance (proxy for E[H])
        etas = np.array([r["eta"] for r in active])
        dists = np.array([r["dist_au"] for r in active])
        # ln(η) vs d_AU slope as proxy
        rho = np.corrcoef(dists, np.log(etas))[0, 1] if len(active) > 2 else 0
        # η trend: does η decrease faster or slower than expected with distance?
        print(
            f"  T{t}: {len(active)} active epochs, η_range=[{min(etas):.4f}, {max(etas):.4f}], ρ(d,ln η)={rho:.3f}"
        )

    # ─────────────────────────────────────────────────────────────────
    # 7. Contact graph complexity
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 7. CONTACT GRAPH COMPLEXITY ──────────────────────────────────────┐")
    print(
        f"  {'Tier':>4} {'n_contacts':>11} {'n_intra':>8} {'n_inter':>8} {'n_relay':>8} {'relay%':>7}"
    )
    print(f"  {'─' * 48}")
    for t in tiers:
        td = tier_data[t]
        nc = np.mean([r["n_contacts"] for r in td])
        ni = np.mean([r["n_intra"] for r in td])
        ne = np.mean([r["n_inter"] for r in td])
        nr = np.mean([r["n_relay"] for r in td])
        pct = 100 * nr / nc if nc > 0 else 0
        print(f"  {t:>4} {nc:>11.0f} {ni:>8.0f} {ne:>8.0f} {nr:>8.0f} {pct:>6.1f}%")

    # ─────────────────────────────────────────────────────────────────
    # 8. Full epoch-by-epoch table
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─ 8. FULL SYNODIC PROFILE ─────────────────────────────────────────┐")
    print(
        f"  {'Day':>5} {'SEP':>5} {'d_AU':>6} "
        f"{'S_T1':>5} {'η_T1':>6} {'DR_T1':>6} "
        f"{'S_T3':>5} {'η_T3':>6} {'DR_T3':>6} "
        f"{'S_T4':>5} {'η_T4':>6} {'DR_T4':>6}"
    )
    print(f"  {'─' * 80}")
    for i in range(n_epochs):
        r1 = tier_data[1][i]
        r3 = tier_data[3][i]
        r4 = tier_data[4][i]
        marker = " ◄" if r1["sep_deg"] < 5 else ""
        print(
            f"  {r1['epoch_day']:>5} {r1['sep_deg']:>5.1f} {r1['dist_au']:>6.2f} "
            f"{r1['S_T']:>5.3f} {r1['eta']:>6.4f} {r1['DR']:>6.4f} "
            f"{r3['S_T']:>5.3f} {r3['eta']:>6.4f} {r3['DR']:>6.4f} "
            f"{r4['S_T']:>5.3f} {r4['eta']:>6.4f} {r4['DR']:>6.4f}{marker}"
        )

    # ─────────────────────────────────────────────────────────────────
    # 9. Archived descriptive summary
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ARCHIVED DESCRIPTIVE SUMMARY — NO CAUSAL OR DESIGN CLAIMS")
    print("=" * 80)

    # Braess
    t1_mean = np.mean([r["DR"] for r in tier_data[1]])
    t2_mean = np.mean([r["DR"] for r in tier_data[2]])
    braess_pct = 100 * (1 - t2_mean / t1_mean) if t1_mean > 0 else 0
    print(f"\n  1. T1/T2 ROW COMPARISON: mean T2 DR is {braess_pct:.1f}% lower than mean T1 DR")
    print(f"     T1 mean DR = {t1_mean:.4f}  |  T2 mean DR = {t2_mean:.4f}")
    print(f"     T1 DR > T2 DR in {t1_wins}/{n_epochs} loaded epochs")

    # Low-separation zero-S_T comparison
    conj_t1_bl = sum(1 for r in tier_data[1] if r["S_T"] == 0)
    conj_t3_bl = sum(1 for r in tier_data[3] if r["S_T"] == 0)
    print(f"\n  2. ZERO-S_T COUNTS: T1 has {conj_t1_bl}; T3 has {conj_t3_bl} in the loaded rows")
    min_sep_r3 = [r for r in tier_data[3] if r["epoch_day"] == min_sep_day][0]
    min_sep_r4 = [r for r in tier_data[4] if r["epoch_day"] == min_sep_day][0]
    print(
        f"     At SEP={min_sep_r3['sep_deg']:.1f}°: T3 DR={min_sep_r3['DR']:.4f}, T4 DR={min_sep_r4['DR']:.4f}"
    )

    # T4 bookkeeping means
    t4_active = [r for r in tier_data[4] if r["eta"] > 0]
    st4 = np.mean([r["S_T"] for r in t4_active])
    eta4 = np.mean([r["eta"] for r in t4_active])
    print("\n  3. T4 BOOKKEEPING MEANS:")
    print(f"     mean S_T = {st4:.4f}")
    print(f"     mean η   = {eta4:.4f}")
    print(f"     normalized difference: {100 * (st4 - eta4) / st4:.1f}%")
    print("     The difference is descriptive and is not assigned to a routing cause.")

    # Maximum/minimum loaded epochs
    best = max(tier_data[4], key=lambda r: r["DR"])
    worst = min(tier_data[4], key=lambda r: r["DR"])
    print(
        f"\n  4. MAXIMUM LOADED T4 DR: day {best['epoch_day']}, DR={best['DR']:.4f}, d={best['dist_au']:.2f} AU"
    )
    print(
        f"     MINIMUM LOADED T4 DR: day {worst['epoch_day']}, DR={worst['DR']:.4f}, d={worst['dist_au']:.2f} AU"
    )
    print(f"     Dynamic range: {best['DR'] / max(worst['DR'], 1e-6):.1f}×")

    print()


if __name__ == "__main__":
    main()
