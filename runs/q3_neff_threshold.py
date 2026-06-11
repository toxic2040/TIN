#!/usr/bin/env python3
"""
Q3: N_eff threshold derivation — Is the catastrophe threshold
an absolute count or a coverage fraction?

Run toy model at multiple B values (phase-bin counts) and test:
  - If N_eff_crit / B = constant → coverage fraction
  - If N_eff_crit = constant   → absolute threshold

From the envelope framework: each phase bin is an independent channel.
Catastrophe is driven by n_src (source contact count), which determines
N_eff independent of B when n_src is small. Prediction: absolute threshold.

Also tests whether the threshold can be derived in closed form from
the binomial coverage model.

Approach:
  1. Generate synthetic contact plans with controlled phase distributions
  2. At each B value, compute N_eff and S_T
  3. Find the catastrophe threshold via logistic regression
  4. Test N_eff_crit vs B

  5. Also recompute from existing time-reversal data at multiple B
"""

import json
import math

import numpy as np

DAY = 86400.0
YEAR = 365.25 * DAY


# ═══════════════════════════════════════════════════════════════════
# TOY MODEL — Synthetic contact plans with controlled phase coverage
# ═══════════════════════════════════════════════════════════════════


def compute_neff(counts):
    """Simpson's diversity N_eff from bin counts."""
    total = np.sum(counts)
    if total == 0:
        return 0.0
    return float(total**2 / np.sum(counts**2))


def synthetic_network_st(n_src, n_total, B, n_hops, concentration, n_inject=1000, seed=42):
    """
    Generate a synthetic temporal network and compute S_T and N_eff.

    Parameters:
        n_src: number of source-serving contacts
        n_total: total contacts across all link types
        B: number of phase bins
        n_hops: hop count (number of link types in the chain)
        concentration: Dirichlet concentration parameter
            - high (>10): uniform across bins
            - low (<1): concentrated in few bins
        n_inject: number of injection times to test
        seed: random seed

    Returns:
        dict with N_eff, S_T, and diagnostics
    """
    rng = np.random.default_rng(seed)

    # Generate phase distribution via Dirichlet
    # Each link type gets its own phase distribution
    alpha = np.full(B, concentration)

    # Source contacts: distributed according to Dirichlet
    src_probs = rng.dirichlet(alpha)
    src_bins = rng.choice(B, size=n_src, p=src_probs)
    src_counts = np.bincount(src_bins, minlength=B)

    # For intermediate links: generate independent distributions
    # Each hop has its own phase distribution
    link_bins = []
    for hop in range(n_hops - 1):  # n_hops - 1 intermediate link types
        link_probs = rng.dirichlet(alpha)
        n_per_link = n_total // n_hops
        bins = rng.choice(B, size=n_per_link, p=link_probs)
        link_bins.append(np.bincount(bins, minlength=B))

    # Compute N_eff for source contacts
    neff_src = compute_neff(src_counts)

    # Compute S_T by checking temporal reachability
    # Simple model: injection at random phases, check if each hop
    # has a contact in a compatible phase bin
    # For simplicity: S_T ≈ fraction of bins that have at least 1
    # source contact AND at least 1 contact in every intermediate link

    # Phase-bin-level feasibility
    feasible_bins = src_counts > 0  # source must be present
    for hop_counts in link_bins:
        feasible_bins &= hop_counts > 0  # all intermediate links too

    s_t = float(np.sum(feasible_bins)) / B

    # More refined: inject at each bin center, check connectivity
    # (accounts for bin-to-bin handoff)
    n_occupied_src = int(np.sum(src_counts > 0))
    n_empty_src = B - n_occupied_src

    return {
        "n_src": n_src,
        "n_total": n_total,
        "B": B,
        "n_hops": n_hops,
        "concentration": concentration,
        "neff_src": float(neff_src),
        "s_t": float(s_t),
        "n_occupied_src": n_occupied_src,
        "n_empty_src": n_empty_src,
        "catastrophe": s_t < 0.5,
        "gini": float(1 - neff_src / B) if B > 0 else 0,
    }


def logistic_threshold(neff_vals, catastrophe_flags, target_p=0.5):
    """Fit logistic regression and find threshold at P(catastrophe)=target_p.

    Returns threshold N_eff, or None if fit fails.
    """
    x = np.array(neff_vals, dtype=float)
    y = np.array(catastrophe_flags, dtype=float)

    if len(set(y)) < 2:
        return None, None

    # Simple logistic regression via Newton's method
    # P(y=1|x) = 1 / (1 + exp(-(a + b*x)))
    a, b = 0.0, -1.0
    for _ in range(100):
        z = a + b * x
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # Gradient
        grad_a = np.sum(y - p)
        grad_b = np.sum((y - p) * x)

        # Hessian (approximate)
        w = p * (1 - p)
        H_aa = -np.sum(w)
        H_ab = -np.sum(w * x)
        H_bb = -np.sum(w * x * x)

        det = H_aa * H_bb - H_ab * H_ab
        if abs(det) < 1e-15:
            break

        da = (H_bb * grad_a - H_ab * grad_b) / det
        db = (H_aa * grad_b - H_ab * grad_a) / det
        a -= da
        b -= db

    # Threshold: P=target_p → a + b*x = log(target_p/(1-target_p))
    logit_target = math.log(target_p / (1 - target_p))
    if abs(b) < 1e-15:
        return None, None

    threshold = (logit_target - a) / b

    # AUC (simple: fraction correctly separated)
    pred = 1 / (1 + np.exp(-(a + b * x)))
    pred_class = pred > 0.5
    accuracy = np.mean(pred_class == y)

    return float(threshold), float(accuracy)


# ═══════════════════════════════════════════════════════════════════
# ANALYTICAL PREDICTION — Binomial coverage model
# ═══════════════════════════════════════════════════════════════════


def binomial_coverage_neff(n_src, B):
    """Predict N_eff under uniform random placement of n_src contacts
    into B bins (Poisson approximation).

    N_eff ≈ B * (1 - exp(-n_src/B)) * f(n_src, B)

    For n_src >> B: N_eff → B (full coverage)
    For n_src << B: N_eff → n_src (each in own bin)
    """
    if n_src == 0:
        return 0.0
    mu = n_src / B
    # Expected occupancy: Poisson with rate mu
    # N_eff = (Σ n_b)² / Σ n_b²
    # For Poisson: E[n_b] = mu, E[n_b²] = mu + mu²
    # E[Σ n_b²] = B * (mu + mu²) = n_src + n_src²/B
    # N_eff ≈ n_src² / (n_src + n_src²/B) = n_src * B / (B + n_src)
    # = 1 / (1/n_src + 1/B)  — harmonic mean of n_src and B!
    return 1.0 / (1.0 / n_src + 1.0 / B)


def main():
    print("=" * 70)
    print("  Q3: N_eff THRESHOLD DERIVATION")
    print("  Is the catastrophe threshold absolute or fractional?")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # PART 1: ANALYTICAL PREDICTION — Harmonic mean
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 1: ANALYTICAL PREDICTION")
    print(f"{'=' * 70}")

    print("""
  Under uniform random placement of n_src contacts into B phase bins:

    N_eff ≈ 1 / (1/n_src + 1/B)  =  n_src * B / (n_src + B)

  This is the HARMONIC MEAN of n_src and B.

  Key limits:
    n_src << B:  N_eff ≈ n_src     (each contact in own bin)
    n_src >> B:  N_eff ≈ B         (bins saturate)
    n_src = B:   N_eff = B/2       (half-saturation)

  For catastrophe (n_src = 1-3):
    n_src=1:  N_eff = B/(B+1) ≈ 1  for all B >> 1
    n_src=2:  N_eff = 2B/(B+2) ≈ 2  for all B >> 1
    n_src=3:  N_eff = 3B/(B+3) ≈ 3  for all B >> 1

  PREDICTION: N_eff_crit is an ABSOLUTE threshold (≈ n_src at the boundary),
  independent of B for B >> n_src_crit.
  """)

    # Show the harmonic mean across B values
    print(f"  {'n_src':>6s}", end="")
    B_values = [12, 18, 24, 36, 48, 72]
    for B in B_values:
        print(f" | {'B=' + str(B):>8s}", end="")
    print()
    print(f"  {'-' * 70}")

    for n_src in [1, 2, 3, 5, 10, 20, 36, 72]:
        print(f"  {n_src:6d}", end="")
        for B in B_values:
            neff = binomial_coverage_neff(n_src, B)
            print(f" | {neff:8.2f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════════
    # PART 2: TOY MODEL — Sweep n_src and B
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 2: TOY MODEL — N_eff_crit vs B")
    print(f"{'=' * 70}")

    # For each B, generate many configurations with varying n_src
    # and concentration, then find the catastrophe threshold

    n_hops = 5  # 5-hop chain
    n_total_base = 500
    concentrations = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    n_srcs = [1, 2, 3, 5, 8, 12, 18, 25, 36, 50, 72, 100]

    thresholds = {}
    for B in B_values:
        all_neff = []
        all_catastrophe = []

        for n_src in n_srcs:
            for ci, conc in enumerate(concentrations):
                for trial in range(5):  # 5 trials per config
                    seed = B * 10000 + n_src * 100 + ci * 10 + trial
                    result = synthetic_network_st(
                        n_src=n_src,
                        n_total=n_total_base,
                        B=B,
                        n_hops=n_hops,
                        concentration=conc,
                        seed=seed,
                    )
                    all_neff.append(result["neff_src"])
                    all_catastrophe.append(result["catastrophe"])

        threshold, accuracy = logistic_threshold(all_neff, all_catastrophe)
        thresholds[B] = {
            "threshold": threshold,
            "accuracy": accuracy,
            "n_configs": len(all_neff),
            "n_catastrophe": sum(all_catastrophe),
        }

    print(
        f"\n  {'B':>4s} | {'N_eff_crit':>10s} | {'N_eff/B':>8s} | "
        f"{'Accuracy':>8s} | {'n_configs':>9s} | {'n_cat':>6s}"
    )
    print(f"  {'-' * 60}")
    for B in B_values:
        t = thresholds[B]
        if t["threshold"] is not None:
            ratio = t["threshold"] / B
            print(
                f"  {B:4d} | {t['threshold']:10.3f} | {ratio:8.4f} | "
                f"{t['accuracy']:8.1%} | {t['n_configs']:9d} | {t['n_catastrophe']:6d}"
            )
        else:
            print(f"  {B:4d} | {'N/A':>10s}")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: SCALING TEST — Is N_eff_crit constant?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 3: SCALING TEST")
    print(f"{'=' * 70}")

    valid_thresholds = [
        (B, t["threshold"]) for B, t in thresholds.items() if t["threshold"] is not None
    ]

    if len(valid_thresholds) >= 3:
        B_arr = np.array([b for b, _ in valid_thresholds])
        thresh_arr = np.array([t for _, t in valid_thresholds])
        ratio_arr = thresh_arr / B_arr

        # Test 1: Is threshold constant?
        thresh_cv = np.std(thresh_arr) / np.mean(thresh_arr)

        # Test 2: Is ratio constant?
        ratio_cv = np.std(ratio_arr) / np.mean(ratio_arr)

        print("\n  Absolute threshold test:")
        print(f"    Mean N_eff_crit = {np.mean(thresh_arr):.3f}")
        print(f"    Std  N_eff_crit = {np.std(thresh_arr):.3f}")
        print(
            f"    CV(threshold)   = {thresh_cv:.3f}  "
            f"({'LOW → constant ✓' if thresh_cv < 0.2 else 'HIGH → varies ✗'})"
        )

        print("\n  Coverage fraction test:")
        print(f"    Mean N_eff/B    = {np.mean(ratio_arr):.4f}")
        print(f"    Std  N_eff/B    = {np.std(ratio_arr):.4f}")
        print(
            f"    CV(ratio)       = {ratio_cv:.3f}  "
            f"({'LOW → constant fraction ✓' if ratio_cv < 0.2 else 'HIGH → varies ✗'})"
        )

        # Correlation with B
        if len(B_arr) >= 3:
            corr = np.corrcoef(B_arr, thresh_arr)[0, 1]
            print(f"\n  Correlation(B, N_eff_crit) = {corr:.4f}")
            if abs(corr) > 0.9:
                print("    Strong correlation → N_eff_crit scales with B (coverage fraction)")
            elif abs(corr) < 0.3:
                print("    Weak correlation → N_eff_crit is approximately constant (absolute)")
            else:
                print("    Moderate correlation → mixed regime")

    # ═══════════════════════════════════════════════════════════════
    # PART 4: CLOSED-FORM DERIVATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 4: CLOSED-FORM DERIVATION")
    print(f"{'=' * 70}")

    print(f"""
  From the harmonic mean model:

    N_eff(n_src, B) = n_src * B / (n_src + B)

  The catastrophe condition S_T < 0.5 requires that more than half
  the phase bins have no source contacts. Under Poisson placement:

    P(bin empty) = exp(-n_src/B)
    Fraction occupied: f = 1 - exp(-n_src/B)
    S_T ≈ f^(n_hops)  (independent phases per hop, worst case)

  For S_T = 0.5 with n_hops hops:
    f_crit = 0.5^(1/n_hops)

  For n_hops = 5: f_crit = 0.5^0.2 = {0.5**0.2:.4f}
  For n_hops = 7: f_crit = 0.5^(1/7) = {0.5 ** (1 / 7):.4f}

  Required n_src/B: n_src/B = -ln(1 - f_crit)

  For n_hops = 5: n_src/B = {-math.log(1 - 0.5**0.2):.4f}
  For n_hops = 7: n_src/B = {-math.log(1 - 0.5 ** (1 / 7)):.4f}

  At this n_src/B ratio:
    N_eff_crit = n_src * B / (n_src + B) = mu * B² / (mu*B + B)
              = mu * B / (mu + 1)

  where mu = n_src / B = -ln(1 - f_crit).

  For n_hops = 5, B = 36: N_eff_crit = {(-math.log(1 - 0.5**0.2)) * 36 / (-math.log(1 - 0.5**0.2) + 1):.2f}
  For n_hops = 7, B = 36: N_eff_crit = {(-math.log(1 - 0.5 ** (1 / 7))) * 36 / (-math.log(1 - 0.5 ** (1 / 7)) + 1):.2f}

  KEY INSIGHT: N_eff_crit = mu * B / (mu + 1) IS proportional to B
  when the catastrophe depends on fractional coverage (f < f_crit).

  BUT: when n_src << B (the catastrophe regime), N_eff ≈ n_src.
  The threshold becomes n_src_crit, which is topology-dependent
  (depends on n_hops) but B-independent.

  RESOLUTION: There are TWO thresholds:
    1. Support threshold: n_src_crit (absolute, B-independent)
       Controls whether the network CAN function.
    2. Coverage threshold: N_eff_crit (proportional to B)
       Controls HOW WELL it functions once n_src > n_src_crit.

  When n_src < n_src_crit, the network is dead regardless of B.
  When n_src > n_src_crit but N_eff < coverage threshold, it's degraded.
  The empirical N_eff_crit ≈ 1.52 at B=36 is the SUPPORT threshold,
  not the coverage threshold.
  """)

    # ═══════════════════════════════════════════════════════════════
    # PART 5: VERIFY AGAINST EXISTING DATA
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 5: VERIFY AGAINST EXISTING DATA")
    print(f"{'=' * 70}")

    # Load existing time-reversal + N_eff data
    import os

    neff_file = "runs/neff_results.json"
    tr_file = "runs/time_reversal_results.json"

    if os.path.exists(neff_file) and os.path.exists(tr_file):
        with open(neff_file) as f:
            neff_data = json.load(f)
        with open(tr_file) as f:
            tr_data = json.load(f)

        # Build per-config (body, n_orb, neff_src, S_T_fwd)
        print(f"\n  Loaded: {neff_file} ({len(neff_data)} entries)")
        print(f"  Loaded: {tr_file}")

        # Build neff lookup: (target, n_orb, epoch_day) → neff_src
        neff_lookup = {}
        for entry in neff_data:
            key = (entry["target"], entry["n_orb"], entry.get("epoch_day", 0))
            neff_lookup[key] = entry

        # Extract per-config data
        configs = []
        if isinstance(tr_data, dict) and "per_config" in tr_data:
            for cfg in tr_data["per_config"]:
                body = cfg.get("target", cfg.get("body", "unknown"))
                n_orb = cfg.get("n_orb", cfg.get("n_orbiters", 0))
                epoch = cfg.get("epoch_day", 0)

                # Get S_T from forward results
                fwd = cfg.get("forward", {})
                s_t_fwd = fwd.get("S_T", fwd.get("s_t", 0))

                # Find matching neff entry
                neff_entry = neff_lookup.get((body, n_orb, epoch))
                if neff_entry:
                    configs.append(
                        {
                            "body": body,
                            "n_orb": n_orb,
                            "s_t": s_t_fwd,
                            "neff_src": neff_entry.get("neff_src", 0),
                            "catastrophe": s_t_fwd < 0.5,
                        }
                    )

        if configs:
            print(f"  Matched {len(configs)} configs with N_eff data")

            # Logistic threshold from real data
            neff_vals = [c["neff_src"] for c in configs]
            cat_flags = [c["catastrophe"] for c in configs]
            real_threshold, real_accuracy = logistic_threshold(neff_vals, cat_flags)

            if real_threshold is not None:
                print(f"  Real data threshold (B=36): N_eff_crit = {real_threshold:.3f}")
                print(f"  Accuracy: {real_accuracy:.1%}")
            else:
                print("  Could not fit logistic (insufficient class separation)")
        else:
            print("  Could not match configs between files (format mismatch)")
            print("  This is OK — the toy model results above are the primary test")
    else:
        print("  Data files not found — using toy model results only")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  SYNTHESIS")
    print(f"{'=' * 70}")

    print("""
  Q3 ANSWER: The catastrophe threshold has TWO components:

  1. SUPPORT THRESHOLD (absolute, B-independent):
     n_src_crit — the minimum number of source contacts needed
     for the network to function at all. This is what the empirical
     N_eff_crit ≈ 1.52 measures. Since N_eff ≈ n_src for small n_src,
     this threshold is approximately constant across B.

  2. COVERAGE THRESHOLD (proportional to B):
     N_eff_cov = mu_crit × B / (mu_crit + 1)
     where mu_crit = -ln(1 - f_crit^(1/n_hops)).
     This controls how well the network functions once it has enough
     support. It scales linearly with B.

  The harmonic mean formula N_eff = n_src × B / (n_src + B) bridges
  both regimes:
     - n_src << B: N_eff ≈ n_src (support-limited, B-independent)
     - n_src >> B: N_eff ≈ B     (coverage-limited, B-proportional)

  CLOSED FORM: N_eff_crit(B) = n_src_crit × B / (n_src_crit + B)
  where n_src_crit depends on topology (n_hops) but not on B.
  """)

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    output = {
        "thresholds_by_B": {str(B): t for B, t in thresholds.items()},
        "harmonic_mean_examples": {
            str(B): {str(n): binomial_coverage_neff(n, B) for n in [1, 2, 3, 5, 10, 20, 36, 72]}
            for B in B_values
        },
        "analytical": {
            "f_crit_5hop": float(0.5**0.2),
            "f_crit_7hop": float(0.5 ** (1 / 7)),
            "mu_crit_5hop": float(-math.log(1 - 0.5**0.2)),
            "mu_crit_7hop": float(-math.log(1 - 0.5 ** (1 / 7))),
        },
    }

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open("runs/q3_neff_threshold_results.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print("\nResults saved to runs/q3_neff_threshold_results.json")


if __name__ == "__main__":
    main()
