#!/usr/bin/env python3
"""run_boltzmann_gof.py — Boltzmann goodness-of-fit test for hop-count routing.

Tests whether oracle path routing frequencies follow a Boltzmann distribution
in hop-count utility: f_k ∝ exp(β · H_k · λ), where H_k is the hop count
and λ is the Lyapunov exponent (mean log link probability).

Since λ is constant within a config, the test reduces to:
    ln(f_k) linear in H_k ?

Three models are compared:
  1. Boltzmann: p_k ∝ exp(b · H_k),  1 free parameter (b)
  2. Uniform:   p_k = 1/K,            0 free parameters
  3. Power-law: p_k ∝ H_k^α,         1 free parameter (α)

GOF metrics: χ², p-value, KL divergence, AIC.
Only configs with ≥3 distinct hop counts are testable (≥1 DOF after Boltzmann fit).

Provenance
----------
Input:  runs/hop_histogram_survey_results.json
        runs/epyc_results/ (for Lyapunov exponent join)
Output: runs/boltzmann_gof_results.json

Usage:
    python -m runs.run_boltzmann_gof             # full
    python -m runs.run_boltzmann_gof --smoke     # first 30 configs
"""

from __future__ import annotations

import glob
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

_HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def _load_hop_histograms():
    """Load hop histogram survey results."""
    path = _HERE / "hop_histogram_survey_results.json"
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def _load_lyapunov_index():
    """Build lookup: (body, family, n_sats, alt_km, band, seed) → lyapunov."""
    index = {}
    for path in sorted(
        glob.glob(str(_HERE / "epyc_results/production_2026_03_11/production_*.json"))
    ):
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict) and "results" in raw:
            data = raw["results"]
        else:
            data = [v for v in raw.values() if isinstance(v, dict)]
        for d in data:
            if not isinstance(d, dict):
                continue
            key = (
                d.get("body"),
                d.get("family"),
                d.get("n_sats"),
                d.get("alt_km"),
                d.get("band"),
                d.get("seed"),
            )
            lyap = d.get("lyapunov")
            if lyap is not None and not (isinstance(lyap, float) and math.isnan(lyap)):
                index[key] = lyap

    for path in sorted(glob.glob(str(_HERE / "epyc_results/campaign_2026_03_11/*.json"))):
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict) and "results" in raw:
            data = raw["results"]
        else:
            continue
        for d in data:
            if not isinstance(d, dict):
                continue
            key = (
                d.get("body"),
                d.get("family"),
                d.get("n_sats"),
                d.get("alt_km"),
                d.get("band"),
                d.get("seed"),
            )
            lyap = d.get("lyapunov")
            if lyap is not None and not (isinstance(lyap, float) and math.isnan(lyap)):
                index[key] = lyap
    return index


# ---------------------------------------------------------------------------
# Boltzmann fit: MLE for p_k ∝ exp(b · H_k)
# ---------------------------------------------------------------------------


def _boltzmann_mle(hop_counts, frequencies):
    """Fit b in p_k ∝ exp(b · H_k) via MLE.

    MLE condition: E_observed[H] = E_model[H] = Σ H_k · softmax(b·H_k)

    Returns (b, p_predicted) or (None, None) if degenerate.
    """
    H = np.array(hop_counts, dtype=float)
    f = np.array(frequencies, dtype=float)
    N = f.sum()
    f_norm = f / N
    E_obs = np.dot(H, f_norm)

    def _mean_H(b):
        """Model mean hop count at inverse temperature b."""
        logits = b * H
        logits -= logits.max()  # numerical stability
        w = np.exp(logits)
        w /= w.sum()
        return np.dot(H, w) - E_obs

    # Bracket search: at b=0, model mean = mean(H) over uniform support.
    # As b → +∞, model concentrates on max(H); b → -∞ on min(H).
    # Find b where model mean = observed mean.
    H_min, H_max = H.min(), H.max()
    if H_min == H_max:
        return None, None

    # Check if E_obs is achievable
    if E_obs <= H_min or E_obs >= H_max:
        # Degenerate: all mass on one endpoint
        return None, None

    try:
        b_opt = optimize.brentq(_mean_H, -50.0, 50.0, xtol=1e-10)
    except ValueError:
        return None, None

    logits = b_opt * H
    logits -= logits.max()
    p_pred = np.exp(logits)
    p_pred /= p_pred.sum()
    return b_opt, p_pred


def _power_law_mle(hop_counts, frequencies):
    """Fit α in p_k ∝ H_k^α via MLE.

    MLE condition: E_observed[ln H] = E_model[ln H] = Σ ln(H_k) · softmax(α·ln(H_k))

    Returns (α, p_predicted) or (None, None) if degenerate.
    """
    H = np.array(hop_counts, dtype=float)
    f = np.array(frequencies, dtype=float)
    N = f.sum()
    f_norm = f / N
    lnH = np.log(H)
    E_obs_lnH = np.dot(lnH, f_norm)

    def _mean_lnH(alpha):
        logits = alpha * lnH
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        return np.dot(lnH, w) - E_obs_lnH

    if lnH.min() == lnH.max():
        return None, None

    try:
        alpha_opt = optimize.brentq(_mean_lnH, -100.0, 100.0, xtol=1e-10)
    except ValueError:
        return None, None

    logits = alpha_opt * lnH
    logits -= logits.max()
    p_pred = np.exp(logits)
    p_pred /= p_pred.sum()
    return alpha_opt, p_pred


# ---------------------------------------------------------------------------
# GOF metrics
# ---------------------------------------------------------------------------


def _chi2_test(observed, expected, n_params):
    """χ² test. Returns (chi2, p_value, dof)."""
    K = len(observed)
    dof = K - 1 - n_params  # K bins, 1 normalization constraint, n_params fitted
    if dof <= 0:
        return float("nan"), float("nan"), dof

    obs = np.array(observed, dtype=float)
    exp = np.array(expected, dtype=float)
    N = obs.sum()
    exp_counts = exp * N

    # Pool bins with expected < 5 (standard χ² requirement)
    mask = exp_counts >= 5.0
    if mask.sum() < 3:
        # Not enough bins after pooling
        return float("nan"), float("nan"), 0

    # Use only bins with sufficient expected counts
    chi2 = np.sum((obs[mask] - exp_counts[mask]) ** 2 / exp_counts[mask])
    dof_eff = mask.sum() - 1 - n_params
    if dof_eff <= 0:
        return chi2, float("nan"), dof_eff

    p_val = 1.0 - stats.chi2.cdf(chi2, dof_eff)
    return float(chi2), float(p_val), int(dof_eff)


def _kl_divergence(observed_freq, predicted_prob):
    """KL(observed || predicted). observed_freq are counts, predicted_prob are probs."""
    obs = np.array(observed_freq, dtype=float)
    pred = np.array(predicted_prob, dtype=float)
    obs_norm = obs / obs.sum()
    # Avoid log(0)
    mask = obs_norm > 0
    kl = np.sum(obs_norm[mask] * np.log(obs_norm[mask] / pred[mask]))
    return float(kl)


def _aic(log_likelihood, n_params):
    """Akaike Information Criterion."""
    return 2 * n_params - 2 * log_likelihood


def _log_likelihood(observed_freq, predicted_prob):
    """Multinomial log-likelihood."""
    obs = np.array(observed_freq, dtype=float)
    pred = np.array(predicted_prob, dtype=float)
    pred = np.clip(pred, 1e-300, None)
    return float(np.sum(obs * np.log(pred)))


# ---------------------------------------------------------------------------
# Per-config analysis
# ---------------------------------------------------------------------------


def _analyze_config(rec, lyapunov):
    """Run Boltzmann GOF test on one config.

    Returns result dict or None if insufficient data.
    """
    hist = rec.get("hop_histogram", {})
    if len(hist) < 3:
        return None

    hop_counts = sorted(int(k) for k in hist.keys())
    frequencies = [hist[str(h)] for h in hop_counts]
    N = sum(frequencies)
    K = len(hop_counts)

    if N < 20:
        return None

    # ── Model 1: Boltzmann  p_k ∝ exp(b · H_k) ─────────────────────────
    b_opt, p_boltz = _boltzmann_mle(hop_counts, frequencies)
    if b_opt is None:
        return None

    chi2_b, pval_b, dof_b = _chi2_test(frequencies, p_boltz, n_params=1)
    kl_b = _kl_divergence(frequencies, p_boltz)
    ll_b = _log_likelihood(frequencies, p_boltz)
    aic_b = _aic(ll_b, n_params=1)

    # ── Model 2: Uniform  p_k = 1/K ─────────────────────────────────────
    p_unif = np.ones(K) / K
    chi2_u, pval_u, dof_u = _chi2_test(frequencies, p_unif, n_params=0)
    kl_u = _kl_divergence(frequencies, p_unif)
    ll_u = _log_likelihood(frequencies, p_unif)
    aic_u = _aic(ll_u, n_params=0)

    # ── Model 3: Power-law  p_k ∝ H_k^α ────────────────────────────────
    alpha_opt, p_power = _power_law_mle(hop_counts, frequencies)
    if alpha_opt is not None:
        chi2_p, pval_p, dof_p = _chi2_test(frequencies, p_power, n_params=1)
        kl_p = _kl_divergence(frequencies, p_power)
        ll_p = _log_likelihood(frequencies, p_power)
        aic_p = _aic(ll_p, n_params=1)
    else:
        chi2_p = pval_p = kl_p = ll_p = aic_p = alpha_opt = float("nan")
        dof_p = 0
        p_power = None

    # ── β_eff extraction ────────────────────────────────────────────────
    beta_eff = None
    if lyapunov is not None and abs(lyapunov) > 1e-15:
        beta_eff = b_opt / lyapunov

    # ── Winner by AIC ───────────────────────────────────────────────────
    models = {"boltzmann": aic_b, "uniform": aic_u}
    if not math.isnan(aic_p):
        models["power_law"] = aic_p
    winner = min(models, key=models.get)

    return {
        "body": rec["body"],
        "family": rec["family"],
        "n_sats": rec["n_sats"],
        "alt_km": rec.get("alt_km"),
        "n_unique_hops": K,
        "n_paths": N,
        "hop_counts": hop_counts,
        "frequencies": frequencies,
        "E_H": rec.get("E_H_recomputed"),
        "lyapunov": lyapunov,
        # Boltzmann
        "b_opt": float(b_opt),
        "beta_eff": float(beta_eff) if beta_eff is not None else None,
        "p_boltzmann": [float(x) for x in p_boltz],
        "chi2_boltzmann": chi2_b,
        "pval_boltzmann": pval_b,
        "dof_boltzmann": dof_b,
        "kl_boltzmann": kl_b,
        "aic_boltzmann": aic_b,
        # Uniform
        "chi2_uniform": chi2_u,
        "pval_uniform": pval_u,
        "kl_uniform": kl_u,
        "aic_uniform": aic_u,
        # Power-law
        "alpha_opt": float(alpha_opt)
        if alpha_opt is not None and not math.isnan(alpha_opt)
        else None,
        "chi2_power_law": chi2_p,
        "pval_power_law": pval_p,
        "kl_power_law": kl_p,
        "aic_power_law": aic_p,
        # Summary
        "winner_aic": winner,
        "boltzmann_rejected_05": pval_b < 0.05 if not math.isnan(pval_b) else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    smoke = "--smoke" in sys.argv

    print("=" * 72)
    print("BOLTZMANN GOF TEST — Is oracle routing Boltzmann in hop count?")
    print("=" * 72)

    t0 = time.time()
    records = _load_hop_histograms()
    print(f"\n  Loaded {len(records)} hop histogram records")

    lyap_index = _load_lyapunov_index()
    print(f"  Loaded {len(lyap_index)} Lyapunov values from EPYC data")

    # Filter to ≥3 distinct hop counts
    testable = [r for r in records if r.get("n_unique_hops", 0) >= 3]
    print(f"  Testable (≥3 hop families): {len(testable)}")

    if smoke:
        testable = testable[:30]
        print(f"  Smoke mode: using first {len(testable)}")

    # Run analysis
    results = []
    n_no_lyap = 0
    n_skip = 0
    for rec in testable:
        key = (
            rec.get("body"),
            rec.get("family"),
            rec.get("n_sats"),
            rec.get("alt_km"),
            rec.get("band"),
            rec.get("seed"),
        )
        lyap = lyap_index.get(key)
        if lyap is None:
            n_no_lyap += 1

        res = _analyze_config(rec, lyap)
        if res is None:
            n_skip += 1
            continue
        results.append(res)

    wall_s = time.time() - t0
    print(f"\n  Analyzed: {len(results)}")
    print(f"  Skipped (degenerate): {n_skip}")
    print(f"  Missing lyapunov: {n_no_lyap}")
    print(f"  Wall time: {wall_s:.1f}s")

    if not results:
        print("  No results to report.")
        return

    # ── Summary statistics ────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("RESULTS")
    print(f"{'=' * 72}")

    # AIC winners
    winner_counts = Counter(r["winner_aic"] for r in results)
    print("\n  AIC winner distribution:")
    for w, cnt in winner_counts.most_common():
        pct = cnt / len(results) * 100
        print(f"    {w:15s}: {cnt:4d} ({pct:.1f}%)")

    # Boltzmann rejection rate
    testable_pval = [r for r in results if r["boltzmann_rejected_05"] is not None]
    if testable_pval:
        n_rejected = sum(1 for r in testable_pval if r["boltzmann_rejected_05"])
        n_not = len(testable_pval) - n_rejected
        print("\n  Boltzmann χ² test (α=0.05):")
        print(
            f"    Not rejected (consistent): {n_not:4d} ({n_not / len(testable_pval) * 100:.1f}%)"
        )
        print(
            f"    Rejected:                  {n_rejected:4d} "
            f"({n_rejected / len(testable_pval) * 100:.1f}%)"
        )

    # KL divergence comparison
    kl_b = [r["kl_boltzmann"] for r in results if not math.isnan(r["kl_boltzmann"])]
    kl_u = [r["kl_uniform"] for r in results if not math.isnan(r["kl_uniform"])]
    kl_p = [
        r["kl_power_law"] for r in results if not math.isnan(r.get("kl_power_law", float("nan")))
    ]
    print("\n  KL divergence (observed || model):")
    if kl_b:
        print(
            f"    Boltzmann:  mean={np.mean(kl_b):.4f}  "
            f"median={np.median(kl_b):.4f}  max={max(kl_b):.4f}"
        )
    if kl_u:
        print(
            f"    Uniform:    mean={np.mean(kl_u):.4f}  "
            f"median={np.median(kl_u):.4f}  max={max(kl_u):.4f}"
        )
    if kl_p:
        print(
            f"    Power-law:  mean={np.mean(kl_p):.4f}  "
            f"median={np.median(kl_p):.4f}  max={max(kl_p):.4f}"
        )

    # Boltzmann beats uniform?
    boltz_beats_uniform = sum(1 for r in results if r["kl_boltzmann"] < r["kl_uniform"])
    print(
        f"\n  Boltzmann KL < Uniform KL: {boltz_beats_uniform}/{len(results)} "
        f"({boltz_beats_uniform / len(results) * 100:.1f}%)"
    )

    # Boltzmann beats power-law?
    both_valid = [r for r in results if not math.isnan(r.get("kl_power_law", float("nan")))]
    if both_valid:
        boltz_beats_power = sum(1 for r in both_valid if r["kl_boltzmann"] < r["kl_power_law"])
        print(
            f"  Boltzmann KL < Power-law KL: {boltz_beats_power}/{len(both_valid)} "
            f"({boltz_beats_power / len(both_valid) * 100:.1f}%)"
        )

    # β_eff statistics
    beta_vals = [r["beta_eff"] for r in results if r["beta_eff"] is not None]
    if beta_vals:
        print(f"\n  β_eff statistics (n={len(beta_vals)}):")
        print(
            f"    mean={np.mean(beta_vals):.3f}  "
            f"median={np.median(beta_vals):.3f}  "
            f"std={np.std(beta_vals):.3f}"
        )
        print(f"    min={min(beta_vals):.3f}  max={max(beta_vals):.3f}")
        print(f"    IQR=[{np.percentile(beta_vals, 25):.3f}, {np.percentile(beta_vals, 75):.3f}]")

        # β_eff by body
        body_beta = defaultdict(list)
        for r in results:
            if r["beta_eff"] is not None:
                body_beta[r["body"]].append(r["beta_eff"])
        print("\n  β_eff by body:")
        for body in sorted(body_beta.keys()):
            vals = body_beta[body]
            print(
                f"    {body:10s}: n={len(vals):3d}  "
                f"mean={np.mean(vals):+.3f}  "
                f"std={np.std(vals):.3f}  "
                f"median={np.median(vals):+.3f}"
            )

    # b_opt (reduced slope) statistics
    b_vals = [r["b_opt"] for r in results]
    print("\n  b_opt (slope in ln f_k vs H_k) statistics:")
    print(
        f"    mean={np.mean(b_vals):.4f}  median={np.median(b_vals):.4f}  std={np.std(b_vals):.4f}"
    )
    print(
        f"    fraction negative: "
        f"{sum(1 for b in b_vals if b < 0)}/{len(b_vals)} "
        f"({sum(1 for b in b_vals if b < 0) / len(b_vals) * 100:.1f}%)"
    )

    # Breakdown by n_unique_hops (DOF proxy)
    by_k = defaultdict(list)
    for r in results:
        by_k[r["n_unique_hops"]].append(r)
    print("\n  Results by # distinct hop counts:")
    print(f"    {'K':>3s}  {'n':>5s}  {'Boltz win':>10s}  {'not rej':>8s}  {'mean KL_b':>10s}")
    for k in sorted(by_k.keys()):
        recs = by_k[k]
        n = len(recs)
        boltz_wins = sum(1 for r in recs if r["winner_aic"] == "boltzmann")
        not_rej = sum(
            1
            for r in recs
            if r["boltzmann_rejected_05"] is not None and not r["boltzmann_rejected_05"]
        )
        testable_k = sum(1 for r in recs if r["boltzmann_rejected_05"] is not None)
        kl_mean = np.mean([r["kl_boltzmann"] for r in recs if not math.isnan(r["kl_boltzmann"])])
        print(
            f"    {k:3d}  {n:5d}  {boltz_wins:5d}/{n:<4d}  "
            f"{not_rej:4d}/{testable_k:<3d}  {kl_mean:10.4f}"
        )

    # Worst fits — what does Boltzmann miss?
    ranked = sorted(
        [r for r in results if not math.isnan(r["kl_boltzmann"])],
        key=lambda r: r["kl_boltzmann"],
        reverse=True,
    )
    print("\n  Top 10 worst Boltzmann fits (by KL):")
    for r in ranked[:10]:
        print(
            f"    {r['body']:10s} {r['family']:20s} n_sats={r['n_sats']:2d}  "
            f"K={r['n_unique_hops']}  KL={r['kl_boltzmann']:.4f}  "
            f"χ²p={r['pval_boltzmann']:.3f}  "
            f"winner={r['winner_aic']}"
        )

    # Best fits
    ranked_best = sorted(
        [r for r in results if not math.isnan(r["kl_boltzmann"])], key=lambda r: r["kl_boltzmann"]
    )
    print("\n  Top 10 best Boltzmann fits (by KL):")
    for r in ranked_best[:10]:
        print(
            f"    {r['body']:10s} {r['family']:20s} n_sats={r['n_sats']:2d}  "
            f"K={r['n_unique_hops']}  KL={r['kl_boltzmann']:.6f}  "
            f"χ²p={r['pval_boltzmann']:.3f}  b={r['b_opt']:+.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = _HERE / "boltzmann_gof_results.json"
    output = {
        "provenance": {
            "script": "runs/run_boltzmann_gof.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input": "runs/hop_histogram_survey_results.json",
            "n_testable": len(testable),
            "n_analyzed": len(results),
            "n_skipped": n_skip,
            "n_no_lyapunov": n_no_lyap,
            "wall_s": wall_s,
            "smoke": smoke,
        },
        "summary": {
            "aic_winners": dict(winner_counts),
            "boltzmann_not_rejected_05": (n_not if testable_pval else None),
            "boltzmann_rejected_05": (n_rejected if testable_pval else None),
            "kl_boltzmann_mean": float(np.mean(kl_b)) if kl_b else None,
            "kl_uniform_mean": float(np.mean(kl_u)) if kl_u else None,
            "kl_power_law_mean": float(np.mean(kl_p)) if kl_p else None,
            "beta_eff_mean": float(np.mean(beta_vals)) if beta_vals else None,
            "beta_eff_std": float(np.std(beta_vals)) if beta_vals else None,
            "b_opt_mean": float(np.mean(b_vals)),
            "b_opt_std": float(np.std(b_vals)),
        },
        "results": results,
    }

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    with open(out_path, "w") as f:
        json.dump(_sanitize(output), f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
