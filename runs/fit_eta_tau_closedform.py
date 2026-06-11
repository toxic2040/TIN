"""fit_eta_tau_closedform.py — Path 3: η_tau closed-form derivation and fit.

Theory (CTMC with TTL barrier):
  η_tau(T_eff) = η_∞ · (1 - exp(-r · T_eff))
where T_eff = max(0, TTL - OWLT) is the effective routing time budget,
η_∞ = s/(s+1) from normal-mode, r = total CTMC exit rate.

Datasets:
  1. Distance sweep (6 points): fixed TTL=3600s, varying OWLT (0.5-2.5 AU)
  2. TTL universality (5 configs × 8 TTL values): varying TTL, fixed OWLT

Models tested:
  A: Fixed-asymptote CTMC — η_tau = s/(s+1)·(1 - exp(-r·T_eff))  [1 param: r]
  B: Free-asymptote CTMC  — η_tau = η_∞·(1 - exp(-r·T_eff))      [2 params: η_∞, r]
  C: Stretched exponential — η_tau = η_∞·(1 - exp(-(r·T_eff)^α))  [3 params]

Key finding: the simple CTMC model fails for TTL sweeps due to oracle
conditioning (at small TTL, only easy bundles are feasible → η_tau biased
upward). The fit diagnostic identifies this as a systematic residual.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
with open(_HERE / "distance_sweep_results.json") as f:
    DIST_DATA = json.load(f)

with open(_HERE / "ttl_universality_results.json") as f:
    TTL_DATA = json.load(f)

# Known s values from soft penalty / instrumented runs
S_KNOWN = {
    "Moon K=1 (8 polar)": 52.3,
    "Moon K=2 (8P+ELFO)": 6.1,
    "Moon K=3 (8P+ELFO+halo)": 8.9,
    "Mars K=1 (6 polar)": 85.2,
    "Mars K=2 (6P+ecc)": 68.8,
}

# OWLT for TTL configs
_C_KM_S = 299_792.458
_AU_KM = 149_597_870.7
OWLT_MOON = 384_400.0 / _C_KM_S  # ~1.28 s
OWLT_MARS = 2.25 * _AU_KM / _C_KM_S  # ~1123 s

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def model_A(T_eff, r, *, s):
    """Fixed-asymptote CTMC: η = s/(s+1) · (1 - exp(-r·T_eff))."""
    eta_inf = s / (s + 1.0)
    return eta_inf * (1.0 - np.exp(-r * np.clip(T_eff, 0, None)))


def model_B(T_eff, eta_inf, r):
    """Free-asymptote CTMC: η = η_∞ · (1 - exp(-r·T_eff))."""
    return eta_inf * (1.0 - np.exp(-r * np.clip(T_eff, 0, None)))


def model_C(T_eff, eta_inf, r, alpha):
    """Stretched exponential: η = η_∞ · (1 - exp(-(r·T_eff)^α))."""
    x = r * np.clip(T_eff, 0, None)
    return eta_inf * (1.0 - np.exp(-np.power(x, alpha)))


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------


def _r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _rmse(y_obs, y_pred):
    return float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))


def _aic(n, k, rss):
    """AIC for least-squares regression."""
    if rss <= 0 or n <= k + 1:
        return float("inf")
    return n * np.log(rss / n) + 2 * k


def fit_config(T_eff, eta_obs, s, label):
    """Fit all three models to one config. Returns dict of fit results."""
    T_eff = np.asarray(T_eff, dtype=float)
    eta_obs = np.asarray(eta_obs, dtype=float)

    # Filter out zero-TTL points (T_eff <= 0 or eta_obs == 0)
    mask = (T_eff > 0) & (eta_obs > 0)
    T = T_eff[mask]
    y = eta_obs[mask]
    n = len(T)

    if n < 2:
        return {"label": label, "n_points": n, "error": "insufficient data"}

    eta_norm = s / (s + 1.0)
    results = {"label": label, "n_points": n, "s": s, "eta_norm": eta_norm}

    # --- Model A: fixed asymptote ---
    try:

        def _mA(t, r):
            return model_A(t, r, s=s)

        popt, pcov = curve_fit(_mA, T, y, p0=[5e-4], bounds=(0, np.inf))
        y_pred = _mA(T, *popt)
        results["model_A"] = {
            "r": float(popt[0]),
            "eta_inf": eta_norm,
            "R2": _r2(y, y_pred),
            "RMSE": _rmse(y, y_pred),
            "AIC": _aic(n, 1, np.sum((y - y_pred) ** 2)),
            "residuals": (y - y_pred).tolist(),
        }
    except Exception as e:
        results["model_A"] = {"error": str(e)}

    # --- Model B: free asymptote ---
    try:
        popt, pcov = curve_fit(model_B, T, y, p0=[eta_norm, 5e-4], bounds=([0, 0], [1.5, 1.0]))
        y_pred = model_B(T, *popt)
        results["model_B"] = {
            "eta_inf": float(popt[0]),
            "r": float(popt[1]),
            "R2": _r2(y, y_pred),
            "RMSE": _rmse(y, y_pred),
            "AIC": _aic(n, 2, np.sum((y - y_pred) ** 2)),
            "residuals": (y - y_pred).tolist(),
        }
    except Exception as e:
        results["model_B"] = {"error": str(e)}

    # --- Model C: stretched exponential ---
    try:
        popt, pcov = curve_fit(
            model_C,
            T,
            y,
            p0=[eta_norm, 5e-4, 0.5],
            bounds=([0, 0, 0.01], [1.5, 1.0, 5.0]),
            maxfev=10000,
        )
        y_pred = model_C(T, *popt)
        results["model_C"] = {
            "eta_inf": float(popt[0]),
            "r": float(popt[1]),
            "alpha": float(popt[2]),
            "R2": _r2(y, y_pred),
            "RMSE": _rmse(y, y_pred),
            "AIC": _aic(n, 3, np.sum((y - y_pred) ** 2)),
            "residuals": (y - y_pred).tolist(),
        }
    except Exception as e:
        results["model_C"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Dataset 1: Distance sweep
# ---------------------------------------------------------------------------


def fit_distance_sweep():
    """Fit models to the distance sweep data."""
    print("\n" + "=" * 70)
    print("  DATASET 1: Distance Sweep (Mars 6P, TTL=3600s, varying OWLT)")
    print("=" * 70)

    TTL = 3600.0
    T_eff = np.array([TTL - d["owlt_s"] for d in DIST_DATA])
    eta_tau = np.array([d["eta_tau"] for d in DIST_DATA])

    # Use mean η_norm as s estimate
    eta_norm_mean = np.mean([d["eta_norm"] for d in DIST_DATA])
    s = eta_norm_mean / (1 - eta_norm_mean)

    print(f"  η_norm = {eta_norm_mean:.4f} → s = {s:.1f}")
    print(f"  T_eff range: {T_eff[-1]:.0f} – {T_eff[0]:.0f} s")
    print(f"  η_tau range: {eta_tau[-1]:.4f} – {eta_tau[0]:.4f}")
    print()

    result = fit_config(T_eff, eta_tau, s, "Distance sweep")

    for model_name in ["model_A", "model_B", "model_C"]:
        m = result.get(model_name, {})
        if "error" in m:
            print(f"  {model_name}: FAILED — {m['error']}")
            continue
        params = {k: v for k, v in m.items() if k not in ("R2", "RMSE", "AIC", "residuals")}
        print(f"  {model_name}: R²={m['R2']:.6f}  RMSE={m['RMSE']:.6f}  AIC={m['AIC']:.1f}")
        for k, v in params.items():
            if isinstance(v, float):
                print(f"    {k} = {v:.6f}")
        print(f"    residuals: {['%+.4f' % r for r in m['residuals']]}")
        print()

    return result


# ---------------------------------------------------------------------------
# Dataset 2: TTL universality
# ---------------------------------------------------------------------------


def fit_ttl_sweep():
    """Fit models to each TTL universality config."""
    print("\n" + "=" * 70)
    print("  DATASET 2: TTL Universality Sweep")
    print("=" * 70)

    results = {}
    for key, val in TTL_DATA.items():
        label = val["label"]
        owlt = OWLT_MOON if "Moon" in label else OWLT_MARS

        # Find matching s
        s = None
        for s_key, s_val in S_KNOWN.items():
            if s_key == label:
                s = s_val
                break
        if s is None:
            print(f"\n  {label}: no known s, skipping")
            continue

        ttls = np.array([r["ttl"] for r in val["ttl_sweep"]])
        T_eff = ttls - owlt
        eta_tau = np.array([r["eta_tau"] for r in val["ttl_sweep"]])

        print(f"\n  --- {label} (s={s:.1f}, OWLT={owlt:.1f}s) ---")
        print(f"  TTL range: {ttls[0]:.0f} – {ttls[-1]:.0f} s")
        print(f"  T_eff range: {T_eff[0]:.0f} – {T_eff[-1]:.0f} s")

        result = fit_config(T_eff, eta_tau, s, label)
        results[key] = result

        for model_name in ["model_A", "model_B", "model_C"]:
            m = result.get(model_name, {})
            if "error" in m:
                print(f"  {model_name}: FAILED — {m['error']}")
                continue
            params = {k: v for k, v in m.items() if k not in ("R2", "RMSE", "AIC", "residuals")}
            print(f"  {model_name}: R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  AIC={m['AIC']:.1f}")
            for k, v in params.items():
                if isinstance(v, float):
                    print(f"    {k} = {v:.6f}")

    return results


# ---------------------------------------------------------------------------
# Cross-config r comparison
# ---------------------------------------------------------------------------


def print_r_table(dist_result, ttl_results):
    """Print comparison table of fitted r values across configs."""
    print("\n" + "=" * 70)
    print("  CROSS-CONFIG PARAMETER TABLE (Model B — free asymptote)")
    print("=" * 70)
    print(f"  {'Config':<32} {'s':>6} {'η_∞(fit)':>9} {'r':>10} {'R²':>7} {'RMSE':>7}")
    print("  " + "─" * 66)

    # Distance sweep
    m = dist_result.get("model_B", {})
    if "error" not in m:
        print(
            f"  {'Distance sweep (Mars 6P)':<32} "
            f"{dist_result['s']:>6.1f} {m['eta_inf']:>9.4f} "
            f"{m['r']:>10.6f} {m['R2']:>7.4f} {m['RMSE']:>7.4f}"
        )

    # TTL configs
    for key, res in ttl_results.items():
        m = res.get("model_B", {})
        if "error" in m:
            continue
        print(
            f"  {res['label']:<32} "
            f"{res['s']:>6.1f} {m['eta_inf']:>9.4f} "
            f"{m['r']:>10.6f} {m['R2']:>7.4f} {m['RMSE']:>7.4f}"
        )

    print()


# ---------------------------------------------------------------------------
# Diagnostic: oracle conditioning analysis
# ---------------------------------------------------------------------------


def diagnose_conditioning(ttl_results):
    """Analyze systematic residual patterns indicating oracle conditioning."""
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC: Oracle Conditioning Effect")
    print("=" * 70)
    print()
    print("  If Model A residuals are systematically POSITIVE at small TTL")
    print("  and NEGATIVE at large TTL, the oracle conditioning effect is")
    print("  present: easy bundles dominate at small TTL → η_tau biased up.")
    print()

    for key, res in ttl_results.items():
        m = res.get("model_A", {})
        if "error" in m or "residuals" not in m:
            continue

        resid = np.array(m["residuals"])
        n = len(resid)
        if n < 4:
            continue

        # Split into first half (small TTL) and second half (large TTL)
        first_half = resid[: n // 2]
        second_half = resid[n // 2 :]
        bias_small = float(np.mean(first_half))
        bias_large = float(np.mean(second_half))

        pattern = "CONDITIONING" if bias_small > 0.01 and bias_large < -0.01 else "WEAK/NONE"

        print(
            f"  {res['label']:<32}  small-TTL bias={bias_small:+.4f}  "
            f"large-TTL bias={bias_large:+.4f}  → {pattern}"
        )

    print()
    print("  Interpretation:")
    print("  - CONDITIONING: simple CTMC fails; need oracle-integrated model")
    print("  - WEAK/NONE: CTMC model adequate for this architecture")


# ---------------------------------------------------------------------------
# Closed-form prediction formula
# ---------------------------------------------------------------------------


def derive_prediction_formula(dist_result, ttl_results):
    """Derive the best-fit prediction formula from all data."""
    print("\n" + "=" * 70)
    print("  DERIVED PREDICTION FORMULA")
    print("=" * 70)

    # Collect all Model C fits (stretched exponential — best flexibility)
    all_r = []
    all_alpha = []
    all_eta_inf_ratio = []

    for res in [dist_result] + list(ttl_results.values()):
        m = res.get("model_C", {})
        if "error" in m or m.get("R2", 0) < 0.9:
            continue
        all_r.append(m["r"])
        all_alpha.append(m["alpha"])
        if "eta_norm" in res:
            all_eta_inf_ratio.append(m["eta_inf"] / res["eta_norm"])

    if all_alpha:
        print("\n  Stretched exponential (Model C) summary:")
        print(f"    α across configs: {[f'{a:.3f}' for a in all_alpha]}")
        print(f"    r across configs: {[f'{r:.6f}' for r in all_r]}")
        mean_alpha = np.mean(all_alpha)
        print(f"    mean α = {mean_alpha:.3f}")
        if all_eta_inf_ratio:
            print(f"    η_∞/η_norm ratios: {[f'{r:.3f}' for r in all_eta_inf_ratio]}")

    # Best simple formula: use Model B with config-specific r
    print("\n  Recommended formula:")
    print("    η_tau(TTL, OWLT) = η_∞ · (1 - exp(-r · max(0, TTL - OWLT)))")
    print("    where η_∞ and r are fitted per-architecture from ONE TTL sweep")
    print()

    # For the distance sweep specifically (the cleanest dataset)
    m_dist = dist_result.get("model_B", {})
    if "error" not in m_dist:
        print("  Mars 6-orbiter prediction:")
        print(
            f"    η_tau(TTL, OWLT) = {m_dist['eta_inf']:.4f} · "
            f"(1 - exp(-{m_dist['r']:.6f} · (TTL - OWLT)))"
        )
        print(f"    R² = {m_dist['R2']:.6f},  RMSE = {m_dist['RMSE']:.6f}")

        # Validate: predict distance sweep
        print("\n  Distance sweep validation:")
        print(f"  {'AU':>5} {'OWLT':>7} {'T_eff':>6} {'η_obs':>7} {'η_pred':>7} {'resid':>7}")
        print("  " + "─" * 42)
        for d in DIST_DATA:
            T_eff = 3600.0 - d["owlt_s"]
            pred = model_B(T_eff, m_dist["eta_inf"], m_dist["r"])
            resid = d["eta_tau"] - pred
            print(
                f"  {d['distance_au']:>5.2f} {d['owlt_s']:>7.0f} "
                f"{T_eff:>6.0f} {d['eta_tau']:>7.4f} "
                f"{pred:>7.4f} {resid:>+7.4f}"
            )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_all(dist_result, ttl_results):
    """Generate comparison figures."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Panel (0,0): Distance sweep Model B ---
    ax = axes[0, 0]
    T_eff_obs = np.array([3600 - d["owlt_s"] for d in DIST_DATA])
    eta_obs = np.array([d["eta_tau"] for d in DIST_DATA])

    ax.plot(T_eff_obs, eta_obs, "ko", markersize=8, label="Observed", zorder=5)

    T_dense = np.linspace(1500, 4000, 200)
    for model_name, color, ls in [
        ("model_A", "C0", "--"),
        ("model_B", "C1", "-"),
        ("model_C", "C3", ":"),
    ]:
        m = dist_result.get(model_name, {})
        if "error" in m:
            continue
        if model_name == "model_A":
            s = dist_result["s"]
            y = model_A(T_dense, m["r"], s=s)
        elif model_name == "model_B":
            y = model_B(T_dense, m["eta_inf"], m["r"])
        else:
            y = model_C(T_dense, m["eta_inf"], m["r"], m["alpha"])
        ax.plot(T_dense, y, color=color, ls=ls, label=f"{model_name} (R²={m['R2']:.4f})")

    ax.set_xlabel("T_eff = TTL − OWLT (s)")
    ax.set_ylabel("η_τ")
    ax.set_title("A: Distance Sweep (Mars 6P)")
    ax.legend(fontsize=7)

    # --- Panels (0,1)-(1,2): TTL sweep configs ---
    config_panels = [
        ("Moon_K=1_8_polar", (0, 1)),
        ("Moon_K=2_8P+ELFO", (0, 2)),
        ("Moon_K=3_8P+ELFO+halo", (1, 0)),
        ("Mars_K=1_6_polar", (1, 1)),
        ("Mars_K=2_6P+ecc", (1, 2)),
    ]

    for ttl_key, (row, col) in config_panels:
        ax = axes[row, col]
        val = TTL_DATA.get(ttl_key)
        res = ttl_results.get(ttl_key)
        if val is None or res is None:
            ax.set_visible(False)
            continue

        label = val["label"]
        owlt = OWLT_MOON if "Moon" in label else OWLT_MARS

        ttls = np.array([r["ttl"] for r in val["ttl_sweep"]])
        T_eff_obs = ttls - owlt
        eta_obs = np.array([r["eta_tau"] for r in val["ttl_sweep"]])
        eta_err = np.array([r["eta_tau_std"] for r in val["ttl_sweep"]])

        # Filter valid points for plotting
        mask = (T_eff_obs > 0) & (eta_obs > 0)
        ax.errorbar(
            T_eff_obs[mask],
            eta_obs[mask],
            yerr=eta_err[mask],
            fmt="ko",
            markersize=6,
            capsize=3,
            label="Observed",
            zorder=5,
        )

        T_dense = np.linspace(max(0, T_eff_obs[mask].min() * 0.8), T_eff_obs[mask].max() * 1.1, 200)

        for model_name, color, ls in [
            ("model_A", "C0", "--"),
            ("model_B", "C1", "-"),
            ("model_C", "C3", ":"),
        ]:
            m = res.get(model_name, {})
            if "error" in m:
                continue
            if model_name == "model_A":
                s = res["s"]
                y = model_A(T_dense, m["r"], s=s)
            elif model_name == "model_B":
                y = model_B(T_dense, m["eta_inf"], m["r"])
            else:
                y = model_C(T_dense, m["eta_inf"], m["r"], m["alpha"])
            ax.plot(T_dense, y, color=color, ls=ls, label=f"{model_name[-1]} R²={m['R2']:.3f}")

        ax.set_xlabel("T_eff (s)")
        ax.set_ylabel("η_τ")
        ax.set_title(label)
        ax.legend(fontsize=6)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_eta_tau_closedform.{ext}", dpi=200)
    print("  fig_eta_tau_closedform saved")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not HAS_SCIPY:
        print("ERROR: scipy required for curve fitting")
        print("  pip install scipy")
        sys.exit(1)

    print("=" * 70)
    print("  η_tau Closed-Form Derivation and Fit")
    print("=" * 70)

    dist_result = fit_distance_sweep()
    ttl_results = fit_ttl_sweep()
    print_r_table(dist_result, ttl_results)
    diagnose_conditioning(ttl_results)
    derive_prediction_formula(dist_result, ttl_results)

    # Save all results
    output = {
        "distance_sweep": dist_result,
        "ttl_configs": {k: v for k, v in ttl_results.items()},
        "model_descriptions": {
            "A": "Fixed-asymptote CTMC: η = s/(s+1)·(1-exp(-r·T_eff)), 1 param",
            "B": "Free-asymptote CTMC:  η = η_∞·(1-exp(-r·T_eff)), 2 params",
            "C": "Stretched exponential: η = η_∞·(1-exp(-(r·T_eff)^α)), 3 params",
        },
    }
    out_path = _HERE / "eta_tau_closedform_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path.name}")

    plot_all(dist_result, ttl_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
