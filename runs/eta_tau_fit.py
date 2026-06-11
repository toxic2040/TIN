"""run_eta_tau_fit.py -- Fit CTMC absorbing-barrier model to TTL surface data.

Two models fitted to the 45-point TTL x Distance surface from
ttl_surface_results.json:

Model 1 — CTMC absorbing barrier:
  eta_tau = eta_inf * (1 - exp(-mu * T_eff))

  where eta_inf = s/(s+1) is the asymptotic efficiency (from normal-mode),
  mu is the effective routing rate (contact opportunities per second),
  T_eff = TTL - OWLT is the effective routing time budget.

  Note: the full CTMC form (s/(s+1))*(1-exp(-(s+1)*mu*T_eff)) is
  over-parameterised (s and mu are degenerate). We fit the reduced
  form with eta_inf and mu as free parameters.

Model 2 — Stretched exponential:
  eta_tau = eta_inf * (1 - exp(-(T_eff / tau)^alpha))

  where tau is the characteristic routing time, alpha is the stretching
  exponent (alpha < 1 = sub-exponential saturation).

Both models are fit to the active points (T_eff > 0) using
scipy.optimize.curve_fit with measurement uncertainties.

Output:
  - eta_tau_fit_results.json
  - fig_eta_tau_fit.pdf    (residual diagnostic)
  - fig_eta_tau_collapse.pdf (money plot: collapse + fitted curves)
"""

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from scipy.optimize import curve_fit

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def model_ctmc(t_eff, eta_inf, mu):
    """CTMC absorbing barrier: eta = eta_inf * (1 - exp(-mu * t_eff))."""
    return eta_inf * (1.0 - np.exp(-mu * t_eff))


def model_stretched(t_eff, eta_inf, tau, alpha):
    """Stretched exponential: eta = eta_inf * (1 - exp(-(t_eff/tau)^alpha))."""
    return eta_inf * (1.0 - np.exp(-np.power(t_eff / tau, alpha)))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_data():
    with open(_HERE / "ttl_surface_results.json") as f:
        grid = json.load(f)

    t_eff = []
    eta_tau = []
    eta_std = []
    ttl_vals = []
    dist_vals = []

    for key, rec in grid.items():
        eff = rec["effective_ttl"]
        if eff <= 0:
            continue  # blocked by OWLT
        t_eff.append(eff)
        eta_tau.append(rec["eta_tau"])
        eta_std.append(rec["eta_std"] if rec["eta_std"] > 0 else 0.01)
        ttl_vals.append(rec["ttl_s"])
        dist_vals.append(rec["dist_au"])

    return (
        np.array(t_eff),
        np.array(eta_tau),
        np.array(eta_std),
        np.array(ttl_vals),
        np.array(dist_vals),
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_models(t_eff, eta_tau, eta_std):
    results = {}

    # --- Model 1: CTMC ---
    try:
        p0_ctmc = [0.91, 1e-4]
        bounds_ctmc = ([0.5, 1e-7], [1.0, 1e-1])
        popt_c, pcov_c = curve_fit(
            model_ctmc,
            t_eff,
            eta_tau,
            p0=p0_ctmc,
            sigma=eta_std,
            absolute_sigma=True,
            bounds=bounds_ctmc,
            maxfev=10000,
        )
        perr_c = np.sqrt(np.diag(pcov_c))

        eta_pred_c = model_ctmc(t_eff, *popt_c)
        ss_res = np.sum((eta_tau - eta_pred_c) ** 2)
        ss_tot = np.sum((eta_tau - np.mean(eta_tau)) ** 2)
        r2_c = 1.0 - ss_res / ss_tot

        results["ctmc"] = {
            "eta_inf": round(float(popt_c[0]), 6),
            "eta_inf_err": round(float(perr_c[0]), 6),
            "mu": float(popt_c[1]),
            "mu_err": float(perr_c[1]),
            "tau_routing_s": round(1.0 / float(popt_c[1]), 1),
            "R2": round(r2_c, 6),
            "residual_rms": round(float(np.sqrt(np.mean((eta_tau - eta_pred_c) ** 2))), 6),
            "residual_max": round(float(np.max(np.abs(eta_tau - eta_pred_c))), 6),
        }
        print("  CTMC fit:")
        print(f"    eta_inf = {popt_c[0]:.4f} ± {perr_c[0]:.4f}")
        print(f"    mu      = {popt_c[1]:.6e} ± {perr_c[1]:.6e}")
        print(f"    tau_routing = 1/mu = {1.0 / popt_c[1]:.0f} s")
        print(f"    R²      = {r2_c:.6f}")
        print(f"    RMS     = {np.sqrt(np.mean((eta_tau - eta_pred_c) ** 2)):.4f}")
    except Exception as e:
        print(f"  CTMC fit FAILED: {e}")
        popt_c = None
        r2_c = None

    # --- Model 2: Stretched exponential ---
    try:
        p0_se = [0.91, 5000.0, 0.7]
        bounds_se = ([0.5, 100.0, 0.1], [1.0, 100000.0, 2.0])
        popt_s, pcov_s = curve_fit(
            model_stretched,
            t_eff,
            eta_tau,
            p0=p0_se,
            sigma=eta_std,
            absolute_sigma=True,
            bounds=bounds_se,
            maxfev=10000,
        )
        perr_s = np.sqrt(np.diag(pcov_s))

        eta_pred_s = model_stretched(t_eff, *popt_s)
        ss_res = np.sum((eta_tau - eta_pred_s) ** 2)
        ss_tot = np.sum((eta_tau - np.mean(eta_tau)) ** 2)
        r2_s = 1.0 - ss_res / ss_tot

        results["stretched_exp"] = {
            "eta_inf": round(float(popt_s[0]), 6),
            "eta_inf_err": round(float(perr_s[0]), 6),
            "tau": round(float(popt_s[1]), 1),
            "tau_err": round(float(perr_s[1]), 1),
            "alpha": round(float(popt_s[2]), 4),
            "alpha_err": round(float(perr_s[2]), 4),
            "R2": round(r2_s, 6),
            "residual_rms": round(float(np.sqrt(np.mean((eta_tau - eta_pred_s) ** 2))), 6),
            "residual_max": round(float(np.max(np.abs(eta_tau - eta_pred_s))), 6),
        }
        print("\n  Stretched exponential fit:")
        print(f"    eta_inf = {popt_s[0]:.4f} ± {perr_s[0]:.4f}")
        print(f"    tau     = {popt_s[1]:.0f} ± {perr_s[1]:.0f} s")
        print(f"    alpha   = {popt_s[2]:.4f} ± {perr_s[2]:.4f}")
        print(f"    R²      = {r2_s:.6f}")
        print(f"    RMS     = {np.sqrt(np.mean((eta_tau - eta_pred_s) ** 2)):.4f}")
    except Exception as e:
        print(f"  Stretched exponential fit FAILED: {e}")
        popt_s = None
        r2_s = None

    return results, popt_c, popt_s


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_fit_diagnostic(t_eff, eta_tau, eta_std, dist_vals, popt_c, popt_s):
    """Residual diagnostic plot."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color by distance
    dists_unique = sorted(set(dist_vals))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dists_unique)))
    dist_color = {d: colors[i] for i, d in enumerate(dists_unique)}
    c = [dist_color[d] for d in dist_vals]

    # (0,0) Measured vs Predicted — CTMC
    ax = axes[0, 0]
    if popt_c is not None:
        pred_c = model_ctmc(t_eff, *popt_c)
        ax.scatter(pred_c, eta_tau, c=c, s=30, zorder=3)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Predicted $\\eta_\\tau$ (CTMC)")
        ax.set_ylabel("Measured $\\eta_\\tau$")
        ax.set_title("CTMC: Predicted vs Measured")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    # (0,1) Measured vs Predicted — Stretched
    ax = axes[0, 1]
    if popt_s is not None:
        pred_s = model_stretched(t_eff, *popt_s)
        ax.scatter(pred_s, eta_tau, c=c, s=30, zorder=3)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Predicted $\\eta_\\tau$ (Stretched Exp)")
        ax.set_ylabel("Measured $\\eta_\\tau$")
        ax.set_title("Stretched Exp: Predicted vs Measured")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    # (1,0) Residuals vs T_eff — CTMC
    ax = axes[1, 0]
    if popt_c is not None:
        res_c = eta_tau - pred_c
        ax.scatter(t_eff, res_c, c=c, s=30, zorder=3)
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("$T_{eff}$ (s)")
        ax.set_ylabel("Residual")
        ax.set_title(f"CTMC Residuals (RMS={np.sqrt(np.mean(res_c**2)):.4f})")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)

    # (1,1) Residuals vs T_eff — Stretched
    ax = axes[1, 1]
    if popt_s is not None:
        res_s = eta_tau - pred_s
        ax.scatter(t_eff, res_s, c=c, s=30, zorder=3)
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("$T_{eff}$ (s)")
        ax.set_ylabel("Residual")
        ax.set_title(f"Stretched Exp Residuals (RMS={np.sqrt(np.mean(res_s**2)):.4f})")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)

    # Legend for distance colors
    for d in dists_unique:
        axes[0, 0].scatter([], [], c=[dist_color[d]], label=f"{d:.1f} AU", s=30)
    axes[0, 0].legend(title="Distance", fontsize=7, loc="lower right")

    fig.suptitle("$\\eta_\\tau$ Model Fit Diagnostics — Mars K=1", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_eta_tau_fit.{ext}", dpi=200)
    print("  fig_eta_tau_fit saved")
    plt.close()


def plot_collapse(t_eff, eta_tau, eta_std, dist_vals, ttl_vals, popt_c, popt_s):
    """Money plot: all 33 points vs T_eff with fitted curves."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by distance
    dists_unique = sorted(set(dist_vals))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dists_unique)))
    dist_color = {d: colors[i] for i, d in enumerate(dists_unique)}

    # Plot data points colored by distance
    for d in dists_unique:
        mask = dist_vals == d
        ax.errorbar(
            t_eff[mask],
            eta_tau[mask],
            yerr=eta_std[mask],
            fmt="D",
            ms=6,
            color=dist_color[d],
            capsize=3,
            label=f"{d:.1f} AU",
            zorder=3,
        )

    # Fitted curves
    t_curve = np.linspace(10, 50000, 500)

    if popt_c is not None:
        eta_curve_c = model_ctmc(t_curve, *popt_c)
        r2_c = 1.0 - np.sum((eta_tau - model_ctmc(t_eff, *popt_c)) ** 2) / np.sum(
            (eta_tau - np.mean(eta_tau)) ** 2
        )
        ax.plot(
            t_curve,
            eta_curve_c,
            "-",
            color="#F44336",
            lw=2.5,
            label=f"CTMC: $\\eta_\\infty$={popt_c[0]:.3f}, "
            f"$\\mu$={popt_c[1]:.1e} ($R^2$={r2_c:.4f})",
        )

    if popt_s is not None:
        eta_curve_s = model_stretched(t_curve, *popt_s)
        r2_s = 1.0 - np.sum((eta_tau - model_stretched(t_eff, *popt_s)) ** 2) / np.sum(
            (eta_tau - np.mean(eta_tau)) ** 2
        )
        ax.plot(
            t_curve,
            eta_curve_s,
            "--",
            color="#2196F3",
            lw=2.5,
            label=f"Stretched: $\\eta_\\infty$={popt_s[0]:.3f}, "
            f"$\\tau$={popt_s[1]:.0f}s, "
            f"$\\alpha$={popt_s[2]:.2f} ($R^2$={r2_s:.4f})",
        )

    ax.set_xlabel("Effective Routing Time $T_{eff}$ = TTL $-$ OWLT  (seconds)", fontsize=12)
    ax.set_ylabel("$\\eta_\\tau$", fontsize=14)
    ax.set_title(
        "TTL Surface Collapse: $\\eta_\\tau$ vs $T_{eff}$  (Mars K=1, 5 distances)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlim(30, 60000)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    # Annotate OWLT
    ax.axvline(499.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(520, 0.05, "OWLT(1 AU)", fontsize=7, color="gray", rotation=90)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(_HERE / f"fig_eta_tau_collapse.{ext}", dpi=200)
    print("  fig_eta_tau_collapse saved")
    plt.close()


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def print_interpretation(results, t_eff, eta_tau):
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    # Collapse quality
    # Group by effective TTL bins (1000s bins) and check spread
    bins = {}
    for i, te in enumerate(t_eff):
        bucket = round(te / 500) * 500
        bins.setdefault(bucket, []).append(eta_tau[i])

    print("\n  Collapse test (500s bins with >=2 points):")
    print(f"    {'T_eff bin':>10}  {'n':>3}  {'mean':>7}  {'std':>7}  {'CV%':>6}")
    print("    " + "-" * 40)
    collapse_cvs = []
    for b in sorted(bins.keys()):
        if len(bins[b]) >= 2:
            m = np.mean(bins[b])
            s = np.std(bins[b], ddof=1)
            cv = 100 * s / m if m > 0 else 0
            collapse_cvs.append(cv)
            print(f"    {b:>9.0f}s  {len(bins[b]):>3}  {m:>7.4f}  {s:>7.4f}  {cv:>5.1f}%")

    if collapse_cvs:
        mean_cv = np.mean(collapse_cvs)
        print(f"\n    Mean CV across bins: {mean_cv:.1f}%")
        if mean_cv < 5:
            print("    -> STRONG collapse: T_eff is the correct independent variable")
        elif mean_cv < 10:
            print("    -> MODERATE collapse: T_eff captures most variance")
        else:
            print("    -> WEAK collapse: residual distance dependence remains")

    # Model comparison
    if "ctmc" in results and "stretched_exp" in results:
        r2_c = results["ctmc"]["R2"]
        r2_s = results["stretched_exp"]["R2"]
        print("\n  Model comparison:")
        print(f"    CTMC R²              = {r2_c:.6f}")
        print(f"    Stretched Exp R²     = {r2_s:.6f}")
        print(f"    Delta R²             = {r2_s - r2_c:.6f}")

        if r2_s - r2_c > 0.01:
            print("    -> Stretched exponential significantly better")
        elif abs(r2_s - r2_c) < 0.005:
            print("    -> Models equivalent; prefer simpler CTMC (2 params)")
        else:
            print("    -> CTMC better; exponential saturation is adequate")

    # Asymptotic eta
    if "ctmc" in results:
        eta_inf = results["ctmc"]["eta_inf"]
        tau_r = results["ctmc"]["tau_routing_s"]
        print("\n  Physical parameters (CTMC):")
        print(
            f"    eta_inf = {eta_inf:.4f}  "
            f"(-> s = eta_inf/(1-eta_inf) = {eta_inf / (1 - eta_inf):.1f})"
        )
        print(f"    tau_routing = 1/mu = {tau_r:.0f}s  ({tau_r / 3600:.1f} hours)")
        print("    For Mars K=1 (6 polar, 70N, DSN=12h)")

    # Non-monotonicity note
    long_etas = eta_tau[t_eff > 25000]
    mid_etas = eta_tau[(t_eff > 10000) & (t_eff < 20000)]
    if len(long_etas) > 0 and len(mid_etas) > 0:
        if np.mean(long_etas) < np.mean(mid_etas):
            print("\n  NOTE: eta_tau decreases at very long T_eff:")
            print(f"    T_eff 10-20ks: mean eta = {np.mean(mid_etas):.4f}")
            print(f"    T_eff >25ks:   mean eta = {np.mean(long_etas):.4f}")
            print("    -> Greedy dead-end accumulation at long horizons")
            print("    -> Neither model captures this (both monotonically approach eta_inf)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  eta_tau Model Fitting — CTMC + Stretched Exponential")
    print("=" * 60)

    t_eff, eta_tau, eta_std, ttl_vals, dist_vals = load_data()
    n_active = len(t_eff)
    print(f"  Active points: {n_active} (T_eff > 0)")
    print(f"  T_eff range: {t_eff.min():.0f} – {t_eff.max():.0f} s")
    print(f"  eta_tau range: {eta_tau.min():.4f} – {eta_tau.max():.4f}")
    print(f"  Distances: {sorted(set(dist_vals))}")

    results, popt_c, popt_s = fit_models(t_eff, eta_tau, eta_std)

    print_interpretation(results, t_eff, eta_tau)

    # Save results
    out_path = _HERE / "eta_tau_fit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path.name}")

    # Plots
    plot_fit_diagnostic(t_eff, eta_tau, eta_std, dist_vals, popt_c, popt_s)
    plot_collapse(t_eff, eta_tau, eta_std, dist_vals, ttl_vals, popt_c, popt_s)

    print("\nDone.")


if __name__ == "__main__":
    main()
