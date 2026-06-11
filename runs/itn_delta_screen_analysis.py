#!/usr/bin/env python3
"""ITN Analysis: Closed-form for the oracle screening premium Delta_screen.

Loads the dwell diamond results (200-instance ensemble, 3 configs,
18 tau_half values) and derives a closed-form expression for
Delta_screen as a function of dwell distribution parameters and
commodity hazard rate.

Key result:  Delta_screen_j ~ (sigma^2_D_j / H_j) * lambda^2

where sigma^2_D_j = n_intermediate_j * T_window_j^2 / 12 is the
total dwell variance on path j, H_j is the hop count, and
lambda = ln(2) / tau_half is the commodity decay hazard.

This connects the screening premium to the var_log_p scaling law
(Section 3.8): both are driven by sigma^2_t * lambda^2.

References: Whitepaper Section 3.5 (oracle screening premium).
"""

import json
import os

import numpy as np

# Optional: publication figure
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tin_figure_style import apply_style, figsize_double, save_fig

    apply_style("pre")
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

RUNS = os.path.dirname(os.path.abspath(__file__))


def main():
    # ── Load diamond results ──────────────────────────────────────
    with open(os.path.join(RUNS, "dwell_diamond_results.json")) as f:
        data = json.load(f)

    # ── Extract Delta_screen for all (config, path, tau_half) ─────
    records = []  # (label, path_id, H, sigma2_D, tau, lambda, delta_screen)

    for cfg in data:
        label = cfg["label"]
        paths = cfg["paths"]

        for path_id in ("upper", "lower"):
            p = paths[path_id]
            H = p["H"]
            T_w = p["T_window_days"]
            n_int = H - 1  # intermediate nodes = H - 1 for a diamond
            sigma2_D = n_int * T_w**2 / 12.0  # Var(total dwell) for Uniform

            for entry in cfg["sweep_exponential"]:
                tau = entry["tau_half_days"]
                if tau is None:
                    continue  # skip DTN baseline (no decay)

                sel_key = f"subset_{path_id}_sel"
                cf_key = f"counterfactual_{path_id}"

                sel = entry[sel_key]
                cf = entry[cf_key]

                if sel["mean_lambda_eff"] is None or cf["mean_lambda_eff"] is None:
                    continue

                ds = sel["mean_lambda_eff"] - cf["mean_lambda_eff"]
                lam = np.log(2) / tau  # decay hazard (1/day)

                records.append(
                    {
                        "label": label,
                        "path": path_id,
                        "H": H,
                        "T_w": T_w,
                        "n_int": n_int,
                        "sigma2_D": sigma2_D,
                        "tau_half": tau,
                        "lambda": lam,
                        "delta_screen": ds,
                        "n_sel": sel["n_instances"],
                    }
                )

    print(f"Extracted {len(records)} (config, path, tau) data points.\n")

    # ── Candidate model: Delta_screen ~ a * (sigma^2_D / H) * lambda^2
    #
    # Physical reasoning:
    #   - Oracle cherry-picks instances with short dwell
    #   - Benefit scales with dwell variance (sigma^2_D)
    #   - Benefit scales with decay rate^2 (lambda^2) because the
    #     per-hop log-survival perturbation is proportional to lambda*dwell
    #     and its variance is lambda^2 * Var(dwell)
    #   - Per-hop quantity: normalise by H
    # ──────────────────────────────────────────────────────────────

    # Filter to records with nonzero selection (oracle actually uses this path)
    recs = [r for r in records if r["n_sel"] > 0 and r["sigma2_D"] > 0]

    # Predictor: x = (sigma^2_D / H) * lambda^2
    x = np.array([r["sigma2_D"] / r["H"] * r["lambda"] ** 2 for r in recs])
    y = np.array([r["delta_screen"] for r in recs])

    # ── Fit 1: simple linear  y = a * x ───────────────────────────
    # (no intercept — Delta_screen → 0 as lambda → 0 by definition)
    a_hat = np.sum(x * y) / np.sum(x**2)
    y_pred = a_hat * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("=" * 65)
    print("MODEL 1:  Delta_screen = a * (sigma^2_D / H) * lambda^2")
    print(f"  a_hat = {a_hat:.6f}")
    print(f"  R^2   = {r2:.6f}")
    print(f"  RMSE  = {np.sqrt(np.mean((y - y_pred) ** 2)):.6f}")
    print()

    # ── Fit 2: add frac-selected correction ───────────────────────
    # When the oracle selects a small fraction, bias is larger.
    # Try: y = a * x + b * x * (1 - frac) where frac = n_sel / 200
    frac = np.array([r["n_sel"] / 200.0 for r in recs])
    X2 = np.column_stack([x, x * (1.0 - frac)])
    beta2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
    y_pred2 = X2 @ beta2
    ss_res2 = np.sum((y - y_pred2) ** 2)
    r2_2 = 1.0 - ss_res2 / ss_tot if ss_tot > 0 else float("nan")

    print("MODEL 2:  Delta_screen = a * x + b * x * (1 - frac_selected)")
    print(f"  a = {beta2[0]:.6f},  b = {beta2[1]:.6f}")
    print(f"  R^2   = {r2_2:.6f}")
    print(f"  RMSE  = {np.sqrt(np.mean((y - y_pred2) ** 2)):.6f}")
    print()

    # ── Fit 3: pure quadratic in lambda (alternative predictor) ───
    # x3 = sigma^2_D * lambda^2 (without /H normalisation)
    x3 = np.array([r["sigma2_D"] * r["lambda"] ** 2 for r in recs])
    a3 = np.sum(x3 * y) / np.sum(x3**2)
    y_pred3 = a3 * x3
    ss_res3 = np.sum((y - y_pred3) ** 2)
    r2_3 = 1.0 - ss_res3 / ss_tot if ss_tot > 0 else float("nan")

    print("MODEL 3:  Delta_screen = a * sigma^2_D * lambda^2  (no /H)")
    print(f"  a_hat = {a3:.6f}")
    print(f"  R^2   = {r2_3:.6f}")
    print(f"  RMSE  = {np.sqrt(np.mean((y - y_pred3) ** 2)):.6f}")
    print()

    # ── Summary table ──────────────────────────────────────────────
    print("=" * 65)
    print("DELTA_SCREEN BY CONFIG AND TAU_HALF")
    print("-" * 65)
    print(f"{'Config':<35} {'path':>5} {'tau':>6} {'DS':>10} {'pred':>10}")
    print("-" * 65)
    for r in recs:
        pred = a_hat * r["sigma2_D"] / r["H"] * r["lambda"] ** 2
        print(
            f"{r['label'][:35]:<35} {r['path']:>5} "
            f"{r['tau_half']:>6.0f} {r['delta_screen']:>+10.6f} "
            f"{pred:>+10.6f}"
        )

    # ── Physical interpretation ────────────────────────────────────
    print()
    print("=" * 65)
    print("CLOSED-FORM RESULT")
    print()
    print("  Delta_screen_j  ≈  %.4f × (σ²_D_j / H_j) × λ²" % a_hat)
    print()
    print("  where:")
    print("    σ²_D_j = n_intermediate × T²_window / 12  (Uniform dwell variance)")
    print("    H_j    = hop count on path j")
    print("    λ      = ln(2) / τ_half  (commodity hazard rate)")
    print()
    print("  Connection to var_log_p scaling law (Section 3.8):")
    print("    var_log_p(λ) = σ²_hw − 2ρ_ht λ + σ²_t λ²")
    print("    The screening premium is driven by the SAME σ²_t λ² term")
    print("    that controls trap severity.  The oracle screening premium")
    print("    is a selection-bias correction to the per-hop Lyapunov")
    print("    exponent, proportional to the within-path exposure-time")
    print("    variance.")
    print()
    if r2 > 0.7:
        print(f"  ✓ Model fits well (R² = {r2:.3f}).  The closed form is adequate")
        print("    for reporting requirements (Section 3.5).")
    else:
        print(f"  △ Model R² = {r2:.3f} — additional structure may be needed.")

    # ── Save results ───────────────────────────────────────────────
    results = {
        "model": "Delta_screen = a * (sigma2_D / H) * lambda^2",
        "a_hat": float(a_hat),
        "R2": float(r2),
        "RMSE": float(np.sqrt(np.mean((y - y_pred) ** 2))),
        "n_data_points": len(recs),
        "records": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}
            for r in recs
        ],
    }
    out_path = os.path.join(RUNS, "itn_delta_screen_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # ── Figure ─────────────────────────────────────────────────────
    if HAS_PLOT:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double("pre", height_ratio=0.50))
        fig.subplots_adjust(wspace=0.30)

        # Left: Delta_screen vs predictor, colored by config
        config_colors = {
            "Relay Chain": "#4477AA",
            "Deep Chain": "#CC3311",
            "Control": "#009988",
        }
        for r in recs:
            for cname, col in config_colors.items():
                if cname.lower() in r["label"].lower():
                    break
            else:
                col = "#888888"
            marker = "s" if r["path"] == "upper" else "o"
            ax1.plot(
                r["sigma2_D"] / r["H"] * r["lambda"] ** 2,
                r["delta_screen"],
                marker,
                color=col,
                markersize=3,
                alpha=0.6,
            )

        # Fit line
        x_fit = np.linspace(0, x.max() * 1.05, 100)
        ax1.plot(
            x_fit,
            a_hat * x_fit,
            "-",
            color="#333333",
            lw=1.2,
            label=(
                "$\\Delta_{\\mathrm{screen}} = %.4f \\,"
                "\\sigma^2_D / H \\cdot \\lambda^2$\n$R^2 = %.3f$" % (a_hat, r2)
            ),
        )

        ax1.set_xlabel("$(\\sigma^2_D / H) \\cdot \\lambda^2$")
        ax1.set_ylabel("$\\Delta_{\\mathrm{screen}}$")
        ax1.legend(fontsize=5.5, loc="upper left")
        ax1.axhline(0, color="#CCCCCC", lw=0.4)

        # Legend for configs
        for cname, col in config_colors.items():
            ax1.plot([], [], "s", color=col, markersize=3, label=cname)
        ax1.legend(fontsize=5, loc="upper left")

        # Right: Delta_screen vs tau_half for deep chain (both paths)
        for r in recs:
            if "deep" not in r["label"].lower():
                continue
            marker = "s" if r["path"] == "upper" else "o"
            col = "#CC3311" if r["path"] == "upper" else "#4477AA"
            ax2.plot(r["tau_half"], r["delta_screen"], marker, color=col, markersize=3.5, alpha=0.7)

        ax2.set_xlabel("$\\tau_{1/2}$ (days)")
        ax2.set_ylabel("$\\Delta_{\\mathrm{screen}}$")
        ax2.set_xscale("log")
        ax2.axhline(0, color="#CCCCCC", lw=0.4)
        ax2.plot([], [], "s", color="#CC3311", markersize=3.5, label="Upper (2-hop)")
        ax2.plot([], [], "o", color="#4477AA", markersize=3.5, label="Lower (4-hop)")
        ax2.legend(fontsize=5.5)
        ax2.set_title("Deep Chain config", fontsize=7)

        save_fig(fig, "itn_delta_screen_fit")
        plt.close(fig)


if __name__ == "__main__":
    main()
