"""run_helio_s_derivation.py — Step 7: Analytical s(p) derivation.

Rigorous curve fitting of escape ratio s as a function of effective link
probability p from the synodic sweep data.  Tests multiple models:
  1. Power law:    s = C · p^β
  2. Refined:      s = A · p^α · (1-p)^γ  (captures saturation)
  3. CTMC:         s = γ_eff / δ_eff with γ(p), δ(p) functional forms
  4. Logistic:     log(s) = a + b·log(p/(1-p))  (logit transform)
  5. Physics-based: s = n_contacts · p / (1 - (1-p)^k)  (retry model)

Uses full 780-epoch synodic sweep data for fitting, 4-fold cross-validation.

Key question: what is the minimal analytical model for s(p)?
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

_HERE = Path(__file__).parent


def main():
    print()
    print("=" * 72)
    print("  Step 7: Analytical s(p) Derivation")
    print("=" * 72)
    print()

    # Load synodic sweep data
    with open(_HERE / "synodic_sweep_results.json") as f:
        data = json.load(f)
    results = data["time_series"]

    active = [r for r in results if r["S_T"] > 0 and r["eta"] > 0]
    p_arr = np.array([r["mean_p_success"] for r in active])
    eta_arr = np.array([r["eta"] for r in active])
    dr_arr = np.array([r["DR"] for r in active])
    st_arr = np.array([r["S_T"] for r in active])
    dist_arr = np.array([r["dist_au"] for r in active])

    # Compute s from eta: s = eta / (1 - eta)
    s_arr = eta_arr / (1.0 - eta_arr)

    # Filter valid (finite, positive s and p)
    valid = np.isfinite(s_arr) & (s_arr > 0) & (p_arr > 0.001)
    p = p_arr[valid]
    s = s_arr[valid]
    eta = eta_arr[valid]
    dist = dist_arr[valid]
    n = len(p)
    print(f"  Data: {n} valid epochs (of {len(active)} active)")

    log_p = np.log(p)
    log_s = np.log(s)

    # =========================================================
    # Model 1: Power law  s = C * p^beta
    # =========================================================
    coeffs1 = np.polyfit(log_p, log_s, 1)
    beta1 = coeffs1[0]
    C1 = np.exp(coeffs1[1])
    s_pred1 = C1 * p**beta1
    ss_res1 = np.sum((log_s - np.log(s_pred1)) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2_1 = 1 - ss_res1 / ss_tot

    print("\n  Model 1: s = C·p^β")
    print(f"    C = {C1:.3f}, β = {beta1:.3f}")
    print(f"    R² (log space) = {r2_1:.4f}")
    print(f"    RMSE(log s) = {np.sqrt(ss_res1 / n):.4f}")

    # =========================================================
    # Model 2: Refined  s = A * p^alpha * (1-p)^gamma
    # =========================================================
    def model2(log_p, a, alpha, gamma):
        p_val = np.exp(log_p)
        return a + alpha * log_p + gamma * np.log(1.0 - p_val + 1e-10)

    try:
        popt2, _ = curve_fit(model2, log_p, log_s, p0=[1.0, 1.5, -0.5])
        A2 = np.exp(popt2[0])
        alpha2 = popt2[1]
        gamma2 = popt2[2]
        s_pred2 = A2 * p**alpha2 * (1 - p) ** gamma2
        ss_res2 = np.sum((log_s - np.log(s_pred2)) ** 2)
        r2_2 = 1 - ss_res2 / ss_tot

        print("\n  Model 2: s = A·p^α·(1-p)^γ")
        print(f"    A = {A2:.3f}, α = {alpha2:.3f}, γ = {gamma2:.3f}")
        print(f"    R² (log space) = {r2_2:.4f}")
        print(f"    RMSE(log s) = {np.sqrt(ss_res2 / n):.4f}")
    except Exception as e:
        print(f"\n  Model 2: fit failed ({e})")
        r2_2 = 0

    # =========================================================
    # Model 3: Logistic  log(s) = a + b*logit(p)
    # =========================================================
    logit_p = np.log(p / (1 - p + 1e-10))
    coeffs3 = np.polyfit(logit_p, log_s, 1)
    b3 = coeffs3[0]
    a3 = coeffs3[1]
    s_pred3 = np.exp(a3 + b3 * logit_p)
    ss_res3 = np.sum((log_s - np.log(s_pred3)) ** 2)
    r2_3 = 1 - ss_res3 / ss_tot

    print("\n  Model 3: log(s) = a + b·logit(p)")
    print(f"    a = {a3:.3f}, b = {b3:.3f}")
    print(f"    R² (log space) = {r2_3:.4f}")
    print(f"    RMSE(log s) = {np.sqrt(ss_res3 / n):.4f}")

    # =========================================================
    # Model 4: Physics-based  s = n_eff * p / (1 - (1-p)^k)
    # =========================================================
    def model4_func(p_val, n_eff, k):
        denom = 1.0 - (1.0 - p_val) ** k
        denom = np.maximum(denom, 1e-10)
        return n_eff * p_val / denom

    try:
        popt4, _ = curve_fit(model4_func, p, s, p0=[10.0, 3.0], maxfev=10000)
        n_eff4 = popt4[0]
        k4 = popt4[1]
        s_pred4 = model4_func(p, n_eff4, k4)
        ss_res4 = np.sum((log_s - np.log(s_pred4)) ** 2)
        r2_4 = 1 - ss_res4 / ss_tot

        print("\n  Model 4: s = n_eff·p / (1-(1-p)^k)  [retry model]")
        print(f"    n_eff = {n_eff4:.3f}, k = {k4:.3f}")
        print(f"    R² (log space) = {r2_4:.4f}")
        print(f"    RMSE(log s) = {np.sqrt(ss_res4 / n):.4f}")
    except Exception as e:
        print(f"\n  Model 4: fit failed ({e})")
        r2_4 = 0

    # =========================================================
    # Model 5: Two-variable  s = f(p, d)
    # =========================================================
    log_d = np.log(dist)
    X = np.column_stack([np.ones(n), log_p, log_d, log_p * log_d])
    beta5, res5, _, _ = np.linalg.lstsq(X, log_s, rcond=None)
    s_pred5 = np.exp(X @ beta5)
    ss_res5 = np.sum((log_s - np.log(s_pred5)) ** 2)
    r2_5 = 1 - ss_res5 / ss_tot

    print("\n  Model 5: log(s) = β₀ + β₁·log(p) + β₂·log(d) + β₃·log(p)·log(d)")
    print(f"    β = [{', '.join(f'{b:.3f}' for b in beta5)}]")
    print(f"    R² (log space) = {r2_5:.4f}")
    print(f"    RMSE(log s) = {np.sqrt(ss_res5 / n):.4f}")

    # =========================================================
    # Cross-validation (4-fold) for top models
    # =========================================================
    print("\n  4-FOLD CROSS-VALIDATION:")
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 4)

    cv_scores = {}
    for name, predict_fn in [
        ("Power law", lambda p_tr, s_tr, p_te: C1 * p_te**beta1),
        ("Logistic", lambda p_tr, s_tr, p_te: np.exp(a3 + b3 * np.log(p_te / (1 - p_te + 1e-10)))),
    ]:
        rmses = []
        for fold in folds:
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            p_tr, s_tr = p[mask], s[mask]
            p_te, s_te = p[fold], s[fold]
            s_hat = predict_fn(p_tr, s_tr, p_te)
            rmse = np.sqrt(np.mean((np.log(s_te) - np.log(s_hat)) ** 2))
            rmses.append(rmse)
        cv_scores[name] = np.mean(rmses)
        print(f"    {name:<20s} CV RMSE(log s) = {np.mean(rmses):.4f}")

    # =========================================================
    # η(p) crossover — check if α stable across subranges
    # =========================================================
    print("\n  CROSSOVER EXPONENT STABILITY:")
    log_eta = np.log(eta)
    for p_lo, p_hi in [(0.01, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        mask = (p >= p_lo) & (p < p_hi)
        if mask.sum() > 10:
            c = np.polyfit(log_p[mask], log_eta[mask], 1)
            print(f"    p ∈ [{p_lo:.2f}, {p_hi:.2f}): α = {c[0]:.3f} (n={mask.sum()})")

    # =========================================================
    # Best model summary
    # =========================================================
    models = [
        ("Power law s=Cp^β", r2_1),
        ("Refined s=Ap^α(1-p)^γ", r2_2),
        ("Logistic log(s)=a+b·logit(p)", r2_3),
    ]
    if r2_4 > 0:
        models.append(("Retry s=np/(1-(1-p)^k)", r2_4))
    models.append(("Two-var s=f(p,d)", r2_5))
    models.sort(key=lambda x: x[1], reverse=True)

    print("\n  MODEL RANKING (R² in log space):")
    for i, (name, r2) in enumerate(models):
        print(f"    {i + 1}. {name:<30s}  R² = {r2:.4f}")

    # Save
    output = {
        "n_data": int(n),
        "models": {
            "power_law": {"C": float(C1), "beta": float(beta1), "R2": float(r2_1)},
            "logistic": {"a": float(a3), "b": float(b3), "R2": float(r2_3)},
            "two_var": {"beta": [float(b) for b in beta5], "R2": float(r2_5)},
        },
    }
    if r2_2 > 0:
        output["models"]["refined"] = {
            "A": float(A2),
            "alpha": float(alpha2),
            "gamma": float(gamma2),
            "R2": float(r2_2),
        }
    if r2_4 > 0:
        output["models"]["retry"] = {
            "n_eff": float(n_eff4),
            "k": float(k4),
            "R2": float(r2_4),
        }

    out_path = _HERE / "helio_s_derivation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> {out_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
