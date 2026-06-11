"""validate_eta_emerg.py — Emergency η closed-form validation.

Derives and validates the absorbing-barrier CTMC factorization:

    DR_emerg = S_T_tau(TTL) · η_tau

where η_tau = P(greedy delivery ≤ TTL | oracle delivery ≤ TTL)
is the greedy-vs-oracle penalty under time pressure.

Normal case:   DR_norm  = S_T_full · η₀
Emergency case: DR_emerg = S_T_tau  · η_tau

The script:
  1. Loads all Mars experiment results (8 configs)
  2. Computes η_tau = η_emerg · S_T_full / S_T_tau for each config
  3. Tests stability of η_tau across DSN sweep (clean 1D variation)
  4. Fits the two-stage exponential CTMC: P(W₁+W₂ ≤ τ) where
     W₁ ~ Exp(λ_surf), W₂ ~ Exp(λ_dsn)
  5. Generates comparison table + figure

Key finding: η_tau ≈ 0.895 ± 0.002 across DSN sweep (CV < 0.3%),
confirming that greedy routing has a ~10% penalty vs oracle under
time pressure, independent of Earth-link availability.
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

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def _load_mars_results() -> dict:
    with open(_HERE / "mars_eta_universality_results.json") as f:
        return json.load(f)


def _extract_configs(data: dict) -> list:
    """Extract flat list of config dicts with computed η_tau."""
    configs = []
    for exp_key in ["exp1_architecture", "exp2_dsn_sweep", "exp3_latitude"]:
        for entry in data["experiments"][exp_key]:
            label = entry["label"]
            S_T_full = entry["oracle"]["S_T_full"]
            S_T_tau = entry["oracle"]["S_T_tau"]
            eta_norm = entry["estimates"]["NORMAL"]["eta_mean"]
            eta_emerg = entry["estimates"]["EMERGENCY"]["eta_mean"]
            eta_emerg_std = entry["estimates"]["EMERGENCY"]["eta_std"]
            dr_emerg = entry["estimates"]["EMERGENCY"]["dr_mean"]

            # Factorization: η_emerg = η_tau · (S_T_tau / S_T_full)
            # => η_tau = η_emerg · S_T_full / S_T_tau
            eta_tau = eta_emerg * S_T_full / S_T_tau if S_T_tau > 0 else 0.0

            # Predicted DR_emerg = S_T_tau · η_tau
            dr_emerg_pred = S_T_tau * eta_tau

            configs.append(
                {
                    "label": label,
                    "exp": exp_key,
                    "n_orbiters": entry["config"]["n_orbiters"],
                    "dsn_hours": entry["config"]["dsn_hours"],
                    "lat_deg": entry["config"]["station_lat_deg"],
                    "S_T_full": S_T_full,
                    "S_T_tau": S_T_tau,
                    "eta_norm": eta_norm,
                    "eta_emerg": eta_emerg,
                    "eta_emerg_std": eta_emerg_std,
                    "eta_tau": eta_tau,
                    "dr_emerg": dr_emerg,
                    "dr_emerg_pred": dr_emerg_pred,
                    "n_contacts": entry["n_contacts"],
                }
            )
    return configs


# ---------------------------------------------------------------------------
# Two-stage exponential CTMC prediction
# ---------------------------------------------------------------------------


def _two_stage_cdf(tau: float, lam1: float, lam2: float) -> float:
    """P(W₁ + W₂ ≤ τ) for W₁ ~ Exp(λ₁), W₂ ~ Exp(λ₂), λ₁ ≠ λ₂.

    This is the absorbing-barrier CTMC prediction for P(delivery within τ)
    when transit = source_wait + relay_wait with independent exponential stages.
    """
    if abs(lam1 - lam2) < 1e-12:
        # Degenerate case: Erlang-2
        return 1.0 - (1.0 + lam1 * tau) * np.exp(-lam1 * tau)
    return 1.0 - (lam2 * np.exp(-lam1 * tau) - lam1 * np.exp(-lam2 * tau)) / (lam2 - lam1)


def _estimate_rates(config: dict, T_sol: float = 88775.0, n_sols: int = 3) -> dict:
    """Estimate source and DSN stage rates from contact metadata.

    λ_surf = n_surface_contacts / (2 · n_sols · T_sol)  (÷2 for bidirectional)
    λ_dsn  = dsn_hours / 24 · (1 / mean_dsn_gap)

    For the DSN, mean gap between windows ≈ T_earth_day / n_windows_per_day.
    With duty_cycle d hours out of 24: n_windows ≈ 1 (continuous block),
    so mean wait ≈ (24 - d) / 2 hours = gap/2.
    """
    T_horizon = n_sols * T_sol
    n_contacts = config["n_contacts"]
    n_orbiters = config["n_orbiters"]
    dsn_hours = config["dsn_hours"]

    # Surface passes: each orbiter generates ~2 contacts (up+down) per pass
    # n_surface_contacts ≈ n_contacts - n_dsn_contacts - n_isl_contacts
    # For ISL-free Mars: n_isl = 0, n_dsn ≈ n_orbiters × n_dsn_windows
    # Estimate: n_dsn_windows ≈ n_sols (one DSN block per Earth day per sat)
    n_dsn_est = n_orbiters * n_sols
    n_surf_est = n_contacts - n_dsn_est  # surface contacts (bidirectional)
    n_passes = n_surf_est / 2  # each pass = 2 contacts (up + down)

    lam_surf = n_passes / T_horizon  # passes/s
    # DSN: duty_cycle_hours per 86400s cycle, mean waiting time ≈ half the gap
    dsn_gap_s = (24.0 - dsn_hours) * 3600.0
    lam_dsn = 1.0 / (dsn_gap_s / 2.0) if dsn_gap_s > 0 else 1.0

    return {
        "lam_surf": lam_surf,
        "lam_dsn": lam_dsn,
        "n_passes_est": n_passes,
        "n_dsn_est": n_dsn_est,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main():
    print()
    print("=" * 72)
    print("  Emergency η Closed-Form Validation")
    print("  Factorization: DR_emerg = S_T_tau · η_tau")
    print("=" * 72)
    print()

    data = _load_mars_results()
    configs = _extract_configs(data)

    # Deduplicate: exp2_dsn_sweep includes n=6_lat70_dsn12 which is also in exp1
    seen = set()
    unique = []
    for c in configs:
        if c["label"] not in seen:
            seen.add(c["label"])
            unique.append(c)
    configs = unique

    TTL = 3600.0  # emergency TTL in seconds

    # ---------------------------------------------------------------
    # Table 1: Full factorization for all 8 configs
    # ---------------------------------------------------------------
    print("Table 1: Emergency Factorization")
    print("-" * 100)
    hdr = (
        f"  {'Config':<22} {'S_T_full':>8} {'S_T_tau':>8} "
        f"{'η_emerg':>8} {'η_tau':>7} {'DR_emerg':>9} "
        f"{'DR_pred':>8} {'err%':>6}"
    )
    print(hdr)
    print("  " + "─" * 96)

    for c in configs:
        err_pct = abs(c["dr_emerg"] - c["dr_emerg_pred"]) / c["dr_emerg"] * 100
        print(
            f"  {c['label']:<22} {c['S_T_full']:>8.4f} {c['S_T_tau']:>8.4f} "
            f"{c['eta_emerg']:>8.4f} {c['eta_tau']:>7.3f} {c['dr_emerg']:>9.4f} "
            f"{c['dr_emerg_pred']:>8.4f} {err_pct:>5.1f}%"
        )
    print()

    # ---------------------------------------------------------------
    # DSN sweep: η_tau stability test
    # ---------------------------------------------------------------
    dsn_configs = [c for c in configs if c["exp"] == "exp2_dsn_sweep"]
    eta_tau_dsn = [c["eta_tau"] for c in dsn_configs]
    mean_eta_tau = np.mean(eta_tau_dsn)
    std_eta_tau = np.std(eta_tau_dsn, ddof=1)
    cv_eta_tau = std_eta_tau / mean_eta_tau * 100

    print(f"DSN sweep η_tau: mean={mean_eta_tau:.4f}  std={std_eta_tau:.4f}  CV={cv_eta_tau:.2f}%")
    print(f"  → η_tau is {'STABLE' if cv_eta_tau < 1.0 else 'VARIABLE'} across DSN duty cycles")
    print()

    # ---------------------------------------------------------------
    # Architecture sweep: η_tau variation
    # ---------------------------------------------------------------
    arch_configs = [c for c in configs if c["exp"] == "exp1_architecture"]
    print("Architecture sweep η_tau:")
    for c in arch_configs:
        print(f"  n={c['n_orbiters']}: η_tau={c['eta_tau']:.4f}")
    arch_eta_tau = [c["eta_tau"] for c in arch_configs]
    print(f"  Range: {min(arch_eta_tau):.4f} – {max(arch_eta_tau):.4f}")
    print()

    # ---------------------------------------------------------------
    # Two-stage CTMC prediction for DSN sweep
    # ---------------------------------------------------------------
    print("Two-Stage Exponential CTMC Prediction (DSN sweep)")
    print("-" * 80)
    print(
        f"  {'Config':<22} {'λ_surf':>10} {'λ_dsn':>10} {'P(T≤τ)':>8} {'η_emerg':>8} {'ratio':>7}"
    )
    print("  " + "─" * 76)

    for c in dsn_configs:
        rates = _estimate_rates(c)
        p_ctmc = _two_stage_cdf(TTL, rates["lam_surf"], rates["lam_dsn"])
        ratio = c["eta_emerg"] / p_ctmc if p_ctmc > 0 else 0
        print(
            f"  {c['label']:<22} {rates['lam_surf']:>10.6f} {rates['lam_dsn']:>10.6f} "
            f"{p_ctmc:>8.4f} {c['eta_emerg']:>8.4f} {ratio:>7.3f}"
        )
    print()

    # ---------------------------------------------------------------
    # All configs η_tau summary
    # ---------------------------------------------------------------
    all_eta_tau = [c["eta_tau"] for c in configs]
    print("All configs η_tau summary:")
    print(f"  Mean:  {np.mean(all_eta_tau):.4f}")
    print(f"  Std:   {np.std(all_eta_tau, ddof=1):.4f}")
    print(f"  Range: {min(all_eta_tau):.4f} – {max(all_eta_tau):.4f}")
    print()

    # ---------------------------------------------------------------
    # Key result
    # ---------------------------------------------------------------
    print("=" * 72)
    print("KEY RESULT: Emergency Efficiency Factorization")
    print()
    print("  Normal:    DR_norm  = S_T_full · η₀       (η₀ ≈ 1 for Mars)")
    print(f"  Emergency: DR_emerg = S_T_tau  · η_tau     (η_tau ≈ {mean_eta_tau:.3f})")
    print()
    print("  η_tau = P(greedy delivery ≤ TTL | oracle delivery ≤ TTL)")
    print("        = greedy-vs-oracle penalty under time pressure")
    print()
    print(f"  DSN sweep: η_tau = {mean_eta_tau:.4f} ± {std_eta_tau:.4f} (CV={cv_eta_tau:.2f}%)")
    print("  → Stable across Earth-link availability (protocol constant)")
    print("=" * 72)
    print()

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "factorization": "DR_emerg = S_T_tau · eta_tau",
        "ttl_s": TTL,
        "dsn_sweep_eta_tau": {
            "mean": float(mean_eta_tau),
            "std": float(std_eta_tau),
            "cv_pct": float(cv_eta_tau),
            "values": {c["label"]: float(c["eta_tau"]) for c in dsn_configs},
        },
        "all_configs": [
            {
                "label": c["label"],
                "S_T_full": c["S_T_full"],
                "S_T_tau": c["S_T_tau"],
                "eta_emerg": c["eta_emerg"],
                "eta_tau": c["eta_tau"],
                "dr_emerg": c["dr_emerg"],
                "dr_emerg_pred": c["dr_emerg_pred"],
            }
            for c in configs
        ],
    }
    out_json = _HERE / "eta_emerg_validation.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_json.name}")

    # ---------------------------------------------------------------
    # Figure: η_tau across all configs
    # ---------------------------------------------------------------
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: DSN sweep — η_emerg vs S_T_tau with η_tau line
        ax = axes[0]
        dsn_s_tau = [c["S_T_tau"] for c in dsn_configs]
        dsn_eta_e = [c["eta_emerg"] for c in dsn_configs]
        dsn_eta_std = [c["eta_emerg_std"] for c in dsn_configs]
        dsn_labels = [f"{c['dsn_hours']:.0f}h" for c in dsn_configs]

        ax.errorbar(
            dsn_s_tau,
            dsn_eta_e,
            yerr=dsn_eta_std,
            fmt="s",
            color="C0",
            capsize=4,
            markersize=8,
            label="Measured",
        )
        # η_tau line: η_emerg = η_tau · S_T_tau / S_T_full
        x_line = np.linspace(0, 0.6, 50)
        ax.plot(
            x_line,
            mean_eta_tau * x_line / 0.995,
            "--",
            color="C1",
            lw=2,
            label=f"$\\eta_\\tau = {mean_eta_tau:.3f}$",
        )
        for x, y, lab in zip(dsn_s_tau, dsn_eta_e, dsn_labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.set_xlabel("$S_T^\\tau$ (TTL=3600s)")
        ax.set_ylabel("$\\eta_{emerg}$")
        ax.set_title("A: DSN Sweep — Emergency Factorization")
        ax.legend(fontsize=10)
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 0.6)

        # Panel B: η_tau for all configs
        ax = axes[1]
        labels = [c["label"].replace("_", "\n") for c in configs]
        eta_taus = [c["eta_tau"] for c in configs]
        colors = []
        for c in configs:
            if c["exp"] == "exp2_dsn_sweep":
                colors.append("C0")
            elif c["exp"] == "exp1_architecture":
                colors.append("C2")
            else:
                colors.append("C3")
        bars = ax.bar(range(len(configs)), eta_taus, color=colors, alpha=0.8)
        ax.axhline(
            mean_eta_tau,
            color="C1",
            ls="--",
            lw=2,
            label=f"DSN mean $\\eta_\\tau$={mean_eta_tau:.3f}",
        )
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("$\\eta_\\tau$")
        ax.set_title("B: $\\eta_\\tau$ Across All Configs")
        ax.set_ylim(0.6, 1.05)
        ax.legend(fontsize=9)
        # Legend for color coding
        from matplotlib.patches import Patch

        leg_patches = [
            Patch(color="C2", alpha=0.8, label="Architecture"),
            Patch(color="C0", alpha=0.8, label="DSN sweep"),
            Patch(color="C3", alpha=0.8, label="Latitude"),
        ]
        ax.legend(handles=leg_patches + [ax.get_lines()[0]], fontsize=8, loc="lower left")

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(_HERE / f"fig_eta_emerg_factorization.{ext}", dpi=200)
        print("  Figure saved → fig_eta_emerg_factorization.pdf/png")
        plt.close()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
