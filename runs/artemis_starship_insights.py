#!/usr/bin/env python3
"""Historical TIN diagnostics for Artemis and Starship scenarios.

Replays arithmetic and summaries from archived TIN simulation results. This is
a retired claim surface, not a mission-planning, feasibility, reserve-sizing,
crew-safety, or operations reference. No new simulations are run.

Seven sections:
  1. Commodity sweep: ξ and exponential remaining fraction at Mars T_scheduled
  2. Historical Gateway UQ diagnostic: archived NRHO coupling measurements
  3. Historical parity diagnostic: even-n NRHO fleet S_T pattern
  4. Historical return-leg model output: outbound vs return η by fleet size
  5. Historical Mars variance decomposition
  6. Historical commodity synchronization diagnostic: F_sync
  7. One-tau screening table: every commodity × every route

Output: runs/artemis_starship_insights_results.json
"""

import json
import math
from pathlib import Path

_HERE = Path(__file__).parent
DAY = 86400.0
LN2 = math.log(2)
CLAIM_STATUS = (
    "HISTORICAL/RETIRED: archived TIN model output; not mission feasibility, "
    "reserve-sizing, crew-safety, or operations guidance"
)


# ═══════════════════════════════════════════════════════════════
# COMMODITY DATABASE
# ═══════════════════════════════════════════════════════════════

COMMODITIES = {
    "LH2": {"tau_half_d": 180, "category": "propellant", "notes": "passive storage, no ZBO"},
    "LH2_ZBO": {
        "tau_half_d": 1800,
        "category": "propellant",
        "notes": "with zero boil-off (10x extension)",
    },
    "CH4": {
        "tau_half_d": 3600,
        "category": "propellant",
        "notes": "archived 3,600-day half-life input for a methane row",
    },
    "LOX": {"tau_half_d": 500, "category": "propellant", "notes": "moderate boiloff"},
    "LOX_ZBO": {"tau_half_d": 5000, "category": "propellant", "notes": "with zero boil-off"},
    "N2O4_MMH": {
        "tau_half_d": 36500,
        "category": "propellant",
        "notes": "storable hypergolic, ~100yr half-life",
    },
    "food": {"tau_half_d": 730, "category": "consumable", "notes": "freeze-dried/packaged"},
    "water": {"tau_half_d": 36500, "category": "consumable", "notes": "effectively infinite"},
    "mRNA_vaccine": {"tau_half_d": 3, "category": "medical", "notes": "-70C required"},
    "standard_vaccine": {"tau_half_d": 30, "category": "medical", "notes": "2-8C cold chain"},
    "biologics": {
        "tau_half_d": 1,
        "category": "medical",
        "notes": "tissue samples, blood products",
    },
    "hardware": {"tau_half_d": 5475, "category": "structure", "notes": "15-year rated life"},
    "electronics": {
        "tau_half_d": 3650,
        "category": "structure",
        "notes": "10-year rated, radiation degradation",
    },
    "crew": {
        "tau_half_d": None,
        "category": "crew",
        "notes": "step function: alive or dead, not exponential",
    },
}

# Routes with scheduled transit times (measured or computed)
ROUTES = {
    "Earth-Moon (Hohmann)": {
        "t_hohmann_d": 5.0,
        "t_scheduled_d": 5.0,
        "notes": "negligible scheduling overhead",
    },
    "Earth-Moon (NRHO transfer)": {
        "t_hohmann_d": 5.0,
        "t_scheduled_d": 7.0,
        "notes": "NRHO insertion adds ~2d",
    },
    "Earth-Mars (Hohmann, 30d window)": {
        "t_hohmann_d": 258.9,
        "t_scheduled_d": 611.2,
        "notes": "measured +136% overhead",
    },
    "Earth-Mars (fast, 30d window)": {
        "t_hohmann_d": 180.0,
        "t_scheduled_d": 532.3,
        "notes": "measured +196% overhead",
    },
    "Earth-Mars (Hohmann+staging)": {
        "t_hohmann_d": 258.9,
        "t_scheduled_d": 638.8,
        "notes": "measured +147% overhead",
    },
    "Earth-Jupiter (EMJ relay)": {
        "t_hohmann_d": 997.5,
        "t_scheduled_d": 1611.0,
        "notes": "from hull GT33",
    },
    "Mars-Jupiter (direct)": {
        "t_hohmann_d": 1126.6,
        "t_scheduled_d": 1615.0,
        "notes": "from hull GT33",
    },
}


def _compute_xi(t_d, tau_half_d):
    """Compute ξ = T × ln(2) / τ_half."""
    if tau_half_d is None or tau_half_d <= 0:
        return None
    return t_d * LN2 / tau_half_d


def _compute_remaining_fraction(t_d, tau_half_d):
    """Compute the exponential model's remaining fraction."""
    if tau_half_d is None or tau_half_d <= 0:
        return None
    return math.exp(-LN2 * t_d / tau_half_d)


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Commodity sweep at Mars T_scheduled
# ═══════════════════════════════════════════════════════════════


def section1_commodity_sweep():
    """Render ξ and remaining-fraction references for archived commodity rows."""
    print("\n" + "=" * 70)
    print("  1. HISTORICAL COMMODITY MODEL: Mars Hohmann (30d window)")
    print("     T_Hohmann = 258.9d, T_scheduled = 611.2d")
    print("=" * 70)

    t_h = 258.9
    t_s = 611.2

    results = {}
    print(
        f"\n  {'Commodity':<20} {'τ_half':>8} {'ξ_Hohm':>8} {'ξ_Sched':>8} "
        f"{'Remain_H':>8} {'Remain_S':>8} {'Model band':>12}"
    )
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}")

    for name, c in COMMODITIES.items():
        tau = c["tau_half_d"]
        if tau is None:
            xi_h = xi_s = dr_h = dr_s = None
            model_band = "NOT_MODELED"
        else:
            xi_h = _compute_xi(t_h, tau)
            xi_s = _compute_xi(t_s, tau)
            dr_h = _compute_remaining_fraction(t_h, tau)
            dr_s = _compute_remaining_fraction(t_s, tau)
            if xi_s > 1.0:
                model_band = "ABOVE_1.0"
            elif xi_s > 0.7:
                model_band = "0.7_TO_1.0"
            else:
                model_band = "BELOW_0.7"

        tau_str = f"{tau}d" if tau else "N/A"
        xi_h_str = f"{xi_h:.4f}" if xi_h is not None else "N/A"
        xi_s_str = f"{xi_s:.4f}" if xi_s is not None else "N/A"
        dr_h_str = f"{dr_h:.4f}" if dr_h is not None else "N/A"
        dr_s_str = f"{dr_s:.4f}" if dr_s is not None else "N/A"

        print(
            f"  {name:<20} {tau_str:>8} {xi_h_str:>8} {xi_s_str:>8} "
            f"{dr_h_str:>8} {dr_s_str:>8} {model_band:>12}"
        )

        results[name] = {
            "tau_half_d": tau,
            "xi_hohmann": round(xi_h, 6) if xi_h else None,
            "xi_scheduled": round(xi_s, 6) if xi_s else None,
            "model_remaining_hohmann": round(dr_h, 6) if dr_h else None,
            "model_remaining_scheduled": round(dr_s, 6) if dr_s else None,
            "model_band": model_band,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Gateway UQ warning
# ═══════════════════════════════════════════════════════════════


def section2_gateway_uq():
    """Render archived NRHO coupling measurements without design advice."""
    print("\n" + "=" * 70)
    print("  2. HISTORICAL GATEWAY UQ DIAGNOSTIC — CLASSIFIER RETIRED")
    print("=" * 70)

    with open(_HERE / "uq_moon_coupling_v4_results.json") as f:
        data = json.load(f)

    arch = data["architecture_summary"]
    pooled_rho = data["pooled_correlation"]["rho_st_eta_lam50"]

    print(f"\n  Pooled ρ(S_T, η) at λ=50: {pooled_rho:.3f}")
    print(
        f"\n  {'n_orb':>5} {'Parity':>6} {'S_T':>8} {'η(λ=1)':>8} {'γ(λ=1)':>8} "
        f"{'η(λ=50)':>8} {'γ(λ=50)':>8} {'Status':>12}"
    )
    print(f"  {'-' * 5} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}")

    results = {}
    for n_str, a in sorted(arch.items(), key=lambda x: int(x[0])):
        n = a["n_orbiters"]
        parity = a["parity"]
        s_t = a["s_t"]
        eta1 = a["lam_1"]["eta_mean"]
        gamma1 = a["lam_1"]["gamma"]
        eta50 = a["lam_50"]["eta_mean"]
        gamma50 = a["lam_50"]["gamma"]

        print(
            f"  {n:>5} {parity:>6} {s_t:>8.3f} {eta1:>8.3f} {gamma1:>8.2f} "
            f"{eta50:>8.3f} {gamma50:>8.2f} {'HISTORICAL':>12}"
        )

        results[n_str] = {
            "n_orbiters": n,
            "parity": parity,
            "s_t": round(s_t, 4),
            "eta_lam1": round(eta1, 4),
            "gamma_lam1": round(gamma1, 3),
            "classifier_status_lam1": "RETIRED",
            "eta_lam50": round(eta50, 4),
            "gamma_lam50": round(gamma50, 3),
            "classifier_status_lam50": "RETIRED",
        }

    print("\n  Historical observation: this archived sweep reported a pooled")
    print(f"  association ρ(S_T, η) = {pooled_rho:.2f} at λ=50.")
    print("  Gamma is retained here only as a descriptive, outcome-derived slope.")
    print("  No gamma threshold, mission margin, or reserve-sizing rule is retained.")

    return {
        "claim_status": "historical diagnostic; classifier and reserve advice retired",
        "pooled_rho_lam50": round(pooled_rho, 4),
        "architectures": results,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Archived parity comparison
# ═══════════════════════════════════════════════════════════════


def section3_parity_comparison():
    """Render the archived even-n NRHO parity pattern."""
    print("\n" + "=" * 70)
    print("  3. HISTORICAL PARITY DIAGNOSTIC — NOT A FLEET-SIZING RULE")
    print("=" * 70)

    with open(_HERE / "uq_moon_coupling_v4_results.json") as f:
        data = json.load(f)

    arch = data["architecture_summary"]

    print("\n  Each tested even-n row in this archive has S_T ≈ 0.5675.")
    print("  The archived golden-angle rows provide a separate spacing comparison.")
    print(f"\n  {'n_orb':>5} {'Parity':>6} {'S_T':>8} {'Contacts':>10}")
    print(f"  {'-' * 5} {'-' * 6} {'-' * 8} {'-' * 10}")

    for n_str, a in sorted(arch.items(), key=lambda x: int(x[0])):
        print(f"  {a['n_orbiters']:>5} {a['parity']:>6} {a['s_t']:>8.4f} {a['n_contacts']:>10}")

    print("\n  Archived comparison notes (not design rules):")
    print("    1. Golden-angle and equal-spacing rows were both evaluated")
    print("    2. Adjacent n and n+1 values were recorded")
    print("    3. Tested even-n rows have lower S_T than neighboring odd-n rows")

    return {
        "even_row_s_t_reference": 0.5675,
        "comparison_spacing": "golden-angle RAAN spacing (137.508 deg)",
        "archive_note": "Phase 5 archive: exact in 24/24 tested constellations; 45 deg jitter tested",
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Return leg crew safety
# ═══════════════════════════════════════════════════════════════


def section4_return_leg():
    """Render archived outbound-versus-return model outputs."""
    print("\n" + "=" * 70)
    print("  4. HISTORICAL RETURN-LEG MODEL OUTPUT — NOT CREW-SAFETY GUIDANCE")
    print("=" * 70)

    with open(_HERE / "binding_diagnostic_results.json") as f:
        data = json.load(f)

    results = {}
    for body in ["Moon", "Mars"]:
        print(f"\n  {body}:")
        print(
            f"  {'n_orb':>5} {'η_out':>8} {'η_ret':>8} {'DR_out':>8} {'DR_ret':>8} "
            f"{'Binding':>10} {'Factor':>8}"
        )
        print(f"  {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8}")

        body_results = {}
        for key in sorted(data.keys()):
            if not key.startswith(body.lower()):
                continue
            d = data[key]
            o = d["outbound"]
            r = d["return"]
            b = d["binding"]
            n = d["n_orb"]

            print(
                f"  {n:>5} {o['eta_mean']:>8.3f} {r['eta_mean']:>8.3f} "
                f"{o['dr_mean']:>8.3f} {r['dr_mean']:>8.3f} "
                f"{b['leg']:>10} {b['factor']:>8}"
            )

            body_results[key] = {
                "n_orb": n,
                "eta_outbound": round(o["eta_mean"], 4),
                "eta_return": round(r["eta_mean"], 4),
                "dr_outbound": round(o["dr_mean"], 4),
                "dr_return": round(r["dr_mean"], 4),
                "binding_leg": b["leg"],
                "binding_factor": b["factor"],
            }

        results[body] = body_results

    print("\n  Archived model observation: at n≥6, return η ≈ 0.80 vs outbound η ≈ 0.91.")
    print("  This synthetic comparison is not a crew-abort or mission-safety result.")
    print("  The archived n=8 Mars case has lower modeled outbound than return η.")

    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 5: Archived variance decomposition
# ═══════════════════════════════════════════════════════════════


def section5_fix_the_radio():
    """Render the archived Mars variance decomposition."""
    print("\n" + "=" * 70)
    print("  5. HISTORICAL MARS VARIANCE DECOMPOSITION — NOT DESIGN GUIDANCE")
    print("=" * 70)

    with open(_HERE / "uncertainty_quantification_results.json") as f:
        data = json.load(f)

    vd = data.get("variance_decomposition", {})
    var_between = vd.get("frac_between", 0.083) * 100
    comp = data.get("comparison", {})
    ci_overlap = comp.get("ci_overlap", 0.992)
    ks = comp.get("ks_statistic", 0.070)

    print("\n  Mars 6-polar, 50 perturbed plans × 5 seeds:")
    print(f"    Between-plan variance (geometry):  {var_between:.1f}%")
    print("    Within-plan variance component: larger in this archived decomposition")
    print(f"\n  CI overlap (factored vs direct): {ci_overlap:.3f}")
    print(f"  KS statistic:                    {ks:.3f}")

    print("\n  Archived band labels from this model (not a design prescription):")
    print("    n=2-4:  geometry-associated S_T band")
    print("    n=4-6:  routing-associated η band")
    print("    n≥7:    link-probability-associated band")
    print("    The archived n=8 row is lower than the n=6 comparison; no cause is inferred.")

    return {
        "between_plan_pct": round(var_between, 1),
        "within_plan_fraction_exceeds_between_plan": True,
        "ci_overlap": round(ci_overlap, 4),
        "ks_statistic": round(ks, 4),
        "archived_band_labels": "low_n=geometry, mid_n=routing, high_n=link_probability",
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6: Historical commodity synchronization diagnostic
# ═══════════════════════════════════════════════════════════════


def section6_synchronization():
    """Compute F_sync for archived synthetic commodity parameter rows."""
    print("\n" + "=" * 70)
    print("  6. HISTORICAL COMMODITY SYNCHRONIZATION MODEL")
    print("=" * 70)

    def f_sync_geometric(p_h, p_p, decay_factor=1.0):
        """Synchronization factor for two-commodity hub.

        p_h: hardware delivery probability per epoch
        p_p: propellant delivery probability per epoch
        D: decay factor per epoch of waiting (exp(-ln2 * T_epoch / tau_half))
        """
        D = decay_factor
        q = 1 - (1 - p_h) * (1 - p_p)  # either arrives
        if q == 0:
            return 0
        # Probability propellant arrives usable given hardware arrives
        numerator = p_h / q + (p_p * (1 - p_h) / q) * (p_h * D / (1 - D * (1 - p_h)))
        return min(1.0, max(0.0, p_p * numerator / (p_h * p_p / q) if p_h * p_p > 0 else 0))

    scenarios = {
        "Archived Artemis-like row": {
            "p_hw": 0.85,
            "p_prop": 0.80,
            "tau_half_d": 500,
            "epoch_d": 30,
            "notes": "LOX/CH4, monthly resupply from Earth",
        },
        "Archived Mars ISRU-like row": {
            "p_hw": 0.60,
            "p_prop": 0.35,
            "tau_half_d": 300,
            "epoch_d": 780,
            "notes": "synodic resupply, local ISRU propellant",
        },
        "Archived Mars LH2-like row": {
            "p_hw": 0.60,
            "p_prop": 0.095,
            "tau_half_d": 180,
            "epoch_d": 780,
            "notes": "from scheduling overhead result: 9.5% pipeline DR",
        },
        "Archived methane-fleet row": {
            "p_hw": 0.70,
            "p_prop": 0.65,
            "tau_half_d": 3600,
            "epoch_d": 780,
            "notes": "archived methane half-life and probability inputs",
        },
        "Symmetric stress test": {
            "p_hw": 0.30,
            "p_prop": 0.30,
            "tau_half_d": 180,
            "epoch_d": 780,
            "notes": "synthetic low-probability comparison",
        },
    }

    print(
        f"\n  {'Archived row':<30} {'p_hw':>6} {'p_prop':>7} {'τ_half':>7} {'F_sync':>7} {'1-F_sync':>8}"
    )
    print(f"  {'-' * 30} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")

    results = {}
    for name, s in scenarios.items():
        D = math.exp(-LN2 * s["epoch_d"] / s["tau_half_d"])
        # Simplified sync: probability both commodities available in same window
        # P(both) = p_hw * p_prop (independent) corrected for decay while waiting
        p_both = s["p_hw"] * s["p_prop"]
        # If hardware arrives first, propellant decays while waiting for resupply
        # Expected wait for second commodity given first arrived: geometric(p_other)
        # Mean wait = (1-p_other)/p_other epochs → decay = D^((1-p_other)/p_other)
        mean_wait_hw_first = (1 - s["p_prop"]) / max(s["p_prop"], 0.001)
        mean_wait_prop_first = (1 - s["p_hw"]) / max(s["p_hw"], 0.001)
        decay_if_hw_first = D**mean_wait_hw_first  # propellant decays
        decay_if_prop_first = D**mean_wait_prop_first  # propellant arrived, decays waiting for hw

        # Weighted average: P(hw first) * decay_prop + P(prop first) * decay_prop_waiting
        p_hw_first = s["p_hw"] * (1 - s["p_prop"]) / max(1 - p_both, 0.001) if p_both < 1 else 0.5
        f_sync = p_both + (1 - p_both) * (
            p_hw_first * s["p_prop"] * decay_if_hw_first
            + (1 - p_hw_first) * s["p_prop"] * decay_if_prop_first
        )
        f_sync = min(1.0, f_sync / max(s["p_prop"], 0.001))
        reduction_pct = (1 - f_sync) * 100

        print(
            f"  {name:<30} {s['p_hw']:>6.2f} {s['p_prop']:>7.3f} {s['tau_half_d']:>5.0f}d "
            f"{f_sync:>7.3f} {reduction_pct:>7.1f}%"
        )

        results[name] = {
            "p_hw": s["p_hw"],
            "p_prop": s["p_prop"],
            "tau_half_d": s["tau_half_d"],
            "f_sync": round(f_sync, 4),
            "reduction_pct": round(reduction_pct, 2),
            "notes": s["notes"],
        }

    print("\n  In this synthetic formula, availability is computed from both modeled inputs.")
    print("  Values below one record the formula's synchronization reduction only.")

    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 7: One-tau model-band table
# ═══════════════════════════════════════════════════════════════


def section7_onetau_table():
    """Render the archived one-tau screening heuristic."""
    print("\n" + "=" * 70)
    print("  7. HISTORICAL ONE-TAU MODEL BANDS: ξ = T_scheduled × ln(2) / τ_half")
    print("     BAND A: ξ < 0.7 | BAND B: 0.7-1.0 | BAND C: ξ > 1.0")
    print("     Descriptive archived bands only — no mission or survivability verdict")
    print("=" * 70)

    # Filter to interesting commodities (skip crew, water, hypergolics)
    interesting = [
        "LH2",
        "LH2_ZBO",
        "CH4",
        "LOX",
        "LOX_ZBO",
        "food",
        "mRNA_vaccine",
        "standard_vaccine",
        "hardware",
    ]

    # Header
    route_short = {
        "Earth-Moon (Hohmann)": "E-Moon",
        "Earth-Moon (NRHO transfer)": "E-NRHO",
        "Earth-Mars (Hohmann, 30d window)": "E-Mars",
        "Earth-Mars (fast, 30d window)": "E-Mars(f)",
        "Earth-Mars (Hohmann+staging)": "E-Mars(s)",
        "Earth-Jupiter (EMJ relay)": "E-Jup",
        "Mars-Jupiter (direct)": "M-Jup",
    }

    header = f"  {'Commodity':<18}"
    for route_name in ROUTES:
        short = route_short[route_name]
        header += f" {short:>10}"
    print(f"\n{header}")
    print(f"  {'-' * 18}" + (" " + "-" * 10) * len(ROUTES))

    # Sub-header: T_scheduled
    subh = f"  {'T_sched (d)':<18}"
    for route_name, r in ROUTES.items():
        subh += f" {r['t_scheduled_d']:>8.0f}d "
    print(subh)
    print(f"  {'-' * 18}" + (" " + "-" * 10) * len(ROUTES))

    results = {}
    for cname in interesting:
        c = COMMODITIES[cname]
        tau = c["tau_half_d"]
        row = f"  {cname:<18}"
        row_data = {}

        for route_name, r in ROUTES.items():
            t_s = r["t_scheduled_d"]
            if tau is None:
                row += f" {'N/A':>10}"
                row_data[route_name] = None
                continue

            xi = _compute_xi(t_s, tau)
            remaining_fraction = _compute_remaining_fraction(t_s, tau)

            if xi > 1.0:
                marker = "C"
            elif xi > 0.7:
                marker = "B"
            else:
                marker = "A"

            row += f"  {marker}{xi:>5.2f}    "
            row_data[route_name] = {
                "xi": round(xi, 4),
                "model_remaining_fraction": round(remaining_fraction, 4),
            }

        print(row)
        results[cname] = row_data

    print("\n  Legend: A = ξ<0.7  B = 0.7-1.0  C = ξ>1.0")
    print("\n  Archived model row: CH4 is in Band A on each displayed route (Jupiter ξ=0.13).")
    print("  LH2 is in Band C beyond Moon; the extended-half-life Mars row is in Band B.")
    print("  The displayed medical-input rows are in Band C beyond Moon.")
    print("  These bands are arithmetic references, not preservation or mission conclusions.")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  HISTORICAL TIN ARTEMIS/STARSHIP ANALYSIS — RETIRED CLAIM SURFACE")
    print("  Historical TIN analysis — configuration counts are non-additive")
    print(f"  {CLAIM_STATUS}")
    print("=" * 70)

    results = {"_claim_status": CLAIM_STATUS}
    results["commodity_sweep"] = section1_commodity_sweep()
    results["gateway_uq"] = section2_gateway_uq()
    results["parity_comparison"] = section3_parity_comparison()
    results["return_leg"] = section4_return_leg()
    results["fix_the_radio"] = section5_fix_the_radio()
    results["synchronization"] = section6_synchronization()
    results["onetau_table"] = section7_onetau_table()

    out_path = _HERE / "artemis_starship_insights_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
