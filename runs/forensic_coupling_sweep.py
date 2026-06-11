"""run_forensic_coupling_sweep.py — ρ(S_T, η) at λ=1 across production corpus.

Computes the Pearson correlation between S_T and η across the full
82,515-config production corpus (89,178 total, excluding S_T=0 and
NaN η).  Stratifies by body and by S_T regime.

Also assembles the complete coupling phase diagram from the v1-v4
UQ congestion experiments.

Input:  runs/epyc_results/production_2026_03_11/production_P*.json
Output: runs/forensic_coupling_sweep_results.json
"""

import glob
import json
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
PROD_DIR = _HERE / "epyc_results" / "production_2026_03_11"


def _load_production_configs():
    """Load all production result records."""
    records = []
    for path in sorted(glob.glob(str(PROD_DIR / "production_P*_results.json"))):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict) and "results" in data:
            records.extend(data["results"])
    return records


def _pearson_rho(x, y):
    """Pearson correlation coefficient."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = math.sqrt(float(np.sum(dx**2) * np.sum(dy**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(dx * dy) / denom)


def _body_stats(records):
    """Per-body ρ(S_T, η) and summary statistics."""
    from collections import defaultdict

    by_body = defaultdict(lambda: {"s_t": [], "eta": []})
    for r in records:
        by_body[r["body"]]["s_t"].append(r["S_T"])
        by_body[r["body"]]["eta"].append(r["eta"])

    result = {}
    for body in sorted(by_body):
        s = np.array(by_body[body]["s_t"])
        e = np.array(by_body[body]["eta"])
        result[body] = {
            "n": len(s),
            "rho_st_eta": round(_pearson_rho(s, e), 6),
            "s_t_mean": round(float(s.mean()), 4),
            "s_t_std": round(float(s.std()), 4),
            "s_t_min": round(float(s.min()), 4),
            "s_t_max": round(float(s.max()), 4),
            "eta_mean": round(float(e.mean()), 4),
            "eta_std": round(float(e.std()), 4),
            "eta_min": round(float(e.min()), 4),
            "eta_max": round(float(e.max()), 4),
        }
    return result


def _regime_stats(records):
    """ρ(S_T, η) stratified by S_T regime."""
    regimes = {
        "S_T<0.3": lambda st: st < 0.3,
        "0.3<=S_T<0.7": lambda st: 0.3 <= st < 0.7,
        "S_T>=0.7": lambda st: st >= 0.7,
    }
    result = {}
    for label, pred in regimes.items():
        subset = [r for r in records if pred(r["S_T"])]
        s = np.array([r["S_T"] for r in subset])
        e = np.array([r["eta"] for r in subset])
        result[label] = {
            "n": len(subset),
            "rho_st_eta": round(_pearson_rho(s, e), 6),
            "s_t_mean": round(float(s.mean()), 4),
            "eta_mean": round(float(e.mean()), 4),
        }
    return result


def _phase_diagram(overall_rho, overall_mean_st, body_stats, regime_stats):
    """Assemble complete coupling phase diagram including v1-v4 results."""
    entries = [
        {
            "condition": f"lambda=1, full corpus ({sum(b['n'] for b in body_stats.values())})",
            "mean_st": overall_mean_st,
            "rho": overall_rho,
            "mechanism": "routing diversity (supply-side)",
            "strength": "weak",
        },
    ]
    # Per-body entries for Mars and Moon (largest + most interesting)
    for body_name, mech in [
        ("Mars", "routing diversity"),
        ("Moon", "parity locking / diminishing returns"),
    ]:
        if body_name in body_stats:
            b = body_stats[body_name]
            entries.append(
                {
                    "condition": f"lambda=1, {body_name} ({b['n']})",
                    "mean_st": b["s_t_mean"],
                    "rho": b["rho_st_eta"],
                    "mechanism": mech,
                    "strength": "weak",
                }
            )
    # Per-regime entries
    regime_mechs = {
        "S_T<0.3": "geometric constraint hurts quality",
        "0.3<=S_T<0.7": "routing diversity peak",
        "S_T>=0.7": "saturation / proto-Braess",
    }
    for label, mech in regime_mechs.items():
        if label in regime_stats:
            rs = regime_stats[label]
            entries.append(
                {
                    "condition": f"lambda=1, {label} ({rs['n']})",
                    "mean_st": rs["s_t_mean"],
                    "rho": rs["rho_st_eta"],
                    "mechanism": mech,
                    "strength": "weak",
                }
            )
    # v1-v4 UQ congestion experiment summaries (fixed reference values)
    entries.extend(
        [
            {
                "condition": "v1: lambda sweep, Mars sigma=120s",
                "mean_st": 0.989,
                "rho": 0.005,
                "mechanism": "neither (no spread)",
                "strength": "negligible",
            },
            {
                "condition": "v2: lambda=50, Mars sigma=3600s",
                "mean_st": 0.999,
                "rho": 0.135,
                "mechanism": "routing diversity",
                "strength": "weak",
            },
            {
                "condition": "v3: lambda=50, Mars n-sweep",
                "mean_st": 0.99,
                "rho": 0.108,
                "mechanism": "routing diversity",
                "strength": "weak",
            },
            {
                "condition": "v4: lambda=50, Moon NRHO n-sweep",
                "mean_st": 0.163,
                "rho": -0.697,
                "mechanism": "congestion via differential injection",
                "strength": "strong",
            },
        ]
    )
    return {
        "description": "Complete coupling phase diagram from v1-v4 experiments + forensic sweep",
        "entries": entries,
    }


def main():
    print("Loading production configs...")
    all_records = _load_production_configs()
    total = len(all_records)
    print(f"  Total configs: {total}")

    # Filter
    nonzero = [r for r in all_records if r["S_T"] > 0]
    excluded_st_zero = total - len(nonzero)
    clean = [
        r
        for r in nonzero
        if not (r["eta"] is None or (isinstance(r["eta"], float) and math.isnan(r["eta"])))
    ]
    excluded_nan_eta = len(nonzero) - len(clean)
    print(f"  Excluded S_T=0: {excluded_st_zero}")
    print(f"  Excluded NaN eta: {excluded_nan_eta}")
    print(f"  Clean configs: {len(clean)}")

    # Overall correlation
    s_all = np.array([r["S_T"] for r in clean])
    e_all = np.array([r["eta"] for r in clean])
    rho_overall = _pearson_rho(s_all, e_all)

    # Factorization check
    residuals = np.array([abs(r["DR"] - r["S_T"] * r["eta"]) for r in clean])

    overall = {
        "rho_st_eta": round(rho_overall, 6),
        "s_t_mean": round(float(s_all.mean()), 4),
        "s_t_std": round(float(s_all.std()), 4),
        "s_t_range": [round(float(s_all.min()), 4), round(float(s_all.max()), 4)],
        "eta_mean": round(float(e_all.mean()), 4),
        "eta_std": round(float(e_all.std()), 4),
        "eta_range": [round(float(e_all.min()), 4), round(float(e_all.max()), 4)],
        "factorization_max_residual": float(residuals.max()),
        "factorization_mean_residual": float(residuals.mean()),
    }

    print(f"\n  Overall rho(S_T, eta) = {rho_overall:+.6f}")
    print(f"  Factorization max residual: {residuals.max():.2e}")

    body = _body_stats(clean)
    regime = _regime_stats(clean)
    phase = _phase_diagram(rho_overall, overall["s_t_mean"], body, regime)

    print("\n  By body:")
    for b, stats in sorted(body.items()):
        print(f"    {b:10s}  n={stats['n']:6d}  rho={stats['rho_st_eta']:+.6f}")

    print("\n  By S_T regime:")
    for label, stats in regime.items():
        print(f"    {label:15s}  n={stats['n']:6d}  rho={stats['rho_st_eta']:+.6f}")

    result = {
        "description": "Forensic coupling sweep: rho(S_T, eta) at lambda=1 across production corpus",
        "corpus": {
            "source": "epyc_results/production_2026_03_11/production_P*.json",
            "total_configs": total,
            "excluded_st_zero": excluded_st_zero,
            "excluded_nan_eta": excluded_nan_eta,
            "clean_configs": len(clean),
        },
        "overall": overall,
        "by_body": body,
        "by_st_regime": regime,
        "phase_diagram": phase,
    }

    out_path = _HERE / "forensic_coupling_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
