#!/usr/bin/env python3
"""Verify quantitative claims in classification_theorem.tex against source data.

Usage:
    python runs/verify_paper_claims.py           # Tier 1 only (headline claims)
    python runs/verify_paper_claims.py --full    # All tiers (~35 checks)

Each claim maps a paper statement to a data file and computation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RUNS = Path(__file__).resolve().parent
ROOT = RUNS.parent

# ---------------------------------------------------------------------------
# Claim infrastructure
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    id: str
    tier: int  # 1 = headline, 2 = table/detail
    section: str  # paper section
    line: int  # approximate line number
    description: str
    expected: Any
    actual: Any = None
    passed: bool | None = None
    note: str = ""
    tolerance: float = 0.02  # 2% relative tolerance for floats


def check(c: Claim) -> Claim:
    """Evaluate pass/fail for a claim."""
    if c.actual is None:
        c.passed = None
        c.note = c.note or "COULD NOT COMPUTE"
        return c
    if isinstance(c.expected, bool):
        c.passed = c.actual == c.expected
    elif isinstance(c.expected, (int, float)) and isinstance(c.actual, (int, float)):
        if c.expected == 0:
            c.passed = abs(c.actual) < 1e-10
        else:
            c.passed = abs(c.actual - c.expected) / abs(c.expected) <= c.tolerance
    elif isinstance(c.expected, str):
        c.passed = str(c.actual) == c.expected
    elif isinstance(c.expected, tuple) and len(c.expected) == 2:
        # Range check: actual should be within [lo, hi]
        c.passed = c.expected[0] <= c.actual <= c.expected[1]
    else:
        c.passed = c.actual == c.expected
    return c


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}
_missing: set[str] = set()


def load(name: str) -> Any:
    if name not in _cache:
        # Fresh results in runs/ take precedence; fall back to the curated
        # result-JSON archive shipped with the repo, then to a local results
        # corpus if DATASETS_ROOT is set (absent for public clones; checks
        # then skip honestly).
        candidates = [RUNS / name, ROOT / "docs" / "archive" / "TIN_results_json" / name]
        datasets_root = os.environ.get("DATASETS_ROOT")
        if datasets_root:
            candidates.append(Path(datasets_root) / "helio_sweep_results" / name)
            candidates.append(Path(datasets_root) / name)
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            print(f"  WARNING: {name} not found", file=sys.stderr)
            _missing.add(name)
            _cache[name] = None
            return None
        with open(path) as f:
            _cache[name] = json.load(f)
    return _cache[name]


# ---------------------------------------------------------------------------
# Tier 1 — Headline claims
# ---------------------------------------------------------------------------


def tier1_claims() -> list[Claim]:
    claims = []

    # --- 1. Total simulation runs > 155,000 ---
    total = 0
    # Heliocentric orbital
    n_helio = 0
    phi = load("phi_decompose_results.json")
    if phi:
        n_helio = phi.get("n_configs", 0)
        total += n_helio
    # CRAWDAD
    n_crawdad = 0
    for exp in ["Exp1", "Exp2", "Exp3", "Exp6"]:
        cd = load(f"crawdad_contacts.{exp}_results.json")
        if cd:
            n_crawdad += cd.get("n_configs", 0)
    total += n_crawdad
    # Vehicular
    veh = load("vehicular_gamma_results.json")
    n_veh = veh.get("n_results", 0) if veh else 0
    total += n_veh
    # Mars relay
    syn = load("synodic_sweep_results.json")
    if syn and "time_series" in syn:
        ts = syn["time_series"]
        n_mars = len(ts) * 5 if isinstance(ts, list) else 0  # 4 tiers × 39 epochs × 5 seeds
    else:
        n_mars = 780
    total += n_mars

    claims.append(
        check(
            Claim(
                id="T1-001",
                tier=1,
                section="Abstract",
                line=71,
                description="Total simulation runs > 155,000",
                expected=(155000, 999999),
                actual=total,
                tolerance=0.0,
                note=f"helio={n_helio}, crawdad={n_crawdad}, veh={n_veh}, mars={n_mars}",
            )
        )
    )

    # --- 2. Braess paradox: 35/39 epochs ---
    braess = load("analysis_braess_variance_results.json")
    if braess:
        bs = braess["braess_summary"]
        claims.append(
            check(
                Claim(
                    id="T1-002",
                    tier=1,
                    section="Abstract/Intro",
                    line=88,
                    description="Braess paradox in 35/39 epochs",
                    expected=35,
                    actual=bs["n_braess"],
                )
            )
        )
        claims.append(
            check(
                Claim(
                    id="T1-003",
                    tier=1,
                    section="Abstract/Intro",
                    line=88,
                    description="Total synodic epochs = 39",
                    expected=39,
                    actual=bs["n_total"],
                )
            )
        )

    # --- 3. Classification gap >= 1.95 ---
    bootstrap = load("bootstrap_ci_results.json")
    if bootstrap:
        results = bootstrap["results"]
        # Trap max: Titan
        titan_gamma = results.get("Titan", {}).get("gamma_paper", None)
        # Cluster min: Exp2
        exp2_gamma = results.get("Exp2", {}).get("gamma_paper", None)
        if titan_gamma is not None and exp2_gamma is not None:
            gap = exp2_gamma - titan_gamma
            claims.append(
                check(
                    Claim(
                        id="T1-004",
                        tier=1,
                        section="Conjecture",
                        line=723,
                        description="Classification gap >= 1.95",
                        expected=(1.95, 10.0),
                        actual=gap,
                        note=f"Titan={titan_gamma}, Exp2={exp2_gamma}",
                    )
                )
            )

    # --- 4. SF Cab: 197,938 contacts ---
    if veh:
        ts = veh.get("trace_summary", {})
        claims.append(
            check(
                Claim(
                    id="T1-005",
                    tier=1,
                    section="Validation",
                    line=668,
                    description="SF Cab: 197,938 contacts",
                    expected=197938,
                    actual=ts.get("n_contacts"),
                )
            )
        )
        # Mean contact duration ~120s
        claims.append(
            check(
                Claim(
                    id="T1-006",
                    tier=1,
                    section="Validation",
                    line=669,
                    description="SF Cab: mean contact duration ~120s",
                    expected=120,
                    actual=round(ts.get("mean_contact_duration_s", 0), 1),
                    tolerance=0.01,
                )
            )
        )
        # 200 nodes
        claims.append(
            check(
                Claim(
                    id="T1-007",
                    tier=1,
                    section="Validation",
                    line=665,
                    description="SF Cab: 200 vehicles selected",
                    expected=200,
                    actual=ts.get("n_nodes"),
                )
            )
        )

    # --- 5. SF Cab: 1,200 simulation runs ---
    if veh:
        claims.append(
            check(
                Claim(
                    id="T1-008",
                    tier=1,
                    section="Validation",
                    line=671,
                    description="SF Cab: 1,200 simulation runs",
                    expected=1200,
                    actual=veh.get("n_results"),
                )
            )
        )

    # --- 6. Sign consistency at p_eff=0.1 (verifiable subset of the 49) ---
    # The full 49 count requires the exact curated p_eff subset from the
    # gamma pipeline, which isn't stored in a single file.
    # We verify: all 8 orbital bodies negative, all 5 social/vehicular positive.
    if bootstrap and veh:
        trap_ok = 0
        cluster_ok = 0
        sign_violations = []
        results = bootstrap["results"]
        orbital = ["Mercury", "Venus", "Mars", "Ceres", "Europa", "Jupiter", "Saturn", "Titan"]
        social = ["Exp1", "Exp2", "Exp3", "Exp6"]
        for body in orbital:
            gp = results.get(body, {}).get("gamma_paper", 0)
            if gp < 0:
                trap_ok += 1
            else:
                sign_violations.append(f"{body}={gp}")
        for trace in social:
            gp = results.get(trace, {}).get("gamma_paper", 0)
            if gp > 0:
                cluster_ok += 1
            else:
                sign_violations.append(f"{trace}={gp}")
        # SF Cab at p_eff=0.1
        sf_sign = veh.get("gamma", {}).get("0.1", {}).get("sign", "")
        if sf_sign == "+":
            cluster_ok += 1
        else:
            sign_violations.append(f"SF_Cab={sf_sign}")

        claims.append(
            check(
                Claim(
                    id="T1-009",
                    tier=1,
                    section="Conjecture",
                    line=545,
                    description="Sign consistency: 8/8 trap neg, 5/5 cluster pos at p=0.1",
                    expected=0,
                    actual=len(sign_violations),
                    note=f"trap_ok={trap_ok}/8, cluster_ok={cluster_ok}/5"
                    + (f", violations={sign_violations}" if sign_violations else ""),
                )
            )
        )
        # Flag the 49 count as manual-verify
        claims.append(
            Claim(
                id="T1-010",
                tier=1,
                section="Conjecture",
                line=545,
                description="49 body/trace-p_eff combos (20T+29C) — MANUAL VERIFY",
                expected=49,
                actual=None,
                passed=None,
                note="Exact count requires curated p_eff subset from gamma pipeline",
            )
        )

    # --- 7. Full-corpus gamma_morph sign-overlap check ---
    # The paper claims zero sign overlap at the reference p_eff = 0.1.
    # T1-009 checks 13 curated combos via stored gamma values.
    # This check recomputes gamma_morph from raw Phi data across the
    # full corpus at the reference p_eff (± 0.02), verifying the
    # sign-class assignment for every individual config.
    TRAP_BODIES = {"mercury", "venus", "mars", "ceres", "europa", "jupiter", "saturn", "titan"}
    P_REF = 0.1
    P_TOL = 0.02  # reference p_eff window

    sign_ok_configs = 0
    sign_fail_configs = 0
    sign_fail_groups: list[str] = []
    groups_tested = 0

    # Orbital bodies from phi_decompose (expected TRAP: slope < 0)
    if phi and "results" in phi:
        from collections import defaultdict

        by_target: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in phi["results"]:
            target = r.get("target", "")
            p = r.get("p_eff", 0)
            phi_n = r.get("phi_normal", 0)
            eh = r.get("E_H", 0)
            if not (phi_n and phi_n > 0 and eh > 0 and target):
                continue
            if abs(p - P_REF) > P_TOL:
                continue
            by_target[target].append((eh, np.log(phi_n)))

        for target, pts in by_target.items():
            if len(pts) < 10:
                continue
            groups_tested += 1
            ehs = np.array([x[0] for x in pts])
            lps = np.array([x[1] for x in pts])
            if np.std(ehs) < 0.01:
                continue
            slope = np.polyfit(ehs, lps, 1)[0]
            expected_neg = target in TRAP_BODIES
            if expected_neg and slope < 0:
                sign_ok_configs += len(pts)
            elif not expected_neg and slope > 0:
                sign_ok_configs += len(pts)
            else:
                sign_fail_configs += len(pts)
                sign_fail_groups.append(f"{target}@p={P_REF}:slope={slope:+.4f}")

    # CRAWDAD traces (expected CLUSTER: slope > 0)
    for exp in ["Exp1", "Exp2", "Exp3", "Exp6"]:
        cd = load(f"crawdad_contacts.{exp}_results.json")
        if not cd or "results" not in cd:
            continue
        pts_exp: list[tuple[float, float]] = []
        for r in cd["results"]:
            phi_n = r.get("phi_normal", 0)
            eh = r.get("E_H", 0)
            p = r.get("p_eff", 0)
            if not (phi_n and phi_n > 0 and eh > 0):
                continue
            if abs(p - P_REF) > P_TOL:
                continue
            pts_exp.append((eh, np.log(phi_n)))
        if len(pts_exp) < 10:
            continue
        groups_tested += 1
        ehs = np.array([x[0] for x in pts_exp])
        lps = np.array([x[1] for x in pts_exp])
        if np.std(ehs) < 0.01:
            continue
        slope = np.polyfit(ehs, lps, 1)[0]
        if slope > 0:
            sign_ok_configs += len(pts_exp)
        else:
            sign_fail_configs += len(pts_exp)
            sign_fail_groups.append(f"{exp}@p={P_REF}:slope={slope:+.4f}")

    total_checked = sign_ok_configs + sign_fail_configs
    claims.append(
        check(
            Claim(
                id="T1-011",
                tier=1,
                section="Conjecture",
                line=545,
                description=f"Full-corpus gamma_morph sign overlap = 0 "
                f"at p_eff={P_REF} "
                f"({total_checked} configs, {groups_tested} groups)",
                expected=0,
                actual=sign_fail_configs,
                note=f"ok={sign_ok_configs}, fail={sign_fail_configs}"
                + (f", violations={sign_fail_groups[:10]}" if sign_fail_groups else ""),
            )
        )
    )

    return claims


# ---------------------------------------------------------------------------
# Tier 2 — Table values and detail claims
# ---------------------------------------------------------------------------


def tier2_claims() -> list[Claim]:
    claims = []

    # --- Table 3: gamma values at p_eff=0.1 (13 entries) ---
    bootstrap = load("bootstrap_ci_results.json")
    veh = load("vehicular_gamma_results.json")

    paper_gamma_table3 = {
        "Ceres": -1.20,
        "Jupiter": -1.14,
        "Mercury": -1.01,
        "Saturn": -0.67,
        "Europa": -0.54,
        "Mars": -0.40,
        "Venus": -0.21,
        "Titan": -0.10,
        "Exp1": +1.89,
        "Exp2": +1.85,
        "Exp3": +2.22,
        "Exp6": +2.07,
    }
    if bootstrap:
        results = bootstrap["results"]
        for body, expected_gamma in paper_gamma_table3.items():
            actual_gamma = results.get(body, {}).get("gamma_paper")
            claims.append(
                check(
                    Claim(
                        id=f"T2-T3-{body}",
                        tier=2,
                        section="Table 3",
                        line=777,
                        description=f"Table 3 gamma: {body} = {expected_gamma}",
                        expected=expected_gamma,
                        actual=actual_gamma,
                    )
                )
            )

    # SF Cab gamma (from vehicular results, p_eff=0.1)
    if veh:
        sf_gamma = veh.get("gamma", {}).get("0.1", {}).get("raw_slope")
        claims.append(
            check(
                Claim(
                    id="T2-T3-SFCab",
                    tier=2,
                    section="Table 3",
                    line=787,
                    description="Table 3 gamma: SF Cab = +2.29",
                    expected=2.29,
                    actual=round(sf_gamma, 2) if sf_gamma else None,
                )
            )
        )

    # --- Table 7: Mars tier stats ---
    braess = load("analysis_braess_variance_results.json")
    if braess:
        paper_tiers = {
            "1": {"S_T": 0.869, "DR": 0.475, "eta": 0.527},
            "2": {"S_T": 0.949, "DR": 0.400, "eta": 0.400},
            "3": {"S_T": 1.000, "DR": 0.497, "eta": 0.498},
            "4": {"S_T": 1.000, "DR": 0.591, "eta": 0.591},
        }
        tier_stats = braess["tier_stats"]
        for t, expected in paper_tiers.items():
            actual_st = tier_stats[t]["S_T_mean"]
            actual_dr = tier_stats[t]["DR_mean"]
            actual_eta = tier_stats[t]["eta_mean"]
            claims.append(
                check(
                    Claim(
                        id=f"T2-T7-T{t}-ST",
                        tier=2,
                        section="Table 7",
                        line=968,
                        description=f"Table 7 Tier {t}: S_T = {expected['S_T']:.3f}",
                        expected=expected["S_T"],
                        actual=round(actual_st, 3),
                    )
                )
            )
            claims.append(
                check(
                    Claim(
                        id=f"T2-T7-T{t}-DR",
                        tier=2,
                        section="Table 7",
                        line=968,
                        description=f"Table 7 Tier {t}: DR = {expected['DR']:.3f}",
                        expected=expected["DR"],
                        actual=round(actual_dr, 3),
                    )
                )
            )

    # --- Mars tier upgrades ---
    if braess:
        upgrades = braess["tier_upgrades"]
        # T1->T2: mean_gain = -0.075, std = 0.092
        t12 = upgrades["T1→T2"]
        claims.append(
            check(
                Claim(
                    id="T2-Braess-mean",
                    tier=2,
                    section="Mars Braess",
                    line=1019,
                    description="T1→T2 mean DR change = -0.075",
                    expected=-0.075,
                    actual=round(t12["mean_gain"], 3),
                    tolerance=0.02,
                )
            )
        )
        claims.append(
            check(
                Claim(
                    id="T2-Braess-std",
                    tier=2,
                    section="Mars Braess",
                    line=1019,
                    description="T1→T2 std DR change = 0.092",
                    expected=0.092,
                    actual=round(t12["std_gain"], 3),
                    tolerance=0.02,
                )
            )
        )
        # T2->T3: improvement in 35/39 epochs, mean = +0.097
        t23 = upgrades["T2→T3"]
        claims.append(
            check(
                Claim(
                    id="T2-T23-n",
                    tier=2,
                    section="Mars Braess",
                    line=1028,
                    description="T2→T3 improvement in 35/39 epochs",
                    expected=35,
                    actual=t23["n_positive"],
                )
            )
        )
        claims.append(
            check(
                Claim(
                    id="T2-T23-mean",
                    tier=2,
                    section="Mars Braess",
                    line=1028,
                    description="T2→T3 mean DR change = +0.097",
                    expected=0.097,
                    actual=round(t23["mean_gain"], 3),
                    tolerance=0.02,
                )
            )
        )
        # T3->T4: improvement in 29/39 epochs
        t34 = upgrades["T3→T4"]
        claims.append(
            check(
                Claim(
                    id="T2-T34-n",
                    tier=2,
                    section="Mars Braess",
                    line=1036,
                    description="T3→T4 improvement in 29/39 epochs",
                    expected=29,
                    actual=t34["n_positive"],
                )
            )
        )
        claims.append(
            check(
                Claim(
                    id="T2-T34-mean",
                    tier=2,
                    section="Mars Braess",
                    line=1036,
                    description="T3→T4 mean DR change = +0.094",
                    expected=0.094,
                    actual=round(t34["mean_gain"], 3),
                    tolerance=0.02,
                )
            )
        )

    # --- Bootstrap CIs (Venus, Exp2, Exp3, SF Cab) ---
    if bootstrap:
        results = bootstrap["results"]
        ci_checks = {
            "Venus": {"lo": -0.41, "hi": 0.17, "key": "Venus"},
            "Exp2": {"lo": 1.81, "hi": 1.88, "key": "Exp2"},
            "Exp3": {"lo": 2.19, "hi": 2.25, "key": "Exp3"},
        }
        for name, spec in ci_checks.items():
            info = results.get(spec["key"], {})
            ci_lo = info.get("ci_lower")
            ci_hi = info.get("ci_upper")
            if ci_lo is not None:
                claims.append(
                    check(
                        Claim(
                            id=f"T2-CI-{name}-lo",
                            tier=2,
                            section="Validation",
                            line=762,
                            description=f"Bootstrap CI lower {name} = {spec['lo']}",
                            expected=spec["lo"],
                            actual=round(ci_lo, 2),
                            tolerance=0.03,
                        )
                    )
                )
            if ci_hi is not None:
                claims.append(
                    check(
                        Claim(
                            id=f"T2-CI-{name}-hi",
                            tier=2,
                            section="Validation",
                            line=762,
                            description=f"Bootstrap CI upper {name} = {spec['hi']}",
                            expected=spec["hi"],
                            actual=round(ci_hi, 2),
                            tolerance=0.03,
                        )
                    )
                )
        # SF Cab CI
        sf_ci = results.get("SF Cab", {})
        if sf_ci:
            claims.append(
                check(
                    Claim(
                        id="T2-CI-SFCab-lo",
                        tier=2,
                        section="Validation",
                        line=763,
                        description="Bootstrap CI lower SF Cab = 2.19",
                        expected=2.19,
                        actual=round(sf_ci.get("ci_lower", 0), 2),
                        tolerance=0.03,
                    )
                )
            )
            claims.append(
                check(
                    Claim(
                        id="T2-CI-SFCab-hi",
                        tier=2,
                        section="Validation",
                        line=763,
                        description="Bootstrap CI upper SF Cab = 2.38",
                        expected=2.38,
                        actual=round(sf_ci.get("ci_upper", 0), 2),
                        tolerance=0.03,
                    )
                )
            )

    # --- 226 unique source-destination pairs ---
    mosaic = load("pair_gamma_mosaic_results.json")
    if mosaic:
        pair_gammas = mosaic["pair_gammas"]
        # Count unique (trace, source, dest) tuples
        unique_pairs = set()
        crawdad_pairs = set()
        for item in pair_gammas:
            trace = item["trace"]
            key = (trace, item["source"], item["dest"])
            unique_pairs.add(key)
            if trace in ("Exp1", "Exp2", "Exp3", "Exp6"):
                crawdad_pairs.add(key)

        claims.append(
            check(
                Claim(
                    id="T2-226-total",
                    tier=2,
                    section="Validation",
                    line=743,
                    description="226 unique source-dest pairs (paper claim)",
                    expected=226,
                    actual=len(unique_pairs),
                    note=f"CRAWDAD-only={len(crawdad_pairs)}, all traces={len(unique_pairs)}",
                )
            )
        )
        # Check zero opposite-sign pairs (pooled mean gamma across p_eff)
        from collections import defaultdict

        pair_mean_gamma = defaultdict(list)
        for item in pair_gammas:
            key = (item["trace"], item["source"], item["dest"])
            pair_mean_gamma[key].append(item["gamma_pair"])
        n_negative_mean = sum(
            1 for gammas in pair_mean_gamma.values() if sum(gammas) / len(gammas) < 0
        )
        claims.append(
            check(
                Claim(
                    id="T2-226-sign",
                    tier=2,
                    section="Validation",
                    line=744,
                    description="Zero pairs with negative mean gamma (pooled)",
                    expected=0,
                    actual=n_negative_mean,
                )
            )
        )

    # --- Achievability gap: Moon 35.5x at p_eff=0.3 ---
    achiev = load("achievability_results.json")
    if achiev:
        moon = achiev.get("moon", {})
        p_rows = moon.get("p_eff_rows", [])
        for row in p_rows:
            if abs(row["p_eff"] - 0.3) < 0.01:
                claims.append(
                    check(
                        Claim(
                            id="T2-Achiev-Moon03",
                            tier=2,
                            section="Discussion",
                            line=1097,
                            description="Moon achievability gap at p_eff=0.3 = 35.5x",
                            expected=35.5,
                            actual=round(row["phi_ratio"], 1),
                            tolerance=0.02,
                        )
                    )
                )
                claims.append(
                    check(
                        Claim(
                            id="T2-Achiev-Moon03-pct",
                            tier=2,
                            section="Discussion",
                            line=1097,
                            description="Moon greedy captures 2.8% of capacity",
                            expected=0.028,
                            actual=round(row["eta_greedy"] / row["eta_optimal"], 3),
                            tolerance=0.05,
                        )
                    )
                )
                break

    # --- Appendix: CRAWDAD raw gamma by p_eff ---
    paper_raw_gamma = {
        "Exp1": {0.02: 3.30, 0.05: 2.43, 0.1: 1.89, 0.3: 0.98, 0.5: 0.51},
        "Exp2": {0.02: 3.24, 0.05: 2.39, 0.1: 1.85, 0.3: 1.06, 0.5: 0.65},
        "Exp3": {0.02: 3.82, 0.05: 2.91, 0.1: 2.22, 0.3: 1.16, 0.5: 0.68},
        "Exp6": {0.02: 3.61, 0.05: 2.73, 0.1: 2.07, 0.3: 1.07, 0.5: 0.61},
    }
    for exp, p_gammas in paper_raw_gamma.items():
        cd = load(f"crawdad_contacts.{exp}_results.json")
        if not cd:
            continue
        results = cd.get("results", [])
        for p_eff, expected_gamma in p_gammas.items():
            # Compute gamma = OLS slope of ln(phi_normal) vs E_H at this p_eff
            pts = [
                r
                for r in results
                if abs(r.get("p_eff", 0) - p_eff) < 0.001
                and r.get("phi_normal", 0) > 0
                and r.get("E_H", 0) > 0
            ]
            if len(pts) < 3:
                continue
            E_H = np.array([p["E_H"] for p in pts])
            ln_phi = np.array([np.log(p["phi_normal"]) for p in pts])
            slope, _, _, _, _ = stats.linregress(E_H, ln_phi)
            claims.append(
                check(
                    Claim(
                        id=f"T2-App-{exp}-p{p_eff}",
                        tier=2,
                        section="Appendix",
                        line=1604,
                        description=f"Raw gamma {exp} p_eff={p_eff}: {expected_gamma}",
                        expected=expected_gamma,
                        actual=round(float(slope), 2),
                        tolerance=0.02,
                    )
                )
            )

    return claims


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Run all tiers (default: Tier 1 only)")
    args = parser.parse_args()

    os.chdir(ROOT)

    print("=" * 72)
    print("  PAPER CLAIM VERIFICATION — classification_theorem.tex")
    print("=" * 72)

    claims = tier1_claims()
    if args.full:
        claims.extend(tier2_claims())

    # Print results
    n_pass = n_fail = n_skip = 0
    for c in claims:
        if c.passed is True:
            status = "PASS"
            n_pass += 1
        elif c.passed is False:
            status = "FAIL"
            n_fail += 1
        else:
            status = "SKIP"
            n_skip += 1

        icon = {"PASS": " OK ", "FAIL": "FAIL", "SKIP": " -- "}[status]
        line = f"[{icon}] {c.id:20s} {c.description}"
        if c.passed is False:
            line += f"\n       expected={c.expected}  actual={c.actual}"
        if c.note:
            line += f"\n       note: {c.note}"
        print(line)

    print()
    print("-" * 72)
    tiers_run = "Tier 1 + 2 (full)" if args.full else "Tier 1 (quick)"
    print(f"  {tiers_run}: {len(claims)} claims checked")
    print(f"  PASS: {n_pass}  |  FAIL: {n_fail}  |  SKIP: {n_skip}")
    if _missing:
        print(f"  Not checkable here: claim families gated on {len(_missing)} "
              f"absent result file(s): {', '.join(sorted(_missing))}")
    if n_fail > 0:
        print(f"\n  *** {n_fail} CLAIM(S) FAILED — INVESTIGATE BEFORE SUBMISSION ***")
    else:
        print("\n  All verified claims match source data.")
    print("-" * 72)

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
