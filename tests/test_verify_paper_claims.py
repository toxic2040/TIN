"""Regression tests for fail-closed historical claim reproduction."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

from runs import verify_paper_claims

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "runs" / "verify_paper_claims.py"


def test_public_checkout_skips_missing_aggregate_sources():
    env = os.environ.copy()
    env.pop("DATASETS_ROOT", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--full"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[ -- ] T1-001" in result.stdout
    assert "[ -- ] T1-011" in result.stdout
    assert "(0 contributing rows, 0 groups)" in result.stdout
    assert "vacuous" not in result.stdout.lower()


def _slope_rows(slope: float, target: str | None = None) -> list[dict]:
    rows = []
    for index in range(10):
        expected_hops = 1.0 + index * 0.2
        row = {
            "E_H": expected_hops,
            "p_eff": 0.1,
            "phi_normal": math.exp(slope * expected_hops),
        }
        if target is not None:
            row["target"] = target
        rows.append(row)
    return rows


def test_group_slope_check_skips_incomplete_present_corpus(monkeypatch):
    orbital_targets = [
        "ceres",
        "europa",
        "jupiter",
        "mars",
        "mercury",
        "saturn",
        "venus",
    ]
    orbital_rows = [row for target in orbital_targets for row in _slope_rows(-0.4, target=target)]
    orbital_rows.extend(_slope_rows(+0.4, target="unexpected_target"))

    fixtures = {
        "phi_decompose_results.json": {"n_configs": 80, "results": orbital_rows},
        **{
            f"crawdad_contacts.{exp}_results.json": {
                "n_configs": 10,
                "results": _slope_rows(+0.4),
            }
            for exp in ("Exp1", "Exp2", "Exp3", "Exp6")
        },
    }
    monkeypatch.setattr(verify_paper_claims, "load", fixtures.get)

    claim = next(c for c in verify_paper_claims.tier1_claims() if c.id == "T1-011")

    assert claim.passed is None
    assert "(110 contributing rows, 11 groups)" in claim.description
    assert "missing required eligible groups: orbital=titan" in claim.note
    assert "unexpected_target" not in claim.note


def _count_fixtures() -> dict[str, dict]:
    return {
        "phi_decompose_results.json": {"n_configs": 230_400, "results": []},
        **{
            f"crawdad_contacts.{exp}_results.json": {
                "n_configs": 12_000,
                "results": [],
            }
            for exp in ("Exp1", "Exp2", "Exp3", "Exp6")
        },
        "vehicular_gamma_results.json": {"n_results": 1_200},
        "synodic_sweep_results.json": {"time_series": [None] * 780},
    }


def test_total_run_arithmetic_counts_synodic_rows_once(monkeypatch):
    fixtures = _count_fixtures()
    monkeypatch.setattr(verify_paper_claims, "load", fixtures.get)

    claim = next(c for c in verify_paper_claims.tier1_claims() if c.id == "T1-001")

    assert claim.passed is True
    assert claim.actual == 280_380
    assert "mars=780" in claim.note


def test_total_run_check_skips_present_file_with_missing_count(monkeypatch):
    fixtures = _count_fixtures()
    fixtures["crawdad_contacts.Exp3_results.json"] = {"results": []}
    monkeypatch.setattr(verify_paper_claims, "load", fixtures.get)

    claim = next(c for c in verify_paper_claims.tier1_claims() if c.id == "T1-001")

    assert claim.passed is None
    assert "crawdad_contacts.Exp3_results.json:n_configs" in claim.note


def test_historical_gamma_descriptions_scope_p01_to_supported_rows(monkeypatch):
    orbital = {
        "Ceres": -1.20,
        "Jupiter": -1.14,
        "Mercury": -1.01,
        "Saturn": -0.67,
        "Europa": -0.54,
        "Mars": -0.40,
        "Venus": -0.21,
        "Titan": -0.10,
    }
    social = {"Exp1": 1.89, "Exp2": 1.85, "Exp3": 2.22, "Exp6": 2.07}
    fixtures = {
        "bootstrap_ci_results.json": {
            "results": {
                name: {"gamma_paper": gamma} for name, gamma in {**orbital, **social}.items()
            }
        },
        "vehicular_gamma_results.json": {
            "gamma": {"0.1": {"raw_slope": 2.29, "sign": "+"}},
            "n_results": 0,
            "trace_summary": {},
        },
    }
    monkeypatch.setattr(verify_paper_claims, "load", fixtures.get)

    tier1 = {claim.id: claim for claim in verify_paper_claims.tier1_claims()}
    tier2 = {claim.id: claim for claim in verify_paper_claims.tier2_claims()}

    assert "composite orbital" in tier1["T1-009"].description
    assert "social/vehicular pos at p=0.1" in tier1["T1-009"].description
    assert "p_eff=0.1" not in tier2["T2-T3-Mercury"].description
    assert "composite" in tier2["T2-T3-Mercury"].description
    assert "p_eff=0.1" in tier2["T2-T3-Exp2"].description
    assert "p_eff=0.1" in tier2["T2-T3-SFCab"].description


def test_tier2_present_incomplete_sources_append_explicit_skips(monkeypatch):
    orbital = {
        "Ceres": -1.20,
        "Jupiter": -1.14,
        "Mercury": -1.01,
        "Saturn": -0.67,
        "Europa": -0.54,
        "Mars": -0.40,
        "Venus": -0.21,
        "Titan": -0.10,
    }
    social = {"Exp1": 1.89, "Exp2": 1.85, "Exp3": 2.22, "Exp6": 2.07}
    fixtures = {
        "bootstrap_ci_results.json": {
            "results": {
                name: {"gamma_paper": gamma} for name, gamma in {**orbital, **social}.items()
            }
        },
        "vehicular_gamma_results.json": {
            "gamma": {"0.1": {"raw_slope": 2.29}},
        },
        "achievability_results.json": {"moon": {"p_eff_rows": [{"p_eff": 0.2, "phi_ratio": 10.0}]}},
        **{
            f"crawdad_contacts.{exp}_results.json": {"results": []}
            for exp in ("Exp1", "Exp2", "Exp3", "Exp6")
        },
    }
    monkeypatch.setattr(verify_paper_claims, "load", fixtures.get)

    claims = {claim.id: claim for claim in verify_paper_claims.tier2_claims()}

    ci_ids = {
        f"T2-CI-{name}-{bound}"
        for name in ("Venus", "Exp2", "Exp3", "SFCab")
        for bound in ("lo", "hi")
    }
    moon_ids = {"T2-Achiev-Moon03", "T2-Achiev-Moon03-pct"}
    appendix_ids = {
        f"T2-App-{exp}-p{p_eff}"
        for exp in ("Exp1", "Exp2", "Exp3", "Exp6")
        for p_eff in (0.02, 0.05, 0.1, 0.3, 0.5)
    }
    expected_skips = ci_ids | moon_ids | appendix_ids

    assert expected_skips <= claims.keys()
    assert all(claims[claim_id].passed is None for claim_id in expected_skips)
    assert "required numeric CI field" in claims["T2-CI-Venus-lo"].note
    assert "required Moon p_eff=0.3 row" in claims["T2-Achiev-Moon03"].note
    assert "0 eligible rows" in claims["T2-App-Exp1-p0.1"].note
