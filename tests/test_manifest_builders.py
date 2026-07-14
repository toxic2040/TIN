"""Regression tests for safe manifest generation boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runs import build_config_manifest, build_provenance_manifest


def test_provenance_scan_excludes_manifest_metadata():
    names = {
        path.name
        for path in build_provenance_manifest.collect_jsons(build_provenance_manifest.RUNS_DIR)
    }

    assert "CONFIG_MANIFEST.json" not in names
    assert "PROVENANCE.json" not in names


def test_unresolved_summary_does_not_fall_back_to_epyc_phase5():
    scripts = {"epyc_phase5": Path("runs/epyc_phase5.py")}

    matched = build_provenance_manifest.match_script(
        Path("runs/epyc_results/campaign_2026_03_11/campaign_summary.json"), scripts
    )

    assert matched is None


def test_repro_summary_uses_exact_documented_producer():
    producer = Path("runs/repro_v0_3_8_ensemble.py")
    scripts = {
        "epyc_phase5": Path("runs/epyc_phase5.py"),
        "repro_v0_3_8_ensemble": producer,
    }

    matched = build_provenance_manifest.match_script(
        Path("runs/results/repro_v0_3_8_ensemble_summary.json"), scripts
    )

    assert matched == producer


def test_config_and_provenance_use_same_tracked_json_scope():
    config_paths = build_config_manifest.collect_jsons(build_config_manifest._RUNS)
    provenance_paths = build_provenance_manifest.collect_jsons(build_provenance_manifest.RUNS_DIR)
    config_relative = [path.relative_to(build_config_manifest._RUNS) for path in config_paths]
    provenance_relative = [
        path.relative_to(build_provenance_manifest.RUNS_DIR) for path in provenance_paths
    ]

    assert config_relative == provenance_relative
    assert Path("results/repro_v0_3_8_ensemble_summary.json") in config_relative
    assert len(config_relative) == len(set(config_relative))


def test_archive_fallback_recurses_without_manifest_metadata(tmp_path):
    runs_dir = tmp_path / "runs"
    nested_dir = runs_dir / "results"
    nested_dir.mkdir(parents=True)
    (runs_dir / "top.json").write_text(json.dumps({"results": [1]}))
    (nested_dir / "nested.json").write_text(json.dumps([1, 2]))
    (runs_dir / "CONFIG_MANIFEST.json").write_text("{}")
    (runs_dir / "CONFIG_MANIFEST.current.json").write_text("{}")
    (runs_dir / "PROVENANCE.json").write_text("{}")

    manifest = build_config_manifest.build_manifest(runs_dir)

    assert [entry["file"] for entry in manifest["files"]] == [
        "results/nested.json",
        "top.json",
    ]
    assert manifest["summary"] == {
        "total_files": 2,
        "total_configs": 3,
        "total_size_mb": 0.0,
        "errors": 0,
    }


def test_historical_config_manifest_requires_explicit_overwrite():
    with pytest.raises(SystemExit, match="2"):
        build_config_manifest.main(["--output", str(build_config_manifest._HISTORICAL_MANIFEST)])
