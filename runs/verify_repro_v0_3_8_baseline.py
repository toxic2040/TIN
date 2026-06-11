#!/usr/bin/env python3
"""Verify the committed v0.3.8 seed-42 baseline artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_SCRIPT = REPO_ROOT / "runs" / "repro_v0_3_8.py"
BASELINE = REPO_ROOT / "runs" / "results" / "repro_v0_3_8_baseline.json"
VOLATILE_FIELDS = frozenset({"timestamp_utc"})


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return data


def _strip_volatile(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in VOLATILE_FIELDS}


def _run_repro(output_stem: Path, coverage_workers: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPRO_SCRIPT),
        "--seed",
        "42",
        "--output",
        str(output_stem),
        "--no_coverage_png",
        "--coverage_workers",
        str(coverage_workers),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return _load_json(output_stem.with_suffix(".json"))


def _diff(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    keys = sorted(set(expected) | set(actual))
    for key in keys:
        if key in VOLATILE_FIELDS:
            continue
        if expected.get(key) != actual.get(key):
            rows.append(f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-workers",
        type=int,
        default=1,
        help="Coverage-grid worker count passed to the reproduction run.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for the generated comparison artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = _load_json(BASELINE)

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="tin-repro-") as td:
            actual = _run_repro(Path(td) / "repro_v0_3_8", args.coverage_workers)
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        actual = _run_repro(args.work_dir / "repro_v0_3_8", args.coverage_workers)

    expected_clean = _strip_volatile(expected)
    actual_clean = _strip_volatile(actual)
    if expected_clean == actual_clean:
        print("v0.3.8 baseline comparison passed")
        return 0

    print("v0.3.8 baseline comparison failed", file=sys.stderr)
    for row in _diff(expected, actual):
        print(f"  {row}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
