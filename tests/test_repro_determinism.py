"""tests/test_repro_determinism.py — Determinism smoke test for the v0.3.8 repro entrypoint.

Runs runs/repro_v0_3_8.py twice with the same --seed and verifies the JSON
output is identical on every field except the wall-clock timestamp. Also
runs once with a different seed to confirm the seed plumbed through (i.e.
that bundle generation, storm draws, and link probabilities all flow
through default_rng(seed) rather than a hidden global state).

Kept as a fast smoke test (sim_days=1, bundles=50) so it runs in <1 min on
CI. The full 28-day baseline determinism is verified out-of-band when a new
baseline JSON is committed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "runs" / "repro_v0_3_8.py"


def _run(seed: int, out_dir: Path, label: str) -> dict:
    out = out_dir / label
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--sim_days",
        "1",
        "--bundles",
        "50",
        "--seed",
        str(seed),
        "--output",
        str(out),
        "--no_coverage_png",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    with open(out.with_suffix(".json")) as f:
        return json.load(f)


def _strip_volatile(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "timestamp_utc"}


def test_same_seed_produces_byte_identical_results(tmp_path):
    a = _run(seed=42, out_dir=tmp_path, label="a")
    b = _run(seed=42, out_dir=tmp_path, label="b")
    assert _strip_volatile(a) == _strip_volatile(b), (
        "Two runs with seed=42 produced different results — RNG plumbing has a hole"
    )


def test_different_seed_produces_different_results(tmp_path):
    a = _run(seed=42, out_dir=tmp_path, label="a")
    c = _run(seed=7, out_dir=tmp_path, label="c")
    assert _strip_volatile(a) != _strip_volatile(c), (
        "seed=42 and seed=7 produced identical results — seed is not flowing through"
    )
