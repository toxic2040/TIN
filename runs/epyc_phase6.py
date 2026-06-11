#!/usr/bin/env python3
"""epyc_phase6.py — Paper 2 critical experiments batch.

Runs three experiments sequentially:
  1. Per-path covariance (the sharp mechanism test)
  2. Mars synodic sweep (frac_orbit causality)
  3. Time-reversal symmetry v5 (with parallel Exp6 fix)

All use the FROZEN engine (tin/core/).  No engine code is modified.

Usage:
    conda activate tin
    cd ~/TIN
    nohup python runs/epyc_phase6.py > phase6.log 2>&1 &
    tail -f phase6.log

Reads:  data/traces/ (CRAWDAD Exp1-Exp6), data/vehicular/ (optional)
Writes: runs/per_path_cov_results.json
        runs/mars_synodic_sweep_results.json
        runs/time_reversal_results.json

v2 changes (12 March 2026):
  - Module invocation (-m) instead of direct file execution
  - Per-experiment timeout tuning (not a flat 2h cap)
  - Pre-flight import validation before running each experiment
  - Proper process cleanup (kill process group on timeout)
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent

EXPERIMENTS = [
    {
        "name": "Per-path Covariance",
        "module": "runs.run_per_path_cov",
        "output": "runs/per_path_cov_results.json",
        "paper": "Paper 2 Sec 3 — THE sharp mechanism test",
        "timeout_s": 3600,  # 1h — 256 tasks, multi-family constellations up to n=40
    },
    {
        "name": "Mars Synodic Sweep",
        "module": "runs.run_mars_synodic_sweep",
        "output": "runs/mars_synodic_sweep_results.json",
        "paper": "Paper 2 Sec 4 — frac_orbit causality",
        "timeout_s": 28800,  # 8 hours — 780d windows generate millions of contacts
    },
    {
        "name": "Time-Reversal Symmetry v5",
        "module": "runs.run_time_reversal",
        "output": "runs/time_reversal_results.json",
        "paper": "Paper 2 Sec 4 — temporal asymmetry",
        "timeout_s": 43200,  # 12 hours — 8 bodies × many configs × parallel oracle
    },
]


def _preflight_check(module_name: str) -> str | None:
    """Try importing the module to catch ModuleNotFoundError before committing.

    Returns None on success, error string on failure.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return proc.stderr.strip().split("\n")[-1]
        return None
    except subprocess.TimeoutExpired:
        return "import timed out after 30s"


def main():
    t0 = time.time()
    print("=" * 70)
    print("TIN EPYC Phase 6 — Paper 2 Critical Experiments (v2)")
    print("=" * 70)
    print()
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  Working:   {REPO}")
    print()

    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        output = REPO / exp["output"]
        timeout_s = exp["timeout_s"]
        timeout_h = timeout_s / 3600

        print(f"{'─' * 70}")
        print(f"  [{i}/{len(EXPERIMENTS)}] {exp['name']}")
        print(f"  Module:  {exp['module']}")
        print(f"  For:     {exp['paper']}")
        print(f"  Timeout: {timeout_h:.1f}h ({timeout_s}s)")
        print(f"{'─' * 70}")
        print()

        # Pre-flight: verify the module imports cleanly
        print(f"  Pre-flight import check: {exp['module']} ...", end=" ", flush=True)
        err = _preflight_check(exp["module"])
        if err is not None:
            print("FAIL")
            print(f"  Import error: {err}")
            results.append({"name": exp["name"], "status": f"IMPORT_FAIL: {err}", "elapsed_s": 0})
            continue
        print("OK")

        exp_t0 = time.time()
        try:
            env = {**os.environ, "PYTHONPATH": str(REPO)}
            # Use module invocation (-m) so `runs` is a recognized package.
            # start_new_session=True creates a process group for clean cleanup.
            proc = subprocess.Popen(
                [sys.executable, "-m", exp["module"]],
                cwd=str(REPO),
                env=env,
                start_new_session=True,
            )
            proc.wait(timeout=timeout_s)
            exp_elapsed = time.time() - exp_t0
            status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"

            # Check output file
            if output.exists():
                size_kb = output.stat().st_size / 1024
                print(f"\n  Output: {output.name} ({size_kb:.1f} KB)")
            else:
                print(f"\n  WARNING: Expected output {output.name} not found")

        except subprocess.TimeoutExpired:
            exp_elapsed = time.time() - exp_t0
            status = f"TIMEOUT ({timeout_h:.1f}h)"
            print(f"\n  TIMEOUT after {exp_elapsed:.0f}s — killing process group ...")
            # Kill the entire process group (parent + all child workers)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            # Incremental saves mean partial results are already on disk
            if output.exists():
                size_kb = output.stat().st_size / 1024
                print(f"  Partial output preserved: {output.name} ({size_kb:.1f} KB)")
        except Exception as e:
            exp_elapsed = time.time() - exp_t0
            status = f"ERROR: {e}"
            print(f"\n  ERROR: {e}")

        results.append(
            {
                "name": exp["name"],
                "status": status,
                "elapsed_s": round(exp_elapsed, 1),
            }
        )
        print(f"\n  Status: {status}  ({exp_elapsed:.1f}s)")
        print()

    # Summary
    total = time.time() - t0
    print()
    print("=" * 70)
    print(f"Phase 6 COMPLETE — {total:.1f}s total ({total / 3600:.2f} h)")
    print("=" * 70)
    print()
    for r in results:
        elapsed = r.get("elapsed_s", 0)
        print(f"  {r['name']:<30s}  {r['status']:<20s}  {elapsed:.1f}s")

    print()
    print("Next: scp results back to local machine:")
    print("  scp root@$(hostname):~/TIN/runs/per_path_cov_results.json .")
    print("  scp root@$(hostname):~/TIN/runs/mars_synodic_sweep_results.json .")
    print("  scp root@$(hostname):~/TIN/runs/time_reversal_results.json .")


if __name__ == "__main__":
    main()
