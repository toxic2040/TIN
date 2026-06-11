"""epyc_phase5.py — Session 7 EPYC batch: all four paper gaps in one shot.

Runs:
  1. Expanded Braess (120 tasks, ~4 min) — Paper 3
  2. Mercury Braess deep dive (651 tasks, ~10 min) — Paper 3
  3. Transfer operator pilot (20 tasks, ~5 min) — Paper 2
  4. NRHO extended window (126 tasks, ~15 min) — Paper 4

Total: ~917 tasks, estimated ~30 min on 192+ core EPYC.

Usage:
    nohup python runs/epyc_phase5.py > phase5.log 2>&1 &
    python runs/epyc_phase5.py --smoke   # quick validation
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
RESULTS_DIR = _HERE / "epyc_results" / "phase5_2026_03_12"


def _run_step(name, cmd, timeout_s=3600):
    """Run one experiment step, return (success, elapsed, output_path)."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 5 — STEP: {name}")
    print(f"{'=' * 70}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Timeout: {timeout_s}s")
    print()

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0
        print(f"\n  {'PASS' if success else 'FAIL'} — {name} in {elapsed:.1f}s")
        return success, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"\n  TIMEOUT — {name} after {elapsed:.1f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ERROR — {name}: {e}")
        return False, elapsed


def _collect_result(json_name, step_name):
    """Copy a result JSON to the phase5 results directory."""
    src = _HERE / json_name
    if src.exists():
        dst = RESULTS_DIR / json_name
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"  Collected: {json_name} ({size_kb:.1f} KB)")
    else:
        print(f"  WARNING: {json_name} not found after {step_name}")


def main():
    smoke = "--smoke" in sys.argv
    t0_total = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EPYC PHASE 5 — ALL FOUR PAPER GAPS")
    print("=" * 70)
    print(f"  Mode: {'SMOKE' if smoke else 'FULL'}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Cores: {os.cpu_count()}")
    print()

    summary = {}

    # Step 1: Expanded Braess
    ok, elapsed = _run_step(
        "Expanded Braess (Paper 3)",
        [sys.executable, "-m", "runs.run_expanded_braess"],
        timeout_s=1800,
    )
    summary["expanded_braess"] = {"success": ok, "elapsed_s": round(elapsed, 1)}
    _collect_result("expanded_braess_results.json", "expanded_braess")

    # Step 2: Mercury Braess deep dive
    cmd = [sys.executable, "-m", "runs.run_mercury_braess_deep"]
    if smoke:
        cmd.append("--smoke")
    ok, elapsed = _run_step(
        "Mercury Braess Deep Dive (Paper 3)",
        cmd,
        timeout_s=3600,
    )
    summary["mercury_braess_deep"] = {"success": ok, "elapsed_s": round(elapsed, 1)}
    _collect_result("mercury_braess_deep_results.json", "mercury_braess_deep")

    # Step 3: Transfer operator pilot (Saturn/Jupiter)
    ok, elapsed = _run_step(
        "Transfer Operator Pilot (Paper 2)",
        [sys.executable, "-m", "runs.run_transfer_operator_pilot"],
        timeout_s=1800,
    )
    summary["transfer_operator"] = {"success": ok, "elapsed_s": round(elapsed, 1)}
    _collect_result("transfer_operator_pilot_results.json", "transfer_operator")

    # Step 4: NRHO extended window
    cmd = [sys.executable, "-m", "runs.run_nrho_extended_window"]
    if smoke:
        cmd.append("--smoke")
    ok, elapsed = _run_step(
        "NRHO Extended Window (Paper 4)",
        cmd,
        timeout_s=3600,
    )
    summary["nrho_extended_window"] = {"success": ok, "elapsed_s": round(elapsed, 1)}
    _collect_result("nrho_extended_window_results.json", "nrho_extended_window")

    # Final summary
    total_elapsed = time.time() - t0_total
    summary["total_elapsed_s"] = round(total_elapsed, 1)
    summary["all_passed"] = all(
        s["success"] for s in summary.values() if isinstance(s, dict) and "success" in s
    )

    print(f"\n{'=' * 70}")
    print("PHASE 5 — FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total elapsed: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print()
    for step, info in summary.items():
        if isinstance(info, dict) and "success" in info:
            status = "PASS" if info["success"] else "FAIL"
            print(f"  {step:30s}: {status} ({info['elapsed_s']:.1f}s)")
    print()
    print(f"  ALL PASSED: {summary['all_passed']}")

    # Save summary
    summary_path = RESULTS_DIR / "phase5_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
