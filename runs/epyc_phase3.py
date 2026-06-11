"""epyc_phase3.py — Routing Independence + Vehicular γ (target: ~2 hours)

Two strengthening experiments:

  A) Routing independence — run 5 CRAWDAD traces under 4 routing policies
     (greedy, no-retry, oracle, random). Confirm γ sign agreement across
     all policies for each trace.

  B) Vehicular γ — convert vehicular GPS trace to contacts, compute γ.
     Confirm γ > 0 (cluster class) on a non-Bluetooth trace.

Usage:
    python runs/epyc_phase3.py                # full phase 3
    python runs/epyc_phase3.py --skip-a       # skip routing independence
    python runs/epyc_phase3.py --skip-b       # skip vehicular
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_RESULTS = _ROOT / "runs" / "epyc_results"
_RESULTS.mkdir(exist_ok=True)


def run_one(label, cmd, timeout=14400):
    """Run a single job, return (label, ok, elapsed, returncode)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        log_path = _RESULTS / f"{label}.log"
        with open(log_path, "w") as f:
            f.write(result.stdout)
        return (label, result.returncode == 0, elapsed, result.returncode)
    except subprocess.TimeoutExpired:
        return (label, False, time.time() - t0, -1)
    except Exception as e:
        return (label, False, time.time() - t0, str(e))


def run_routing_independence():
    """Workload A: routing independence across 5 CRAWDAD traces."""
    print("=" * 70)
    print("WORKLOAD A: Routing Independence (5 traces × 4 policies)")
    print("=" * 70)

    n_workers = min(os.cpu_count() or 8, 48)
    label = "routing_independence"
    cmd = [
        sys.executable,
        "runs/run_routing_independence.py",
        "--all",
        "--workers",
        str(n_workers),
    ]

    print(f"  Workers: {n_workers}")
    print(f"  Running: {' '.join(cmd)}")
    print()

    result = run_one(label, cmd, timeout=7200)
    status = "PASS" if result[1] else "FAIL"
    print(f"\n  [{status}] {label}  {result[2]:.0f}s  rc={result[3]}")

    # Check results
    results_found = list(_ROOT.glob("runs/routing_independence_*_results.json"))
    sign_summary = {}

    for rpath in results_found:
        try:
            with open(rpath) as f:
                data = json.load(f)
            trace = data.get("trace", rpath.stem)
            agreement = data.get("sign_agreement", {})
            sign_summary[trace] = agreement
            mark = "PASS" if agreement.get("overall") else "FAIL"
            print(f"    [{mark}] {trace}: {agreement.get('fraction', '?')}")
        except Exception as e:
            print(f"    [ERR] {rpath.name}: {e}")

    return result, sign_summary


def run_vehicular():
    """Workload B: vehicular γ classification."""
    print("\n" + "=" * 70)
    print("WORKLOAD B: Vehicular γ Classification")
    print("=" * 70)

    # Check for vehicular data
    data_dirs = [
        (_ROOT / "data" / "vehicular" / "sfcabs" / "cabspottingdata", "sfcab"),
        (_ROOT / "data" / "vehicular" / "roma", "rome"),
        (_ROOT / "data" / "vehicular" / "tdrive", "tdrive"),
    ]

    data_path = None
    data_format = None
    for dpath, dfmt in data_dirs:
        if dpath.exists() and any(dpath.iterdir()):
            data_path = dpath
            data_format = dfmt
            break

    if data_path is None:
        print("  SKIP: no vehicular GPS data found")
        print("  Checked:")
        for dpath, _ in data_dirs:
            print(f"    {dpath}")
        print("  Run epyc_setup_v2.sh to download vehicular data")
        return None, {"skipped": True, "reason": "no data"}

    n_workers = min(os.cpu_count() or 8, 48)
    label = "vehicular_gamma"
    cmd = [
        sys.executable,
        "runs/run_vehicular_gamma.py",
        "--data",
        str(data_path),
        "--format",
        data_format,
        "--workers",
        str(n_workers),
        "--max-hours",
        "24",
        "--max-vehicles",
        "200",
    ]

    print(f"  Data: {data_path} (format: {data_format})")
    print(f"  Workers: {n_workers}")
    print()

    result = run_one(label, cmd, timeout=7200)
    status = "PASS" if result[1] else "FAIL"
    print(f"\n  [{status}] {label}  {result[2]:.0f}s  rc={result[3]}")

    # Check result
    gamma_path = _ROOT / "runs" / "vehicular_gamma_results.json"
    gamma_summary = {}
    if gamma_path.exists():
        try:
            with open(gamma_path) as f:
                data = json.load(f)
            gamma = data.get("gamma", {})
            if "error" not in gamma:
                signs = [g.get("sign") for g in gamma.values() if isinstance(g, dict)]
                dominant = max(set(signs), key=signs.count) if signs else "?"
                gamma_summary = {
                    "classification": "CLUSTER" if dominant == "+" else "TRAP",
                    "unanimous": len(set(signs)) == 1,
                    "signs": signs,
                    "n_nodes": data.get("trace_summary", {}).get("n_nodes"),
                    "n_contacts": data.get("trace_summary", {}).get("n_contacts"),
                }
                mark = "CLUSTER" if dominant == "+" else "TRAP"
                print(f"  Classification: {mark} (γ {'>' if dominant == '+' else '<'} 0)")
                print(f"  Unanimous: {'YES' if gamma_summary['unanimous'] else 'NO'}")
            else:
                gamma_summary = {"error": gamma.get("error")}
        except Exception as e:
            gamma_summary = {"error": str(e)}

    return result, gamma_summary


def main():
    parser = argparse.ArgumentParser(description="EPYC Phase 3: Strengthening Experiments")
    parser.add_argument("--skip-a", action="store_true", help="Skip routing independence")
    parser.add_argument("--skip-b", action="store_true", help="Skip vehicular γ")
    args = parser.parse_args()

    print("=" * 70)
    print("TIN EPYC — PHASE 3: STRENGTHENING EXPERIMENTS")
    print(f"Machine: {os.cpu_count()} cores")
    print(f"Start:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t_total = time.time()
    results = {}

    if not args.skip_a:
        res_a, sign_summary = run_routing_independence()
        results["routing_independence"] = {
            "passed": res_a[1],
            "elapsed_s": round(res_a[2], 1),
            "sign_agreement": sign_summary,
        }
    else:
        print("\n  [SKIP] Workload A: routing independence")

    if not args.skip_b:
        res_b, gamma_summary = run_vehicular()
        results["vehicular_gamma"] = {
            "passed": res_b[1] if res_b else False,
            "elapsed_s": round(res_b[2], 1) if res_b else 0,
            "gamma": gamma_summary,
        }
    else:
        print("\n  [SKIP] Workload B: vehicular γ")

    total_elapsed = time.time() - t_total

    # Summary
    print(f"\n{'=' * 70}")
    print("PHASE 3 SUMMARY")
    print(f"{'=' * 70}")

    if "routing_independence" in results:
        ri = results["routing_independence"]
        sa = ri.get("sign_agreement", {})
        all_agree = all(
            v.get("overall", False) for v in sa.values() if isinstance(v, dict) and "overall" in v
        )
        print(
            f"  A) Routing independence: {'ALL SIGNS AGREE' if all_agree else 'SIGN DISAGREEMENT'}"
        )

    if "vehicular_gamma" in results:
        vg = results["vehicular_gamma"]
        gamma = vg.get("gamma", {})
        if gamma.get("skipped"):
            print("  B) Vehicular γ: SKIPPED (no data)")
        elif "error" in gamma:
            print(f"  B) Vehicular γ: ERROR ({gamma['error']})")
        else:
            print(
                f"  B) Vehicular γ: {gamma.get('classification', '?')} "
                f"(unanimous={gamma.get('unanimous', '?')})"
            )

    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)")

    # Save summary
    summary_path = _RESULTS / "phase3_summary.json"
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["machine"] = f"{os.cpu_count()} cores"
    results["elapsed_s"] = round(total_elapsed, 1)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary: {summary_path}")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
