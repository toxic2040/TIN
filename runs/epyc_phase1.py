"""epyc_phase1.py — Reproduction + Tests + Bootstrap CIs (target: ~4 hours)

Runs on a single 96-core machine. All 5 original server workloads execute
in parallel via subprocess, then new strengthening work (bootstrap CIs,
Moon factorization, residual diagnostics).

Usage:
    python runs/epyc_phase1.py           # full phase 1
    python runs/epyc_phase1.py --check   # just check results
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_RESULTS = _ROOT / "runs" / "epyc_results"
_RESULTS.mkdir(exist_ok=True)

_TARGETS = ["mercury", "venus", "mars", "ceres", "europa", "jupiter", "saturn", "titan"]

# ---------------------------------------------------------------------------
# Job definitions — flattened from 5 servers into parallel groups
# ---------------------------------------------------------------------------

# Group A: Independent jobs that can ALL run in parallel
PARALLEL_JOBS = [
    # Tests
    ("pytest", [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=short"]),
    # Cloud sweep — single job, internally parallel (~17,760 tasks)
    ("cloud_sweep", [sys.executable, "runs/run_cloud_sweep.py"]),
    # CRAWDAD sweeps (4 parallel)
    (
        "crawdad_exp1",
        [
            sys.executable,
            "runs/run_crawdad_validation.py",
            "--trace",
            "data/traces/Exp1/contacts.Exp1.dat",
            "--format",
            "haggle",
            "--max-node-id",
            "9",
        ],
    ),
    (
        "crawdad_exp2",
        [
            sys.executable,
            "runs/run_crawdad_validation.py",
            "--trace",
            "data/traces/Exp2/contacts.Exp2.dat",
            "--format",
            "haggle",
            "--max-node-id",
            "12",
        ],
    ),
    (
        "crawdad_exp3",
        [
            sys.executable,
            "runs/run_crawdad_validation.py",
            "--trace",
            "data/traces/Exp3/contacts.Exp3.dat",
            "--format",
            "haggle",
            "--max-node-id",
            "41",
        ],
    ),
    (
        "crawdad_exp6",
        [
            sys.executable,
            "runs/run_crawdad_validation.py",
            "--trace",
            "data/traces/Exp6/contacts.Exp6.dat",
            "--format",
            "haggle",
            "--max-node-id",
            "98",
        ],
    ),
    # Classification / achievability
    ("achievability", [sys.executable, "runs/run_achievability.py"]),
    ("chi_completion", [sys.executable, "runs/run_chi_completion.py"]),
    ("susceptibility", [sys.executable, "runs/run_susceptibility.py"]),
    ("gauge_invariance", [sys.executable, "runs/run_gauge_invariance.py"]),
    ("domain_transfer", [sys.executable, "runs/run_domain_transfer.py"]),
    # Helio / orbital
    ("finite_size_scaling", [sys.executable, "runs/run_finite_size_scaling.py"]),
    ("multi_body_sweep", [sys.executable, "runs/run_multi_body_sweep.py"]),
    # load_sweep_v2_results.json is retained as an archived artifact in public main.
    ("synodic_sweep", [sys.executable, "runs/run_synodic_sweep.py"]),
    ("helio_multi_arch", [sys.executable, "runs/run_helio_multi_arch.py"]),
    ("helio_phase_diagram", [sys.executable, "runs/run_helio_phase_diagram.py"]),
    ("cislunar", [sys.executable, "runs/run_cislunar.py"]),
    ("inclination_sweep", [sys.executable, "runs/run_inclination_sweep.py"]),
    ("distance_sweep", [sys.executable, "runs/run_distance_sweep.py"]),
    ("ttl_surface_moon", [sys.executable, "runs/run_ttl_surface_moon.py"]),
    ("link_quality_sweep", [sys.executable, "runs/run_link_quality_sweep.py"]),
    # Note: phi_sweep, phi_decompose, phi_surface, topology_sweep moved to
    # CLOUD_DEPENDENT_JOBS — they all read cloud_sweep_shard_*.json
    # Validation
    ("validation", [sys.executable, "runs/run_validation.py"]),
    # Additional runs
    ("pair_gamma_mosaic", [sys.executable, "runs/run_pair_gamma_mosaic.py"]),
    ("cluster_subset_sweep", [sys.executable, "runs/run_cluster_subset_sweep.py"]),
    ("tail_ratio_survey", [sys.executable, "runs/run_tail_ratio_survey.py"]),
    ("hub_crossover", [sys.executable, "runs/run_hub_crossover.py"]),
    ("cambridge_forensic", [sys.executable, "runs/run_cambridge_forensic.py"]),
    ("cambridge_fixed_relay", [sys.executable, "runs/run_cambridge_fixed_relay.py"]),
]

# Group B: Depends on cloud_sweep output (needs shard splitting first)
CLOUD_DEPENDENT_JOBS = [
    # Phi sweep shards (8 parallel)
    *[(f"phi_{t}", [sys.executable, "runs/run_phi_sweep.py", "--target", t]) for t in _TARGETS],
    ("phi_decompose", [sys.executable, "runs/run_phi_decompose.py"]),
    ("phi_surface", [sys.executable, "runs/run_phi_surface.py"]),
    ("topology_sweep", [sys.executable, "runs/run_topology_sweep.py"]),
]

# Group C: Sequential merges (need shards done first)
MERGE_JOBS = [
    ("phi_merge", [sys.executable, "runs/run_phi_sweep_merge.py"]),
    ("crawdad_cross_trace", [sys.executable, "runs/run_crawdad_cross_trace.py"]),
]


def run_one(label_cmd):
    """Run a single job, return (label, ok, elapsed, return_code)."""
    label, cmd = label_cmd
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=14400,  # 4h timeout per job
        )
        elapsed = time.time() - t0
        # Save output
        log_path = _RESULTS / f"{label}.log"
        with open(log_path, "w") as f:
            f.write(result.stdout)
        return (label, result.returncode == 0, elapsed, result.returncode)
    except subprocess.TimeoutExpired:
        return (label, False, time.time() - t0, -1)
    except Exception as e:
        return (label, False, time.time() - t0, str(e))


def run_parallel_group(jobs, max_workers=None):
    """Run jobs in parallel, return list of (label, ok, elapsed, rc)."""
    if max_workers is None:
        # Use at most 48 workers — each job may itself use multiprocessing
        max_workers = min(48, len(jobs))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                status = "PASS" if result[1] else "FAIL"
                print(f"  [{status}] {result[0]:30s}  {result[2]:>8.1f}s", flush=True)
                results.append(result)
            except Exception as e:
                print(f"  [FAIL] {label:30s}  ERROR: {e}", flush=True)
                results.append((label, False, 0, str(e)))
    return results


def split_cloud_sweep_to_shards():
    """Split cloud_sweep_results.json into per-target shard files for phi_surface."""
    cloud_path = _ROOT / "runs" / "cloud_sweep_results.json"
    if not cloud_path.exists():
        print("  SKIP: cloud_sweep_results.json not found")
        return False

    with open(cloud_path) as f:
        cloud = json.load(f)

    core = cloud.get("core_results", [])
    if not core:
        print("  SKIP: no core_results in cloud_sweep_results.json")
        return False

    for target in _TARGETS:
        target_rows = [r for r in core if r.get("target") == target]
        if not target_rows:
            continue
        shard = {
            "config": cloud.get("config", {}),
            "core_results": target_rows,
        }
        shard_path = _ROOT / "runs" / f"cloud_sweep_shard_{target}.json"
        with open(shard_path, "w") as f:
            json.dump(shard, f, indent=2)
        print(f"  Shard: {shard_path.name} ({len(target_rows)} rows)")

    return True


def run_bootstrap_cis():
    """NEW: Bootstrap confidence intervals on gamma for all configs.

    This addresses S3.1-001 (the most substantive MEDIUM issue from the audit).
    Computes 1000 bootstrap resamples of the OLS gamma slope for each
    body/trace and p_eff combination.
    """
    print("\n" + "=" * 70)
    print("PHASE 1B: Bootstrap CIs on gamma (NEW — addresses S3.1-001)")
    print("=" * 70)

    # Load achievability results (has per-config eta/phi data)
    achiev_path = _ROOT / "runs" / "achievability_results.json"
    if not achiev_path.exists():
        print("  SKIP: achievability_results.json not found yet")
        return {}

    with open(achiev_path) as f:
        achiev = json.load(f)

    bootstrap_results = {}
    n_bootstrap = 1000

    for body_key, body_data in achiev.items():
        if not isinstance(body_data, dict) or "raw_rows" not in body_data:
            continue

        rows = body_data["raw_rows"]
        p_effs = sorted(set(r.get("p_eff") for r in rows if r.get("p_eff") is not None))

        for p_eff in p_effs:
            subset = [r for r in rows if r.get("p_eff") == p_eff]
            if len(subset) < 5:
                continue

            _valid = [
                r
                for r in subset
                if "E_H" in r
                and "phi_greedy" in r
                and r["phi_greedy"] is not None
                and r["phi_greedy"] > 0
            ]
            ehs = np.array([r["E_H"] for r in _valid])
            lps = np.array([np.log(r["phi_greedy"]) for r in _valid])
            if len(ehs) < 5:
                continue

            # Point estimate
            slope, intercept = np.polyfit(ehs, lps, 1)

            # Bootstrap
            rng = np.random.default_rng(42)
            boot_slopes = []
            for _ in range(n_bootstrap):
                idx = rng.choice(len(ehs), size=len(ehs), replace=True)
                try:
                    bs, _ = np.polyfit(ehs[idx], lps[idx], 1)
                    boot_slopes.append(bs)
                except (np.linalg.LinAlgError, ValueError):
                    continue

            if len(boot_slopes) < 100:
                continue

            boot_slopes = np.array(boot_slopes)
            ci_lo = np.percentile(boot_slopes, 2.5)
            ci_hi = np.percentile(boot_slopes, 97.5)
            se = np.std(boot_slopes)
            sign_consistent = np.all(boot_slopes > 0) or np.all(boot_slopes < 0)
            sign_fraction = np.mean(np.sign(boot_slopes) == np.sign(slope))

            key = f"{body_key}_p{p_eff}"
            bootstrap_results[key] = {
                "body": body_key,
                "p_eff": p_eff,
                "n_points": len(ehs),
                "raw_slope": round(float(slope), 6),
                "bootstrap_se": round(float(se), 6),
                "ci_95_lo": round(float(ci_lo), 6),
                "ci_95_hi": round(float(ci_hi), 6),
                "sign_consistent_100pct": bool(sign_consistent),
                "sign_fraction": round(float(sign_fraction), 4),
                "n_bootstrap": n_bootstrap,
            }

            sign_mark = "YES" if sign_consistent else f"NO ({sign_fraction:.1%})"
            print(
                f"  {key:30s}  slope={slope:+.4f}  "
                f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
                f"sign_consistent={sign_mark}"
            )

    # Save
    out_path = _RESULTS / "bootstrap_gamma_cis.json"
    with open(out_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total configs bootstrapped: {len(bootstrap_results)}")

    return bootstrap_results


def run_residual_diagnostics():
    """NEW: OLS residual diagnostics for gamma regression.

    Addresses S3.1-001 (no R², no residual diagnostics).
    """
    print("\n" + "=" * 70)
    print("PHASE 1C: Residual diagnostics on gamma OLS (NEW)")
    print("=" * 70)

    achiev_path = _ROOT / "runs" / "achievability_results.json"
    if not achiev_path.exists():
        print("  SKIP: achievability_results.json not found yet")
        return {}

    with open(achiev_path) as f:
        achiev = json.load(f)

    diag_results = {}

    for body_key, body_data in achiev.items():
        if not isinstance(body_data, dict) or "raw_rows" not in body_data:
            continue

        rows = body_data["raw_rows"]
        p_effs = sorted(set(r.get("p_eff") for r in rows if r.get("p_eff") is not None))

        for p_eff in p_effs:
            subset = [r for r in rows if r.get("p_eff") == p_eff]
            _valid = [
                r
                for r in subset
                if "E_H" in r
                and "phi_greedy" in r
                and r["phi_greedy"] is not None
                and r["phi_greedy"] > 0
            ]
            ehs = np.array([r["E_H"] for r in _valid])
            lps = np.array([np.log(r["phi_greedy"]) for r in _valid])
            if len(ehs) < 5:
                continue

            # OLS
            slope, intercept = np.polyfit(ehs, lps, 1)
            predicted = slope * ehs + intercept
            residuals = lps - predicted

            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((lps - np.mean(lps)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

            # Standard error of slope
            n = len(ehs)
            mse = ss_res / max(n - 2, 1)
            se_slope = np.sqrt(mse / max(np.sum((ehs - np.mean(ehs)) ** 2), 1e-15))

            # Heteroscedasticity: Breusch-Pagan-like (correlation of |residual| with x)
            abs_resid = np.abs(residuals)
            if np.std(abs_resid) > 1e-15 and np.std(ehs) > 1e-15:
                hetero_corr = float(np.corrcoef(ehs, abs_resid)[0, 1])
            else:
                hetero_corr = 0.0

            # Durbin-Watson-like (autocorrelation of residuals)
            if n > 2:
                dw = float(np.sum(np.diff(residuals) ** 2) / max(ss_res, 1e-15))
            else:
                dw = 2.0  # no autocorrelation

            key = f"{body_key}_p{p_eff}"
            diag_results[key] = {
                "body": body_key,
                "p_eff": p_eff,
                "n_points": n,
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 6),
                "r_squared": round(float(r_squared), 6),
                "se_slope": round(float(se_slope), 6),
                "t_statistic": round(float(slope / se_slope) if se_slope > 1e-15 else 0, 4),
                "hetero_corr": round(float(hetero_corr), 4),
                "durbin_watson": round(float(dw), 4),
                "residual_mean": round(float(np.mean(residuals)), 8),
                "residual_std": round(float(np.std(residuals)), 6),
            }

            print(
                f"  {key:30s}  R²={r_squared:.4f}  SE={se_slope:.4f}  "
                f"DW={dw:.3f}  hetero_r={hetero_corr:+.3f}"
            )

    out_path = _RESULTS / "gamma_residual_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(diag_results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return diag_results


def run_checks(all_results):
    """Run validation checks on completed results."""
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    checks = []
    n_pass = sum(1 for r in all_results if r[1])
    n_fail = sum(1 for r in all_results if not r[1])
    checks.append(
        {
            "check": "all_jobs_passed",
            "passed": n_fail == 0,
            "detail": f"{n_pass} passed, {n_fail} failed",
        }
    )

    failed = [r for r in all_results if not r[1]]
    if failed:
        print(f"\n  FAILED JOBS ({len(failed)}):")
        for label, _, elapsed, rc in failed:
            print(f"    {label}: rc={rc} ({elapsed:.0f}s)")

    # Factorization check
    cloud_path = _ROOT / "runs" / "cloud_sweep_results.json"
    if cloud_path.exists():
        with open(cloud_path) as f:
            cloud = json.load(f)
        if "core_results" in cloud:
            max_res = 0.0
            for r in cloud["core_results"]:
                s_t = r.get("S_T", 0)
                eta = r.get("eta", 0)
                dr = r.get("DR", 0)
                max_res = max(max_res, abs(dr - s_t * eta))
            checks.append(
                {
                    "check": "factorization_exact",
                    "passed": max_res < 1e-12,
                    "detail": f"max_residual={max_res:.2e}",
                }
            )
            print(f"  Factorization: max residual = {max_res:.2e}")

    # Print summary
    all_ok = all(c["passed"] for c in checks)
    print(
        f"\n  {'PASS' if all_ok else 'FAIL'}: {sum(c['passed'] for c in checks)}/{len(checks)} checks"
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": f"{os.cpu_count()} cores",
        "checks": checks,
        "job_results": [
            {"label": r[0], "passed": r[1], "elapsed_s": round(r[2], 1)} for r in all_results
        ],
    }
    with open(_RESULTS / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Just run checks on existing results")
    args = parser.parse_args()

    if args.check:
        # Just run checks
        run_checks([])
        return

    print("=" * 70)
    print("TIN EPYC REVALIDATION — PHASE 1")
    print(f"Machine: {os.cpu_count()} cores")
    print(f"Start:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t_total = time.time()

    # ── Stage 1: All parallel jobs (including cloud_sweep) ──
    print(f"\nSTAGE 1: {len(PARALLEL_JOBS)} parallel jobs")
    print("-" * 70)
    results_a = run_parallel_group(PARALLEL_JOBS, max_workers=48)

    # ── Stage 1.5: Split cloud_sweep output into per-target shards ──
    print("\nSTAGE 1.5: Splitting cloud_sweep into per-target shards")
    print("-" * 70)
    split_cloud_sweep_to_shards()

    # ── Stage 2: Cloud-dependent jobs (phi_surface needs shards) ──
    print(f"\nSTAGE 2: {len(CLOUD_DEPENDENT_JOBS)} cloud-dependent jobs")
    print("-" * 70)
    results_dep = run_parallel_group(CLOUD_DEPENDENT_JOBS, max_workers=4)

    # ── Stage 3: Merge jobs ──
    print(f"\nSTAGE 3: {len(MERGE_JOBS)} merge jobs (sequential)")
    print("-" * 70)
    results_b = []
    for job in MERGE_JOBS:
        result = run_one(job)
        status = "PASS" if result[1] else "FAIL"
        print(f"  [{status}] {result[0]:30s}  {result[2]:>8.1f}s")
        results_b.append(result)

    all_results = results_a + results_dep + results_b

    # ── Stage 3: Bootstrap CIs (new work) ──
    bootstrap = run_bootstrap_cis()

    # ── Stage 4: Residual diagnostics (new work) ──
    diagnostics = run_residual_diagnostics()

    # ── Validation ──
    run_checks(all_results)

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print("PHASE 1 COMPLETE")
    print(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)")
    print(f"Results: {_RESULTS}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
