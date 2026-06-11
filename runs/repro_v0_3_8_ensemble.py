"""repro_v0_3_8_ensemble.py — Multi-seed ensemble of the v0.3.8 baseline.

Runs the deterministic v0.3.8 simulation across multiple seeds in parallel
to characterize the distribution of stochastic metrics — primarily
emergency_worst_min, which on the canonical seed=42 run sits at exactly
3.4 minutes (the threshold of the v0.3.1 archive's "under 3.4 minutes
worst-case" banner). A single-seed result samples one point of the
distribution; this entrypoint produces the honest spread that should
backstop any worst-case claim.

Operational safeguards:
- Each seed's result is appended to the JSONL file immediately on
  completion (no in-memory accumulation a crash could lose).
- On startup, the existing JSONL is read and already-completed seeds
  are skipped, allowing resume after Ctrl-C or OOM.
- Each per-seed run is wrapped in try/except so one failure cannot
  kill the batch.
- Outer parallelism uses min(n_seeds, cpu_count - 1) workers; each
  worker's simulation is single-threaded — no nested pools.

Output files (in runs/results/):
- repro_v0_3_8_ensemble.jsonl — one record per completed seed
- repro_v0_3_8_ensemble_summary.json — aggregate distribution stats

Usage:
    python runs/repro_v0_3_8_ensemble.py --seeds 42,0,1,2,3,4,5,6,7,8
    python runs/repro_v0_3_8_ensemble.py --n_seeds 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS_DIR))
from repro_v0_3_8 import run_simulation  # noqa: E402

REPO_ROOT = RUNS_DIR.parent
OUT_DIR = REPO_ROOT / "runs" / "results"
JSONL_PATH = OUT_DIR / "repro_v0_3_8_ensemble.jsonl"
SUMMARY_PATH = OUT_DIR / "repro_v0_3_8_ensemble_summary.json"


def _make_args(seed: int, sim_days: int, bundles: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        sim_days=sim_days,
        bundles=bundles,
        polar_sats=8,
        altitude=400.0,
        inclination=90.0,
        min_elev=5.0,
        include_elfo=True,
        topo_shadow=True,
        solar_storm=True,
        storm_prob=0.08,
        output="unused",
        no_coverage_png=True,
    )


def _run_one_seed(seed: int, sim_days: int, bundles: int) -> dict:
    args = _make_args(seed, sim_days, bundles)
    t0 = time.monotonic()
    results, *_ = run_simulation(args, compute_coverage=False)
    results["wall_seconds"] = round(time.monotonic() - t0, 1)
    return results


def _read_completed_seeds(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    seeds: set[int] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "error" not in d:
                    seeds.add(int(d["seed"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return seeds


def _append_jsonl(jsonl_path: Path, record: dict) -> None:
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()


def _summarize(records: list[dict]) -> dict:
    successful = [r for r in records if "error" not in r]
    failed = [r for r in records if "error" in r]
    metrics_to_summarize = [
        "delivery_pct",
        "overall_avg_h",
        "emergency_avg_min",
        "emergency_worst_min",
        "polar_touches",
        "avg_custody_hops",
        "delivered_count",
        "handoffs",
    ]
    metrics: dict[str, dict[str, float | int]] = {}
    for k in metrics_to_summarize:
        vals = [r[k] for r in successful if k in r and r[k] is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        metrics[k] = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 3),
            "min": round(float(arr.min()), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "max": round(float(arr.max()), 3),
        }
    return {
        "n_seeds": len(records),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "seeds_run": sorted(int(r["seed"]) for r in successful),
        "failed_seeds": sorted(int(r.get("seed", -1)) for r in failed),
        "metrics": metrics,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0.3.8 baseline multi-seed ensemble")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--seeds", type=str, help="Comma-separated list of seeds")
    g.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Use seeds 0..n_seeds-1 (default 10)",
    )
    p.add_argument("--sim_days", type=int, default=28)
    p.add_argument("--bundles", type=int, default=300)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: min(n_seeds, cpu_count - 1))",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    seeds = (
        [int(s) for s in args.seeds.split(",")]
        if args.seeds
        else list(range(args.n_seeds))
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = _read_completed_seeds(JSONL_PATH)
    if completed:
        print(f"Resuming: {len(completed)} seeds already in {JSONL_PATH.name}")
    pending = [s for s in seeds if s not in completed]

    if pending:
        n_workers = args.workers or min(
            len(pending), max(1, (os.cpu_count() or 2) - 1)
        )
        print(
            f"Running {len(pending)} seeds across {n_workers} workers "
            f"(sim_days={args.sim_days}, bundles={args.bundles})"
        )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_run_one_seed, s, args.sim_days, args.bundles): s
                for s in pending
            }
            for fut in as_completed(futs):
                seed = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {
                        "seed": seed,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"  seed={seed}: FAILED ({e})")
                else:
                    print(
                        f"  seed={seed}: emergency_worst_min={rec['emergency_worst_min']} "
                        f"({rec.get('wall_seconds', '?')} s)"
                    )
                _append_jsonl(JSONL_PATH, rec)
    else:
        print("All seeds already complete. Re-aggregating summary.")

    records: list[dict] = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    summary = _summarize(records)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print()
    print(
        f"Summary: {summary['n_successful']} succeeded, {summary['n_failed']} failed"
    )
    if "emergency_worst_min" in summary["metrics"]:
        m = summary["metrics"]["emergency_worst_min"]
        print(
            f"emergency_worst_min: mean={m['mean']} std={m['std']} "
            f"min={m['min']} p50={m['p50']} p95={m['p95']} max={m['max']}"
        )
    print(f"Wrote {JSONL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
