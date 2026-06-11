"""run_timing.py — Pre-flight estimation and progress tracking for runners.

Drop-in replacement for the ProcessPoolExecutor pattern used across runs/.

Usage (minimal change to any runner):

    from run_timing import timed_parallel_map

    # Instead of:
    #   with ProcessPoolExecutor(...) as executor:
    #       results = list(executor.map(_worker, tasks))
    #
    # Use:
    results = timed_parallel_map(_worker, tasks)

For more control:

    from run_timing import probe, estimate_wall, parallel_map_eta

    probe_times = probe(_worker, tasks, n_probe=3)
    est = estimate_wall(probe_times, len(tasks))
    print(f"Estimated: {format_time(est)}")

    results = parallel_map_eta(_worker, tasks)
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_time(seconds):
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Pre-flight probe
# ---------------------------------------------------------------------------


def probe(worker_fn, tasks, n_probe=3):
    """Run a diverse sample of tasks sequentially, return list of wall times.

    Samples evenly spaced indices so heterogeneous task lists (e.g. mixed
    Mercury + Mars) get representative coverage.
    """
    n = len(tasks)
    n_probe = min(n_probe, n)
    if n_probe <= 0:
        return []

    # Evenly spaced indices for diversity across task groups
    indices = [int(i * n / n_probe) for i in range(n_probe)]

    times = []
    for idx in indices:
        t0 = time.monotonic()
        worker_fn(tasks[idx])
        times.append(time.monotonic() - t0)
    return times


def estimate_wall(probe_times, n_tasks, n_workers=None):
    """Estimate total wall time from probe timings.

    Uses mean (not median) because with a work-queue executor, total wall
    time ≈ total_work / n_workers, and mean × n_tasks approximates total
    work better than median when task costs vary.
    """
    if not probe_times:
        return 0.0
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    mean_time = sum(probe_times) / len(probe_times)
    return mean_time * n_tasks / n_workers


# ---------------------------------------------------------------------------
# Progress-reporting parallel map
# ---------------------------------------------------------------------------


def parallel_map_eta(worker_fn, tasks, n_workers=None, label="", report_every_pct=5):
    """ProcessPoolExecutor.map replacement with progress + ETA.

    Returns results in original task order.
    """
    n_total = len(tasks)
    if n_total == 0:
        return []

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, n_total)

    results = [None] * n_total
    report_interval = max(1, n_total * report_every_pct // 100)
    prefix = f"  {label} " if label else "  "
    width = len(str(n_total))
    t_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(worker_fn, task): i for i, task in enumerate(tasks)}

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1

            if completed % report_interval == 0 or completed == n_total:
                elapsed = time.monotonic() - t_start
                rate = elapsed / completed
                remaining = rate * (n_total - completed)
                pct = 100 * completed / n_total
                print(
                    f"{prefix}[{completed:>{width}}/{n_total}]"
                    f"  {pct:5.1f}%  |  elapsed {format_time(elapsed)}"
                    f"  |  ETA {format_time(remaining)}"
                    f"  |  {rate:.2f}s/task",
                    flush=True,
                )

    return results


# ---------------------------------------------------------------------------
# Full workflow: probe → estimate → confirm → run
# ---------------------------------------------------------------------------


def timed_parallel_map(worker_fn, tasks, n_probe=3, n_workers=None, label="", auto_proceed=False):
    """Probe, estimate, optionally confirm, then run with progress.

    Parameters
    ----------
    worker_fn : callable
        Module-level worker (must be picklable).
    tasks : list
        Task argument tuples.
    n_probe : int
        Number of calibration tasks to run before estimating.
    n_workers : int or None
        Defaults to os.cpu_count().
    label : str
        Optional label for progress output.
    auto_proceed : bool
        If True, skip the Y/n confirmation prompt.

    Returns
    -------
    list : Results in original task order, or None if user aborts.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, len(tasks))

    n_total = len(tasks)
    n_p = min(n_probe, n_total)

    print(f"\n  {n_total} tasks, {n_workers} workers")

    # --- Probe phase ---
    print(f"  Pre-flight probe: {n_p} tasks ...", end="", flush=True)
    probe_times = probe(worker_fn, tasks, n_p)

    p_total = sum(probe_times)
    p_min = min(probe_times)
    p_max = max(probe_times)
    p_mean = p_total / len(probe_times)
    spread = p_max / p_min if p_min > 0 else float("inf")

    print(f" done ({format_time(p_total)})")
    print(
        f"  Per-task: min={format_time(p_min)}"
        f"  mean={format_time(p_mean)}"
        f"  max={format_time(p_max)}"
        f"  (spread {spread:.1f}x)"
    )

    est = estimate_wall(probe_times, n_total, n_workers)
    print(f"  Estimated total: ~{format_time(est)}")

    if spread > 10:
        print(
            f"  ⚠  High spread ({spread:.0f}x) — tasks are very heterogeneous."
            f"  Estimate may be rough."
        )

    if not auto_proceed:
        resp = input("  Proceed? [Y/n] ").strip().lower()
        if resp and resp != "y":
            print("  Aborted.")
            return None

    # --- Run phase ---
    return parallel_map_eta(worker_fn, tasks, n_workers=n_workers, label=label)
