"""_chunked_base.py — Chunked parallel oracle for maximum CPU utilization.

Problem
-------
Venus 24-orb takes 60+ min on 1 core: 2000 Dijkstras on 100k+ contacts,
run TWICE (once for feasibility, once for path tracing in predict_s).

Solution
--------
1. Merge feasibility + path tracing into ONE Dijkstra per injection time.
   (Halves total Dijkstra count: 4000 -> 2000 per config.)
2. Split injection times into chunks of ~50, parallelize across all cores.
   (Venus 24-orb: 2000 Dijkstras / 128 cores = ~16 per core.)
3. Contacts shared via fork COW where available; initializer hydration
   on spawn platforms (Windows, macOS default).  Zero-copy on Linux,
   one-copy-per-worker on spawn — works everywhere.

Speedup: Venus 24-orb from 60+ min (1 core) to ~1 min (128 cores).

Usage
-----
    from runs._chunked_base import batch_compute_metrics

    jobs = {
        ("venus", 24, 0, "fwd"): {
            "contacts": contacts,
            "src": "surface", "dst": "earth",
        },
        ...
    }
    results = batch_compute_metrics(jobs, chunk_size=50)
    # results[key] = {"S_T": ..., "E_H": ..., "phi": ..., ...} or None
"""

from __future__ import annotations

import heapq
import multiprocessing
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_INF = float("inf")

# ---------------------------------------------------------------------------
# Module-level shared state
#
# On fork platforms (Linux): populated before fork, inherited COW by
# child processes — zero serialization, zero copy until write.
#
# On spawn platforms (Windows, macOS default): empty at import time.
# The pool initializer (_init_worker) hydrates them once per worker
# from serialized data.  One copy per worker, but only paid once.
# ---------------------------------------------------------------------------

_SHARED_ADJ: dict = {}
_SHARED_SRC_DST: dict = {}


def _can_fork() -> bool:
    """True if the platform supports fork-based multiprocessing."""
    if sys.platform == "win32":
        return False
    try:
        return multiprocessing.get_start_method(allow_none=True) in (None, "fork")
    except RuntimeError:
        return False


def _init_worker(adj_data: dict, src_dst_data: dict) -> None:
    """Initializer for spawn-mode workers.  Hydrates module globals."""
    global _SHARED_ADJ, _SHARED_SRC_DST
    _SHARED_ADJ = adj_data
    _SHARED_SRC_DST = src_dst_data


# ---------------------------------------------------------------------------
# Adjacency builder
# ---------------------------------------------------------------------------


def _build_adj(contacts: list[dict]) -> dict[str, list[tuple]]:
    """Build adjacency list: node -> [(start, end, latency, to_node, p_success)].

    Accepts dicts or Contact objects (duck-typed via hasattr).
    """
    adj: dict[str, list[tuple]] = {}
    for c in contacts:
        if hasattr(c, "from_node"):
            fn, tn = c.from_node, c.to_node
            ss, dur, ls = c.start_s, c.duration_s, c.latency_s
            ps = getattr(c, "p_success", 0.98)
        else:
            fn = c["from_node"]
            tn = c["to_node"]
            ss = c["start_s"]
            dur = c["duration_s"]
            ls = c["latency_s"]
            ps = c.get("p_success", 0.98)
        adj.setdefault(fn, []).append((ss, ss + dur, ls, tn, ps))
    return adj


def _contact_q_bar(contacts: list[dict]) -> float:
    """Duration-weighted mean link success probability (plan-wide)."""
    total_w = 0.0
    total_wp = 0.0
    for c in contacts:
        if hasattr(c, "duration_s"):
            dur = c.duration_s
            ps = getattr(c, "p_success", 0.98)
        else:
            dur = c["duration_s"]
            ps = c.get("p_success", 0.98)
        total_w += dur
        total_wp += dur * ps
    return total_wp / total_w if total_w > 0 else 0.0


# ---------------------------------------------------------------------------
# Merged Dijkstra: feasibility + path tracing in ONE pass
# ---------------------------------------------------------------------------


def _oracle_with_path(
    adj: dict[str, list[tuple]],
    source: str,
    dest: str,
    t_inj: float,
) -> tuple[int, list[float]] | None:
    """Single-injection earliest-arrival Dijkstra returning path info.

    Returns
    -------
    None if dest is unreachable.
    (hop_count, [p_success_per_hop]) if reachable.
    """
    if source == dest:
        return (0, [])

    best: dict[str, float] = {source: t_inj}
    parent: dict[str, tuple[str, float]] = {}
    heap = [(t_inj, source)]

    while heap:
        t_cur, node = heapq.heappop(heap)
        if node == dest:
            # Reconstruct path
            p_values: list[float] = []
            n = dest
            while n in parent:
                prev, ps = parent[n]
                p_values.append(ps)
                n = prev
            p_values.reverse()
            return (len(p_values), p_values)

        if t_cur > best.get(node, _INF):
            continue

        for start_s, end_s, latency_s, to_node, p_success in adj.get(node, []):
            if t_cur > end_s:
                continue
            t_depart = max(t_cur, start_s)
            t_arrive = t_depart + latency_s
            if t_arrive < best.get(to_node, _INF):
                best[to_node] = t_arrive
                parent[to_node] = (node, p_success)
                heapq.heappush(heap, (t_arrive, to_node))

    return None


# ---------------------------------------------------------------------------
# Chunk worker (module-scope for pickle)
# ---------------------------------------------------------------------------


def _chunk_worker(args: tuple) -> tuple:
    """Process a chunk of injection times: merged feasibility + path tracing.

    Returns (job_key, results_list, error_or_none) where each result is:
        (t_inj, feasible, path_or_none)
    """
    job_key, chunk_times = args
    try:
        adj = _SHARED_ADJ[job_key]
        src, dst = _SHARED_SRC_DST[job_key]

        results = []
        for t in chunk_times:
            path = _oracle_with_path(adj, src, dst, float(t))
            feasible = path is not None
            results.append((float(t), feasible, path))
        return job_key, results, None
    except Exception as exc:
        return job_key, [], str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_injection_times(
    contacts: list[dict],
    dt_inject: float = 300.0,
    max_injections: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Generate injection time array from a contact plan."""
    t_min = min(c["start_s"] for c in contacts)
    t_max = max(c["start_s"] + c["duration_s"] for c in contacts)
    inj = np.arange(t_min, t_max, dt_inject)
    if len(inj) > max_injections:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(inj), max_injections, replace=False))
        inj = inj[idx]
    return inj


def compute_metrics_from_paths(
    contacts: list[dict],
    paths_with_feas: list[tuple],
) -> dict | None:
    """Compute metrics from pre-collected oracle paths.

    Replicates predict_s output without re-running Dijkstra.

    Parameters
    ----------
    contacts : list[dict]
        The contact plan (for q_bar computation).
    paths_with_feas : list of (t_inj, feasible, path_or_none)
        Output from the chunk workers.

    Returns
    -------
    dict with keys: S_T, E_H, var_H, eta, phi, lyapunov, n_paths,
                    q_bar, q_bar_oracle, var_log_p
    or None if no injection times were provided.
    """
    n_total = len(paths_with_feas)
    if n_total == 0:
        return None

    q_bar = _contact_q_bar(contacts)
    n_feasible = sum(1 for _, f, _ in paths_with_feas if f)
    s_t = n_feasible / n_total

    # Collect valid paths
    paths = [p for _, f, p in paths_with_feas if f and p is not None]

    if not paths:
        return {
            "S_T": s_t,
            "E_H": float("nan"),
            "eta": float("nan"),
            "phi": float("nan"),
            "lyapunov": float("nan"),
            "n_paths": 0,
            "var_H": float("nan"),
            "q_bar": q_bar,
            "q_bar_oracle": float("nan"),
            "var_log_p": float("nan"),
        }

    hop_counts = [h for h, _ in paths]
    E_H = float(np.mean(hop_counts))
    var_H = float(np.var(hop_counts))

    # Per-hop p_success across all paths
    all_p: list[float] = []
    for _, p_values in paths:
        all_p.extend(p_values)

    if not all_p or E_H == 0:
        return {
            "S_T": s_t,
            "E_H": E_H,
            "eta": 1.0,
            "phi": float("nan"),
            "lyapunov": 0.0,
            "n_paths": len(paths),
            "var_H": var_H,
            "q_bar": q_bar,
            "q_bar_oracle": 1.0,
            "var_log_p": 0.0,
        }

    log_p = np.log(np.clip(all_p, 1e-300, 1.0))
    q_bar_oracle = float(np.exp(np.mean(log_p)))

    # Lyapunov exponent: lambda = E[log p_h]
    lyap = float(np.mean(log_p))
    eta_lyap = float(np.exp(E_H * lyap))
    var_log_p = float(np.var(log_p))

    # OPSP: eta = (1/N) * sum(prod(p_h) per path)
    path_probs: list[float] = []
    for _, p_values in paths:
        if p_values:
            path_probs.append(float(np.prod(p_values)))
        else:
            path_probs.append(1.0)
    eta_opsp = float(np.mean(path_probs))

    # Phi = eta_opsp / eta_lyapunov
    phi = eta_opsp / eta_lyap if eta_lyap > 1e-12 else float("nan")

    return {
        "S_T": s_t,
        "E_H": E_H,
        "var_H": var_H,
        "eta": eta_opsp,
        "phi": phi,
        "lyapunov": lyap,
        "n_paths": len(paths),
        "q_bar": q_bar,
        "q_bar_oracle": q_bar_oracle,
        "var_log_p": var_log_p,
    }


def batch_compute_metrics(
    jobs: dict,
    dt_inject: float = 300.0,
    max_injections: int = 2000,
    chunk_size: int = 50,
    n_workers: int | None = None,
    on_progress=None,
) -> dict:
    """Compute metrics for multiple configs using chunked oracle parallelism.

    Parameters
    ----------
    jobs : dict
        Keys: arbitrary hashable identifiers (e.g. tuples).
        Values: dicts with "contacts" (list[dict]), "src" (str), "dst" (str).
    dt_inject : float
        Injection time spacing in seconds.
    max_injections : int
        Cap on injection times per job.
    chunk_size : int
        Injection times per chunk task.
    n_workers : int or None
        Worker count.  None = os.cpu_count().
    on_progress : callable or None
        Called with (chunks_done, total_chunks, last_job_key).

    Returns
    -------
    dict : same keys as jobs.  Values are metric dicts or None.
    """
    global _SHARED_ADJ, _SHARED_SRC_DST

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    # Contacts kept separately — only needed for metrics, not by workers
    contacts_by_key: dict = {}

    try:
        # ------------------------------------------------------------------
        # Phase 1: populate shared state, build adj lists, generate inj times
        # ------------------------------------------------------------------
        job_inj_times: dict = {}
        skipped: dict = {}

        for key, job in jobs.items():
            contacts = job["contacts"]
            if not contacts:
                skipped[key] = None
                continue

            contacts_by_key[key] = contacts
            _SHARED_ADJ[key] = _build_adj(contacts)
            _SHARED_SRC_DST[key] = (job["src"], job["dst"])

            inj = make_injection_times(contacts, dt_inject, max_injections)
            if len(inj) < 10:
                skipped[key] = None
                continue
            job_inj_times[key] = inj

        # ------------------------------------------------------------------
        # Phase 2: build chunk tasks
        # ------------------------------------------------------------------
        tasks = []
        for key, inj in job_inj_times.items():
            for i in range(0, len(inj), chunk_size):
                chunk = inj[i : i + chunk_size]
                tasks.append((key, chunk))

        total_chunks = len(tasks)

        # ------------------------------------------------------------------
        # Phase 3: execute chunks in parallel
        # ------------------------------------------------------------------
        raw: dict = defaultdict(list)
        errors: dict[object, str] = {}

        if total_chunks > 0:
            use_fork = _can_fork()
            n_w = min(n_workers, total_chunks)

            executor_kwargs: dict = {"max_workers": n_w}
            if not use_fork:
                # Spawn mode: hydrate worker globals via initializer
                executor_kwargs["initializer"] = _init_worker
                executor_kwargs["initargs"] = (
                    dict(_SHARED_ADJ),
                    dict(_SHARED_SRC_DST),
                )

            with ProcessPoolExecutor(**executor_kwargs) as executor:
                futures = {executor.submit(_chunk_worker, t): t for t in tasks}
                done = 0
                for future in as_completed(futures):
                    try:
                        job_key, chunk_results, error = future.result()
                    except Exception as exc:
                        # Identify which task failed from the futures map
                        task_args = futures[future]
                        job_key = task_args[0]
                        errors[job_key] = str(exc)
                        done += 1
                        continue

                    if error is not None:
                        errors.setdefault(job_key, error)
                    else:
                        raw[job_key].extend(chunk_results)

                    done += 1
                    if on_progress:
                        on_progress(done, total_chunks, job_key)

        # Warn about any failed jobs (don't silently swallow)
        for key, msg in errors.items():
            if key not in raw:
                warnings.warn(
                    f"_chunked_base: job {key!r} failed: {msg}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # ------------------------------------------------------------------
        # Phase 4: sort by injection time and compute metrics
        # ------------------------------------------------------------------
        results = dict(skipped)

        for key in job_inj_times:
            if key not in raw:
                results[key] = None
                continue

            # Sort by injection time (as_completed returns chunks out of order)
            entries = sorted(raw[key], key=lambda x: x[0])
            contacts = contacts_by_key[key]
            results[key] = compute_metrics_from_paths(contacts, entries)

        return results

    finally:
        _SHARED_ADJ.clear()
        _SHARED_SRC_DST.clear()
