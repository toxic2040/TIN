# SPDX-License-Identifier: MIT
"""tin.core.oracle — Deterministic temporal reachability oracle.

Earliest-arrival Dijkstra on the contact graph.  Used to compute
F_det(t) (per-bundle feasibility) and S_T^tau (TTL-aware reachability).

Commodity-aware cost-priority Dijkstra for the screening pipeline.
Edge cost = -log(p_e) + lambda_c * exposure_time.  When lambda_c=0,
equivalent to max-reliability oracle.

Parallel sweep functions use ProcessPoolExecutor with an initializer
that builds adjacency ONCE per worker.  Tasks carry only the injection
time — no per-task pickling of the contact plan.

Accepts any iterable of objects with attributes:
    from_node, to_node, start_s, latency_s
(i.e. tin.core.routing.Contact instances or raw contact dicts).
"""

import heapq
import math
import os
from concurrent.futures import ProcessPoolExecutor

_INF = float("inf")


# ═══════════════════════════════════════════════════════════════
# ADJACENCY BUILDERS (public)
# ═══════════════════════════════════════════════════════════════


def build_adjacency(contacts, max_contacts=2_000_000):
    """Build earliest-arrival adjacency from contacts.

    Returns dict mapping from_node to list of
    (start_s, end_s, latency_s, to_node) tuples.

    Accepts Contact objects or plain dicts.

    Parameters
    ----------
    max_contacts : int
        Safety ceiling.  Raises ValueError if the iterable yields more
        contacts than this.  Default 2M is generous for cislunar;
        callers processing interplanetary configs should sub-batch
        by epoch rather than raising this limit.
    """
    adj: dict = {}
    n = 0
    for c in contacts:
        n += 1
        if n > max_contacts:
            raise ValueError(
                f"Contact plan exceeds {max_contacts:,} contacts at entry "
                f"{n:,}. Sub-batch by epoch or increase max_contacts."
            )
        if hasattr(c, "from_node"):
            fn, tn, ss, dur, ls = (
                c.from_node,
                c.to_node,
                c.start_s,
                c.duration_s,
                c.latency_s,
            )
        else:
            fn, tn, ss, dur, ls = (
                c["from_node"],
                c["to_node"],
                c["start_s"],
                c["duration_s"],
                c["latency_s"],
            )
        adj.setdefault(fn, []).append((ss, ss + dur, ls, tn))
    return adj


def build_cost_adjacency(contacts, max_contacts=2_000_000):
    """Build cost-priority adjacency from contacts.

    Returns dict mapping from_node to list of
    (start_s, end_s, latency_s, to_node, p_success, link_type) tuples.

    Accepts Contact objects or plain dicts.

    Parameters
    ----------
    max_contacts : int
        Safety ceiling — same semantics as ``build_adjacency``.
    """
    adj: dict = {}
    n = 0
    for c in contacts:
        n += 1
        if n > max_contacts:
            raise ValueError(
                f"Contact plan exceeds {max_contacts:,} contacts at entry "
                f"{n:,}. Sub-batch by epoch or increase max_contacts."
            )
        if hasattr(c, "from_node"):
            fn, tn = c.from_node, c.to_node
            ss, dur, ls = c.start_s, c.duration_s, c.latency_s
            ps = getattr(c, "p_success", 0.98)
            lt = getattr(c, "link_type", None)
        else:
            fn, tn = c["from_node"], c["to_node"]
            ss, dur, ls = c["start_s"], c["duration_s"], c["latency_s"]
            ps = c.get("p_success", 0.98)
            lt = c.get("link_type", None)
        adj.setdefault(fn, []).append((ss, ss + dur, ls, tn, ps, lt))
    return adj


# ═══════════════════════════════════════════════════════════════
# EARLIEST-ARRIVAL DIJKSTRA
# ═══════════════════════════════════════════════════════════════


def _earliest_arrival_on_adj(
    adj, source, dest, t_inject, ttl, _counters=None
) -> tuple[bool, float]:
    """Earliest-arrival Dijkstra on a pre-built adjacency.

    Core inner loop — no adjacency construction, no attribute dispatch.

    Parameters
    ----------
    _counters : OpCounter | None
        If provided, increments algorithmic operation counters for FPGA
        resource estimation.  None (default) = zero overhead, all existing
        call sites unchanged.
    """
    if source == dest:
        if _counters is not None:
            _counters.feasibility_invocations += 1
            _counters.feasibility_feasible_count += 1
        return True, t_inject

    deadline = t_inject + ttl
    best: dict = {source: t_inject}
    heap = [(t_inject, source)]

    if _counters is not None:
        _counters.feasibility_invocations += 1
        _counters.feasibility_heap_pushes += 1
        _counters.update_heap_peak(1)

    while heap:
        t_cur, node = heapq.heappop(heap)
        if _counters is not None:
            _counters.feasibility_heap_pops += 1

        if node == dest:
            if _counters is not None:
                _counters.feasibility_feasible_count += 1
            return True, t_cur

        if t_cur > best.get(node, _INF):
            continue

        for start_s, end_s, latency_s, to_node in adj.get(node, []):
            if _counters is not None:
                _counters.feasibility_edge_scans += 1
            if t_cur > end_s:
                if _counters is not None:
                    _counters.feasibility_edges_filtered += 1
                continue
            t_depart = max(t_cur, start_s)
            t_arrive = t_depart + latency_s
            if t_arrive > deadline:
                if _counters is not None:
                    _counters.feasibility_edges_filtered += 1
                continue
            if t_arrive < best.get(to_node, _INF):
                if _counters is not None:
                    _counters.feasibility_relaxations += 1
                    _counters.feasibility_heap_pushes += 1
                best[to_node] = t_arrive
                heapq.heappush(heap, (t_arrive, to_node))
                if _counters is not None:
                    _counters.update_heap_peak(len(heap))

    return False, _INF


def earliest_arrival(
    source: str,
    dest: str,
    t_inject: float,
    contacts,
    ttl: float = _INF,
    *,
    _adj=None,
    _counters=None,
) -> tuple[bool, float]:
    """Earliest-arrival Dijkstra on a temporal contact graph.

    Parameters
    ----------
    source    : originating node name
    dest      : destination node name
    t_inject  : bundle injection time (seconds from simulation epoch)
    contacts  : iterable of Contact objects or dicts with keys
                from_node, to_node, start_s, latency_s
    ttl       : time-to-live in seconds; rejects any path where
                arrival_time > t_inject + ttl  (default: unbounded)
    _adj      : pre-built adjacency from build_adjacency(); built
                on the fly if None
    _counters : OpCounter | None
        Profiling counters for FPGA resource estimation.

    Returns
    -------
    (reachable, arrival_time)
        reachable    : True iff dest is reachable within TTL
        arrival_time : earliest arrival time at dest, or inf if not reachable
    """
    if _adj is None:
        _adj = build_adjacency(contacts)
        if _counters is not None:
            _counters.adjacency_builds += 1
    return _earliest_arrival_on_adj(_adj, source, dest, t_inject, ttl, _counters=_counters)


def _earliest_arrival_path_on_adj(
    adj, source, dest, t_inject, ttl
) -> tuple[bool, float, list[str]]:
    """Earliest-arrival Dijkstra with path tracking on pre-built adjacency."""
    if source == dest:
        return True, t_inject, [source]

    deadline = t_inject + ttl
    best: dict = {source: t_inject}
    prev: dict[str, str | None] = {source: None}
    heap = [(t_inject, source)]

    while heap:
        t_cur, node = heapq.heappop(heap)

        if node == dest:
            path: list[str] = []
            n: str | None = dest
            while n is not None:
                path.append(n)
                n = prev.get(n)
            return True, t_cur, list(reversed(path))

        if t_cur > best.get(node, _INF):
            continue

        for start_s, end_s, latency_s, to_node in adj.get(node, []):
            if t_cur > end_s:
                continue
            t_depart = max(t_cur, start_s)
            t_arrive = t_depart + latency_s
            if t_arrive > deadline:
                continue
            if t_arrive < best.get(to_node, _INF):
                best[to_node] = t_arrive
                prev[to_node] = node
                heapq.heappush(heap, (t_arrive, to_node))

    return False, _INF, []


def earliest_arrival_path(
    source: str,
    dest: str,
    t_inject: float,
    contacts,
    ttl: float = _INF,
    *,
    _adj=None,
) -> tuple[bool, float, list[str]]:
    """Earliest-arrival Dijkstra returning the node path.

    Returns
    -------
    (reachable, arrival_time, path)
        path : list of node names [source, ..., dest] on the shortest-time
               path, or [] if not reachable.
    """
    if _adj is None:
        _adj = build_adjacency(contacts)
    return _earliest_arrival_path_on_adj(_adj, source, dest, t_inject, ttl)


def is_feasible(
    source: str,
    dest: str,
    t_inject: float,
    contacts,
    ttl: float = _INF,
    *,
    _adj=None,
) -> bool:
    """Convenience wrapper — returns bool only."""
    reachable, _ = earliest_arrival(source, dest, t_inject, contacts, ttl, _adj=_adj)
    return reachable


# ═══════════════════════════════════════════════════════════════
# PARALLEL FEASIBILITY SWEEP
# ═══════════════════════════════════════════════════════════════

_F_ADJ = None
_F_SOURCE = None
_F_DEST = None
_F_TTL = None


def _init_feasibility_pool(contacts, source, dest, ttl):
    """Pool initializer: build adjacency ONCE per worker."""
    global _F_ADJ, _F_SOURCE, _F_DEST, _F_TTL
    _F_ADJ = build_adjacency(contacts)
    _F_SOURCE = source
    _F_DEST = dest
    _F_TTL = ttl


def _feasibility_worker(t_inject):
    """Module-level worker for parallel feasibility sweep.  Pickle-safe."""
    ok, _ = _earliest_arrival_on_adj(_F_ADJ, _F_SOURCE, _F_DEST, float(t_inject), _F_TTL)
    return ok


def parallel_feasibility_sweep(source, dest, contacts, injection_times, ttl=_INF):
    """Compute reachability at each injection time, in parallel.

    Adjacency is built ONCE per worker via pool initializer.
    Tasks carry only the injection time float — no contact pickling.

    Parameters
    ----------
    source, dest     : node names
    contacts         : iterable of contact dicts/objects
    injection_times  : sequence of injection times (seconds)
    ttl              : time-to-live (seconds), default unbounded

    Returns
    -------
    list[bool] : feasibility at each injection time
    """
    n = len(injection_times)
    times = [float(t) for t in injection_times]

    if n < 16:
        adj = build_adjacency(contacts)
        return [_earliest_arrival_on_adj(adj, source, dest, t, ttl)[0] for t in times]

    n_workers = min(os.cpu_count() or 4, n)
    chunksize = max(1, n // (n_workers * 4))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_feasibility_pool,
        initargs=(contacts, source, dest, ttl),
    ) as executor:
        return list(executor.map(_feasibility_worker, times, chunksize=chunksize))


def parallel_arrival_sweep(source, dest, contacts, injection_times, ttl=_INF):
    """Compute (reachable, arrival_time) at each injection time, in parallel.

    Same initializer pattern as parallel_feasibility_sweep but returns
    full (bool, float) tuples.

    Returns
    -------
    list[tuple[bool, float]] : (reachable, arrival_time) per injection time
    """
    n = len(injection_times)
    times = [float(t) for t in injection_times]

    if n < 16:
        adj = build_adjacency(contacts)
        return [_earliest_arrival_on_adj(adj, source, dest, t, ttl) for t in times]

    n_workers = min(os.cpu_count() or 4, n)
    chunksize = max(1, n // (n_workers * 4))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_feasibility_pool,
        initargs=(contacts, source, dest, ttl),
    ) as executor:
        return list(executor.map(_arrival_worker, times, chunksize=chunksize))


def _arrival_worker(t_inject):
    """Module-level worker returning full (reachable, arrival_time)."""
    return _earliest_arrival_on_adj(_F_ADJ, _F_SOURCE, _F_DEST, float(t_inject), _F_TTL)


# ═══════════════════════════════════════════════════════════════
# PARALLEL PATH SWEEP (own pool, own globals — isolated from feasibility pool)
# ═══════════════════════════════════════════════════════════════

_PP_ADJ = None
_PP_SOURCE = None
_PP_DEST = None
_PP_TTL = None


def _init_path_pool(contacts, source, dest, ttl):
    """Pool initializer for parallel_path_sweep. Own globals, never shared."""
    global _PP_ADJ, _PP_SOURCE, _PP_DEST, _PP_TTL
    _PP_ADJ = build_adjacency(contacts)
    _PP_SOURCE = source
    _PP_DEST = dest
    _PP_TTL = ttl


def _path_worker(t_inject):
    """Module-level worker returning (t_inject, ok, arrival, path)."""
    ok, arr, path = _earliest_arrival_path_on_adj(
        _PP_ADJ,
        _PP_SOURCE,
        _PP_DEST,
        float(t_inject),
        _PP_TTL,
    )
    return (float(t_inject), ok, arr, path)


def parallel_path_sweep(source, dest, contacts, injection_times, ttl=_INF):
    """Compute oracle paths at each injection time, in parallel.

    Own pool, own initializer, own globals.  Self-contained contract:
    caller passes all parameters, pool state is never assumed from
    a prior phase.

    Parameters
    ----------
    source, dest     : node names
    contacts         : iterable of contact dicts/objects
    injection_times  : sequence of injection times (seconds)
    ttl              : time-to-live (seconds), default unbounded

    Returns
    -------
    dict[float, list[str]] : mapping t_inject → path for reachable times
    """
    n = len(injection_times)
    times = [float(t) for t in injection_times]

    if n < 16:
        adj = build_adjacency(contacts)
        result = {}
        for t in times:
            ok, _, path = _earliest_arrival_path_on_adj(adj, source, dest, t, ttl)
            if ok:
                result[t] = path
        return result

    n_workers = min(os.cpu_count() or 4, n)
    chunksize = max(1, n // (n_workers * 4))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_path_pool,
        initargs=(contacts, source, dest, ttl),
    ) as executor:
        raw = list(executor.map(_path_worker, times, chunksize=chunksize))

    return {t: path for t, ok, _, path in raw if ok}


# ═══════════════════════════════════════════════════════════════
# COMMODITY-AWARE COST-PRIORITY ORACLE
# ═══════════════════════════════════════════════════════════════


def _commodity_oracle_on_adj(
    adj, source, dest, t_inject, lambda_c, ttl
) -> tuple[bool, float, float, list[dict]]:
    """Cost-priority Dijkstra on a pre-built 6-field adjacency."""
    if source == dest:
        return (True, 0.0, t_inject, [])

    deadline = t_inject + ttl
    eps = 1e-12
    next_label_id = 1
    labels_by_node: dict = {source: [(0.0, t_inject, 0)]}
    parents: dict[int, tuple[int, dict] | None] = {0: None}
    active_labels = {0}
    heap = [(0.0, t_inject, 0, source)]

    while heap:
        cost_cur, t_cur, label_id, node = heapq.heappop(heap)
        if label_id not in active_labels:
            continue

        if node == dest:
            hops = []
            cur_id = label_id
            while True:
                parent_entry = parents[cur_id]
                if parent_entry is None:
                    break
                prev_id, hop = parent_entry
                hops.append(hop)
                cur_id = prev_id
            hops.reverse()
            return (True, cost_cur, t_cur, hops)

        for start_s, end_s, latency_s, to_node, p_success, link_type in adj.get(node, []):
            if t_cur > end_s:
                continue
            t_depart = max(t_cur, start_s)
            t_arrive = t_depart + latency_s
            if t_arrive > deadline:
                continue

            edge_nlp = -math.log(max(p_success, 1e-300))
            edge_cost = edge_nlp + lambda_c * (t_arrive - t_cur)
            new_cost = cost_cur + edge_cost

            existing = labels_by_node.get(to_node, [])
            if any(cost <= new_cost + eps and time <= t_arrive + eps for cost, time, _ in existing):
                continue

            kept = []
            for cost, time, existing_id in existing:
                if new_cost <= cost + eps and t_arrive <= time + eps:
                    active_labels.discard(existing_id)
                else:
                    kept.append((cost, time, existing_id))

            new_label_id = next_label_id
            next_label_id += 1
            kept.append((new_cost, t_arrive, new_label_id))
            labels_by_node[to_node] = kept
            active_labels.add(new_label_id)
            parents[new_label_id] = (
                label_id,
                {
                    "from_node": node,
                    "to_node": to_node,
                    "t_depart": t_depart,
                    "t_arrive": t_arrive,
                    "p_success": p_success,
                    "link_type": link_type,
                },
            )
            heapq.heappush(heap, (new_cost, t_arrive, new_label_id, to_node))

    return (False, _INF, _INF, [])


def commodity_oracle(
    source: str,
    dest: str,
    t_inject: float,
    contacts,
    lambda_c: float = 0.0,
    ttl: float = _INF,
    *,
    _adj=None,
) -> tuple[bool, float, float, list[dict]]:
    """Cost-priority Dijkstra: edge cost = -log(p_e) + lambda_c * exposure.

    Selects paths by minimising total cost, where cost accumulates
    link unreliability (-log p) and commodity-specific time exposure
    (lambda_c * time).  When lambda_c=0, equivalent to max-reliability
    oracle.

    Parameters
    ----------
    source      : originating node name
    dest        : destination node name
    t_inject    : bundle injection time (seconds from epoch)
    contacts    : iterable of Contact objects or dicts with keys
                  from_node, to_node, start_s, duration_s, latency_s, p_success
    lambda_c    : commodity hazard rate (per second).  0.0 = hardware (no decay).
    ttl         : time-to-live in seconds (default: unbounded)
    _adj        : pre-built adjacency from build_cost_adjacency(); built
                  on the fly if None

    Returns
    -------
    (reachable, cost, arrival_time, hops)
        reachable    : True iff dest is reachable within TTL
        cost         : total path cost (sum of edge costs)
        arrival_time : arrival time at dest, or inf if not reachable
        hops         : list of dicts with from_node, to_node, t_depart,
                       t_arrive, p_success, link_type
    """
    if _adj is None:
        _adj = build_cost_adjacency(contacts)
    return _commodity_oracle_on_adj(_adj, source, dest, t_inject, lambda_c, ttl)


# ═══════════════════════════════════════════════════════════════
# Hand-verifiable unit tests (run directly: python -m tin.core.oracle)
# ═══════════════════════════════════════════════════════════════


def _run_tests():
    """Five deterministic cases with known correct answers."""

    # Shared contact set: two-hop path S→R→D
    #   S→R: opens at 100s, dur 60s (closes 160s), latency 10s → arrive R at 110s
    #   R→D: opens at 120s, dur 60s (closes 180s), latency  5s → arrive D at 125s
    two_hop = [
        {"from_node": "S", "to_node": "R", "start_s": 100.0, "duration_s": 60.0, "latency_s": 10.0},
        {"from_node": "R", "to_node": "D", "start_s": 120.0, "duration_s": 60.0, "latency_s": 5.0},
    ]

    # Additional direct contact for Test 4
    direct = [
        {"from_node": "S", "to_node": "D", "start_s": 50.0, "duration_s": 60.0, "latency_s": 5.0},
    ]

    results = []

    # Test 1: basic two-hop, inject at t=0, no TTL — expect True, arrival=125s
    ok, arr = earliest_arrival("S", "D", 0.0, two_hop)
    pass1 = ok and arr == 125.0
    results.append(("T1 two-hop reachable", pass1, f"arr={arr}"))

    # Test 2: inject after contact CLOSED (t=170) — S→R closed at 160s
    ok, arr = earliest_arrival("S", "D", 170.0, two_hop)
    pass2 = not ok
    results.append(("T2 inject after contact close", pass2, f"reachable={ok}"))

    # Test 3a: TTL just too short (deadline=124, arrival=125) — expect False
    ok, arr = earliest_arrival("S", "D", 0.0, two_hop, ttl=124.0)
    pass3a = not ok
    results.append(("T3a TTL too short by 1s", pass3a, f"reachable={ok}"))

    # Test 3b: TTL exactly long enough (deadline=125, arrival=125) — expect True
    ok, arr = earliest_arrival("S", "D", 0.0, two_hop, ttl=125.0)
    pass3b = ok and arr == 125.0
    results.append(("T3b TTL exact boundary", pass3b, f"arr={arr}"))

    # Test 4: direct path (50+5=55s) beats relay (125s) — expect True, arrival=55s
    ok, arr = earliest_arrival("S", "D", 0.0, two_hop + direct)
    pass4 = ok and arr == 55.0
    results.append(("T4 direct path beats relay", pass4, f"arr={arr}"))

    # Test 5: source == dest — expect True immediately
    ok, arr = earliest_arrival("S", "S", 42.0, two_hop)
    pass5 = ok and arr == 42.0
    results.append(("T5 source == dest", pass5, f"arr={arr}"))

    # Test 6: inject mid-contact (t=130, S→R open until 160) — bundle catches it
    #   t_depart = max(130, 100) = 130, arrive R at 130+10=140
    #   R→D opens 120, still open at t=140 (closes 180): t_depart=max(140,120)=140, arrive=145
    ok, arr = earliest_arrival("S", "D", 130.0, two_hop)
    pass6 = ok and arr == 145.0
    results.append(("T6 catch in-progress contact", pass6, f"arr={arr}"))

    # Test 7: pre-built adjacency — same result as T1
    adj = build_adjacency(two_hop)
    ok, arr = earliest_arrival("S", "D", 0.0, two_hop, _adj=adj)
    pass7 = ok and arr == 125.0
    results.append(("T7 pre-built adjacency", pass7, f"arr={arr}"))

    # Test 8: parallel feasibility sweep
    times = [0.0, 130.0, 170.0]
    feas = parallel_feasibility_sweep("S", "D", two_hop, times)
    pass8 = feas == [True, True, False]
    results.append(("T8 parallel feasibility sweep", pass8, f"feas={feas}"))

    # Report
    passed = sum(1 for _, p, _ in results if p)
    print(f"\ntin.core.oracle — {passed}/{len(results)} tests passed\n")
    for name, p, detail in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}]  {name:<38}  {detail}")
    print()
    if passed < len(results):
        raise SystemExit("oracle self-test FAILED")


# ═══════════════════════════════════════════════════════════════
# CANONICAL PAIR SELECTION (public)
# ═══════════════════════════════════════════════════════════════


def select_canonical_pair(
    contacts,
    strategy: str = "first_reachable_by_degree",
    top_k: int = 20,
) -> tuple[str, str]:
    """Select a canonical (source, dest) pair from a contact plan.

    Strategies
    ----------
    "top_degree"
        Two highest-degree nodes regardless of reachability.
    "first_reachable_by_degree"  (default)
        Iterate top-k nodes by degree; return first pair with a
        time-respecting path (earliest_arrival check with full TTL).

    Returns (source, dest).  Raises ValueError if no pair found.
    """
    # Build degree map
    degree: dict[str, int] = {}
    t_min = _INF
    t_max = -_INF
    for c in contacts:
        fn = c["from_node"] if isinstance(c, dict) else c.from_node
        tn = c["to_node"] if isinstance(c, dict) else c.to_node
        ts = c["start_s"] if isinstance(c, dict) else c.start_s
        dur = c.get("duration_s", 0) if isinstance(c, dict) else getattr(c, "duration_s", 0)
        degree[fn] = degree.get(fn, 0) + 1
        degree[tn] = degree.get(tn, 0) + 1
        if ts < t_min:
            t_min = ts
        end = ts + dur
        if end > t_max:
            t_max = end

    top = sorted(degree, key=lambda node: degree[node], reverse=True)

    if strategy == "top_degree":
        if len(top) < 2:
            raise ValueError("Fewer than 2 nodes in contact plan")
        return top[0], top[1]

    if strategy == "first_reachable_by_degree":
        adj = build_adjacency(contacts)
        ttl = t_max - t_min if t_max > t_min else _INF
        k = min(top_k, len(top))
        for i in range(k):
            for j in range(i + 1, min(i + top_k, len(top))):
                ok, _ = earliest_arrival(
                    top[i],
                    top[j],
                    t_min,
                    contacts,
                    ttl=ttl,
                    _adj=adj,
                )
                if ok:
                    return top[i], top[j]
        raise ValueError(f"No reachable pair among top-{k} nodes by degree")

    raise ValueError(f"Unknown strategy: {strategy!r}")


if __name__ == "__main__":
    _run_tests()
