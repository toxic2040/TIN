# SPDX-License-Identifier: MIT
"""tin.core.optimal_router — Optimal DTN routing via backward induction on the contact DAG.

Formulates bundle routing as a finite-horizon stochastic dynamic program.
Backward induction gives the value function V(node, t) = optimal delivery
probability starting from state (node, t).

The DP recurrence
-----------------
    V(dest,  t ≤ TTL) = 1
    V(node,  t > TTL) = 0
    V(node,  t)       = max over contacts c with c.from_node==node, c.start_s≥t:
                            p_c · V(c.to_node, arrival_c)
                          + (1−p_c) · V(node, end_c)

where  t_depart  = max(t, c.start_s)                  (wait for window)
       arrival_c = t_depart + c.latency_s              (latency_s = transfer time)
       end_c     = c.start_s + c.duration_s            (window close)

Note: duration_s is the contact *window* length, not a component of travel
time.  latency_s encapsulates propagation + data transfer overhead.

Custody assumptions: the DP assumes perfect custody transfer (no storage
limits, no refusals, no fragmentation).  This gives the theoretical upper
bound on delivery probability — the achievability frontier.

The DAG is strictly time-forward (arrival > departure), so the backward pass
is a topological sort on the time axis.  No Monte Carlo needed: V(source, t)
IS the optimal delivery probability for injection at t.

Achievability interpretation
-----------------------------
    Φ_optimal → 1  :  the chain law IS the capacity; Φ_greedy measures
                       protocol suboptimality (gap is closable).
    Φ_optimal ≠ 1  :  even optimal control can't reach the chain law;
                       the distortion is intrinsic to the temporal graph
                       (irreducible capacity limit).

Usage
-----
    from tin.core.optimal_router import backward_induction, compute_phi_optimal
    # backward_induction takes an absolute ``deadline`` (= injection_time + ttl).
    V = backward_induction(contacts, "surface", "earth",
                           deadline=t_inj + 86400.0, injection_time=t_inj)
    result = compute_phi_optimal(contacts, "surface", "earth", p_eff=0.3,
                                 injection_times=t_inj_list, ttl=86400.0,
                                 eta_lyapunov=0.42, eta_greedy=0.35, s_t=0.91)
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OptimalRouterResult:
    """Achievability metrics for one (source, dest, p_eff) configuration."""

    source: str
    dest: str
    p_eff: float
    s_t: float  # oracle reachability fraction
    eta_greedy: float  # η from greedy router (EfficiencyEstimator)
    eta_optimal: float  # η from DP-optimal router
    dr_greedy: float  # s_t × eta_greedy
    dr_optimal: float  # mean V(source, t_inject) over all injection times
    eta_lyapunov: float  # η_lyap (chain-law reference)
    phi_greedy: float  # Φ_greedy  = eta_greedy  / eta_lyapunov
    phi_optimal: float  # Φ_optimal = eta_optimal / eta_lyapunov
    phi_ratio: float  # Φ_optimal / Φ_greedy  (gap-closability index)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "dest": self.dest,
            "p_eff": self.p_eff,
            "s_t": self.s_t,
            "eta_greedy": self.eta_greedy,
            "eta_optimal": self.eta_optimal,
            "eta_lyapunov": self.eta_lyapunov,
            "dr_greedy": self.dr_greedy,
            "dr_optimal": self.dr_optimal,
            "phi_greedy": self.phi_greedy,
            "phi_optimal": self.phi_optimal,
            "phi_ratio": self.phi_ratio,
        }


# ---------------------------------------------------------------------------
# Backward induction
# ---------------------------------------------------------------------------


def backward_induction(
    contacts: list[dict],
    source: str,
    dest: str,
    deadline: float,
    injection_time: float | None = None,
    max_contacts: int = 2_000_000,
) -> dict[tuple[str, float], float]:
    """Compute V(node, t) = optimal delivery probability via backward induction.

    Processes contacts in reverse chronological order so that V values at
    later times are always available when computing earlier states.

    Parameters
    ----------
    contacts        : contact plan (TIN dict format); p_success must be set.
    source          : source node label.
    dest            : destination node label.
    deadline        : absolute time after which the bundle is dead.  For a
                      bundle injected at ``t_inject`` with TTL ``ttl``, pass
                      ``deadline = t_inject + ttl``.  All time comparisons
                      are absolute; mixing absolute and elapsed time
                      collapses the DP (silent zero values).
    injection_time  : extra query time at the source node to precompute, so
                      the caller can read ``V[(source, injection_time)]``
                      directly.  One run of this DP corresponds to one
                      bundle deadline; multiple injections with different
                      deadlines need separate calls.
    max_contacts    : safety ceiling (4 states per contact → 4× memory).

    Returns
    -------
    dict mapping (node, time) → V value in [0, 1].
    """
    if len(contacts) > max_contacts:
        raise ValueError(
            f"backward_induction: {len(contacts):,} contacts exceeds "
            f"max_contacts={max_contacts:,}. Sub-batch or reduce time window."
        )

    # Build per-node contact lists sorted by start_s (ascending) for fast scan
    nc: dict[str, list[dict]] = defaultdict(list)
    for c in contacts:
        nc[c["from_node"]].append(c)
    for n in nc:
        nc[n].sort(key=lambda x: x["start_s"])

    # Enumerate all (node, time) states that require explicit computation.
    # Every contact contributes three states:
    #   (from_node, start_s)     — decision point: bundle at from_node may attempt c
    #   (from_node, end_s)       — post-failure: bundle still at from_node
    #   (to_node,   arrival_s)   — post-success: bundle arrived at to_node
    states: set[tuple[str, float]] = set()
    for c in contacts:
        end_s = c["start_s"] + c["duration_s"]
        # Arrival when bundle departs at start (contact not yet open → wait):
        arrival_at_start = c["start_s"] + c.get("latency_s", 0.0)
        states.add((c["from_node"], c["start_s"]))
        states.add((c["from_node"], end_s))
        states.add((c["to_node"], arrival_at_start))
        states.add((c["to_node"], end_s + c.get("latency_s", 0.0)))

    if injection_time is not None:
        states.add((source, float(injection_time)))

    # V dict — value function table
    V: dict[tuple[str, float], float] = {}

    # Per-node sorted event times for bisect interpolation.
    # V is piecewise constant between events, so for any (node, t) not in
    # states we look up the next precomputed time >= t.
    node_times: dict[str, list[float]] = defaultdict(list)
    for node, t in states:
        node_times[node].append(t)
    for n in node_times:
        node_times[n].sort()

    def _get_v(node: str, t: float) -> float:
        """Look up V(node, t) — exact if in table, interpolated otherwise."""
        if node == dest and t <= deadline:
            return 1.0
        if t > deadline:
            return 0.0
        key = (node, t)
        cached = V.get(key)
        if cached is not None:
            return cached
        # V is piecewise constant; find smallest precomputed t' >= t.
        times = node_times.get(node)
        if times:
            idx = bisect.bisect_left(times, t)
            if idx < len(times):
                return V.get((node, times[idx]), 0.0)
        return 0.0

    # Iterative backward pass — no recursion.
    for node, t in sorted(states, key=lambda x: x[1], reverse=True):
        if node == dest and t <= deadline:
            V[(node, t)] = 1.0
            continue
        if t > deadline:
            V[(node, t)] = 0.0
            continue
        best = 0.0
        for c in nc.get(node, []):
            cs = c["start_s"]
            end_s = cs + c["duration_s"]
            if end_s <= t + 1e-9:
                continue
            if cs > deadline:
                break
            p = c["p_success"]
            t_depart = max(t, cs)
            arrival_s = t_depart + c.get("latency_s", 0.0)
            val = p * _get_v(c["to_node"], arrival_s) + (1.0 - p) * _get_v(node, end_s)
            if val > best:
                best = val
        V[(node, t)] = best

    return V


# ---------------------------------------------------------------------------
# Full achievability pipeline
# ---------------------------------------------------------------------------


def compute_phi_optimal(
    contacts: list[dict],
    source: str,
    dest: str,
    p_eff: float,
    injection_times: list[float],
    ttl: float,
    eta_lyapunov: float,
    eta_greedy: float,
    s_t: float,
) -> OptimalRouterResult:
    """Compute Φ_optimal via the DP value function.

    Parameters
    ----------
    contacts        : contact plan (p_success overwritten with p_eff).
    source, dest    : endpoints.
    p_eff           : flat link success probability.
    injection_times : all injection times (feasible + infeasible).
    ttl             : bundle TTL (seconds).
    eta_lyapunov    : η_lyap from analytic_s.predict_s (chain-law reference).
    eta_greedy      : η from EfficiencyEstimator.
    s_t             : oracle reachability fraction.

    Returns
    -------
    OptimalRouterResult
    """
    annotated = [{**c, "p_success": p_eff} for c in contacts]

    # Each injected bundle has its own absolute deadline (t_inject + ttl).
    # The DP value V(node, t) depends on that deadline, so a single shared
    # DP cannot represent multiple injections with different deadlines —
    # run one DP per injection and read V(source, t_inject).
    n = len(injection_times)
    if n:
        per_inj = []
        for t_inj in injection_times:
            t_inj_f = float(t_inj)
            V = backward_induction(
                annotated,
                source,
                dest,
                deadline=t_inj_f + ttl,
                injection_time=t_inj_f,
            )
            per_inj.append(V.get((source, t_inj_f), 0.0))
        dr_opt = float(np.mean(per_inj))
    else:
        dr_opt = 0.0
    eta_opt = dr_opt / s_t if s_t > 1e-12 else 0.0

    dr_greedy = s_t * eta_greedy

    phi_opt = eta_opt / eta_lyapunov if eta_lyapunov > 1e-12 else float("nan")
    phi_gr = eta_greedy / eta_lyapunov if eta_lyapunov > 1e-12 else float("nan")

    if not math.isnan(phi_gr) and phi_gr > 1e-12:
        phi_ratio = phi_opt / phi_gr
    else:
        phi_ratio = float("nan")

    return OptimalRouterResult(
        source=source,
        dest=dest,
        p_eff=p_eff,
        s_t=s_t,
        eta_greedy=eta_greedy,
        eta_optimal=eta_opt,
        eta_lyapunov=eta_lyapunov,
        dr_greedy=dr_greedy,
        dr_optimal=dr_opt,
        phi_greedy=phi_gr,
        phi_optimal=phi_opt,
        phi_ratio=phi_ratio,
    )
