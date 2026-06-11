# SPDX-License-Identifier: MIT
"""tin.core.routing — Stochastic RW-CGR router (B2 closed).

Utility = p_success * exp(-λ * latency). Top-k caching.
Integrates with 5-state FSM + halo.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from tin.core.dtn import Bundle


@dataclass
class Contact:
    """Probabilistic contact window (B2)."""

    from_node: str
    to_node: str
    start_s: float
    duration_s: float
    p_success: float = 0.98
    data_rate_kbps: float = 1000.0
    latency_s: float = 0.0  # propagation + transfer overhead
    link_layer: str = "rf"  # 'rf' or 'optical' for hybrid routing
    failure_domain: str = ""  # correlation group for copula failure model


class RW_CGR:
    """Stochastic Contact Graph Router per study B2.

    Monte-Carlo random-walk sampling over the contact graph to find
    top-k lowest expected-latency paths. Emergency bundles get biased
    toward high-reliability relay nodes (e.g., EM-L2 halo).
    """

    ROUTE_CACHE_MAX = 10_000

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.contact_graph: dict[str, list[Contact]] = {}
        self.route_cache: OrderedDict[
            tuple[str, str, int, int, float, int, int], list[list[str]]
        ] = OrderedDict()

    def add_contact(self, c: Contact) -> None:
        """Append a contact to the graph.

        Mutates ``contact_graph`` and invalidates ``route_cache`` because
        a new contact may dominate a cached path (cheaper hop, higher
        reliability, or a previously-unreachable destination). Pre-fix
        the cache was not cleared, so callers that added a superior
        direct contact kept getting the old indirect path back until
        ``clear_cache()`` was invoked manually.
        """
        self.contact_graph.setdefault(c.from_node, []).append(c)
        self.route_cache.clear()

    def clear_cache(self) -> None:
        """Invalidate route cache (call when contact plan updates)."""
        self.route_cache.clear()

    def _path_reliability(self, path: list[str]) -> float:
        """Cumulative p_success along path (product of hop reliabilities)."""
        p = 1.0
        for i in range(len(path) - 1):
            from_n, to_n = path[i], path[i + 1]
            contacts = [c for c in self.contact_graph.get(from_n, []) if c.to_node == to_n]
            if contacts:
                p *= max(contacts, key=lambda c: c.p_success).p_success
            else:
                return 0.0
        return p

    def _expected_latency(self, path: list[str], t_s: float, bundle: Bundle) -> float:
        """E[T] along path, including waits for future contact windows."""
        total = 0.0
        current_t = t_s
        for i in range(len(path) - 1):
            from_n, to_n = path[i], path[i + 1]
            candidates = [
                c
                for c in self.contact_graph.get(from_n, [])
                if c.to_node == to_n and current_t <= c.start_s + c.duration_s
            ]
            if not candidates:
                return np.inf

            def arrival_time(c: Contact) -> float:
                t_depart = max(current_t, c.start_s)
                transfer_s = bundle.size_bytes / max(c.data_rate_kbps * 125, 1)
                return t_depart + transfer_s + c.latency_s

            c = min(candidates, key=arrival_time)
            t_arrive = arrival_time(c)
            total += t_arrive - current_t
            current_t = t_arrive
        return total

    def find_routes(
        self, src: str, dst: str, bundle: Bundle, t_s: float, n_samples: int = 200, top_k: int = 3
    ) -> list[list[str]]:
        """RW-CGR Monte-Carlo sampling → top-k lowest E[T] paths."""
        key = (src, dst, bundle.priority, bundle.size_bytes, float(t_s), int(n_samples), int(top_k))
        if key in self.route_cache:
            self.route_cache.move_to_end(key)
            return self.route_cache[key]

        paths: list[tuple[list[str], float, float]] = []
        for _ in range(n_samples):
            path = [src]
            current = src
            visited = {src}
            while current != dst and len(path) < 12:
                neighbors = self.contact_graph.get(current, [])
                if not neighbors:
                    break
                # Biased random walk: higher utility for high-p, low-latency
                utils = np.array(
                    [
                        c.p_success * np.exp(-bundle.priority * (c.latency_s + 10) / 600)
                        for c in neighbors
                    ]
                )
                total_u = utils.sum()
                if total_u < 1e-12:
                    break
                utils /= total_u
                next_idx = self.rng.choice(len(neighbors), p=utils)
                next_n = neighbors[next_idx].to_node
                if next_n in visited:
                    break
                visited.add(next_n)
                path.append(next_n)
                current = next_n

            if path[-1] == dst:
                lat = self._expected_latency(path, t_s, bundle)
                if not np.isfinite(lat):
                    continue
                p_path = self._path_reliability(path)
                # B2 composite utility: higher is better
                # λ scales with priority (emergency=0 → λ=0.1, bulk=3 → λ=1.0)
                lam = 0.1 + 0.3 * bundle.priority
                utility = p_path * np.exp(-lam * lat)
                paths.append((path, utility, lat))

        # Sort by utility descending (higher = better)
        paths.sort(key=lambda x: x[1], reverse=True)
        top_paths = [p[0] for p in paths[:top_k]]
        self.route_cache[key] = top_paths
        while len(self.route_cache) > self.ROUTE_CACHE_MAX:
            self.route_cache.popitem(last=False)
        return top_paths

    def route_bundle(self, bundle: Bundle, src: str, dst: str, t_s: float) -> list[str] | None:
        """Public API: returns best stochastic route or None."""
        routes = self.find_routes(src, dst, bundle, t_s)
        return routes[0] if routes else None
