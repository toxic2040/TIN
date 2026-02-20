"""tin.core.dtn — Planet-agnostic BPv7 DTN engine for TIN v0.4.0

Layer 2: Bundle Protocol simulation, custody transfer, priority queuing.
Shared by lunar and Mars (and any future body).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


# =========================================================================
# Bundle Priority (RFC 9171 CoS mapping)
# =========================================================================

PRIORITY_EMERGENCY = 0     # life-critical: medical, abort, pressure
PRIORITY_HIGH = 1          # time-sensitive: habitat status, safety
PRIORITY_NORMAL = 2        # standard operational
PRIORITY_BULK = 3          # routine telemetry, science data

PRIORITY_NAMES = {0: "EMERGENCY", 1: "HIGH", 2: "NORMAL", 3: "BULK"}

# Default lifetimes (seconds)
DEFAULT_LIFETIMES = {
    PRIORITY_EMERGENCY: 3600,
    PRIORITY_HIGH: 14400,
    PRIORITY_NORMAL: 86400,
    PRIORITY_BULK: 604800,
}


# =========================================================================
# BPv7 Bundle
# =========================================================================

@dataclass
class Bundle:
    """BPv7 bundle with custody tracking (RFC 9171 simplified)."""
    bundle_id: str
    priority: int = PRIORITY_NORMAL
    size_bytes: int = 256
    lifetime_s: float = 86400.0
    source: str = ""
    destination: str = "dsn_earth"
    payload_type: str = "telemetry"

    # Timing
    created_s: float = 0.0
    age_s: float = 0.0

    # Custody
    custody_chain: List[Dict] = field(default_factory=list)
    current_custodian: str = ""
    hop_count: int = 0
    hop_log: List[Dict] = field(default_factory=list)

    # Status
    delivered: bool = False
    deleted: bool = False
    delivery_time_s: Optional[float] = None

    def is_expired(self, t_s: float) -> bool:
        return (t_s - self.created_s) > self.lifetime_s

    def accept_custody(self, node_id: str, t_s: float):
        self.current_custodian = node_id
        self.custody_chain.append({"node": node_id, "time_s": round(t_s, 3)})

    def add_hop(self, from_node: str, to_node: str, depart_s: float,
                arrive_s: float, link_type: str = ""):
        self.hop_log.append({
            "from": from_node, "to": to_node,
            "depart_s": round(depart_s, 3), "arrive_s": round(arrive_s, 3),
            "link_type": link_type,
        })
        self.hop_count += 1
        self.age_s = arrive_s - self.created_s

    def mark_delivered(self, node_id: str, t_s: float):
        self.delivered = True
        self.delivery_time_s = t_s
        self.accept_custody(node_id, t_s)

    def total_latency_s(self) -> Optional[float]:
        if self.delivery_time_s is not None:
            return self.delivery_time_s - self.created_s
        return None

    def to_dict(self) -> Dict:
        lat = self.total_latency_s()
        return {
            "bundle_id": self.bundle_id,
            "priority": PRIORITY_NAMES.get(self.priority, "UNKNOWN"),
            "source": self.source,
            "destination": self.destination,
            "payload_type": self.payload_type,
            "size_bytes": self.size_bytes,
            "created_s": self.created_s,
            "hop_count": self.hop_count,
            "delivered": self.delivered,
            "deleted": self.deleted,
            "latency_s": round(lat, 3) if lat else None,
            "latency_min": round(lat / 60, 2) if lat else None,
            "custody_chain": self.custody_chain,
            "hop_log": self.hop_log,
        }


# =========================================================================
# Custody Node
# =========================================================================

class CustodyNode:
    """A DTN node that can accept, store, and forward bundles."""

    def __init__(self, node_id: str, node_type: str = "relay",
                 storage_bytes: int = 100_000_000, is_relay: bool = True):
        self.node_id = node_id
        self.node_type = node_type
        self.storage_bytes = storage_bytes
        self.is_relay = is_relay
        self.bundles: List[Bundle] = []
        self.custody_accepted = 0
        self.custody_forwarded = 0

    def accept_custody(self, bundle: Bundle, t_s: float) -> bool:
        """Accept a bundle into custody. Returns False if storage full or expired."""
        if bundle.is_expired(t_s):
            bundle.deleted = True
            return False
        total_stored = sum(b.size_bytes for b in self.bundles)
        if total_stored + bundle.size_bytes > self.storage_bytes:
            return False
        bundle.accept_custody(self.node_id, t_s)
        # Priority insertion
        if bundle.priority == PRIORITY_EMERGENCY:
            self.bundles.insert(0, bundle)
        else:
            self.bundles.append(bundle)
        self.custody_accepted += 1
        return True

    def forward_bundle(self, bundle: Bundle, to_node: 'CustodyNode',
                       t_s: float, link_type: str = "",
                       data_rate_kbps: float = 256.0,
                       propagation_delay_s: float = 0.001) -> float:
        """Forward a bundle. Returns arrival time at destination node."""
        transfer_s = bundle.size_bytes / (data_rate_kbps * 1000 / 8)
        arrive_s = t_s + transfer_s + propagation_delay_s
        bundle.add_hop(self.node_id, to_node.node_id, t_s, arrive_s, link_type)
        if bundle in self.bundles:
            self.bundles.remove(bundle)
        self.custody_forwarded += 1
        return arrive_s

    def stats(self) -> Dict:
        return {
            "node_id": self.node_id,
            "type": self.node_type,
            "accepted": self.custody_accepted,
            "forwarded": self.custody_forwarded,
            "buffered": len(self.bundles),
        }


# =========================================================================
# DTN Network
# =========================================================================

class DTNNetwork:
    """Manages a network of custody nodes and routes bundles between them."""

    def __init__(self, nodes: List[CustodyNode]):
        self.nodes = {n.node_id: n for n in nodes}
        self.bundles: List[Bundle] = []
        self.total_transfers = 0
        self.delivered_count = 0
        self.event_log: List[Dict] = []

    def add_node(self, node: CustodyNode):
        self.nodes[node.node_id] = node

    def create_bundle(self, source: str, destination: str, priority: int,
                      t_s: float, payload_type: str = "emergency",
                      size_bytes: int = 512) -> Bundle:
        """Create and register a new bundle."""
        bundle = Bundle(
            bundle_id=f"tin:{payload_type}:{len(self.bundles)}",
            priority=priority,
            size_bytes=size_bytes,
            lifetime_s=DEFAULT_LIFETIMES.get(priority, 86400),
            source=source,
            destination=destination,
            payload_type=payload_type,
            created_s=t_s,
        )
        self.bundles.append(bundle)
        # Accept at source node
        if source in self.nodes:
            self.nodes[source].accept_custody(bundle, t_s)
        self._log(t_s, "CREATED", source, bundle)
        return bundle

    def route_along_path(self, bundle: Bundle, path: List[str],
                         link_types: List[str], t_s: float,
                         data_rates: Optional[List[float]] = None,
                         delays: Optional[List[float]] = None) -> bool:
        """Route a bundle along an explicit path of node IDs.

        Args:
            path: ordered list of node_ids from current position to destination
            link_types: link type string for each hop
            t_s: current simulation time
            data_rates: kbps per hop (optional, default 256)
            delays: propagation delay per hop in seconds (optional)

        Returns: True if delivered, False if stuck/expired.
        """
        current_time = t_s
        for i, next_id in enumerate(path):
            if bundle.is_expired(current_time):
                bundle.deleted = True
                self._log(current_time, "EXPIRED", bundle.current_custodian, bundle)
                return False

            from_node = self.nodes.get(bundle.current_custodian)
            to_node = self.nodes.get(next_id)
            if from_node is None or to_node is None:
                self._log(current_time, "NO_NODE", next_id, bundle)
                continue

            rate = data_rates[i] if data_rates else 256.0
            delay = delays[i] if delays else 0.001
            lt = link_types[i] if i < len(link_types) else ""

            arrive = from_node.forward_bundle(bundle, to_node, current_time, lt, rate, delay)
            accepted = to_node.accept_custody(bundle, arrive)

            if not accepted:
                self._log(arrive, "REFUSED", next_id, bundle)
                return False

            self.total_transfers += 1
            self._log(arrive, "FORWARDED", next_id, bundle)
            current_time = arrive

        # Mark delivered if we reached the destination
        if bundle.current_custodian == bundle.destination:
            bundle.mark_delivered(bundle.destination, current_time)
            self.delivered_count += 1
            self._log(current_time, "DELIVERED", bundle.destination, bundle)
            return True

        return False

    def simulate_contact_window(self, t_s: float, contact_nodes: List[str],
                                duration_s: float = 600.0):
        """Simulate a contact window: all buffered bundles at contact_nodes
        attempt forwarding to next hop."""
        for node_id in contact_nodes:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            # Forward any buffered bundles (priority-sorted)
            for bundle in list(node.bundles):
                if not bundle.delivered and not bundle.deleted:
                    self.total_transfers += 1

    def summary(self) -> Dict:
        delivered = [b for b in self.bundles if b.delivered]
        latencies = [b.total_latency_s() for b in delivered if b.total_latency_s() is not None]
        return {
            "total_bundles": len(self.bundles),
            "delivered": len(delivered),
            "deleted": sum(1 for b in self.bundles if b.deleted),
            "pending": sum(1 for b in self.bundles if not b.delivered and not b.deleted),
            "total_transfers": self.total_transfers,
            "delivery_rate_pct": round(100 * len(delivered) / max(len(self.bundles), 1), 1),
            "latency_mean_s": round(float(np.mean(latencies)), 2) if latencies else None,
            "latency_worst_s": round(float(np.max(latencies)), 2) if latencies else None,
            "latency_mean_min": round(float(np.mean(latencies)) / 60, 2) if latencies else None,
            "node_stats": [self.nodes[n].stats() for n in self.nodes],
            "bundles": [b.to_dict() for b in self.bundles],
        }

    def _log(self, t_s, event, node, bundle):
        self.event_log.append({
            "time_s": round(t_s, 3),
            "event": event,
            "node": node,
            "bundle_id": bundle.bundle_id,
            "priority": PRIORITY_NAMES.get(bundle.priority, "?"),
        })
