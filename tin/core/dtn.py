# SPDX-License-Identifier: MIT
"""tin.core.dtn — Planet-agnostic BPv7 DTN engine (B1 + N1 merged)

Layer 2: 5-state custody FSM + fragment-group custody (Deep Dive B1 + N1),
adaptive custody timeout and phantom-custody detection (B1 §1.3–§1.4).
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# =========================================================================
# Priority & Lifetimes
# =========================================================================
PRIORITY_EMERGENCY = 0
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2
PRIORITY_BULK = 3
PRIORITY_NAMES = {0: "EMERGENCY", 1: "HIGH", 2: "NORMAL", 3: "BULK"}
DEFAULT_LIFETIMES = {0: 3600, 1: 14400, 2: 86400, 3: 604800}

# B1 §1.4 custody inventory digest parameters
BLOOM_M = 4096  # filter size, bits
BLOOM_K = 7  # hash functions
RELEASE_CONFIRMATIONS = 3  # heartbeat sightings before backup release


# =========================================================================
# B1 §1.3: Adaptive Custody Timeout τ(t)
# =========================================================================
def adaptive_custody_timeout(
    rtlt_s: float,
    ber: float,
    t_proc_s: float = 5.0,
    ack_bits: int = 1024,
) -> float:
    """B1 §1.3: τ = RTLT + T_proc + N_ret × T_ACK_slot.

    N_ret = 1 / p_ACK with p_ACK = (1 − BER)^ack_bits;
    T_ACK_slot = max(RTLT/2, 10 s). Returns inf when the link cannot
    carry an ACK (p_ACK → 0), which forces LOCAL_HOLD in the policy.
    """
    p_ack = (1.0 - ber) ** ack_bits
    if p_ack <= 0.0:
        return float("inf")
    n_ret = 1.0 / p_ack
    t_ack_slot = max(rtlt_s / 2.0, 10.0)
    return rtlt_s + t_proc_s + n_ret * t_ack_slot


def custody_transfer_mode(tau_s: float, t_contact_s: float) -> str:
    """B1 §1.3 policy: transfer mode from τ vs remaining contact window."""
    if tau_s > t_contact_s:
        return "LOCAL_HOLD"
    if tau_s > 0.5 * t_contact_s:
        return "CAUTIOUS"
    return "NORMAL"


# =========================================================================
# B1 §1.4: Custody Inventory Digest (Bloom filter)
# =========================================================================
def _bloom_indices(bundle_id: str, m: int, k: int):
    for i in range(k):
        h = hashlib.sha256(f"{i}:{bundle_id}".encode()).digest()
        yield int.from_bytes(h[:8], "big") % m


def bloom_digest(bundle_ids, m: int = BLOOM_M, k: int = BLOOM_K) -> np.ndarray:
    """B1 §1.4: custody inventory digest over held bundle IDs."""
    bits = np.zeros(m, dtype=bool)
    for bid in bundle_ids:
        for idx in _bloom_indices(bid, m, k):
            bits[idx] = True
    return bits


def bloom_contains(digest: np.ndarray, bundle_id: str, k: int = BLOOM_K) -> bool:
    m = int(digest.shape[0])
    return all(digest[idx] for idx in _bloom_indices(bundle_id, m, k))


# =========================================================================
# 5-State Custody FSM (B1) + Fragment Groups (N1)
# =========================================================================
class CustodyState(Enum):
    """5-state custody FSM per Deep Dive B1 (extended for N1 fragments)."""

    H = "HOLDING"
    O = "OUTSTANDING"
    P = "PENDING"
    R = "RECEIVED"
    X = "EXPIRED_LOST"


# Separate sentinel for aggregate partial state (not a valid FSM state)
PARTIAL_AGGREGATE = "PARTIAL"


@dataclass
class FragmentGroup:
    """N1: One proactive BPv7 fragment group (RFC 9171 §5.8)."""

    group_id: str
    offset_start: int
    offset_end: int
    state: CustodyState = CustodyState.H
    custody_record: list[dict] = field(default_factory=list)


@dataclass
class Bundle:
    """BPv7 bundle with 5-state FSM and optional fragment groups (N1)."""

    bundle_id: str
    priority: int = PRIORITY_NORMAL
    size_bytes: int = 256
    lifetime_s: float = 86400.0
    source: str = ""
    destination: str = "dsn_earth"
    payload_type: str = "telemetry"
    created_s: float = 0.0
    age_s: float = 0.0

    # Core custody
    custody_state: CustodyState = CustodyState.H
    current_custodian: str = ""
    custody_chain: list[dict] = field(default_factory=list)
    hop_count: int = 0
    hop_log: list[dict] = field(default_factory=list)
    max_hops: int = 100  # safety ceiling; DTNNetwork sets to len(nodes)

    # N1 fragment groups (empty → non-fragmented bundle)
    is_fragmented: bool = False
    fragment_groups: list[FragmentGroup] = field(default_factory=list)

    delivered: bool = False
    deleted: bool = False
    delivery_time_s: float | None = None

    def is_expired(self, t_s: float) -> bool:
        return (t_s - self.created_s) > self.lifetime_s

    def aggregate_state(self) -> str:
        """N1 aggregate state for fragmented bundles.

        Returns CustodyState value string, or PARTIAL_AGGREGATE for mixed states.
        Fixed: does not construct invalid CustodyState("PARTIAL").
        """
        if not self.is_fragmented or not self.fragment_groups:
            return self.custody_state.value
        states = [g.state for g in self.fragment_groups]
        if all(s == CustodyState.H for s in states):
            return CustodyState.H.value
        if any(s == CustodyState.X for s in states):
            return CustodyState.X.value
        if all(s == CustodyState.R for s in states):
            return CustodyState.R.value
        if any(s == CustodyState.R for s in states):
            return PARTIAL_AGGREGATE
        if any(s == CustodyState.P for s in states):
            return CustodyState.P.value
        return CustodyState.O.value

    def accept_custody(self, node_id: str, t_s: float) -> None:
        self.current_custodian = node_id
        self.custody_chain.append({"node": node_id, "time_s": round(t_s, 3)})
        self.custody_state = CustodyState.H
        if self.is_fragmented:
            for g in self.fragment_groups:
                g.state = CustodyState.H
                g.custody_record.append({"node": node_id, "time_s": round(t_s, 3)})

    def mark_delivered(self, t_s: float) -> None:
        self.delivered = True
        self.delivery_time_s = t_s
        self.custody_state = CustodyState.R
        if self.is_fragmented:
            for g in self.fragment_groups:
                g.state = CustodyState.R

    def add_hop(
        self,
        from_node: str,
        to_node: str,
        t_s: float,
        link_type: str = "",
        data_rate_kbps: float = 256.0,
    ) -> None:
        if self.hop_count >= self.max_hops:
            raise RuntimeError(
                f"Bundle {self.bundle_id} exceeded max_hops={self.max_hops}. "
                f"Likely a forwarding loop — check visited-set logic."
            )
        self.hop_count += 1
        self.hop_log.append(
            {
                "hop": self.hop_count,
                "from": from_node,
                "to": to_node,
                "time_s": round(t_s, 3),
                "link_type": link_type,
                "data_rate_kbps": data_rate_kbps,
            }
        )

    def total_latency_s(self) -> float:
        if not self.hop_log:
            return 0.0
        return float(self.hop_log[-1]["time_s"]) - self.created_s

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "priority": PRIORITY_NAMES.get(self.priority, str(self.priority)),
            "size_bytes": self.size_bytes,
            "source": self.source,
            "destination": self.destination,
            "custody_state": self.aggregate_state(),
            "current_custodian": self.current_custodian,
            "hop_count": self.hop_count,
            "delivered": self.delivered,
            "fragmented": self.is_fragmented,
            "num_groups": len(self.fragment_groups),
        }


# =========================================================================
# CustodyNode
# =========================================================================
class CustodyNode:
    MAX_BUNDLES = 100_000  # safety ceiling independent of storage_bytes

    def __init__(
        self,
        node_id: str,
        node_type: str = "relay",
        storage_bytes: int = 100_000_000,
        is_relay: bool = True,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.storage_bytes = storage_bytes
        self.is_relay = is_relay
        self.used_bytes: int = 0
        self.bundles: list[Bundle] = []
        self.custody_accepted: int = 0
        self.custody_refused: int = 0
        # B1 §1.4: backups retained after custody transfer, keyed by bundle_id
        self.pending_release: dict[str, dict] = {}

    def _can_accept(self, bundle: Bundle, t_s: float) -> str:
        """Check acceptance preconditions without mutating any state.

        Returns
        -------
        "ok"       — bundle can be accepted
        "expired"  — bundle lifetime exceeded
        "storage"  — insufficient storage capacity
        """
        if bundle.is_expired(t_s):
            return "expired"
        if self.used_bytes + bundle.size_bytes > self.storage_bytes:
            return "storage"
        if len(self.bundles) >= self.MAX_BUNDLES:
            return "storage"
        return "ok"

    def accept_custody(self, bundle: Bundle, t_s: float) -> bool:
        if bundle in self.bundles:
            return True
        if bundle.current_custodian and bundle.current_custodian != self.node_id:
            self.custody_refused += 1
            return False

        reason = self._can_accept(bundle, t_s)
        if reason == "expired":
            bundle.deleted = True
            bundle.custody_state = CustodyState.X
            if bundle.is_fragmented:
                for g in bundle.fragment_groups:
                    g.state = CustodyState.X
            return False
        if reason == "storage":
            self.custody_refused += 1
            bundle.custody_state = CustodyState.P
            if bundle.is_fragmented:
                for g in bundle.fragment_groups:
                    g.state = CustodyState.P
            return False
        bundle.accept_custody(self.node_id, t_s)
        self.used_bytes += bundle.size_bytes
        if bundle.priority == PRIORITY_EMERGENCY:
            self.bundles.insert(0, bundle)
        else:
            self.bundles.append(bundle)
        self.custody_accepted += 1
        return True

    def _remove_bundle(self, bundle: Bundle) -> None:
        """Remove bundle from this node's storage, freeing capacity."""
        try:
            self.bundles.remove(bundle)
            self.used_bytes -= bundle.size_bytes
        except ValueError:
            pass  # bundle not in this node's list (defensive)

    def forward_bundle(
        self,
        bundle: Bundle,
        to_node: "CustodyNode",
        t_s: float,
        link_type: str = "",
        data_rate_kbps: float = 256.0,
        propagation_delay_s: float = 0.001,
        retain_backup: bool = False,
    ) -> float:
        """Forward bundle to next custodian. Returns arrival time or -1.

        Atomic receipt pattern: no bundle state mutation occurs until the
        receiver confirms acceptance. On refusal, bundle state is unchanged
        (remains H at sender). On expiry, bundle is marked X and freed.

        With retain_backup (B1 §1.4), the sender keeps a storage-charged
        backup in pending_release until heartbeat_check confirms receiver
        custody RELEASE_CONFIRMATIONS times.
        """
        if bundle not in self.bundles or bundle.current_custodian != self.node_id:
            return -1.0

        transfer_s = bundle.size_bytes / (data_rate_kbps * 125)  # kbps → bytes/s
        arrival_s = t_s + transfer_s + propagation_delay_s

        # --- Check preconditions without mutating bundle state ---
        reason = to_node._can_accept(bundle, arrival_s)

        if reason == "expired":
            # Bundle expired in transit — mark and free sender storage
            bundle.deleted = True
            bundle.custody_state = CustodyState.X
            if bundle.is_fragmented:
                for g in bundle.fragment_groups:
                    g.state = CustodyState.X
            self._remove_bundle(bundle)
            return -1.0

        if reason == "storage":
            # Receiver has no room — sender retains custody, state unchanged
            to_node.custody_refused += 1
            return -1.0

        # --- All preconditions passed: commit atomically ---
        # 1. Record hop and update age
        bundle.add_hop(self.node_id, to_node.node_id, arrival_s, link_type, data_rate_kbps)
        bundle.age_s = arrival_s - bundle.created_s

        # 2. Release from sender FIRST (single-custodian invariant:
        #    bundle must never be in two nodes' lists simultaneously)
        self._remove_bundle(bundle)

        # 3. Transfer custody to receiver
        bundle.accept_custody(to_node.node_id, arrival_s)
        to_node.used_bytes += bundle.size_bytes
        if bundle.priority == PRIORITY_EMERGENCY:
            to_node.bundles.insert(0, bundle)
        else:
            to_node.bundles.append(bundle)
        to_node.custody_accepted += 1

        # 4. B1 §1.4: optionally retain a storage-charged backup copy
        if retain_backup:
            self.used_bytes += bundle.size_bytes
            self.pending_release[bundle.bundle_id] = {
                "bundle": bundle,
                "to_node": to_node.node_id,
                "confirmations": 0,
            }

        return arrival_s

    def custody_digest(self) -> np.ndarray:
        """B1 §1.4: Bloom-filter digest of bundle IDs currently in custody."""
        return bloom_digest(b.bundle_id for b in self.bundles)

    def heartbeat_check(self, peer_digests: dict, t_s: float) -> dict:
        """B1 §1.4: check pending-release backups against peer digests.

        A backup is released after RELEASE_CONFIRMATIONS sightings in the
        receiver's digest. A missing entry is a phantom custody event: the
        receiver never had the bundle, so it is re-queued here from the
        backup. Peers with no digest this cycle are skipped (no evidence).
        """
        report: dict[str, list[str]] = {"released": [], "phantoms": []}
        for bid in list(self.pending_release):
            entry = self.pending_release[bid]
            digest = peer_digests.get(entry["to_node"])
            if digest is None:
                continue
            bundle = entry["bundle"]
            if bloom_contains(digest, bid):
                entry["confirmations"] += 1
                if entry["confirmations"] >= RELEASE_CONFIRMATIONS:
                    self.used_bytes -= bundle.size_bytes
                    del self.pending_release[bid]
                    report["released"].append(bid)
            else:
                bundle.accept_custody(self.node_id, t_s)
                if bundle.priority == PRIORITY_EMERGENCY:
                    self.bundles.insert(0, bundle)
                else:
                    self.bundles.append(bundle)
                del self.pending_release[bid]
                report["phantoms"].append(bid)
        return report

    def pre_flush_to_holding(self) -> None:
        """B1/N5: Mass transition to H before safe-mode (SPE or conjunction)."""
        for b in self.bundles:
            b.custody_state = CustodyState.H
            if b.is_fragmented:
                for g in b.fragment_groups:
                    g.state = CustodyState.H


# =========================================================================
# DTNNetwork
# =========================================================================
class DTNNetwork:
    def __init__(self):
        self.nodes: dict[str, CustodyNode] = {}
        self.bundles: list[Bundle] = []
        self._bundle_counter: int = 0

    def add_node(self, node: CustodyNode) -> None:
        self.nodes[node.node_id] = node

    def create_bundle(
        self,
        source: str,
        destination: str,
        priority: int = PRIORITY_NORMAL,
        size_bytes: int = 256,
        payload_type: str = "telemetry",
        t_s: float = 0.0,
        lifetime_s: float = 86400.0,
        fragmented: bool = False,
        num_groups: int = 1,
    ) -> Bundle:
        self._bundle_counter += 1
        bid = f"bundle-{self._bundle_counter:04d}"
        bundle = Bundle(
            bundle_id=bid,
            priority=priority,
            size_bytes=size_bytes,
            lifetime_s=lifetime_s,
            source=source,
            destination=destination,
            payload_type=payload_type,
            created_s=t_s,
            max_hops=max(len(self.nodes), 100),
        )
        # Cap fragment groups: can't have more groups than 32-byte fragments
        num_groups = min(num_groups, max(1, size_bytes // 32))
        if fragmented and num_groups > 1:
            bundle.is_fragmented = True
            step = bundle.size_bytes // num_groups
            for i in range(num_groups):
                end = bundle.size_bytes if i == num_groups - 1 else (i + 1) * step
                bundle.fragment_groups.append(
                    FragmentGroup(
                        group_id=f"{bid}-g{i}",
                        offset_start=i * step,
                        offset_end=end,
                    )
                )
        self.bundles.append(bundle)
        return bundle

    def route_along_path(
        self,
        bundle: Bundle,
        path: list[str],
        link_types: list[str] | None = None,
        t_s: float = 0.0,
        data_rate_kbps: float = 256.0,
    ) -> bool:
        """Route a bundle along a given node path."""
        if not path or len(path) < 2:
            return False
        if path[0] != bundle.source or path[-1] != bundle.destination:
            return False
        if any(node_id not in self.nodes for node_id in path):
            return False
        if link_types is None:
            link_types = ["rf"] * (len(path) - 1)

        # Source accepts custody first
        src_node = self.nodes.get(path[0])
        if src_node is None:
            return False
        if not src_node.accept_custody(bundle, t_s):
            return False

        current_t = t_s
        for i in range(len(path) - 1):
            from_id, to_id = path[i], path[i + 1]
            from_node = self.nodes.get(from_id)
            to_node = self.nodes.get(to_id)
            if from_node is None or to_node is None:
                return False
            lt = link_types[i] if i < len(link_types) else "rf"
            arrival = from_node.forward_bundle(
                bundle, to_node, current_t, link_type=lt, data_rate_kbps=data_rate_kbps
            )
            if arrival < 0:
                return False
            current_t = arrival

        # Mark delivered at final node
        bundle.mark_delivered(current_t)
        return True

    def summary(self) -> dict:
        delivered = [b for b in self.bundles if b.delivered]
        lost = [b for b in self.bundles if b.custody_state == CustodyState.X]
        return {
            "total_bundles": len(self.bundles),
            "delivered": len(delivered),
            "lost": len(lost),
            "in_network": len(self.bundles) - len(delivered) - len(lost),
            "avg_latency_s": (
                np.mean([b.total_latency_s() for b in delivered]) if delivered else 0.0
            ),
        }
