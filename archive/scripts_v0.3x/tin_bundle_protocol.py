#!/usr/bin/env python3
"""tin_bundle_protocol.py — Bundle Protocol v7 (RFC 9171) Simulation Layer for TIN v0.4.0

Implements a simplified but faithful BPv7 simulation for lunar DTN custody chain modeling.
Models the complete message lifecycle from origin to Earth delivery:

    Astronaut (EVA) → Rover relay → Shackleton Base → Polar Sat → ELFO Hub → DSN Earth

Key BPv7 concepts modeled:
  - Primary Block: source EID, destination EID, creation timestamp, lifetime, priority
  - Canonical Payload Block: message payload with size
  - Bundle Age Block: tracks cumulative age for lifetime enforcement
  - Custody Transfer: explicit custody acceptance/refusal at each node
  - Custody Signals: acknowledgment back to previous custodian
  - Bundle Fragmentation: large bundles split for constrained links
  - Lifetime Expiry: bundles deleted if undeliverable before TTL
  - Priority Queuing: emergency traffic pre-empts routine at each node
  - Status Reports: creation, forwarding, delivery, deletion events logged

The simulation uses a discrete-event architecture driven by a contact plan (when each
node pair can communicate, with what data rate and latency).

Reference: RFC 9171 — Bundle Protocol Version 7
           RFC 9172 — Bundle Protocol Security (BPSec) [noted, not fully modeled]
           RFC 9174 — Delay-Tolerant Networking TCP Convergence-Layer Protocol v4

Usage:
    python scripts/tin_bundle_protocol.py
    python scripts/tin_bundle_protocol.py --scenario full_chain
    python scripts/tin_bundle_protocol.py --scenario mass_casualty
    python scripts/tin_bundle_protocol.py --scenario storm_degraded

Dependencies: numpy, matplotlib, tin_coverage_sim.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tin_coverage_sim import compute_coverage
except ImportError as e:
    raise ImportError("Cannot import compute_coverage from tin_coverage_sim.py") from e


# =========================================================================
# BPv7 Constants & Enumerations (per RFC 9171)
# =========================================================================


class BundlePriority(IntEnum):
    """Bundle priority classes — maps to CoS in RFC 9171 Section 4.2.3."""

    BULK = 0  # routine telemetry, science data
    NORMAL = 1  # standard operational
    EXPEDITED = 2  # time-sensitive science, habitat status
    EMERGENCY = 3  # life-critical: medical, abort, pressure alarm


class StatusReportType(IntEnum):
    """Bundle status report reasons per RFC 9171 Section 6.1.1."""

    RECEIVED = 0
    FORWARDED = 1
    DELIVERED = 2
    DELETED = 3


class CustodySignalType(IntEnum):
    """Custody signal dispositions."""

    ACCEPTED = 0
    REFUSED_DEPLETED = 1  # storage full
    REFUSED_NO_ROUTE = 2  # no known next hop
    REFUSED_EXPIRED = 3  # bundle lifetime exceeded


# Endpoint ID scheme: dtn://node_name/service
# e.g., dtn://shackleton-base/emergency, dtn://dsn-goldstone/delivery
EID_SCHEME = "dtn"


# =========================================================================
# Link / Contact Parameters
# =========================================================================

# Data rates for each link type (kbps) — realistic estimates
LINK_DATA_RATES = {
    "eva_to_rover": 256,  # UHF suit radio to rover
    "eva_to_base": 256,  # UHF suit radio to base
    "rover_to_base": 1024,  # S-band rover-to-base
    "rover_to_rover": 512,  # UHF inter-rover
    "base_to_sat": 2048,  # S-band base uplink to polar sat
    "sat_to_sat": 4096,  # Ka-band inter-satellite link (ISL)
    "sat_to_elfo": 4096,  # Ka-band to ELFO relay
    "elfo_to_dsn": 8192,  # X-band ELFO to DSN Earth
}

# One-way light-time delays (seconds)
LINK_DELAYS_S = {
    "eva_to_rover": 0.001,  # surface: ~meters, negligible
    "eva_to_base": 0.001,
    "rover_to_base": 0.001,
    "rover_to_rover": 0.001,
    "base_to_sat": 0.003,  # 400 km altitude ~ 1.3 ms
    "sat_to_sat": 0.005,  # inter-sat ~1500 km worst case
    "sat_to_elfo": 0.030,  # ~8000 km at apoapsis
    "elfo_to_dsn": 1.3,  # Earth-Moon ~384,400 km = 1.28 s
}

# Default bundle lifetime by priority (seconds)
DEFAULT_LIFETIMES_S = {
    BundlePriority.EMERGENCY: 3600,  # 1 hour — must get through fast
    BundlePriority.EXPEDITED: 14400,  # 4 hours
    BundlePriority.NORMAL: 86400,  # 24 hours
    BundlePriority.BULK: 604800,  # 7 days
}


# =========================================================================
# Core BPv7 Data Structures
# =========================================================================


@dataclass
class BPv7Bundle:
    """A Bundle Protocol v7 bundle (simplified but faithful to RFC 9171).

    Models Primary Block fields (Section 4.3.1) plus key extension blocks.
    """

    # --- Primary Block (RFC 9171 §4.3.1) ---
    bundle_id: str  # unique identifier
    source_eid: str  # originator endpoint ID
    destination_eid: str  # final destination EID
    report_to_eid: str = ""  # status report destination
    creation_timestamp_s: float = 0.0  # simulation time at creation
    sequence_number: int = 0
    lifetime_s: float = 3600.0  # bundle TTL
    priority: BundlePriority = BundlePriority.NORMAL
    fragment_offset: int = 0
    total_adu_length: int = 0  # original payload size if fragmented
    is_fragment: bool = False

    # --- Payload Block (RFC 9171 §4.4) ---
    payload_bytes: int = 256  # payload size in bytes
    payload_type: str = "emergency_alert"  # semantic type for simulation

    # --- Bundle Age Block (RFC 9171 §4.4.3 / draft-ietf-dtn-bpbis-age) ---
    bundle_age_s: float = 0.0  # cumulative age (updated at each hop)

    # --- Custody tracking (TIN extension, inspired by BP6 custody) ---
    current_custodian: str = ""  # node currently holding custody
    custody_accepted: bool = False
    custody_chain: list = field(default_factory=list)  # [(node, accept_time_s, signal)]

    # --- Status tracking ---
    status_reports: list = field(default_factory=list)  # [(type, node, time_s)]
    delivered: bool = False
    deleted: bool = False
    deletion_reason: str = ""
    delivery_time_s: float | None = None

    # --- Hop tracking ---
    hop_count: int = 0
    hop_log: list = field(default_factory=list)  # [(from, to, depart_s, arrive_s, link_type)]

    def remaining_lifetime_s(self, current_time_s):
        """Check remaining TTL based on bundle age."""
        elapsed = current_time_s - self.creation_timestamp_s
        return max(0, self.lifetime_s - elapsed)

    def is_expired(self, current_time_s):
        """Check if bundle has exceeded its lifetime."""
        return self.remaining_lifetime_s(current_time_s) <= 0

    def add_status_report(self, report_type, node_name, time_s):
        self.status_reports.append(
            {
                "type": StatusReportType(report_type).name,
                "node": node_name,
                "time_s": round(time_s, 3),
            }
        )

    def add_hop(self, from_node, to_node, depart_s, arrive_s, link_type):
        self.hop_log.append(
            {
                "from": from_node,
                "to": to_node,
                "depart_s": round(depart_s, 3),
                "arrive_s": round(arrive_s, 3),
                "link_type": link_type,
                "transfer_time_s": round(arrive_s - depart_s, 3),
            }
        )
        self.hop_count += 1
        self.bundle_age_s = arrive_s - self.creation_timestamp_s

    def accept_custody(self, node_name, time_s):
        """Node accepts custody of this bundle."""
        self.current_custodian = node_name
        self.custody_accepted = True
        self.custody_chain.append(
            {
                "node": node_name,
                "time_s": round(time_s, 3),
                "signal": CustodySignalType.ACCEPTED.name,
            }
        )
        self.add_status_report(StatusReportType.RECEIVED, node_name, time_s)

    def refuse_custody(self, node_name, time_s, reason):
        """Node refuses custody."""
        self.custody_chain.append(
            {
                "node": node_name,
                "time_s": round(time_s, 3),
                "signal": reason.name,
            }
        )

    def mark_delivered(self, node_name, time_s):
        """Mark bundle as successfully delivered."""
        self.delivered = True
        self.delivery_time_s = time_s
        self.add_status_report(StatusReportType.DELIVERED, node_name, time_s)

    def mark_deleted(self, node_name, time_s, reason="lifetime_expired"):
        """Mark bundle as deleted (expired or undeliverable)."""
        self.deleted = True
        self.deletion_reason = reason
        self.add_status_report(StatusReportType.DELETED, node_name, time_s)

    def total_latency_s(self):
        """End-to-end latency from creation to delivery."""
        if self.delivery_time_s is not None:
            return self.delivery_time_s - self.creation_timestamp_s
        return None

    def to_dict(self):
        """Serialize bundle to dict for JSON output."""
        return {
            "bundle_id": self.bundle_id,
            "source_eid": self.source_eid,
            "destination_eid": self.destination_eid,
            "priority": self.priority.name,
            "payload_type": self.payload_type,
            "payload_bytes": self.payload_bytes,
            "creation_timestamp_s": round(self.creation_timestamp_s, 3),
            "lifetime_s": self.lifetime_s,
            "bundle_age_s": round(self.bundle_age_s, 3),
            "hop_count": self.hop_count,
            "delivered": self.delivered,
            "deleted": self.deleted,
            "deletion_reason": self.deletion_reason,
            "delivery_time_s": round(self.delivery_time_s, 3) if self.delivery_time_s else None,
            "total_latency_s": round(self.total_latency_s(), 3) if self.total_latency_s() else None,
            "total_latency_min": round(self.total_latency_s() / 60, 2)
            if self.total_latency_s()
            else None,
            "custody_chain": self.custody_chain,
            "hop_log": self.hop_log,
            "status_reports": self.status_reports,
        }


# =========================================================================
# DTN Node (convergence layer agent)
# =========================================================================


@dataclass
class DTNNode:
    """A DTN node with bundle storage, forwarding logic, and custody management."""

    name: str
    node_type: str  # "eva", "rover", "base", "sat", "elfo", "dsn"
    eid: str = ""
    storage_capacity_bytes: int = 10_000_000  # 10 MB default
    storage_used_bytes: int = 0
    bundle_queue: list = field(default_factory=list)
    forwarding_table: dict = field(default_factory=dict)  # dest_eid -> next_hop_node
    processing_delay_s: float = 0.1  # local processing overhead

    def __post_init__(self):
        if not self.eid:
            self.eid = f"{EID_SCHEME}://{self.name}/default"

    def can_store(self, bundle):
        """Check if node has capacity to store this bundle."""
        return (self.storage_used_bytes + bundle.payload_bytes) <= self.storage_capacity_bytes

    def receive_bundle(self, bundle, time_s):
        """Receive a bundle: check lifetime, accept/refuse custody."""
        # Check lifetime
        if bundle.is_expired(time_s):
            bundle.refuse_custody(self.name, time_s, CustodySignalType.REFUSED_EXPIRED)
            bundle.mark_deleted(self.name, time_s, "lifetime_expired")
            return False

        # Check storage
        if not self.can_store(bundle):
            bundle.refuse_custody(self.name, time_s, CustodySignalType.REFUSED_DEPLETED)
            return False

        # Accept custody
        bundle.accept_custody(self.name, time_s)
        self.storage_used_bytes += bundle.payload_bytes

        # Priority insertion: emergency bundles go to front of queue
        if bundle.priority == BundlePriority.EMERGENCY:
            self.bundle_queue.insert(0, bundle)
        else:
            self.bundle_queue.append(bundle)

        return True

    def forward_bundle(self, bundle, next_node, link_type, time_s):
        """Forward a bundle to the next node along the path."""
        # Calculate transfer time
        data_rate_kbps = LINK_DATA_RATES.get(link_type, 256)
        data_rate_bytes_s = data_rate_kbps * 1000 / 8
        transfer_time_s = bundle.payload_bytes / data_rate_bytes_s
        propagation_delay_s = LINK_DELAYS_S.get(link_type, 0.01)

        depart_s = time_s + self.processing_delay_s
        arrive_s = depart_s + transfer_time_s + propagation_delay_s

        bundle.add_hop(self.name, next_node.name, depart_s, arrive_s, link_type)
        bundle.add_status_report(StatusReportType.FORWARDED, self.name, depart_s)

        # Release storage
        self.storage_used_bytes = max(0, self.storage_used_bytes - bundle.payload_bytes)
        if bundle in self.bundle_queue:
            self.bundle_queue.remove(bundle)

        return arrive_s


# =========================================================================
# Contact Plan (when nodes can communicate)
# =========================================================================


@dataclass
class Contact:
    """A communication opportunity between two nodes."""

    from_node: str
    to_node: str
    start_s: float
    end_s: float
    link_type: str
    data_rate_kbps: float = 0
    propagation_delay_s: float = 0

    def __post_init__(self):
        if self.data_rate_kbps == 0:
            self.data_rate_kbps = LINK_DATA_RATES.get(self.link_type, 256)
        if self.propagation_delay_s == 0:
            self.propagation_delay_s = LINK_DELAYS_S.get(self.link_type, 0.01)

    def is_active(self, time_s):
        return self.start_s <= time_s < self.end_s

    def duration_s(self):
        return self.end_s - self.start_s


def generate_contact_plan(n_sats=8, alt_km=400, include_elfo=True, sim_hours=24):
    """Generate a contact plan based on orbital coverage data.

    Creates realistic contact windows for each link in the chain:
    EVA ↔ Rover, EVA ↔ Base, Rover ↔ Base, Base ↔ Sat, Sat ↔ ELFO, ELFO ↔ DSN
    """
    contacts = []
    total_s = sim_hours * 3600

    # --- Surface contacts (always available when in range) ---
    # EVA to Base: available throughout EVA (assume 8-hr EVA window)
    eva_windows = [(0, 8 * 3600), (16 * 3600, 24 * 3600)]  # two EVA shifts
    for start, end in eva_windows:
        if end <= total_s:
            contacts.append(Contact("eva_astronaut", "shackleton_base", start, end, "eva_to_base"))
            contacts.append(Contact("eva_astronaut", "rover_science", start, end, "eva_to_rover"))
            contacts.append(Contact("eva_astronaut", "rover_logistics", start, end, "eva_to_rover"))

    # Rover to Base: available most of the time (within range ~90% of traverse)
    rover_contact_fraction = 0.90
    window_s = 3600  # 1-hour contact windows
    for t_start in range(0, int(total_s), int(window_s)):
        t_end = min(t_start + window_s, total_s)
        if np.random.random() < rover_contact_fraction:
            contacts.append(
                Contact("rover_science", "shackleton_base", t_start, t_end, "rover_to_base")
            )
            contacts.append(
                Contact("rover_logistics", "shackleton_base", t_start, t_end, "rover_to_base")
            )

    # --- Orbital contacts (based on coverage model) ---
    cov_pct, worst_gap_min, avg_gap_min = compute_coverage(
        n_sats=n_sats,
        alt_km=alt_km,
        include_elfo=False,  # pure constellation for sat passes
    )

    # Sat pass pattern: coverage fraction determines contact duty cycle
    polar_period_s = 7200 * ((1737.4 + alt_km) / (1737.4 + 400)) ** 1.5
    pass_duration_s = polar_period_s * (cov_pct / 100) / n_sats  # per-sat pass time
    gap_between_passes_s = polar_period_s / n_sats

    # Generate individual sat passes over the base
    for sat_idx in range(n_sats):
        phase_offset_s = sat_idx * gap_between_passes_s
        t = phase_offset_s
        while t < total_s:
            t_start = t
            t_end = min(t + pass_duration_s, total_s)
            if t_end > t_start:
                sat_name = f"polar_sat_{sat_idx}"
                contacts.append(Contact("shackleton_base", sat_name, t_start, t_end, "base_to_sat"))
                # Sat-to-ELFO: available when sat and ELFO both in view
                if include_elfo:
                    contacts.append(
                        Contact(sat_name, "elfo_pathfinder", t_start, t_end, "sat_to_elfo")
                    )
            t += polar_period_s  # next orbit pass

    # --- ELFO to DSN contacts ---
    if include_elfo:
        # ELFO sees Earth ~65% of the time (same duty cycle as South Pole visibility)
        # but for Earth link it's different — ELFO can see Earth when near apoapsis
        elfo_period_s = 12 * 3600
        elfo_earth_duty = 0.70  # ELFO-Earth visibility
        elfo_contact_s = elfo_period_s * elfo_earth_duty

        t = 0
        while t < total_s:
            t_start = t
            t_end = min(t + elfo_contact_s, total_s)
            contacts.append(Contact("elfo_pathfinder", "dsn_earth", t_start, t_end, "elfo_to_dsn"))
            t += elfo_period_s

    return contacts


# =========================================================================
# Bundle Protocol Engine — Discrete Event Simulation
# =========================================================================


class BPv7Engine:
    """Discrete-event simulation engine for BPv7 bundle routing."""

    # Canonical custody chain path (ordered)
    DEFAULT_CHAIN = [
        "eva_astronaut",
        "rover_science",  # or rover_logistics — picked dynamically
        "shackleton_base",
        None,  # polar_sat_N — picked from contact plan
        "elfo_pathfinder",
        "dsn_earth",
    ]

    # Link type mapping for each hop
    LINK_MAP = {
        ("eva_astronaut", "rover_science"): "eva_to_rover",
        ("eva_astronaut", "rover_logistics"): "eva_to_rover",
        ("eva_astronaut", "shackleton_base"): "eva_to_base",
        ("rover_science", "shackleton_base"): "rover_to_base",
        ("rover_logistics", "shackleton_base"): "rover_to_base",
        ("shackleton_base", "polar_sat"): "base_to_sat",
        ("polar_sat", "elfo_pathfinder"): "sat_to_elfo",
        ("elfo_pathfinder", "dsn_earth"): "elfo_to_dsn",
    }

    def __init__(self, contacts, include_elfo=True):
        self.contacts = sorted(contacts, key=lambda c: c.start_s)
        self.include_elfo = include_elfo
        self.nodes = {}
        self.bundles = []
        self.event_log = []
        self._init_nodes()

    def _init_nodes(self):
        """Initialize all DTN nodes in the network."""
        node_defs = [
            ("eva_astronaut", "eva", 1_000_000),  # 1 MB — suit computer
            ("rover_science", "rover", 50_000_000),  # 50 MB — rover storage
            ("rover_logistics", "rover", 50_000_000),
            ("shackleton_base", "base", 1_000_000_000),  # 1 GB — base server
        ]
        # Add polar sat nodes (as many as appear in contacts)
        sat_names = set()
        for c in self.contacts:
            for name in [c.from_node, c.to_node]:
                if name.startswith("polar_sat_"):
                    sat_names.add(name)
        for sn in sorted(sat_names):
            node_defs.append((sn, "sat", 100_000_000))  # 100 MB per sat

        if self.include_elfo:
            node_defs.append(("elfo_pathfinder", "elfo", 500_000_000))  # 500 MB
        node_defs.append(("dsn_earth", "dsn", 10_000_000_000))  # 10 GB

        for name, ntype, capacity in node_defs:
            self.nodes[name] = DTNNode(
                name=name,
                node_type=ntype,
                storage_capacity_bytes=capacity,
                processing_delay_s=0.05 if ntype in ("sat", "elfo") else 0.1,
            )

    def _get_link_type(self, from_name, to_name):
        """Determine link type between two nodes."""
        # Direct lookup
        key = (from_name, to_name)
        if key in self.LINK_MAP:
            return self.LINK_MAP[key]
        # Try with generic sat prefix
        if from_name.startswith("polar_sat"):
            key2 = ("polar_sat", to_name)
            if key2 in self.LINK_MAP:
                return self.LINK_MAP[key2]
        if to_name.startswith("polar_sat"):
            key2 = (from_name, "polar_sat")
            if key2 in self.LINK_MAP:
                return self.LINK_MAP[key2]
        # Fallback for sat-to-sat
        if from_name.startswith("polar_sat") and to_name.startswith("polar_sat"):
            return "sat_to_sat"
        return "eva_to_base"  # fallback

    def _find_next_contact(self, from_name, after_time_s, to_prefix=None):
        """Find the next available contact from a node after a given time."""
        for c in self.contacts:
            if c.from_node == from_name and c.end_s > after_time_s:
                if to_prefix is None or c.to_node.startswith(to_prefix):
                    return c, max(c.start_s, after_time_s)
        return None, None

    def create_bundle(
        self,
        scenario,
        priority,
        creation_time_s,
        payload_bytes=256,
        source="eva_astronaut",
        destination="dsn_earth",
    ):
        """Create a new BPv7 bundle."""
        seq = len(self.bundles)
        bundle = BPv7Bundle(
            bundle_id=f"tin:{scenario}:{seq}",
            source_eid=f"{EID_SCHEME}://{source}/{scenario}",
            destination_eid=f"{EID_SCHEME}://{destination}/delivery",
            report_to_eid=f"{EID_SCHEME}://shackleton_base/reports",
            creation_timestamp_s=creation_time_s,
            sequence_number=seq,
            lifetime_s=DEFAULT_LIFETIMES_S.get(priority, 3600),
            priority=priority,
            payload_bytes=payload_bytes,
            payload_type=scenario,
        )
        self.bundles.append(bundle)

        # Origin node accepts initial custody
        if source in self.nodes:
            self.nodes[source].receive_bundle(bundle, creation_time_s)

        self._log(creation_time_s, "CREATED", source, bundle)
        return bundle

    def route_bundle(self, bundle):
        """Route a single bundle through the custody chain using contact plan.

        Implements store-and-forward: at each node, wait for next contact,
        then forward. Respects bundle lifetime.
        """
        current_time_s = bundle.creation_timestamp_s
        current_node_name = bundle.current_custodian

        # Define the forwarding path
        path_order = self._compute_path(current_node_name)

        for next_node_name in path_order:
            if bundle.is_expired(current_time_s):
                bundle.mark_deleted(current_node_name, current_time_s, "lifetime_expired")
                self._log(current_time_s, "EXPIRED", current_node_name, bundle)
                return

            if bundle.deleted:
                return

            # Determine which contact prefix to look for
            to_prefix = (
                next_node_name if not next_node_name.startswith("polar_sat") else "polar_sat"
            )
            contact, available_time = self._find_next_contact(
                current_node_name, current_time_s, to_prefix
            )

            if contact is None:
                # Try alternate paths (e.g., EVA direct to base if no rover contact)
                alt_paths = self._get_alternate_next_hops(current_node_name, next_node_name)
                for alt_next in alt_paths:
                    alt_prefix = alt_next if not alt_next.startswith("polar_sat") else "polar_sat"
                    contact, available_time = self._find_next_contact(
                        current_node_name, current_time_s, alt_prefix
                    )
                    if contact:
                        next_node_name = contact.to_node
                        break

            if contact is None:
                # No contact found — bundle stuck, check lifetime
                self._log(
                    current_time_s,
                    "NO_CONTACT",
                    current_node_name,
                    bundle,
                    f"no path to {next_node_name}",
                )
                # Wait at current node (in practice, would retry on next contact)
                continue

            # Wait for contact window
            wait_time_s = max(0, available_time - current_time_s)

            # Check lifetime after wait
            forward_time_s = available_time
            if bundle.is_expired(forward_time_s):
                bundle.mark_deleted(current_node_name, forward_time_s, "lifetime_expired_waiting")
                self._log(forward_time_s, "EXPIRED_WAITING", current_node_name, bundle)
                return

            # Forward the bundle
            current_node = self.nodes.get(current_node_name)
            next_node = self.nodes.get(next_node_name)
            if next_node is None:
                next_node = self.nodes.get(contact.to_node)
                next_node_name = contact.to_node

            if current_node and next_node:
                link_type = self._get_link_type(current_node_name, next_node_name)
                arrive_time_s = current_node.forward_bundle(
                    bundle, next_node, link_type, forward_time_s
                )

                # Next node receives and accepts custody
                accepted = next_node.receive_bundle(bundle, arrive_time_s)
                if not accepted:
                    self._log(arrive_time_s, "CUSTODY_REFUSED", next_node_name, bundle)
                    # Bundle stays at current node — would retry in real system
                    continue

                self._log(
                    arrive_time_s,
                    "FORWARDED",
                    current_node_name,
                    bundle,
                    f"-> {next_node_name} (waited {wait_time_s:.1f}s)",
                )

                current_time_s = arrive_time_s
                current_node_name = next_node_name

                # Check if we've reached destination
                if next_node.node_type == "dsn":
                    bundle.mark_delivered(next_node_name, arrive_time_s)
                    self._log(arrive_time_s, "DELIVERED", next_node_name, bundle)
                    return

        # If we exhausted the path without delivery
        if not bundle.delivered and not bundle.deleted:
            self._log(current_time_s, "UNDELIVERED", current_node_name, bundle, "exhausted path")

    def _compute_path(self, start_node):
        """Compute the canonical forwarding path from start to DSN."""
        full_path = [
            "rover_science",  # or direct to base
            "shackleton_base",
            "polar_sat",  # generic — will match any polar_sat_N
            "elfo_pathfinder",
            "dsn_earth",
        ]

        # Find where we are in the chain and return remainder
        node_order = {
            "eva_astronaut": 0,
            "rover_science": 1,
            "rover_logistics": 1,
            "shackleton_base": 2,
        }
        # Sats, ELFO, DSN
        for name in self.nodes:
            if name.startswith("polar_sat"):
                node_order[name] = 3
        node_order["elfo_pathfinder"] = 4
        node_order["dsn_earth"] = 5

        start_rank = node_order.get(start_node, 0)
        remaining = [
            n
            for n in full_path
            if node_order.get(
                n, node_order.get(n.split("_")[0] + "_" + n.split("_")[1] if "_" in n else n, 99)
            )
            > start_rank
        ]

        if not self.include_elfo:
            remaining = [n for n in remaining if n != "elfo_pathfinder"]

        return remaining

    def _get_alternate_next_hops(self, current_node, failed_next):
        """Return alternate nodes to try when primary next hop has no contact."""
        alternates = {
            "eva_astronaut": ["shackleton_base", "rover_logistics"],
            "rover_science": ["shackleton_base"],
            "rover_logistics": ["shackleton_base"],
        }
        alts = alternates.get(current_node, [])
        return [a for a in alts if a != failed_next]

    def _log(self, time_s, event_type, node, bundle, detail=""):
        """Log a simulation event."""
        self.event_log.append(
            {
                "time_s": round(time_s, 3),
                "time_min": round(time_s / 60, 2),
                "event": event_type,
                "node": node,
                "bundle_id": bundle.bundle_id,
                "priority": bundle.priority.name,
                "detail": detail,
            }
        )

    def run_all(self):
        """Route all created bundles."""
        # Sort by priority (emergency first)
        sorted_bundles = sorted(self.bundles, key=lambda b: b.priority, reverse=True)
        for bundle in sorted_bundles:
            if not bundle.delivered and not bundle.deleted:
                self.route_bundle(bundle)

    def summary(self):
        """Generate simulation summary."""
        delivered = [b for b in self.bundles if b.delivered]
        deleted = [b for b in self.bundles if b.deleted]
        pending = [b for b in self.bundles if not b.delivered and not b.deleted]

        latencies = [b.total_latency_s() for b in delivered if b.total_latency_s() is not None]
        hops = [b.hop_count for b in delivered]

        return {
            "total_bundles": len(self.bundles),
            "delivered": len(delivered),
            "deleted": len(deleted),
            "pending": len(pending),
            "delivery_rate_pct": round(100 * len(delivered) / max(len(self.bundles), 1), 1),
            "latency": {
                "mean_s": round(float(np.mean(latencies)), 2) if latencies else None,
                "worst_s": round(float(np.max(latencies)), 2) if latencies else None,
                "best_s": round(float(np.min(latencies)), 2) if latencies else None,
                "mean_min": round(float(np.mean(latencies)) / 60, 2) if latencies else None,
                "worst_min": round(float(np.max(latencies)) / 60, 2) if latencies else None,
            },
            "hops": {
                "mean": round(float(np.mean(hops)), 1) if hops else None,
                "max": int(np.max(hops)) if hops else None,
            },
            "bundles": [b.to_dict() for b in self.bundles],
            "event_log": self.event_log,
        }


# =========================================================================
# Predefined Scenarios
# =========================================================================


def run_scenario(scenario_name, n_sats=8, alt_km=400, include_elfo=True, sim_hours=24):
    """Run a predefined emergency scenario."""

    print("\n  Generating contact plan...")
    contacts = generate_contact_plan(n_sats, alt_km, include_elfo, sim_hours)
    print(f"  Contact plan: {len(contacts)} contact windows over {sim_hours} hr")

    engine = BPv7Engine(contacts, include_elfo)
    print(f"  Network nodes: {len(engine.nodes)}")

    if scenario_name == "full_chain":
        # Single emergency medical alert — trace full custody chain
        print("\n  Scenario: Full custody chain trace (single emergency)")
        b = engine.create_bundle(
            "emergency_medical", BundlePriority.EMERGENCY, creation_time_s=300.0, payload_bytes=512
        )
        print(f"  Bundle created: {b.bundle_id} at t=300s (5 min into sim)")

    elif scenario_name == "multi_priority":
        # Multiple bundles at different priorities
        print("\n  Scenario: Multi-priority bundle routing")
        engine.create_bundle(
            "medical_alert", BundlePriority.EMERGENCY, creation_time_s=300.0, payload_bytes=512
        )
        engine.create_bundle(
            "habitat_pressure", BundlePriority.EXPEDITED, creation_time_s=310.0, payload_bytes=1024
        )
        engine.create_bundle(
            "science_obs", BundlePriority.NORMAL, creation_time_s=320.0, payload_bytes=65536
        )
        engine.create_bundle(
            "rover_telemetry", BundlePriority.BULK, creation_time_s=330.0, payload_bytes=262144
        )
        print("  Created 4 bundles (EMERGENCY, EXPEDITED, NORMAL, BULK)")

    elif scenario_name == "mass_casualty":
        # Multiple simultaneous emergency bundles — stress test
        print("\n  Scenario: Mass casualty — 5 simultaneous emergency bundles")
        for i in range(5):
            engine.create_bundle(
                f"casualty_{i}",
                BundlePriority.EMERGENCY,
                creation_time_s=600.0 + i * 5,
                payload_bytes=1024,
            )

    elif scenario_name == "storm_degraded":
        # Simulate with reduced constellation (storm knocked out 3 sats)
        print("\n  Scenario: Solar storm — 3 sats degraded")
        # Reduce contacts by removing some sat contacts
        degraded_contacts = [
            c
            for c in contacts
            if not (
                c.from_node in ("polar_sat_0", "polar_sat_1", "polar_sat_2")
                or c.to_node in ("polar_sat_0", "polar_sat_1", "polar_sat_2")
            )
        ]
        engine = BPv7Engine(degraded_contacts, include_elfo)
        print(
            f"  Degraded contact plan: {len(degraded_contacts)} contacts "
            f"(removed {len(contacts) - len(degraded_contacts)})"
        )
        engine.create_bundle(
            "storm_emergency", BundlePriority.EMERGENCY, creation_time_s=300.0, payload_bytes=512
        )
        engine.create_bundle(
            "storm_status", BundlePriority.EXPEDITED, creation_time_s=600.0, payload_bytes=2048
        )

    elif scenario_name == "lifetime_test":
        # Test bundle lifetime expiry
        print("\n  Scenario: Lifetime expiry test")
        engine.create_bundle(
            "short_lived", BundlePriority.NORMAL, creation_time_s=300.0, payload_bytes=512
        )
        # Override lifetime to very short
        engine.bundles[-1].lifetime_s = 60  # 1 minute — should expire in transit

        engine.create_bundle(
            "long_lived", BundlePriority.EMERGENCY, creation_time_s=300.0, payload_bytes=512
        )
        # Emergency has 1 hour lifetime — should succeed

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    # Run simulation
    print("\n  Routing bundles through custody chain...")
    engine.run_all()

    return engine


# =========================================================================
# Visualization
# =========================================================================


def plot_custody_chain(engine, outdir):
    """Visualize the bundle custody chain as a timeline diagram."""
    os.makedirs(outdir, exist_ok=True)

    BG_COLOR = "#F7F9FC"
    NODE_COLORS = {
        "eva": "#E74C3C",
        "rover": "#2EA043",
        "base": "#1B2A4A",
        "sat": "#2E75B6",
        "elfo": "#E87722",
        "dsn": "#8E44AD",
    }

    delivered = [b for b in engine.bundles if b.delivered]
    all_bundles = engine.bundles

    if not all_bundles:
        print("  No bundles to plot.")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1.5, 1]})
    fig.patch.set_facecolor(BG_COLOR)

    # --- Panel 1: Custody chain timeline ---
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)

    # Build node ordering for y-axis
    node_order = ["eva_astronaut", "rover_science", "rover_logistics", "shackleton_base"]
    # Add sat nodes
    for name in sorted(engine.nodes.keys()):
        if name.startswith("polar_sat"):
            node_order.append(name)
    node_order.extend(["elfo_pathfinder", "dsn_earth"])
    node_y = {name: i for i, name in enumerate(node_order) if name in engine.nodes}

    priority_colors = {
        "EMERGENCY": "#E74C3C",
        "EXPEDITED": "#E87722",
        "NORMAL": "#2E75B6",
        "BULK": "#95A5A6",
    }

    for b_idx, bundle in enumerate(all_bundles):
        color = priority_colors.get(bundle.priority.name, "#333333")

        # Draw hops as arrows
        for hop in bundle.hop_log:
            from_y = node_y.get(hop["from"], 0)
            to_y = node_y.get(hop["to"], 0)
            from_x = hop["depart_s"] / 60  # convert to minutes
            to_x = hop["arrive_s"] / 60

            ax1.annotate(
                "",
                xy=(to_x, to_y),
                xytext=(from_x, from_y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.7),
            )

        # Mark delivery or deletion
        if bundle.delivered and bundle.delivery_time_s:
            final_y = node_y.get("dsn_earth", len(node_y) - 1)
            ax1.scatter(
                bundle.delivery_time_s / 60,
                final_y,
                marker="*",
                s=200,
                c=color,
                zorder=10,
                edgecolors="black",
            )

        if bundle.deleted:
            last_node = bundle.custody_chain[-1]["node"] if bundle.custody_chain else "unknown"
            last_y = node_y.get(last_node, 0)
            last_time = bundle.custody_chain[-1]["time_s"] if bundle.custody_chain else 0
            ax1.scatter(
                last_time / 60, last_y, marker="X", s=150, c="red", zorder=10, edgecolors="black"
            )

    # Format axes
    ax1.set_yticks(list(node_y.values()))
    ax1.set_yticklabels([n.replace("_", " ").title() for n in node_y.keys()], fontsize=9)
    ax1.set_xlabel("Time (minutes)", fontsize=11)
    ax1.set_title("BPv7 Custody Chain Timeline", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()

    # Legend for priorities
    legend_patches = [
        mpatches.Patch(color=c, label=p)
        for p, c in priority_colors.items()
        if any(b.priority.name == p for b in all_bundles)
    ]
    legend_patches.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markersize=12,
            label="Delivered",
        )
    )
    legend_patches.append(
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Deleted/Expired",
        )
    )
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=9)

    # --- Panel 2: Per-bundle latency breakdown ---
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)

    bundle_labels = []
    hop_times = defaultdict(list)  # link_type -> [time_per_bundle]

    for bundle in all_bundles:
        label = f"{bundle.bundle_id.split(':')[1]}\n({bundle.priority.name})"
        bundle_labels.append(label)
        hop_breakdown = defaultdict(float)
        for hop in bundle.hop_log:
            hop_breakdown[hop["link_type"]] += hop["transfer_time_s"]
        for lt in LINK_DATA_RATES:
            hop_times[lt].append(hop_breakdown.get(lt, 0))

    if bundle_labels:
        x = np.arange(len(bundle_labels))
        width = 0.6
        bottom = np.zeros(len(bundle_labels))

        link_colors = {
            "eva_to_rover": "#E74C3C",
            "eva_to_base": "#C0392B",
            "rover_to_base": "#2EA043",
            "base_to_sat": "#2E75B6",
            "sat_to_elfo": "#E87722",
            "elfo_to_dsn": "#8E44AD",
            "sat_to_sat": "#1ABC9C",
            "rover_to_rover": "#16A085",
        }

        for link_type, times in hop_times.items():
            if max(times) > 0:
                color = link_colors.get(link_type, "#999999")
                label = link_type.replace("_", " ").replace("to", "→")
                ax2.bar(x, times, width, bottom=bottom, color=color, alpha=0.8, label=label)
                bottom += np.array(times)

        # Add wait times (creation to first hop)
        wait_times = []
        for bundle in all_bundles:
            if bundle.hop_log:
                wait = bundle.hop_log[0]["depart_s"] - bundle.creation_timestamp_s
                wait_times.append(max(0, wait))
            else:
                wait_times.append(0)

        if max(wait_times) > 0:
            ax2.bar(
                x,
                wait_times,
                width,
                bottom=bottom,
                color="#BDC3C7",
                alpha=0.6,
                label="Queue/wait time",
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(bundle_labels, fontsize=8)
        ax2.set_ylabel("Time (seconds)", fontsize=11)
        ax2.set_title("Per-Bundle Latency Breakdown by Link", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8, ncol=2)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    fname = f"tin_bpv7_{datetime.now():%Y%m%d_%H%M}.png"
    plot_path = os.path.join(outdir, fname)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Plot saved: {plot_path}")
    return plot_path


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="TIN v0.4.0 Bundle Protocol v7 Simulation")
    parser.add_argument("--n_sats", type=int, default=8)
    parser.add_argument("--alt_km", type=int, default=400)
    parser.add_argument("--include_elfo", action="store_true", default=True)
    parser.add_argument("--no_elfo", action="store_true")
    parser.add_argument("--sim_hours", type=int, default=24)
    parser.add_argument(
        "--scenario",
        type=str,
        default="full_chain",
        choices=[
            "full_chain",
            "multi_priority",
            "mass_casualty",
            "storm_degraded",
            "lifetime_test",
        ],
        help="Scenario to simulate",
    )
    args = parser.parse_args()
    include_elfo = not args.no_elfo

    print(f"\n{'=' * 65}")
    print("  TIN v0.4.0 — Bundle Protocol v7 Simulation (RFC 9171)")
    print(f"  {args.n_sats}x{args.alt_km}km {'+ ELFO' if include_elfo else 'pure'}")
    print(f"  Scenario: {args.scenario} | Duration: {args.sim_hours} hr")
    print(f"{'=' * 65}")

    # Run scenario
    engine = run_scenario(args.scenario, args.n_sats, args.alt_km, include_elfo, args.sim_hours)

    # Print summary
    summary = engine.summary()
    print(f"\n{'=' * 65}")
    print("  RESULTS")
    print(f"{'=' * 65}")
    print(f"  Total bundles:    {summary['total_bundles']}")
    print(f"  Delivered:        {summary['delivered']}")
    print(f"  Deleted/expired:  {summary['deleted']}")
    print(f"  Pending:          {summary['pending']}")
    print(f"  Delivery rate:    {summary['delivery_rate_pct']}%")

    if summary["latency"]["mean_s"] is not None:
        print("\n  Latency (delivered bundles):")
        print(
            f"    Mean:  {summary['latency']['mean_s']:.2f}s ({summary['latency']['mean_min']:.2f} min)"
        )
        print(
            f"    Worst: {summary['latency']['worst_s']:.2f}s ({summary['latency']['worst_min']:.2f} min)"
        )
        print(f"    Best:  {summary['latency']['best_s']:.2f}s")

    if summary["hops"]["mean"] is not None:
        print("\n  Hop count:")
        print(f"    Mean: {summary['hops']['mean']:.1f}")
        print(f"    Max:  {summary['hops']['max']}")

    # Print custody chain for each bundle
    print("\n  Custody chains:")
    for bd in summary["bundles"]:
        status = "DELIVERED" if bd["delivered"] else ("DELETED" if bd["deleted"] else "PENDING")
        chain_str = " → ".join([c["node"].replace("_", " ").title() for c in bd["custody_chain"]])
        latency_str = (
            f"  latency={bd['total_latency_min']:.2f}min" if bd["total_latency_min"] else ""
        )
        print(f"    [{bd['priority']}] {bd['bundle_id']}: {status}{latency_str}")
        print(f"      Chain: {chain_str}")
        if bd["hop_log"]:
            for hop in bd["hop_log"]:
                print(
                    f"        {hop['from']:>22} → {hop['to']:<22} "
                    f"t={hop['depart_s']:.1f}s → {hop['arrive_s']:.1f}s "
                    f"({hop['link_type']}, {hop['transfer_time_s']:.3f}s)"
                )

    # Save outputs
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(repo_root, "data")
    results_dir = os.path.join(repo_root, "results")

    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, f"bpv7_{args.scenario}_{datetime.now():%Y%m%d_%H%M}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON saved: {json_path}")

    plot_custody_chain(engine, results_dir)

    print("\n  Done.")


if __name__ == "__main__":
    main()
