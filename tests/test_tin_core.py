"""tests/test_tin_core.py — Validation tests for tin.core

Mapped to study gap IDs:
  B1  — 5-state custody FSM transitions
  N1  — Fragment-group custody + aggregate state
  N5  — SPE pre-flush
  B2  — RW-CGR stochastic routing
  N8  — Lunar constellation config integrity
  A5  — This file IS the A5 closure (CI test suite)
"""

import pytest

from tin.config.lunar_default import LUNAR_BODY, get_lunar_constellation
from tin.core.dtn import (
    DEFAULT_LIFETIMES,
    PARTIAL_AGGREGATE,
    PRIORITY_BULK,
    PRIORITY_EMERGENCY,
    PRIORITY_NAMES,
    PRIORITY_NORMAL,
    RELEASE_CONFIRMATIONS,
    Bundle,
    CustodyNode,
    CustodyState,
    DTNNetwork,
    FragmentGroup,
    adaptive_custody_timeout,
    bloom_contains,
    bloom_digest,
    custody_transfer_mode,
)
from tin.core.routing import RW_CGR, Contact


# =========================================================================
# B1: 5-State Custody FSM
# =========================================================================
class TestCustodyFSM:
    """B1: Validate all 5 state transitions."""

    def test_initial_state_is_holding(self):
        b = Bundle(bundle_id="b1")
        assert b.custody_state == CustodyState.H

    def test_accept_custody_sets_holding(self):
        b = Bundle(bundle_id="b1")
        b.custody_state = CustodyState.O  # simulate outstanding
        b.accept_custody("node-A", 100.0)
        assert b.custody_state == CustodyState.H
        assert b.current_custodian == "node-A"

    def test_custody_chain_records(self):
        b = Bundle(bundle_id="b1")
        b.accept_custody("node-A", 10.0)
        b.accept_custody("node-B", 20.0)
        assert len(b.custody_chain) == 2
        assert b.custody_chain[0]["node"] == "node-A"
        assert b.custody_chain[1]["node"] == "node-B"

    def test_expired_bundle_transitions_to_X(self):
        node = CustodyNode("n1")
        b = Bundle(bundle_id="b1", created_s=0.0, lifetime_s=100.0)
        accepted = node.accept_custody(b, 200.0)  # expired
        assert not accepted
        assert b.custody_state == CustodyState.X
        assert b.deleted

    def test_storage_overflow_transitions_to_P(self):
        node = CustodyNode("n1", storage_bytes=100)
        b = Bundle(bundle_id="b1", size_bytes=200)
        accepted = node.accept_custody(b, 0.0)
        assert not accepted
        assert b.custody_state == CustodyState.P

    def test_accept_custody_is_idempotent_for_same_node(self):
        node = CustodyNode("n1")
        b = Bundle(bundle_id="b1", size_bytes=100)
        assert node.accept_custody(b, 0.0)
        assert node.accept_custody(b, 1.0)
        assert node.bundles == [b]
        assert node.used_bytes == 100
        assert node.custody_accepted == 1
        assert len(b.custody_chain) == 1

    def test_forward_requires_sender_custody(self):
        sender = CustodyNode("sender")
        receiver = CustodyNode("receiver")
        b = Bundle(bundle_id="b1", size_bytes=100)
        arrival = sender.forward_bundle(b, receiver, t_s=0.0)
        assert arrival == -1.0
        assert b not in receiver.bundles
        assert receiver.used_bytes == 0
        assert b.current_custodian == ""

    def test_forward_success_transfers_custody(self):
        n1 = CustodyNode("n1")
        n2 = CustodyNode("n2")
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0)
        n1.accept_custody(b, 0.0)
        assert b.custody_state == CustodyState.H
        n1.forward_bundle(b, n2, t_s=1.0, data_rate_kbps=256.0)
        # Atomic receipt: bundle goes directly H→H at new custodian
        assert b.custody_state == CustodyState.H
        assert b.current_custodian == "n2"

    def test_mark_delivered_transitions_to_R(self):
        b = Bundle(bundle_id="b1")
        b.mark_delivered(100.0)
        assert b.custody_state == CustodyState.R
        assert b.delivered
        assert b.delivery_time_s == 100.0

    def test_forward_refusal_restores_to_sender(self):
        """P0: Bundle must return to sender if next hop refuses custody."""
        sender = CustodyNode("n1", storage_bytes=10_000)
        receiver = CustodyNode("n2", storage_bytes=50)  # too small
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0)
        sender.accept_custody(b, 0.0)
        assert len(sender.bundles) == 1
        assert sender.used_bytes == 1000

        arrival = sender.forward_bundle(b, receiver, t_s=1.0, data_rate_kbps=256.0)
        assert arrival == -1.0  # refusal
        # Bundle must be back in sender, state HOLDING, storage accounted
        assert b in sender.bundles
        assert sender.used_bytes == 1000
        assert b.custody_state == CustodyState.H
        assert b not in receiver.bundles

    def test_forward_refusal_leaves_hop_log_clean(self):
        """P0: Refused forward must not contaminate hop_log or age_s."""
        sender = CustodyNode("n1", storage_bytes=10_000)
        receiver = CustodyNode("n2", storage_bytes=50)  # too small
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0)
        sender.accept_custody(b, 0.0)
        age_before = b.age_s

        arrival = sender.forward_bundle(b, receiver, t_s=1.0, data_rate_kbps=256.0)
        assert arrival == -1.0
        assert b.hop_count == 0
        assert b.hop_log == []
        assert b.age_s == age_before

    def test_forward_success_records_hop(self):
        """Successful forward must record exactly one hop with correct custodian."""
        sender = CustodyNode("n1", storage_bytes=10_000)
        receiver = CustodyNode("n2", storage_bytes=10_000)
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0)
        sender.accept_custody(b, 0.0)

        arrival = sender.forward_bundle(b, receiver, t_s=1.0, data_rate_kbps=256.0)
        assert arrival > 0
        assert b.hop_count == 1
        assert len(b.hop_log) == 1
        assert b.hop_log[0]["from"] == "n1"
        assert b.hop_log[0]["to"] == "n2"
        assert b.age_s > 0

    def test_forward_expired_frees_sender_storage(self):
        """Bundle that expires in transit should free sender storage."""
        sender = CustodyNode("n1")
        receiver = CustodyNode("n2")
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0, lifetime_s=0.5)
        sender.accept_custody(b, 0.0)
        assert sender.used_bytes == 1000

        arrival = sender.forward_bundle(
            b, receiver, t_s=1.0, data_rate_kbps=256.0, propagation_delay_s=1.0
        )
        assert arrival == -1.0
        assert b.deleted
        assert b.custody_state == CustodyState.X
        assert b not in sender.bundles
        assert sender.used_bytes == 0

    def test_forward_refusal_no_state_mutation(self):
        """Atomic receipt: storage refusal must not touch bundle state at all."""
        sender = CustodyNode("n1", storage_bytes=10_000)
        receiver = CustodyNode("n2", storage_bytes=50)  # too small
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0)
        sender.accept_custody(b, 0.0)
        chain_before = len(b.custody_chain)
        state_before = b.custody_state
        custodian_before = b.current_custodian

        arrival = sender.forward_bundle(b, receiver, t_s=1.0, data_rate_kbps=256.0)
        assert arrival == -1.0
        # Bundle state must be identical to before the attempt
        assert b.custody_state == state_before
        assert b.current_custodian == custodian_before
        assert len(b.custody_chain) == chain_before

    def test_custody_chain_length_equals_hops_plus_one(self):
        """Invariant: custody_chain has source acceptance + one entry per hop."""
        n1 = CustodyNode("n1")
        n2 = CustodyNode("n2")
        n3 = CustodyNode("n3")
        b = Bundle(bundle_id="b1", size_bytes=100, created_s=0.0)
        n1.accept_custody(b, 0.0)
        assert len(b.custody_chain) == 1  # source acceptance
        assert b.hop_count == 0

        n1.forward_bundle(b, n2, t_s=1.0)
        assert len(b.custody_chain) == 2  # source + hop 1
        assert b.hop_count == 1

        n2.forward_bundle(b, n3, t_s=2.0)
        assert len(b.custody_chain) == 3  # source + hop 1 + hop 2
        assert b.hop_count == 2
        # Chain always equals hops + 1
        assert len(b.custody_chain) == b.hop_count + 1

    def test_can_accept_does_not_mutate(self):
        """_can_accept is a pure check — no side effects."""
        node = CustodyNode("n1", storage_bytes=50)
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0)
        state_before = b.custody_state
        deleted_before = b.deleted
        used_before = node.used_bytes
        refused_before = node.custody_refused

        reason = node._can_accept(b, 0.0)
        assert reason == "storage"
        # Nothing changed
        assert b.custody_state == state_before
        assert b.deleted == deleted_before
        assert node.used_bytes == used_before
        assert node.custody_refused == refused_before

    def test_single_custodian_invariant(self):
        """Bundle must be in exactly one node's list after forward."""
        n1 = CustodyNode("n1")
        n2 = CustodyNode("n2")
        n3 = CustodyNode("n3")
        b = Bundle(bundle_id="b1", size_bytes=100, created_s=0.0)
        n1.accept_custody(b, 0.0)
        assert b in n1.bundles
        assert b not in n2.bundles

        n1.forward_bundle(b, n2, t_s=1.0)
        assert b not in n1.bundles
        assert b in n2.bundles
        assert b not in n3.bundles

        n2.forward_bundle(b, n3, t_s=2.0)
        assert b not in n1.bundles
        assert b not in n2.bundles
        assert b in n3.bundles

    def test_storage_conservation(self):
        """Sender bytes freed must equal receiver bytes gained."""
        n1 = CustodyNode("n1")
        n2 = CustodyNode("n2")
        b = Bundle(bundle_id="b1", size_bytes=1000, created_s=0.0)
        n1.accept_custody(b, 0.0)
        total_before = n1.used_bytes + n2.used_bytes
        assert n1.used_bytes == 1000
        assert n2.used_bytes == 0

        n1.forward_bundle(b, n2, t_s=1.0)
        total_after = n1.used_bytes + n2.used_bytes
        assert total_before == total_after
        assert n1.used_bytes == 0
        assert n2.used_bytes == 1000

    def test_fragment_expiry_during_transit(self):
        """All fragment groups must transition to X on transit expiry."""
        sender = CustodyNode("n1")
        receiver = CustodyNode("n2")
        b = Bundle(
            bundle_id="b1",
            size_bytes=1000,
            created_s=0.0,
            lifetime_s=0.5,
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 500),
                FragmentGroup("g1", 500, 1000),
            ],
        )
        sender.accept_custody(b, 0.0)
        # Forward with enough delay to expire
        arrival = sender.forward_bundle(
            b, receiver, t_s=1.0, data_rate_kbps=256.0, propagation_delay_s=1.0
        )
        assert arrival == -1.0
        assert b.deleted
        assert b.custody_state == CustodyState.X
        for g in b.fragment_groups:
            assert g.state == CustodyState.X
        assert b not in sender.bundles
        assert b not in receiver.bundles

    def test_emergency_priority_insertion(self):
        node = CustodyNode("n1")
        b_normal = Bundle(bundle_id="b1", priority=PRIORITY_NORMAL)
        b_emergency = Bundle(bundle_id="b2", priority=PRIORITY_EMERGENCY)
        node.accept_custody(b_normal, 0.0)
        node.accept_custody(b_emergency, 1.0)
        assert node.bundles[0].bundle_id == "b2"  # emergency at front


# =========================================================================
# B1 §1.3: Adaptive Custody Timeout τ(t)
# =========================================================================
class TestAdaptiveCustodyTimeout:
    """B1 §1.3: τ(t) = RTLT + T_proc + N_ret × T_ACK_slot, three-mode policy."""

    RTLT = 1320.0  # 22 min relay-to-Earth

    def test_clean_link_floor(self):
        # BER=0 → p_ACK=1 → N_ret=1: τ = RTLT + 5 + max(RTLT/2, 10)
        tau = adaptive_custody_timeout(self.RTLT, ber=0.0)
        assert tau == pytest.approx(1320.0 + 5.0 + 660.0)

    def test_ack_slot_floor_for_short_rtlt(self):
        # T_ACK_slot = max(RTLT/2, 10 s): floor binds at short RTLT
        tau = adaptive_custody_timeout(10.0, ber=0.0)
        assert tau == pytest.approx(10.0 + 5.0 + 10.0)

    def test_monotonic_in_ber(self):
        taus = [
            adaptive_custody_timeout(self.RTLT, ber=b)
            for b in (0.0, 1e-6, 1e-4, 1e-3)
        ]
        assert taus == sorted(taus)
        assert taus[-1] > taus[0]

    def test_dead_link_is_infinite(self):
        assert adaptive_custody_timeout(self.RTLT, ber=1.0) == float("inf")

    def test_mode_normal(self):
        # τ ≤ 0.5 × T_contact → NORMAL
        assert custody_transfer_mode(100.0, t_contact_s=300.0) == "NORMAL"

    def test_mode_cautious(self):
        # 0.5 × T_contact < τ ≤ T_contact → CAUTIOUS
        assert custody_transfer_mode(200.0, t_contact_s=300.0) == "CAUTIOUS"

    def test_mode_local_hold(self):
        # τ > T_contact → LOCAL_HOLD: do not attempt transfer
        assert custody_transfer_mode(400.0, t_contact_s=300.0) == "LOCAL_HOLD"

    def test_mode_boundaries(self):
        assert custody_transfer_mode(300.0, t_contact_s=300.0) == "CAUTIOUS"
        assert custody_transfer_mode(150.0, t_contact_s=300.0) == "NORMAL"


# =========================================================================
# B1 §1.4: Phantom Custody Detection (Bloom heartbeat)
# =========================================================================
class TestPhantomCustodyDetection:
    """B1 §1.4: custody inventory digest + heartbeat detection + recovery."""

    def test_bloom_membership(self):
        digest = bloom_digest(["b1", "b2", "b3"])
        assert bloom_contains(digest, "b1")
        assert bloom_contains(digest, "b3")

    def test_bloom_empty_has_no_members(self):
        digest = bloom_digest([])
        assert not bloom_contains(digest, "b1")

    def test_node_custody_digest_covers_held_bundles(self):
        node = CustodyNode("n1")
        b = Bundle(bundle_id="b1", size_bytes=100)
        node.accept_custody(b, 0.0)
        assert bloom_contains(node.custody_digest(), "b1")

    def test_retain_backup_keeps_sender_storage_charged(self):
        a = CustodyNode("A", storage_bytes=10_000)
        c = CustodyNode("C", storage_bytes=10_000)
        b = Bundle(bundle_id="b1", size_bytes=100, source="A", destination="E")
        a.accept_custody(b, 0.0)
        used_before = a.used_bytes
        arrival = a.forward_bundle(b, c, 0.0, retain_backup=True)
        assert arrival > 0
        assert a.used_bytes == used_before  # backup copy still charged
        assert "b1" in a.pending_release

    def test_backup_released_after_three_confirmations(self):
        a = CustodyNode("A", storage_bytes=10_000)
        c = CustodyNode("C", storage_bytes=10_000)
        b = Bundle(bundle_id="b1", size_bytes=100, source="A", destination="E")
        a.accept_custody(b, 0.0)
        a.forward_bundle(b, c, 0.0, retain_backup=True)
        released = []
        for i in range(RELEASE_CONFIRMATIONS):
            report = a.heartbeat_check({"C": c.custody_digest()}, t_s=60.0 * (i + 1))
            released += report["released"]
        assert "b1" in released
        assert "b1" not in a.pending_release
        assert a.used_bytes == 0  # backup freed

    def test_phantom_detected_and_recovered(self):
        # Node C "ACKed" but never actually stored the bundle (false ACK).
        a = CustodyNode("A", storage_bytes=10_000)
        c = CustodyNode("C", storage_bytes=10_000)
        b = Bundle(bundle_id="b1", size_bytes=100, source="A", destination="E")
        a.accept_custody(b, 0.0)
        a.forward_bundle(b, c, 0.0, retain_backup=True)
        c._remove_bundle(b)  # simulate: C never had it
        report = a.heartbeat_check({"C": c.custody_digest()}, t_s=60.0)
        assert "b1" in report["phantoms"]
        assert b in a.bundles  # re-queued from backup
        assert b.current_custodian == "A"
        assert b.custody_state == CustodyState.H
        assert "b1" not in a.pending_release

    def test_heartbeat_skips_peers_without_digest(self):
        a = CustodyNode("A", storage_bytes=10_000)
        c = CustodyNode("C", storage_bytes=10_000)
        b = Bundle(bundle_id="b1", size_bytes=100, source="A", destination="E")
        a.accept_custody(b, 0.0)
        a.forward_bundle(b, c, 0.0, retain_backup=True)
        report = a.heartbeat_check({}, t_s=60.0)  # no digest from C this cycle
        assert report["released"] == [] and report["phantoms"] == []
        assert a.pending_release["b1"]["confirmations"] == 0

    def test_forward_without_backup_unchanged(self):
        # Default path (retain_backup=False) keeps original semantics
        a = CustodyNode("A", storage_bytes=10_000)
        c = CustodyNode("C", storage_bytes=10_000)
        b = Bundle(bundle_id="b1", size_bytes=100, source="A", destination="E")
        a.accept_custody(b, 0.0)
        a.forward_bundle(b, c, 0.0)
        assert a.used_bytes == 0
        assert a.pending_release == {}


# =========================================================================
# N1: Fragment-Group Custody
# =========================================================================
class TestFragmentation:
    """N1: Fragment groups + aggregate state logic."""

    def test_non_fragmented_aggregate(self):
        b = Bundle(bundle_id="b1")
        assert b.aggregate_state() == CustodyState.H.value

    def test_all_holding_aggregate(self):
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100, state=CustodyState.H),
                FragmentGroup("g1", 100, 200, state=CustodyState.H),
            ],
        )
        assert b.aggregate_state() == CustodyState.H.value

    def test_all_received_aggregate(self):
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100, state=CustodyState.R),
                FragmentGroup("g1", 100, 200, state=CustodyState.R),
            ],
        )
        assert b.aggregate_state() == CustodyState.R.value

    def test_partial_aggregate(self):
        """N1: S_PARTIAL when some groups received but not all."""
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100, state=CustodyState.R),
                FragmentGroup("g1", 100, 200, state=CustodyState.H),
            ],
        )
        assert b.aggregate_state() == PARTIAL_AGGREGATE

    def test_any_expired_aggregate(self):
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100, state=CustodyState.R),
                FragmentGroup("g1", 100, 200, state=CustodyState.X),
            ],
        )
        assert b.aggregate_state() == CustodyState.X.value

    def test_accept_custody_propagates_to_fragments(self):
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100, state=CustodyState.O),
                FragmentGroup("g1", 100, 200, state=CustodyState.O),
            ],
        )
        b.accept_custody("node-A", 50.0)
        for g in b.fragment_groups:
            assert g.state == CustodyState.H
            assert len(g.custody_record) == 1
            assert g.custody_record[0]["node"] == "node-A"

    def test_mark_delivered_propagates_to_fragments(self):
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100),
                FragmentGroup("g1", 100, 200),
            ],
        )
        b.mark_delivered(100.0)
        for g in b.fragment_groups:
            assert g.state == CustodyState.R

    def test_network_creates_fragment_groups(self):
        net = DTNNetwork()
        net.add_node(CustodyNode("src"))
        b = net.create_bundle("src", "dst", size_bytes=6500000, fragmented=True, num_groups=10)
        assert b.is_fragmented
        assert len(b.fragment_groups) == 10
        assert b.fragment_groups[0].offset_start == 0
        assert b.fragment_groups[-1].offset_end == 6500000

    def test_non_divisible_fragment_size(self):
        """P2: Last fragment must cover remainder bytes when size % groups != 0."""
        net = DTNNetwork()
        net.add_node(CustodyNode("src"))
        b = net.create_bundle("src", "dst", size_bytes=100, fragmented=True, num_groups=3)
        assert len(b.fragment_groups) == 3
        # step = 100 // 3 = 33
        assert b.fragment_groups[0].offset_start == 0
        assert b.fragment_groups[0].offset_end == 33
        assert b.fragment_groups[1].offset_start == 33
        assert b.fragment_groups[1].offset_end == 66
        assert b.fragment_groups[2].offset_start == 66
        assert b.fragment_groups[2].offset_end == 100  # remainder covered
        # No gaps, no overlaps
        for i in range(len(b.fragment_groups) - 1):
            assert b.fragment_groups[i].offset_end == b.fragment_groups[i + 1].offset_start

    def test_to_dict_includes_fragment_info(self):
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100),
            ],
        )
        d = b.to_dict()
        assert d["fragmented"] is True
        assert d["num_groups"] == 1

    def test_fragmented_storage_refusal_reports_pending(self):
        node = CustodyNode("n1", storage_bytes=50)
        b = Bundle(
            bundle_id="b1",
            size_bytes=1000,
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 500),
                FragmentGroup("g1", 500, 1000),
            ],
        )
        assert not node.accept_custody(b, 0.0)
        assert b.custody_state == CustodyState.P
        assert b.aggregate_state() == CustodyState.P.value
        assert all(g.state == CustodyState.P for g in b.fragment_groups)

    def test_fragmented_accept_expiry_reports_expired(self):
        node = CustodyNode("n1")
        b = Bundle(
            bundle_id="b1",
            created_s=0.0,
            lifetime_s=1.0,
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 500),
                FragmentGroup("g1", 500, 1000),
            ],
        )
        assert not node.accept_custody(b, 2.0)
        assert b.custody_state == CustodyState.X
        assert b.aggregate_state() == CustodyState.X.value
        assert all(g.state == CustodyState.X for g in b.fragment_groups)


# =========================================================================
# N5: SPE Pre-Flush
# =========================================================================
class TestSPEPreFlush:
    """N5: Validate mass state transition before safe-mode."""

    def test_pre_flush_resets_outstanding_to_holding(self):
        node = CustodyNode("relay-1")
        b1 = Bundle(bundle_id="b1")
        b2 = Bundle(bundle_id="b2")
        node.accept_custody(b1, 0.0)
        node.accept_custody(b2, 0.0)
        b1.custody_state = CustodyState.O
        b2.custody_state = CustodyState.P
        node.pre_flush_to_holding()
        assert b1.custody_state == CustodyState.H
        assert b2.custody_state == CustodyState.H

    def test_pre_flush_resets_fragment_groups(self):
        node = CustodyNode("relay-1")
        b = Bundle(
            bundle_id="b1",
            is_fragmented=True,
            fragment_groups=[
                FragmentGroup("g0", 0, 100, state=CustodyState.O),
                FragmentGroup("g1", 100, 200, state=CustodyState.P),
            ],
        )
        node.accept_custody(b, 0.0)
        # Manually set back to O/P to simulate mid-transfer
        b.fragment_groups[0].state = CustodyState.O
        b.fragment_groups[1].state = CustodyState.P
        node.pre_flush_to_holding()
        for g in b.fragment_groups:
            assert g.state == CustodyState.H


# =========================================================================
# DTNNetwork Integration
# =========================================================================
class TestDTNNetwork:
    """Integration tests for multi-hop routing."""

    def _build_3hop_network(self) -> DTNNetwork:
        net = DTNNetwork()
        for nid in ["surface", "polar-0", "elfo", "dsn_earth"]:
            net.add_node(CustodyNode(nid))
        return net

    def test_route_along_path_delivers(self):
        net = self._build_3hop_network()
        b = net.create_bundle("surface", "dsn_earth", priority=PRIORITY_EMERGENCY)
        ok = net.route_along_path(
            b,
            ["surface", "polar-0", "elfo", "dsn_earth"],
            link_types=["rf", "rf", "optical"],
            t_s=0.0,
            data_rate_kbps=1000.0,
        )
        assert ok
        assert b.delivered
        assert b.hop_count == 3
        assert b.custody_state == CustodyState.R

    def test_route_along_path_fragmented(self):
        net = self._build_3hop_network()
        b = net.create_bundle(
            "surface", "dsn_earth", size_bytes=6500000, fragmented=True, num_groups=5
        )
        ok = net.route_along_path(
            b,
            ["surface", "polar-0", "elfo", "dsn_earth"],
            t_s=0.0,
            data_rate_kbps=1000.0,
        )
        assert ok
        assert b.delivered
        assert b.aggregate_state() == CustodyState.R.value

    def test_summary_counts(self):
        net = self._build_3hop_network()
        b1 = net.create_bundle("surface", "dsn_earth")
        b2 = net.create_bundle("surface", "dsn_earth")
        net.route_along_path(b1, ["surface", "polar-0", "elfo", "dsn_earth"], t_s=0.0)
        # b2 not routed
        s = net.summary()
        assert s["total_bundles"] == 2
        assert s["delivered"] == 1

    def test_hop_log_records_link_types(self):
        net = self._build_3hop_network()
        b = net.create_bundle("surface", "dsn_earth")
        net.route_along_path(
            b,
            ["surface", "polar-0", "dsn_earth"],
            link_types=["rf", "optical"],
            t_s=0.0,
        )
        assert b.hop_log[0]["link_type"] == "rf"
        assert b.hop_log[1]["link_type"] == "optical"

    def test_route_along_path_rejects_wrong_destination(self):
        net = self._build_3hop_network()
        b = net.create_bundle("surface", "dsn_earth")
        ok = net.route_along_path(b, ["surface", "polar-0", "elfo"], t_s=0.0)
        assert not ok
        assert not b.delivered
        assert b.current_custodian == ""
        assert all(b not in node.bundles for node in net.nodes.values())


# =========================================================================
# B2: RW-CGR Stochastic Routing
# =========================================================================
class TestRWCGR:
    """B2: Stochastic CGR routing validation."""

    def _build_router_with_halo(self) -> RW_CGR:
        """Build a contact graph matching the N8 lunar baseline:
        surface → polar → halo → earth (far-side emergency path).
        """
        router = RW_CGR(seed=42)
        # surface → polar (RF, short contact)
        router.add_contact(
            Contact(
                "lunar_surface",
                "polar-0",
                start_s=0,
                duration_s=600,
                p_success=0.95,
                data_rate_kbps=256,
                latency_s=0.01,
            )
        )
        # polar → halo (optical, high reliability)
        router.add_contact(
            Contact(
                "polar-0",
                "EM-L2-HALO",
                start_s=0,
                duration_s=3600,
                p_success=0.98,
                data_rate_kbps=1000,
                latency_s=0.5,
            )
        )
        # halo → earth (optical, permanent visibility per N8)
        router.add_contact(
            Contact(
                "EM-L2-HALO",
                "earth_dsn",
                start_s=0,
                duration_s=86400,
                p_success=0.99,
                data_rate_kbps=10000,
                latency_s=1.3,
            )
        )
        # Also add a direct polar → earth (when visible, lower reliability)
        router.add_contact(
            Contact(
                "polar-0",
                "earth_dsn",
                start_s=0,
                duration_s=300,
                p_success=0.85,
                data_rate_kbps=500,
                latency_s=1.3,
            )
        )
        return router

    def test_finds_routes(self):
        router = self._build_router_with_halo()
        b = Bundle(bundle_id="b1", priority=PRIORITY_EMERGENCY, size_bytes=256)
        routes = router.find_routes("lunar_surface", "earth_dsn", b, t_s=0)
        assert len(routes) >= 1

    def test_emergency_prefers_halo(self):
        """N8 + B2: Emergency traffic from far-side should prefer halo relay.

        With composite utility ranking (p_success * exp(-λ*latency)),
        the higher-reliability halo path should be top-ranked for emergency.
        """
        router = self._build_router_with_halo()
        b = Bundle(bundle_id="b1", priority=PRIORITY_EMERGENCY, size_bytes=256)
        routes = router.find_routes("lunar_surface", "earth_dsn", b, t_s=0, n_samples=500)
        halo_routes = [r for r in routes if "EM-L2-HALO" in r]
        assert len(halo_routes) >= 1, "Halo should appear in top routes for emergency"
        # For emergency, halo path should be #1 due to higher reliability
        assert "EM-L2-HALO" in routes[0], "Halo should be top-ranked for emergency"

    def test_route_bundle_returns_best(self):
        router = self._build_router_with_halo()
        b = Bundle(bundle_id="b1", priority=PRIORITY_NORMAL, size_bytes=256)
        best = router.route_bundle(b, "lunar_surface", "earth_dsn", t_s=0)
        assert best is not None
        assert best[0] == "lunar_surface"
        assert best[-1] == "earth_dsn"

    def test_no_route_returns_none(self):
        router = RW_CGR()
        b = Bundle(bundle_id="b1", size_bytes=256)
        result = router.route_bundle(b, "nowhere", "earth_dsn", t_s=0)
        assert result is None

    def test_cache_invalidation(self):
        router = self._build_router_with_halo()
        b = Bundle(bundle_id="b1", priority=PRIORITY_NORMAL, size_bytes=256)
        router.find_routes("lunar_surface", "earth_dsn", b, t_s=0)
        assert len(router.route_cache) == 1
        router.clear_cache()
        assert len(router.route_cache) == 0

    def test_add_contact_invalidates_cache(self):
        """Regression: add_contact must drop stale cached routes.

        Pre-fix the cache survived graph mutation, so a caller that
        added a superior direct contact kept getting the old indirect
        path back until clear_cache() was called manually.
        """
        from tin.core.routing import Contact

        router = self._build_router_with_halo()
        b = Bundle(bundle_id="b1", priority=PRIORITY_NORMAL, size_bytes=256)
        router.find_routes("lunar_surface", "earth_dsn", b, t_s=0)
        assert len(router.route_cache) == 1
        # Add a new contact — cache must be invalidated automatically.
        router.add_contact(
            Contact(
                from_node="lunar_surface",
                to_node="earth_dsn",
                start_s=0.0,
                duration_s=600.0,
                p_success=0.99,
                latency_s=1.0,
            )
        )
        assert len(router.route_cache) == 0

    def test_route_cache_includes_bundle_size(self):
        """Different bundle sizes can prefer different paths."""
        router = RW_CGR(seed=11)
        router.add_contact(
            Contact(
                "A",
                "D",
                start_s=0.0,
                duration_s=100_000.0,
                p_success=0.99,
                data_rate_kbps=1.0,
                latency_s=0.1,
            )
        )
        router.add_contact(
            Contact(
                "A",
                "B",
                start_s=0.0,
                duration_s=100_000.0,
                p_success=0.99,
                data_rate_kbps=1000.0,
                latency_s=0.1,
            )
        )
        router.add_contact(
            Contact(
                "B",
                "D",
                start_s=0.0,
                duration_s=100_000.0,
                p_success=0.99,
                data_rate_kbps=1000.0,
                latency_s=0.1,
            )
        )

        small = Bundle(bundle_id="small", priority=PRIORITY_NORMAL, size_bytes=1)
        large = Bundle(bundle_id="large", priority=PRIORITY_NORMAL, size_bytes=10_000_000)

        assert router.find_routes("A", "D", small, t_s=0.0, n_samples=1000, top_k=1)[0] == [
            "A",
            "D",
        ]
        assert router.find_routes("A", "D", large, t_s=0.0, n_samples=1000, top_k=1)[0] == [
            "A",
            "B",
            "D",
        ]
        assert len(router.route_cache) == 2

    def test_waits_for_future_contacts(self):
        router = RW_CGR(seed=7)
        router.add_contact(Contact("A", "B", start_s=100.0, duration_s=10.0, latency_s=1.0))
        router.add_contact(Contact("B", "D", start_s=200.0, duration_s=10.0, latency_s=1.0))
        b = Bundle(bundle_id="b1", size_bytes=256)
        routes = router.find_routes("A", "D", b, t_s=0.0, n_samples=50)
        assert routes
        assert routes[0] == ["A", "B", "D"]
        assert router._expected_latency(routes[0], 0.0, b) < float("inf")

    def test_rejects_temporally_impossible_routes(self):
        router = RW_CGR(seed=7)
        router.add_contact(Contact("A", "B", start_s=0.0, duration_s=1.0, latency_s=1.0))
        router.add_contact(Contact("B", "D", start_s=0.0, duration_s=1.0, latency_s=1.0))
        b = Bundle(bundle_id="b1", size_bytes=256)
        assert router.find_routes("A", "D", b, t_s=10.0, n_samples=50) == []

    def test_loop_avoidance(self):
        """Verify RW-CGR doesn't produce loops."""
        router = self._build_router_with_halo()
        b = Bundle(bundle_id="b1", priority=PRIORITY_NORMAL, size_bytes=256)
        routes = router.find_routes("lunar_surface", "earth_dsn", b, t_s=0, n_samples=500)
        for route in routes:
            assert len(route) == len(set(route)), f"Loop detected in {route}"


# =========================================================================
# N8: Lunar Constellation Config
# =========================================================================
class TestLunarConfig:
    """N8: Validate lunar baseline constellation integrity."""

    def test_lunar_body_constants(self):
        assert LUNAR_BODY.name == "Moon"
        assert LUNAR_BODY.radius_km == pytest.approx(1737.4)
        assert LUNAR_BODY.mu_km3s2 == pytest.approx(4902.8)

    def test_constellation_node_count(self):
        """Study §N8: 8 polar + 1 ELFO + 1 halo = 10 nodes."""
        cfg = get_lunar_constellation()
        total = len(cfg.satellites) + len(cfg.relay_hubs) + len(cfg.halo_satellites)
        assert total == 10

    def test_polar_sat_parameters(self):
        cfg = get_lunar_constellation()
        assert len(cfg.satellites) == 8
        for i, sat in enumerate(cfg.satellites):
            assert sat.a_km == pytest.approx(1737.4 + 400)
            assert sat.i_deg == pytest.approx(89.5)
            assert sat.raan_deg == pytest.approx(i * 45)

    def test_elfo_parameters(self):
        cfg = get_lunar_constellation()
        assert len(cfg.relay_hubs) == 1
        elfo = cfg.relay_hubs[0]
        assert elfo.a_km == 5740
        assert elfo.e == pytest.approx(0.58)
        assert elfo.i_deg == pytest.approx(56.0)
        assert elfo.sat_type == "elfo"

    def test_halo_parameters(self):
        """N8: EM-L2 halo at z=8000 km, period ~14.5 days."""
        cfg = get_lunar_constellation()
        assert len(cfg.halo_satellites) == 1
        halo = cfg.halo_satellites[0]
        assert halo.z_amplitude_km == pytest.approx(8000.0)
        assert halo.period_days == pytest.approx(14.5)

    def test_sim_parameters(self):
        cfg = get_lunar_constellation()
        assert cfg.sim_days == 28
        assert cfg.dt_s == 300.0


# =========================================================================
# Constants & Utilities
# =========================================================================
class TestConstants:
    def test_priority_names(self):
        assert PRIORITY_NAMES[0] == "EMERGENCY"
        assert PRIORITY_NAMES[3] == "BULK"

    def test_default_lifetimes(self):
        assert DEFAULT_LIFETIMES[PRIORITY_EMERGENCY] == 3600
        assert DEFAULT_LIFETIMES[PRIORITY_BULK] == 604800

    def test_bundle_expiry(self):
        b = Bundle(bundle_id="b1", created_s=0.0, lifetime_s=100.0)
        assert not b.is_expired(50.0)
        assert b.is_expired(150.0)

    def test_total_latency(self):
        b = Bundle(bundle_id="b1", created_s=0.0)
        b.add_hop("a", "b", 1.5)
        b.add_hop("b", "c", 3.0)
        assert b.total_latency_s() == pytest.approx(3.0)


# =========================================================================
# Commodity-Aware Oracle
# =========================================================================
class TestCommodityOracle:
    """Validate cost-priority Dijkstra with commodity hazard rates."""

    TWO_PATH = [
        # Fast direct: T=10, p=0.50 → nlQ=0.693
        {
            "from_node": "S",
            "to_node": "D",
            "start_s": 0.0,
            "duration_s": 200.0,
            "latency_s": 10.0,
            "p_success": 0.50,
            "link_type": "direct",
        },
        # Reliable relay: T~100, p=0.95*0.95 → nlQ=0.103
        {
            "from_node": "S",
            "to_node": "R",
            "start_s": 0.0,
            "duration_s": 200.0,
            "latency_s": 50.0,
            "p_success": 0.95,
            "link_type": "relay_leg1",
        },
        {
            "from_node": "R",
            "to_node": "D",
            "start_s": 50.0,
            "duration_s": 200.0,
            "latency_s": 50.0,
            "p_success": 0.95,
            "link_type": "relay_leg2",
        },
    ]

    def test_lambda_zero_picks_reliable(self):
        """lambda_c=0 → max reliability → picks 2-hop relay."""
        import math

        from tin.core.oracle import commodity_oracle

        ok, cost, arr, hops = commodity_oracle("S", "D", 0.0, self.TWO_PATH, lambda_c=0.0)
        assert ok
        assert len(hops) == 2
        assert cost == pytest.approx(-math.log(0.95) - math.log(0.95), abs=1e-6)

    def test_large_lambda_picks_fast(self):
        """Large lambda_c → time dominates → picks 1-hop direct."""
        from tin.core.oracle import commodity_oracle

        ok, cost, arr, hops = commodity_oracle("S", "D", 0.0, self.TWO_PATH, lambda_c=1.0)
        assert ok
        assert len(hops) == 1

    def test_source_equals_dest(self):
        from tin.core.oracle import commodity_oracle

        ok, cost, arr, hops = commodity_oracle("S", "S", 42.0, self.TWO_PATH)
        assert ok
        assert cost == 0.0
        assert arr == 42.0
        assert hops == []

    def test_unreachable(self):
        from tin.core.oracle import commodity_oracle

        contacts = [
            {
                "from_node": "S",
                "to_node": "R",
                "start_s": 0.0,
                "duration_s": 10.0,
                "latency_s": 5.0,
                "p_success": 0.9,
            }
        ]
        ok, cost, arr, hops = commodity_oracle("S", "D", 0.0, contacts)
        assert not ok
        assert hops == []

    def test_ttl_respected(self):
        from tin.core.oracle import commodity_oracle

        contacts = [
            {
                "from_node": "S",
                "to_node": "D",
                "start_s": 0.0,
                "duration_s": 100.0,
                "latency_s": 10.0,
                "p_success": 0.9,
            }
        ]
        ok, _, _, _ = commodity_oracle("S", "D", 0.0, contacts, ttl=9.0)
        assert not ok
        ok, _, arr, _ = commodity_oracle("S", "D", 0.0, contacts, ttl=10.0)
        assert ok
        assert arr == pytest.approx(10.0)

    def test_cost_includes_dwell(self):
        """Dwell time at intermediate node contributes lambda_c * dwell."""
        from tin.core.oracle import commodity_oracle

        contacts = [
            {
                "from_node": "S",
                "to_node": "R",
                "start_s": 0.0,
                "duration_s": 100.0,
                "latency_s": 5.0,
                "p_success": 0.95,
                "link_type": "leg1",
            },
            {
                "from_node": "R",
                "to_node": "D",
                "start_s": 20.0,
                "duration_s": 100.0,
                "latency_s": 10.0,
                "p_success": 0.90,
                "link_type": "leg2",
            },
        ]
        ok0, cost0, _, _ = commodity_oracle("S", "D", 0.0, contacts, lambda_c=0.0)
        ok1, cost1, _, _ = commodity_oracle("S", "D", 0.0, contacts, lambda_c=0.1)
        assert ok0 and ok1
        time_contribution = cost1 - cost0
        # arrive R at 5, wait to 20 (dwell=15), arrive D at 30
        assert time_contribution == pytest.approx(0.1 * 30.0, abs=1e-6)

    def test_hops_contain_link_type(self):
        from tin.core.oracle import commodity_oracle

        contacts = [
            {
                "from_node": "S",
                "to_node": "D",
                "start_s": 0.0,
                "duration_s": 100.0,
                "latency_s": 10.0,
                "p_success": 0.9,
                "link_type": "em_coast",
            }
        ]
        ok, _, _, hops = commodity_oracle("S", "D", 0.0, contacts)
        assert ok
        assert hops[0]["link_type"] == "em_coast"
        assert hops[0]["from_node"] == "S"
        assert hops[0]["to_node"] == "D"

    def test_non_dominated_late_low_cost_label_does_not_hide_early_path(self):
        from tin.core.oracle import commodity_oracle

        contacts = [
            {
                "from_node": "S",
                "to_node": "R",
                "start_s": 0.0,
                "duration_s": 5.0,
                "latency_s": 10.0,
                "p_success": 0.1,
                "link_type": "early_high_cost",
            },
            {
                "from_node": "S",
                "to_node": "R",
                "start_s": 100.0,
                "duration_s": 5.0,
                "latency_s": 0.0,
                "p_success": 1.0,
                "link_type": "late_low_cost",
            },
            {
                "from_node": "R",
                "to_node": "D",
                "start_s": 20.0,
                "duration_s": 5.0,
                "latency_s": 0.0,
                "p_success": 1.0,
                "link_type": "rd",
            },
        ]
        ok, _, arr, hops = commodity_oracle("S", "D", 0.0, contacts, lambda_c=0.0)
        assert ok
        assert arr == pytest.approx(20.0)
        assert [h["link_type"] for h in hops] == ["early_high_cost", "rd"]


class TestOptimalRouterDeadline:
    """Regression: backward_induction uses absolute deadlines, not raw ttl.

    Pre-fix the function compared absolute event times to ttl as if ttl
    were itself absolute, so any path injected at t > ttl collapsed to
    V=0 even when the path completed well within the bundle's lifetime.
    """

    def _single_path_contact(self, start_s=100.0, duration_s=10.0, latency_s=10.0):
        # latency_s is transfer time, NOT window length — arrival = depart + latency.
        # With latency=10, a bundle injected at t=100 reaches B at t=110.
        return {
            "from_node": "A",
            "to_node": "B",
            "start_s": start_s,
            "duration_s": duration_s,
            "latency_s": latency_s,
            "p_success": 1.0,
        }

    def test_path_inside_deadline_delivers(self):
        """Single 10s deterministic path injected at t=100, ttl=20 → deliver."""
        from tin.core.optimal_router import backward_induction

        contacts = [self._single_path_contact()]
        V = backward_induction(
            contacts,
            "A",
            "B",
            deadline=100.0 + 20.0,
            injection_time=100.0,
        )
        assert V[("A", 100.0)] == 1.0

    def test_path_outside_deadline_fails(self):
        """Same path but ttl=5 (deadline=105) — arrival at 110 misses → V=0."""
        from tin.core.optimal_router import backward_induction

        contacts = [self._single_path_contact()]
        V = backward_induction(
            contacts,
            "A",
            "B",
            deadline=100.0 + 5.0,
            injection_time=100.0,
        )
        assert V[("A", 100.0)] == 0.0

    def test_compute_phi_optimal_per_injection_deadlines(self):
        """compute_phi_optimal scores each injection against its own deadline."""
        from tin.core.optimal_router import compute_phi_optimal

        contacts = [self._single_path_contact()]
        # t=100 succeeds (deadline 120 covers arrival 110); t=200 has no contact.
        result = compute_phi_optimal(
            contacts,
            "A",
            "B",
            p_eff=1.0,
            injection_times=[100.0, 200.0],
            ttl=20.0,
            eta_lyapunov=1.0,
            eta_greedy=1.0,
            s_t=1.0,
        )
        assert abs(result.dr_optimal - 0.5) < 1e-9
