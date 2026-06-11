import argparse
import json
from collections import defaultdict

import numpy as np

print("TIN v0.3.7-fix4 — DTN + AI Custody Router")

parser = argparse.ArgumentParser()
parser.add_argument("--polar_sats", type=int, default=8)
parser.add_argument("--sim_days", type=int, default=28)
parser.add_argument("--bundles", type=int, default=300)
parser.add_argument("--storm_prob", type=float, default=0.12)
parser.add_argument("--output", type=str, default="tin_dtn_results_v0.3.7")
args = parser.parse_args()

# Nodes
NODES = ["SouthPole", "ELFO"] + [f"Polar{i + 1}" for i in range(args.polar_sats)]
SOUTHPOLE = 0
ELFO = 1
POLARS = list(range(2, len(NODES)))

np.random.seed(42)
T_sim = args.sim_days * 86400.0

# === Robust Contact Generation ===
contacts = []

# 1. SouthPole → Polar (frequent short)
for p in POLARS:
    for day in range(args.sim_days):
        for rev in range(16):
            t = day * 86400 + rev * 5400 + np.random.uniform(0, 3600)
            dur = np.random.uniform(480, 960)
            if t + dur < T_sim:
                contacts.append((t, t + dur, SOUTHPOLE, p, 450))

# 2. Polar → ELFO (critical relay)
for p in POLARS:
    for day in range(args.sim_days):
        for slot in range(12):
            t = day * 86400 + slot * 7200 + np.random.uniform(0, 1800)
            dur = np.random.uniform(720, 1440)
            if t + dur < T_sim:
                contacts.append((t, t + dur, p, ELFO, 850))

# 3. Direct SouthPole → ELFO (ELFO advantage)
for day in range(args.sim_days):
    t = day * 86400 + np.random.uniform(0, 86400)
    dur = np.random.uniform(10800, 21600)
    if t + dur < T_sim:
        contacts.append((t, t + dur, SOUTHPOLE, ELFO, 1800))

contacts.sort(key=lambda x: x[0])
print(f"Generated {len(contacts):,} realistic contact windows")


# === Bundles ===
class Bundle:
    def __init__(self, bid, size, prio, create_t):
        self.id = bid
        self.size = size
        self.prio = prio
        self.create_t = create_t
        self.custody_at = None
        self.custody_chain = []
        self.delivered = False
        self.delivery_t = None
        self.retransmits = 0
        self.ttl = 86400 * 3


bundles = []
for i in range(args.bundles):
    create_t = np.random.uniform(0, T_sim * 0.75)
    size = np.random.choice([50 * 1024, 500 * 1024, 2 * 1024 * 1024], p=[0.45, 0.4, 0.15])
    prio = np.random.choice([0, 1, 2], p=[0.12, 0.33, 0.55])
    bundles.append(Bundle(i, size, prio, create_t))

bundles.sort(key=lambda b: b.create_t)

custody = defaultdict(list)
pending_idx = 0

# Stats
delivered_count = 0
retrans_total = 0
handoffs = 0
bundles_reached_polar = 0


def predict_has_path(from_node, current_t, ttl_left):
    for c in contacts:
        if c[0] > current_t and c[2] == from_node and (c[0] - current_t) < ttl_left:
            return True
    return False


# Event-driven loop
for contact in contacts:
    t_start, t_end, fr, to, rate = contact
    duration = t_end - t_start

    # Release bundles
    while pending_idx < len(bundles) and bundles[pending_idx].create_t <= t_start:
        b = bundles[pending_idx]
        b.custody_at = SOUTHPOLE
        b.custody_chain = [NODES[SOUTHPOLE]]
        custody[SOUTHPOLE].append(b.id)
        pending_idx += 1

    if np.random.rand() < args.storm_prob * 0.75:
        continue

    ready = sorted(
        [bid for bid in custody[fr] if not bundles[bid].delivered],
        key=lambda x: (-bundles[x].prio, bundles[x].create_t),
    )

    # FIXED CAPACITY — now correct KiB
    capacity_kib = (rate * duration) / 8.0

    for bid in ready:
        b = bundles[bid]
        if b.size / 1024.0 > capacity_kib:
            break

        # AI custody decision
        if to == ELFO:
            accept = True
        else:
            ttl_left = b.ttl - (t_start - b.create_t)
            accept = predict_has_path(to, t_end, ttl_left) if ttl_left > 3600 else False

        custody[fr].remove(bid)

        if accept:
            custody[to].append(bid)
            b.custody_at = to
            b.custody_chain.append(NODES[to])
            handoffs += 1

            if to == ELFO:
                b.delivered = True
                b.delivery_t = t_end
                delivered_count += 1
                if bid in custody[ELFO]:
                    custody[ELFO].remove(bid)
            elif to in POLARS:
                bundles_reached_polar += 1
        else:
            custody[fr].append(bid)
            b.retransmits += 1
            retrans_total += 1

        capacity_kib -= b.size / 1024.0

# Results
success_rate = (delivered_count / args.bundles * 100) if args.bundles > 0 else 0
delivered_bundles = [b for b in bundles if b.delivered]
avg_hops = np.mean([len(b.custody_chain) for b in delivered_bundles]) if delivered_bundles else 0
positive_lat = [
    b.delivery_t - b.create_t
    for b in delivered_bundles
    if b.delivery_t and b.delivery_t > b.create_t
]
avg_latency = np.mean(positive_lat) / 3600 if positive_lat else 0

print("\nTIN DTN v0.3.7-fix4 RESULTS")
print(f"   Bundles generated: {args.bundles}")
print(f"   Delivered to ELFO: {delivered_count} ({success_rate:.1f}%)")
print(f"   Avg custody hops: {avg_hops:.2f}")
print(f"   Avg latency: {avg_latency:.1f} hours")
print(f"   Total retransmits: {retrans_total}")
print(f"   Bundles that reached a Polar: {bundles_reached_polar}")
print(f"   Total handoff events: {handoffs}")

# Save JSON
results = {
    "version": "0.3.7-fix4",
    "success_rate_pct": round(success_rate, 2),
    "avg_hops": round(avg_hops, 2),
    "avg_latency_hours": round(avg_latency, 2),
    "retransmits": retrans_total,
    "storm_prob": args.storm_prob,
    "contacts_generated": len(contacts),
    "timestamp": "2026-02-19",
}
with open(f"{args.output}.json", "w") as f:
    json.dump(results, f, indent=2)

print("Run complete; results written.")
