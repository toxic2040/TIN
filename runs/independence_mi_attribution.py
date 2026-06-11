#!/usr/bin/env python3
"""Mean/variance attribution of the S_T-eta mutual information.

Reconstructs the 2026-03-15 within-bin shuffle attribution behind the
doctrine sentence "MI = 0.545 nats, 77% mean-driven": shuffle eta
values within each S_T bin (preserving each bin's conditional eta
distribution, hence its conditional mean) and recompute pooled MI with
the estimator from independence_tests.py. The retained fraction is the
share of MI carried by bin-level conditional structure ("mean-driven");
the destroyed fraction is within-bin fine structure ("variance-driven").

Historical anchor (2026-03-15 ensemble analysis session; full source
chain in the workspace provenance ledger): MI dropped 0.545 -> 0.421
nats, 77.2% retained. The session notes prescribed deciles, but its
pooled-bin convention was 20 bins and only the 20-bin shuffle
reproduces the anchor (0.421, 77.2%); the decile variant gives 0.3552
(65.2%). The attribution is convention-dependent — the sensitivity
sweep is part of the output and any citation should state the
construction.

Sources: runs/epyc_results/production_2026_03_11/ + campaign_2026_03_11/
Output:  runs/independence_mi_attribution.json
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from independence_tests import load_production_data, mutual_information_binned

_HERE = Path(__file__).parent

N_BINS_HISTORICAL = 20
N_BINS_PRESCRIBED = 10
SEEDS = [42, 43, 44, 45, 46]
SWEEP = [8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 25]


def shuffled_mi(st, eta, n_bins, seed):
    edges = np.unique(np.percentile(st, np.linspace(0, 100, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, len(edges) - 2)
    rng = np.random.default_rng(seed)
    es = eta.copy()
    for b in range(len(edges) - 1):
        m = idx == b
        es[m] = rng.permutation(es[m])
    mi, *_ = mutual_information_binned(st, es)
    return mi


def main():
    t0 = time.time()

    configs = load_production_data()
    valid = [
        (d["S_T"], d["eta"])
        for d in configs
        if isinstance(d, dict)
        and not math.isnan(d.get("S_T", float("nan")))
        and not math.isnan(d.get("eta", float("nan")))
        and d.get("S_T", 0) > 0
        and d.get("eta", 0) > 0
    ]
    st = np.array([v[0] for v in valid])
    eta = np.array([v[1] for v in valid])
    print(f"n = {len(st)}")

    mi_full, *_ = mutual_information_binned(st, eta)
    print(f"I_full = {mi_full:.4f} nats")

    blocks = {}
    for label, nb in [("historical_20bin", N_BINS_HISTORICAL),
                      ("prescribed_decile", N_BINS_PRESCRIBED)]:
        per_seed = [shuffled_mi(st, eta, nb, s) for s in SEEDS]
        mean_mi = float(np.mean(per_seed))
        print(
            f"{label} (bins={nb}): I_shuf = {mean_mi:.4f} "
            f"(seed spread {min(per_seed):.4f}-{max(per_seed):.4f}), "
            f"retained = {mean_mi / mi_full * 100:.1f}%"
        )
        blocks[label] = {
            "n_bins": nb,
            "mi_shuffled_per_seed": {str(s): round(v, 6)
                                     for s, v in zip(SEEDS, per_seed)},
            "mi_shuffled_mean": round(mean_mi, 6),
            "retained_fraction": round(mean_mi / mi_full, 4),
        }

    sweep = {}
    for nb in SWEEP:
        mi_s = shuffled_mi(st, eta, nb, SEEDS[0])
        sweep[str(nb)] = {
            "mi_shuffled": round(mi_s, 6),
            "retained_fraction": round(mi_s / mi_full, 4),
        }
    print("sensitivity (seed 42): retained "
          + ", ".join(f"{nb}:{v['retained_fraction']*100:.1f}%"
                      for nb, v in sweep.items()))

    output = {
        "description": (
            "Within-S_T-bin shuffle attribution of pooled MI(S_T;eta); "
            "retained share = bin-level conditional ('mean') structure. "
            "Historical 2026-03-15 anchor reproduces at the session's "
            "20-bin convention, not the prescribed deciles."
        ),
        "n_valid": len(st),
        "mi_full_nats": round(mi_full, 6),
        "historical_anchor": {
            "source": "2026-03-15 ensemble analysis session; "
                      "source chain in the workspace provenance ledger",
            "mi_full": 0.545,
            "mi_shuffled": 0.421,
            "retained_fraction": round(0.421 / 0.545, 4),
        },
        "attribution": blocks,
        "bin_count_sensitivity_seed42": sweep,
        "wall_time_s": round(time.time() - t0, 1),
    }
    outpath = _HERE / "independence_mi_attribution.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {outpath.name}")


if __name__ == "__main__":
    main()
