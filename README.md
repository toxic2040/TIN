# TIN — Tolerant Interplanetary Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/toxic2040/TIN/actions/workflows/ci.yml/badge.svg)](https://github.com/toxic2040/TIN/actions/workflows/ci.yml)
[![Zenodo](https://img.shields.io/badge/Zenodo-classification%20working%20paper-blue)](https://doi.org/10.5281/zenodo.18851385)

## The problem

Every relay architecture is a timetable. A lunar surface crew waiting for a data uplink doesn't care about orbital mechanics — they care whether the next contact window opens before the bundle expires. Whether bundles arrive depends on *when* contacts open, *how many hops* exist, and *what the routing policy does* with uncertainty.

TIN provides a reproducible way to ask: **given this constellation, this ground station schedule, and this link quality, what fraction of messages will arrive?**

## What TIN does

TIN takes an architecture specification (orbits, ground stations, link budgets) and returns a delivery-ratio estimate.

The engine decomposes delivery ratio into measurable factors:

**DR = S_T &middot; &eta;**

| Factor | What it measures | How |
|--------|-----------------|-----|
| S_T | Can a path exist at all? (temporal reachability) | Oracle sweep on the contact DAG |
| &eta; | What fraction of feasible bundles actually arrive? | Simulation with stochastic routing |

## Quick start

```bash
git clone https://github.com/toxic2040/TIN.git
cd TIN
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -x -q
```

89 tests. The only required dependency is NumPy.

## Reproducibility

The flagship scenario is the v0.3.8 lunar baseline: 8 polar smallsats plus an ELFO halo relay over 28 days, 300 bundles spanning three priority classes, with stochastic solar storms and topographic shadow.

```bash
python runs/repro_v0_3_8.py --seed 42                          # single seeded run (serial; expect tens of minutes)
python runs/repro_v0_3_8.py --seed 42 --coverage_workers 8     # same result with parallel coverage
python runs/repro_v0_3_8_ensemble.py --n_seeds 20              # multi-seed ensemble
```

The single-seed entrypoint uses `np.random.default_rng(seed)` for every random operation (bundle generation, storm draws, link probabilities, custody success). Two runs at the same seed produce JSON output identical on every field except the run timestamp. The canonical seed=42 baseline is committed at `runs/results/repro_v0_3_8_baseline.json`.

The coverage-grid loop is the dominant per-run cost; it parallelises trivially over timesteps (the reduction is an integer sum). `--coverage_workers N` splits the loop across N processes and matches the serial output field-for-field regardless of N.

The ensemble runner resumes from the committed `runs/results/repro_v0_3_8_ensemble.jsonl`; move or delete that file to regenerate from scratch. The committed 21-record set covers seeds 0-19 plus the canonical 42.

For byte-exact reproduction across machines, install from the lock file rather than from `pyproject.toml`'s loose constraint:

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

The committed baseline JSON was produced under numpy 2.4.4 (the locked version). Different numpy versions may produce slightly different float-precision results in transcendental ops; the SeedSequence-based RNG is stable since numpy 1.17.

Worst-case emergency latency is genuinely stochastic — a single seed samples one point of the distribution rather than a property of the architecture. A 21-record ensemble at sim_days=28, covering seeds 0-19 plus the canonical seed 42, yields:

| Metric | Mean | Std | p50 | p95 | Range |
|--------|-----:|----:|----:|----:|------:|
| emergency_worst_min (min) | 3.929 | 0.652 | 3.9 | 4.7 | 2.9 - 5.2 |
| emergency_avg_min (min)   | 1.805 | 0.12 | 1.8 | 2.0 | 1.6 - 2.1 |
| delivery_pct              | 100 | 0    | 100 | 100  | 100 – 100 |
| polar_touches             | 300 | 0    | 300 | 300  | 300 – 300 |

All deterministic metrics (delivery, polar_touches, custody hops, handoffs) are seed-invariant. The per-seed JSONL and aggregate stats live in `runs/results/repro_v0_3_8_ensemble.jsonl` and `runs/results/repro_v0_3_8_ensemble_summary.json`.

`pytest tests/test_repro_determinism.py` is a one-minute smoke test that checks seed plumbing without running the full simulation.

`python runs/verify_paper_claims.py --full` re-checks the quantitative claims of the Zenodo paper against the shipped result data (38 checks; four further claim families need the full simulation corpus and are reported as not checkable).

## Scope of this repository

This repository contains `tin` — the DTN simulation foundation: bundle FSM, routing,
oracle, base configs.

- `tin/core/dtn.py` — Custody FSM with fragment-group aggregation
- `tin/core/routing.py` — Stochastic RW-CGR composite-utility router
- `tin/core/oracle.py` — Earliest-arrival Dijkstra oracle with path extraction
- `tin/core/optimal_router.py` — Backward-induction DP on the contact DAG
- `tin/core/base.py` — Body, satellite, and orbit configuration types

This repository intentionally contains only the public `tin` package and
curated reproducibility materials. Internal engine code is not distributed here.

## Project structure

```
TIN/
├── tin/                    # DTN simulator (this package)
│   ├── core/               #   dtn, routing, oracle, base, optimal_router
│   ├── config/             #   lunar default config
│   └── scenarios/          #   scenario definitions
├── tests/                  # 89 tests (pytest)
├── runs/                   # reproducibility entrypoints + committed baselines (runs/results/)
├── docs/                   # archived result data backing the claim verifier
├── archive/                # curated v0.3.x era record
├── data/                   # dataset license; SPICE kernels optional (see pyproject.toml [spice] extra)
├── .github/workflows/      # CI
├── pyproject.toml
├── LICENSE                 # MIT
└── LICENSING.md            # License details
```

## Scope boundaries

- **Orbital propagation**: Baseline is Keplerian with secular J2. Coverage figures are unvalidated against GMAT/STK/Orekit.
- **Surface occlusion**: Hard-sphere geometric LOS. No terrain or atmospheric models.
- **Protocol fidelity**: Custody FSM (BPv7-inspired). Does not implement BPSec, LTP, or wire-level encoding.
- **v1.0 scope**: Cislunar. A Mars scenario definition ships as exploratory and unvalidated.

## Zenodo record

The current public research record is the Zenodo working-paper deposit. It is
not a journal publication or an accepted IEEE article.

**A Classification Framework for Temporal Contact Graphs: Morphology,
Confinement, and the Routing Efficiency Frontier**

J. Councilman — Zenodo working paper, version 7.5, 2026.

DOI: [10.5281/zenodo.18851385](https://doi.org/10.5281/zenodo.18851385)

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

## Contributing

Contributions, reproductions, and independent validations are welcome. Run `pytest tests/ -x -q` and open an issue if results differ.

Areas where external input would be most valuable:

- Independent coverage analysis using Orekit, GMAT, or STK
- BPv7/ION/HDTN interoperability review of the custody model
- Link budget validation for the EM-L2 halo relay path

## License

- **Code** — [MIT License](LICENSE). Use, modify, distribute freely with attribution.
- **Experimental data** — [CC-BY-4.0](data/LICENSE_DATA). See [LICENSING.md](LICENSING.md) for details.
