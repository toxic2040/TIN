# TIN — Tolerant Interplanetary Network

**Planet-agnostic DTN architecture for cislunar and interplanetary emergency communications.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![v0.4.0](https://img.shields.io/badge/version-0.4.0-orange.svg)](https://github.com/toxic2040/TIN)

---

## What is TIN?

TIN is an open-source simulation engine for Delay-Tolerant Networking (DTN) architectures designed to provide persistent, life-critical communications at the lunar South Pole and Mars. It models orbital constellations, surface relay nodes, and Bundle Protocol v7 (RFC 9171) custody chains — then produces validated coverage, latency, and cost data for mission planning.

**Validated headline results:**

| Body | Config | Coverage | Worst-Case Gap | E2E Emergency Latency |
|------|--------|----------|----------------|-----------------------|
| Moon (Shackleton) | 8×400km polar + ELFO hub | **98.6%** | **5.6 min** | **1.36s** (5 hops to Earth) |
| Mars (equatorial) | 6pol + 2areo + 24 aerostats + Phobos + Deimos | **86.5%** | **1.0 min** | **10.7 sols** (conjunction) |

---

## Architecture

TIN uses a clean layered architecture — planet-agnostic core with body-specific config drops:

```
tin/
├── core/                  # Layer 0-2: Planet-agnostic
│   ├── __init__.py        #   Orbital mechanics, geometry, visibility (Layer 0)
│   ├── dtn.py             #   BPv7 bundles, custody, priority queuing (Layer 2)
│   └── viz.py             #   Interactive Plotly 3D visualization (Layer N)
├── lunar/                 # Moon specialization
│   └── __init__.py        #   LunarBody, 8×400km constellation, ELFO, Shackleton nodes
├── mars/                  # Mars specialization
│   └── __init__.py        #   MarsBody, areostationary, Phobos/Deimos, Aerostat Armada
├── cli/
│   └── demo.py            #   Unified runner: tin-demo --body lunar|mars|both
└── pyproject.toml
```

### Lunar (v0.3.9 heritage → v0.4.0)
- **8 near-polar smallsats** at 400 km altitude (89.5° inclination)
- **ELFO relay hub** (Lunar Pathfinder analog — elliptical frozen orbit, apoapsis over South Pole)
- **Surface nodes**: Shackleton Habitat, 2 rovers, rim relay
- **BPv7 custody chain**: EVA Astronaut → Rover → Habitat → Polar Sat → ELFO → DSN Earth

### Mars (v0.4.0)
- **6 polar smallsats** at 350 km
- **2-4 areostationary relays** at stable Mars longitudes (17,032 km)
- **Phobos + Deimos** as natural custody depot nodes
- **24 Aerostat Armada** — wind-driven super-pressure balloons at 22 km (MCD/GCM wind model)
- **Solar conjunction handling** — predictive bundle parking on Deimos during ~13-day blackout

---

## Quickstart

```bash
# Clone
git clone https://github.com/toxic2040/TIN.git
cd TIN

# Install (editable, all deps)
pip install -e .

# Optional: interactive Plotly visualizations
pip install plotly

# Run
tin-demo --body lunar --days 1 --emergency
tin-demo --body mars --days 1 --emergency --conjunction
tin-demo --body both --all --viz
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--body lunar\|mars\|both` | Target body (default: lunar) |
| `--days N` | Simulation duration in days/sols |
| `--emergency` | Run BPv7 emergency custody chain (default: on) |
| `--trade` | Run N-satellite trade study sweep |
| `--conjunction` | Run Mars solar conjunction simulation |
| `--viz` | Generate interactive Plotly HTML visualizations |
| `--all` | Enable all of the above |

---

## Key Results

### Lunar Trade Study (N-sat sweep)

| Config | Coverage | Worst Gap | Cost Proxy |
|--------|----------|-----------|------------|
| 4×400km | 72.9% | 15,293 min | $96M |
| 4×400km + ELFO | 90.5% | 16.3 min | $96M |
| 6×400km + ELFO | 94.6% | 6.2 min | $144M |
| **8×400km + ELFO** | **98.6%** | **5.6 min** | **$192M** |
| 10×400km + ELFO | 99.2% | 1.5 min | $240M |

**Insight:** ELFO relay is the force multiplier — transforms 96% pure constellation into 98.6% with sub-6-minute emergency latency.

### Mars Trade Study

| Config | Equatorial Cov | Gap | Cost |
|--------|----------------|-----|------|
| 6pol + 2areo | 82.0% | 6.5 min | $1,218M |
| 6pol + 2areo + **24 aerostats** | **86.5%** | **1.0 min** | **$1,302M** |
| 8pol + 4areo (no aerostats) | 98.5% | 1.0 min | $2,114M |

**Insight:** Aerostats deliver the same latency as doubling the orbital constellation, at $812M less.

### Mars Conjunction Emergency

Emergency during worst-case 13-day solar blackout:

```
Astronaut → Aerostat (laser mesh) → Polar Sat → Areo Relay → Deimos Depot (11 days) → Earth (DSN)
```

Total end-to-end: **10.7 sols**. Patient stabilized locally in <6 minutes. Full telemetry reaches Earth surgeons before sol 11.

---

## BPv7 Custody Chain (RFC 9171)

TIN models faithful Bundle Protocol v7 custody transfers:

- **Priority queuing**: Emergency bundles pre-empt all others
- **Custody signals**: ACCEPTED / REFUSED with reason codes
- **Bundle lifetime**: TTL enforcement with aging at each hop
- **Status reports**: RECEIVED, FORWARDED, DELIVERED, DELETED
- **Realistic link budgets**: 256 kbps (UHF suit radio) → 8,192 kbps (X-band ELFO→DSN)
- **Earth-Moon light time**: 1.3s propagation delay modeled

---

## Visualizations

With `--viz` flag and plotly installed, TIN generates interactive HTML:

- **3D Constellation Viewer** — Rotate/zoom Moon globe with neon orbit rings, satellite positions, ELFO hub, and emergency bundle trails
- **BPv7 Custody Gantt** — Timeline showing bundle hops with priority color coding and hover details
- **Shackleton Heatmap** — Polar coverage contour with permanently shadowed region (PSR) overlay

All outputs saved to `results/` as standalone HTML files.

---

## Heritage & References

- NASA JPL Interplanetary Network (IPN) / DTN architecture
- ESA Lunar Pathfinder (ELFO relay concept)
- RFC 9171 — Bundle Protocol Version 7
- RFC 9174 — DTN TCP Convergence-Layer Protocol v4
- Lynn Harper et al. — Lunar communications studies
- NASA ULDB / Mars 2001 balloon concepts (Aerostat Armada heritage)

---

## Alignment

- **NASA LSIC** — Communications focus area (LOI submitted)
- **Artemis** — South Pole surface operations safety
- **NASA SCaN** — DTN architecture standards
- **NASA STMD** — Small spacecraft technology

---

## Contributing

TIN is MIT licensed and welcomes contributions. Key areas:

- Orbit optimization (Walker phasing, asymmetric constellations)
- 3D visualization enhancements (animated bundle flow, real DEM)
- Bundle Protocol extensions (fragmentation, BPSec)
- Solar storm resilience modeling
- Far-side coverage detailed mapping
- Architecture Decision Records (ADRs)

See [Issues](https://github.com/toxic2040/TIN/issues) for scoped contribution opportunities.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Building the interplanetary internet backbone, one custody transfer at a time.*
