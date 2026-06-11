"""repro_v0_3_8.py — Deterministic reproducibility entrypoint for the v0.3.8 lunar baseline.

Mirrors the simulation in archive/scripts_v0.3x/tin_v0.3.8_final_intergrated.py
(physics + DTN + coverage map for an 8-polar + ELFO Moon constellation) but
with the random-number generation made deterministic per --seed.

The archived v0.3.x scripts seeded numpy's global RNG only AFTER bundle
generation, so bundle creation times, sizes, and priorities varied
run-to-run — making emergency_worst_min stochastic across reruns of the
same command. This entrypoint uses np.random.default_rng(seed) for every
random op, so a given seed produces byte-identical results.

The output JSON also includes the seed actually used and a real run
timestamp (the archive script hardcoded "2026-02-19"), and adds the
avg_custody_hops metric that was printed but missing from the JSON.

Usage:
    python runs/repro_v0_3_8.py --seed 42 --output runs/results/repro_v0_3_8

Default arguments reproduce the v0.3.8 baseline scenario.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ====================== CONSTANTS ======================
R_MOON = 1737.4
MU_MOON = 4902.8
DT_SECONDS = 60.0


# ====================== ORBITAL ELEMENTS ======================
class Satellite:
    def __init__(self, a, e, inc, omega, raan, m0, name):
        self.a = a
        self.e = e
        self.inc = np.deg2rad(inc)
        self.omega = np.deg2rad(omega)
        self.raan = np.deg2rad(raan)
        self.m0 = np.deg2rad(m0)
        self.name = name

    def position_at(self, t):
        n = np.sqrt(MU_MOON / self.a**3)
        m = self.m0 + n * t
        ecc_anom = m + self.e * np.sin(m) if self.e > 0.01 else m
        for _ in range(4):
            ecc_anom = m + self.e * np.sin(ecc_anom)
        nu = 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(ecc_anom / 2),
            np.sqrt(1 - self.e) * np.cos(ecc_anom / 2),
        )
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(nu))
        arg_lat = nu + self.omega
        cos_o, sin_o = np.cos(self.raan), np.sin(self.raan)
        cos_i, sin_i = np.cos(self.inc), np.sin(self.inc)
        cos_arg, sin_arg = np.cos(arg_lat), np.sin(arg_lat)
        x = r * (cos_arg * cos_o - sin_arg * sin_o * cos_i)
        y = r * (cos_arg * sin_o + sin_arg * cos_o * cos_i)
        z = r * (sin_arg * sin_i)
        return np.array([x, y, z])


# ====================== BUNDLE ======================
@dataclass
class Bundle:
    bid: int
    size: int
    prio: int
    create_t: float
    custody_at: str | None = None
    custody_chain: list[str] = field(default_factory=list)
    delivered: bool = False
    delivery_t: float | None = None


def make_bundles(rng: np.random.Generator, n_bundles: int) -> list[Bundle]:
    bundles: list[Bundle] = []
    for i in range(n_bundles):
        create_t = float(rng.uniform(0, 5 * 86400))
        size = int(rng.choice([10 * 1024, 100 * 1024, 1000 * 1024], p=[0.5, 0.3, 0.2]))
        prio = int(rng.choice([0, 1, 2], p=[0.15, 0.35, 0.5]))
        bundles.append(Bundle(bid=i, size=size, prio=prio, create_t=create_t))
    return bundles


# ====================== SIMULATION ======================
def build_constellation(args: argparse.Namespace) -> tuple[list[Satellite], Satellite | None]:
    polar_sats = [
        Satellite(
            R_MOON + args.altitude,
            0.0,
            args.inclination,
            0.0,
            i * 360 / args.polar_sats,
            i * 360 / args.polar_sats,
            f"Polar{i + 1}",
        )
        for i in range(args.polar_sats)
    ]
    elfo = (
        Satellite(5740.0, 0.58, 55.0, 86.0, 0.0, 0.0, "ELFO")
        if args.include_elfo
        else None
    )
    return polar_sats, elfo


def _coverage_chunk(
    polar_sats: list[Satellite],
    elfo: Satellite | None,
    pos_gs: np.ndarray,
    min_elev: float,
    step_start: int,
    step_end: int,
) -> np.ndarray:
    """Return covered_count partial sum for steps in [step_start, step_end).

    Pure function, no RNG, no I/O. The reduction across chunks is an int sum
    which is associative and order-independent, so any chunking produces a
    byte-identical final coverage_pct_grid.
    """
    n_cells = pos_gs.shape[0]
    covered_count = np.zeros(n_cells, dtype=int)
    for step in range(step_start, step_end):
        t = step * DT_SECONDS
        visible = np.zeros(n_cells, dtype=bool)
        sat_pos = {s.name: s.position_at(t) for s in polar_sats}
        if elfo:
            sat_pos["ELFO"] = elfo.position_at(t)
        for _name, pos in sat_pos.items():
            vec = pos - pos_gs
            dist = np.linalg.norm(vec, axis=1)
            cos_zen = np.clip(
                np.sum(vec * pos_gs, axis=1) / (dist * R_MOON), -1, 1
            )
            elev = 90 - np.rad2deg(np.arccos(cos_zen))
            visible |= elev > min_elev
        covered_count += visible.astype(int)
    return covered_count


def compute_coverage_grid(
    args: argparse.Namespace,
    polar_sats: list[Satellite],
    elfo: Satellite | None,
    n_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat_grid, lon_grid, coverage_pct_grid). Purely deterministic; no RNG.

    When n_workers > 1, the timestep loop is split into n_workers contiguous
    chunks and each chunk runs in its own process. The reduction is an int
    sum, so the parallel and serial paths produce byte-identical output.
    """
    lats = np.arange(-90, -59, 0.25)
    lons = np.arange(0, 360, 1.0)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()
    pos_gs = R_MOON * np.column_stack(
        (
            np.cos(np.deg2rad(flat_lats)) * np.cos(np.deg2rad(flat_lons)),
            np.cos(np.deg2rad(flat_lats)) * np.sin(np.deg2rad(flat_lons)),
            np.sin(np.deg2rad(flat_lats)),
        )
    )
    num_steps = int(args.sim_days * 86400.0 / DT_SECONDS)

    n_workers = max(1, int(n_workers))
    # Skip parallel path when the per-worker share would be tiny — the
    # process spin-up overhead would dominate for short simulations.
    if n_workers > 1 and num_steps >= 500 * n_workers:
        from concurrent.futures import ProcessPoolExecutor

        chunk = num_steps // n_workers
        bounds = [
            (i * chunk, (i + 1) * chunk if i < n_workers - 1 else num_steps)
            for i in range(n_workers)
        ]
        covered_count = np.zeros(len(flat_lats), dtype=int)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            partials = list(
                pool.map(
                    _coverage_chunk,
                    [polar_sats] * n_workers,
                    [elfo] * n_workers,
                    [pos_gs] * n_workers,
                    [args.min_elev] * n_workers,
                    [lo for lo, _ in bounds],
                    [hi for _, hi in bounds],
                )
            )
        for p in partials:
            covered_count += p
    else:
        covered_count = _coverage_chunk(
            polar_sats, elfo, pos_gs, args.min_elev, 0, num_steps
        )

    coverage_pct_grid = (covered_count / num_steps * 100).reshape(lat_grid.shape)
    return lat_grid, lon_grid, coverage_pct_grid


def run_simulation(
    args: argparse.Namespace,
    compute_coverage: bool = True,
    coverage_workers: int = 1,
) -> tuple[dict, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Run the v0.3.8 lunar baseline.

    Returns the results dict plus (lat_grid, lon_grid, coverage_pct_grid) so
    the caller can plot the coverage map without re-running the integration.

    When compute_coverage is False, the coverage-map loop is skipped (saves
    roughly half the per-run wall time) and south_pole_coverage_pct is set
    to None in the results dict. Used by the multi-seed ensemble runner,
    which only needs the DTN-side metrics.
    """
    rng = np.random.default_rng(args.seed)

    t_sim = args.sim_days * 86400.0
    num_steps = int(t_sim / DT_SECONDS)

    polar_sats, elfo = build_constellation(args)
    gs_pos = np.array([0.0, 0.0, -R_MOON])

    bundles = make_bundles(rng, args.bundles)
    future_bundles = sorted(bundles, key=lambda b: b.create_t)
    custody: dict[str, list[int]] = defaultdict(list)

    emerg_lat_min: list[float] = []
    norm_lat_h: list[float] = []
    max_emerg_min = 0.0
    delivered_count = 0
    polar_touches = 0
    handoffs = 0

    for step in range(num_steps):
        t = step * DT_SECONDS

        while future_bundles and future_bundles[0].create_t <= t:
            b = future_bundles.pop(0)
            b.custody_at = "SouthPole"
            b.custody_chain = ["SouthPole"]
            custody["SouthPole"].append(b.bid)

        storm = args.solar_storm and rng.random() < args.storm_prob

        sat_pos = {s.name: s.position_at(t) for s in polar_sats}
        if elfo:
            sat_pos["ELFO"] = elfo.position_at(t)

        active_links: list[tuple[str, str, int]] = []
        for name, pos in sat_pos.items():
            vec = pos - gs_pos
            dist = np.linalg.norm(vec)
            cos_zen = np.clip(np.dot(vec, -gs_pos) / (dist * R_MOON), -1, 1)
            elev = 90 - np.rad2deg(np.arccos(cos_zen))
            if elev > args.min_elev and not (
                args.topo_shadow and rng.random() < 0.08
            ):
                rate = 2200 if name == "ELFO" else 950
                active_links.append(("SouthPole", name, rate))

        if elfo and not storm:
            elfo_pos = sat_pos["ELFO"]
            for p_name, p_pos in {
                k: v for k, v in sat_pos.items() if k.startswith("Polar")
            }.items():
                if (
                    np.linalg.norm(elfo_pos - p_pos) < 13000
                    and rng.random() > 0.12
                ):
                    active_links.append((p_name, "ELFO", 1350))

        if storm:
            active_links = []

        for fr, to, rate in active_links:
            ready = (
                [bid for bid in custody[fr] if not bundles[bid].delivered]
                if fr in custody
                else []
            )
            ready.sort(key=lambda bid: (-bundles[bid].prio, bundles[bid].create_t))

            capacity_bytes = (rate * 1000 * DT_SECONDS) / 8.0

            for bid in ready:
                b = bundles[bid]
                if b.size > capacity_bytes and b.prio > 0:
                    break

                if (b.prio == 0 or rng.random() < 0.92) and t >= b.create_t + 60:
                    custody[fr].remove(bid)
                    custody[to].append(bid)
                    b.custody_at = to
                    b.custody_chain.append(to)
                    handoffs += 1
                    if to.startswith("Polar"):
                        polar_touches += 1

                    if to == "ELFO":
                        tx_seconds = b.size / (rate * 125.0)
                        b.delivery_t = t + tx_seconds
                        b.delivered = True
                        delivered_count += 1
                        if bid in custody.get("ELFO", []):
                            custody["ELFO"].remove(bid)
                        lat = b.delivery_t - b.create_t
                        if b.prio == 0:
                            emerg_lat_min.append(lat / 60)
                            max_emerg_min = max(max_emerg_min, lat / 60)
                        else:
                            norm_lat_h.append(lat / 3600)
                    break

                capacity_bytes -= b.size

    if compute_coverage:
        lat_grid, lon_grid, coverage_pct_grid = compute_coverage_grid(
            args, polar_sats, elfo, n_workers=coverage_workers
        )
        lats = np.arange(-90, -59, 0.25)
        south_pole_avg: float | None = float(np.mean(coverage_pct_grid[lats < -85]))
    else:
        lat_grid = lon_grid = coverage_pct_grid = None
        south_pole_avg = None

    success_rate = delivered_count / args.bundles * 100
    overall_h = float(np.mean(norm_lat_h)) if norm_lat_h else 0.0
    emerg_avg_min = float(np.mean(emerg_lat_min)) if emerg_lat_min else 0.0
    avg_custody_hops = (
        float(np.mean([len(b.custody_chain) - 1 for b in bundles if b.delivered]))
        if delivered_count
        else 0.0
    )

    results = {
        "version": "0.3.8-repro",
        "seed": int(args.seed),
        "sim_days": int(args.sim_days),
        "delivery_pct": round(success_rate, 2),
        "overall_avg_h": round(overall_h, 2),
        "emergency_avg_min": round(emerg_avg_min, 1),
        "emergency_worst_min": round(max_emerg_min, 1),
        "south_pole_coverage_pct": (
            round(south_pole_avg, 1) if south_pole_avg is not None else None
        ),
        "polar_touches": polar_touches,
        "avg_custody_hops": round(avg_custody_hops, 2),
        "delivered_count": delivered_count,
        "total_bundles": int(args.bundles),
        "handoffs": handoffs,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return results, lat_grid, lon_grid, coverage_pct_grid


# ====================== CLI ======================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TIN v0.3.8 lunar baseline — deterministic reproducibility entrypoint",
    )
    p.add_argument("--polar_sats", type=int, default=8)
    p.add_argument("--altitude", type=float, default=400.0)
    p.add_argument("--inclination", type=float, default=90.0)
    p.add_argument("--min_elev", type=float, default=5.0)
    p.add_argument("--sim_days", type=int, default=28)
    p.add_argument("--include_elfo", action="store_true", default=True)
    p.add_argument("--topo_shadow", action="store_true", default=True)
    p.add_argument("--solar_storm", action="store_true", default=True)
    p.add_argument("--bundles", type=int, default=300)
    p.add_argument("--storm_prob", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="runs/results/repro_v0_3_8")
    p.add_argument(
        "--no_coverage_png",
        action="store_true",
        help="Skip writing the coverage map PNG (still computes coverage stats)",
    )
    p.add_argument(
        "--coverage_workers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes for the coverage timestep loop "
            "(default 1 = serial). Output is byte-identical regardless of "
            "this value (int reduction). On a 12-core box, --coverage_workers 8 "
            "drops a 28-day run from minutes to seconds."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"TIN v0.3.8 reproduction — seed={args.seed} sim_days={args.sim_days} "
        f"bundles={args.bundles}"
    )

    results, lat_grid, lon_grid, coverage_pct_grid = run_simulation(
        args, coverage_workers=args.coverage_workers
    )

    print(f"  delivery_pct          = {results['delivery_pct']}")
    print(f"  overall_avg_h         = {results['overall_avg_h']}")
    print(f"  emergency_avg_min     = {results['emergency_avg_min']}")
    print(f"  emergency_worst_min   = {results['emergency_worst_min']}")
    print(f"  south_pole_coverage   = {results['south_pole_coverage_pct']}%")
    print(f"  polar_touches         = {results['polar_touches']}")
    print(f"  avg_custody_hops      = {results['avg_custody_hops']}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"  wrote {json_path}")

    if not args.no_coverage_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(11, 9))
            plt.contourf(lon_grid, lat_grid, coverage_pct_grid, levels=30, cmap="viridis")
            plt.colorbar(label=f"Coverage % over {args.sim_days} days")
            plt.xlabel("Longitude (deg E)")
            plt.ylabel("Latitude (deg)")
            plt.title(
                f"TIN v0.3.8 reproduction — South Pole coverage (seed={args.seed})"
            )
            plt.grid(True, alpha=0.3)
            png_path = out_path.with_suffix(".png")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  wrote {png_path}")
        except ImportError:
            print("  matplotlib not available; skipping coverage PNG")


if __name__ == "__main__":
    main()
