"""plot_eh_three_regimes.py — E[H] three-regime visualization.

Top panel: Toy temporal contact graph demonstrating WHY E[H] is
non-monotonic with density. Uses the real earliest-arrival oracle
on synthetic contacts to show the scheduling overhead mechanism.

Bottom panel: Empirical E[H] from Starlink data, binned by distance.

Output: figures/eh_three_regimes.png
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from tin.core.oracle import earliest_arrival_path

_HERE = Path(__file__).parent
_FIG = _HERE.parent / "figures"
_FIG.mkdir(exist_ok=True)


# =====================================================================
# TOY MODEL: temporal ring with earliest-arrival oracle
# =====================================================================


def _build_toy_contacts(n_planes, n_positions=16, t_end=1000.0):
    """Build temporal contacts for a ring of positions.

    Each plane creates a ring of contacts between adjacent positions,
    but with periodic time windows offset by phase. More planes =
    more temporal options = the oracle can sometimes arrive faster
    by taking longer routes.

    Contacts:
      - Surface: source/dest ↔ nearby sats (always available)
      - ISL: sat ↔ adjacent sat (periodic, phase-shifted per plane)
    """
    contacts = []
    rng = np.random.default_rng(42)

    # Orbital period for each plane (slightly different to create phasing)
    base_period = t_end / 8.0  # ~8 passes per simulation
    window = base_period * 0.3  # 30% duty cycle per pass

    for p in range(n_planes):
        # Phase offset for this plane
        phase = p * base_period / n_planes

        # Which positions this plane's sats cover (evenly distributed)
        plane_offset = (p * n_positions // max(n_planes, 1)) % n_positions

        # ISL contacts along the ring for this plane
        for pos in range(n_positions):
            sat_id = f"P{p}_S{pos}"
            next_sat = f"P{p}_S{(pos + 1) % n_positions}"

            # Multiple windows per simulation period
            t = phase
            while t < t_end:
                start = t % t_end
                dur = min(window, t_end - start)
                if dur > 5:
                    # ISL: this sat to next sat in same plane (along ring)
                    contacts.append(
                        {
                            "from_node": sat_id,
                            "to_node": next_sat,
                            "start_s": start,
                            "duration_s": dur,
                            "latency_s": 0.01,
                        }
                    )
                    contacts.append(
                        {
                            "from_node": next_sat,
                            "to_node": sat_id,
                            "start_s": start,
                            "duration_s": dur,
                            "latency_s": 0.01,
                        }
                    )
                t += base_period

        # Surface contacts: each plane's sats near source/dest
        # Source at position 0, dest at position n//2
        src_pos = 0
        dst_pos = n_positions // 2

        for target_pos, station in [(src_pos, "source"), (dst_pos, "dest")]:
            # Sats within ±2 positions of the station can see it
            for offset in range(-2, 3):
                pos = (target_pos + offset) % n_positions
                sat_id = f"P{p}_S{pos}"
                t = phase + offset * 5.0  # slight time offset per position
                while t < t_end:
                    start = max(0, t % t_end)
                    dur = min(window * 0.8, t_end - start)
                    if dur > 5:
                        contacts.append(
                            {
                                "from_node": station,
                                "to_node": sat_id,
                                "start_s": start,
                                "duration_s": dur,
                                "latency_s": 0.01,
                            }
                        )
                        contacts.append(
                            {
                                "from_node": sat_id,
                                "to_node": station,
                                "start_s": start,
                                "duration_s": dur,
                                "latency_s": 0.01,
                            }
                        )
                    t += base_period

    # Inter-plane ISL: connect same-position sats across adjacent planes
    # (only if planes are close enough — models the ISL range limit)
    for p in range(n_planes):
        for q in range(p + 1, min(p + 3, n_planes)):
            for pos in range(n_positions):
                sat_p = f"P{p}_S{pos}"
                sat_q = f"P{q}_S{pos}"
                # Available during overlap of both planes' windows
                phase_p = p * base_period / n_planes
                phase_q = q * base_period / n_planes
                mid_phase = (phase_p + phase_q) / 2
                t = mid_phase
                while t < t_end:
                    start = t % t_end
                    dur = min(window * 0.5, t_end - start)
                    if dur > 5:
                        contacts.append(
                            {
                                "from_node": sat_p,
                                "to_node": sat_q,
                                "start_s": start,
                                "duration_s": dur,
                                "latency_s": 0.02,
                            }
                        )
                        contacts.append(
                            {
                                "from_node": sat_q,
                                "to_node": sat_p,
                                "start_s": start,
                                "duration_s": dur,
                                "latency_s": 0.02,
                            }
                        )
                    t += base_period

    return contacts


def _toy_model():
    """Compute E[H] vs plane count using real earliest-arrival oracle."""
    N_POS = 16
    T_END = 1000.0
    DT_INJ = 10.0

    plane_counts = list(range(1, 17))
    mean_hops = []
    mean_ehs = []

    t_injs = np.arange(0, T_END, DT_INJ)

    for n_planes in plane_counts:
        contacts = _build_toy_contacts(n_planes, N_POS, T_END)

        hop_counts = []
        for t in t_injs:
            ok, arr, path = earliest_arrival_path("source", "dest", float(t), contacts, ttl=T_END)
            if ok and len(path) > 1:
                hop_counts.append(len(path) - 1)

        if hop_counts:
            mean_ehs.append(np.mean(hop_counts))
        else:
            mean_ehs.append(0)

    return plane_counts, mean_ehs


# =====================================================================
# EMPIRICAL DATA
# =====================================================================


def _load_empirical():
    """Load E[H] data from oracle_retry_control results."""
    path = _HERE / "starlink_oracle_retry_control_results.json"
    with open(path) as f:
        data = json.load(f)

    by_planes = {}
    for r in data.get("eh_collapse", []):
        np_ = r["n_planes"]
        eh = r["E_H"]
        gc = r["gc_km"]
        if eh > 0:
            by_planes.setdefault(np_, []).append((gc, eh))

    return by_planes


# =====================================================================
# PLOT
# =====================================================================


def main():
    print("Computing toy model (real oracle, ~100 injection times × 16 densities)...", flush=True)
    toy_planes, toy_hops = _toy_model()
    print("  Done.", flush=True)

    print("Toy model E[H]:", flush=True)
    peak_val = max(toy_hops)
    for p, h in zip(toy_planes, toy_hops):
        marker = " <-- PEAK" if h == peak_val and h > 0 else ""
        print(f"  {p:3d} planes: E[H] = {h:.2f}{marker}", flush=True)

    print("\nLoading empirical data...", flush=True)
    empirical = _load_empirical()

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"hspace": 0.38})

    # --- Top: Toy model ---
    ax = axes[0]
    ax.plot(toy_planes, toy_hops, "o-", color="#332288", markersize=6, linewidth=1.8, zorder=5)

    # Find peak region
    peak_idx = np.argmax(toy_hops)
    if peak_idx > 0:
        # Shade regimes based on peak location
        t1 = max(1, peak_idx - 2)
        t2 = min(len(toy_planes), peak_idx + 3)
        ax.axvspan(0.5, toy_planes[t1] + 0.5, alpha=0.12, color="#88CCEE", label="Constrained")
        ax.axvspan(
            toy_planes[t1] + 0.5,
            toy_planes[min(t2, len(toy_planes) - 1)] + 0.5,
            alpha=0.12,
            color="#DDCC77",
            label="Congested",
        )
        ax.axvspan(
            toy_planes[min(t2, len(toy_planes) - 1)] + 0.5,
            toy_planes[-1] + 0.5,
            alpha=0.12,
            color="#44AA99",
            label="Saturated",
        )

    ax.set_xlabel("Number of planes (connectivity bands)", fontsize=11)
    ax.set_ylabel("Mean oracle hop count E[H]", fontsize=11)
    ax.set_title(
        "Toy Model: Temporal Ring Lattice (16 nodes, earliest-arrival oracle)\n"
        "More planes → more temporal options → scheduling overhead peak",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0.5, len(toy_planes) + 0.5)
    ax.grid(True, alpha=0.3)

    # --- Bottom: Empirical ---
    ax = axes[1]

    SHORT = 6000
    LONG = 12000

    colors = ["#332288", "#882255", "#CC6677"]
    labels = ["< 6,000 km", "6–12,000 km", "> 12,000 km"]
    bounds = [(0, SHORT), (SHORT, LONG), (LONG, 50000)]

    plane_counts_sorted = sorted(empirical.keys())

    for (lo, hi), color, label in zip(bounds, colors, labels):
        xs, ys, yerrs = [], [], []
        for np_ in plane_counts_sorted:
            points = [(gc, eh) for gc, eh in empirical[np_] if lo <= gc < hi]
            if points:
                ehs = [eh for _, eh in points]
                xs.append(np_)
                ys.append(np.mean(ehs))
                yerrs.append(np.std(ehs))

        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                fmt="o-",
                color=color,
                markersize=5,
                linewidth=1.5,
                capsize=3,
                label=label,
            )

    # Regime shading (empirical)
    ax.axvspan(0.8, 4.5, alpha=0.12, color="#88CCEE")
    ax.axvspan(4.5, 14, alpha=0.12, color="#DDCC77")
    ax.axvspan(14, 80, alpha=0.12, color="#44AA99")

    ymax = ax.get_ylim()[1]
    ax.text(
        2.2, ymax * 0.93, "Constrained", fontsize=9, ha="center", color="#117733", style="italic"
    )
    ax.text(8.5, ymax * 0.93, "Congested", fontsize=9, ha="center", color="#882255", style="italic")
    ax.text(40, ymax * 0.93, "Saturated", fontsize=9, ha="center", color="#117733", style="italic")

    ax.set_xlabel("Number of active planes", fontsize=11)
    ax.set_ylabel("Mean oracle E[H] (hops)", fontsize=11)
    ax.set_title(
        "Empirical: Starlink Shell 1 (28 station pairs, 7 plane counts)\n"
        "Binned by great-circle distance between endpoints",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 12, 24, 72])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.8, 80)
    ax.grid(True, alpha=0.3)

    plt.savefig(_FIG / "eh_three_regimes.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved → {_FIG / 'eh_three_regimes.png'}", flush=True)


if __name__ == "__main__":
    main()
