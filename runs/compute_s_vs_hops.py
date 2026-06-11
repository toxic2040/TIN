#!/usr/bin/env python3
"""
compute_s_vs_hops.py — CTMC rate ratio s = η/(1−η) across all architectures.

Key question: does s decrease predictably with relay depth?

Theory: η₀ = s/(s+1)  ⟹  s = η/(1-η)
If δ (per-hop overhead) accumulates over the relay chain, then
s = γ/(n·δ₁) and η = s/(s+1) → η falls with hop count.

Reads:
  - mars_eta_universality_results.json    (Mars, 2-hop, no ISL)
  - cislunar_purepolar_results.json       (Moon, 2-hop, with ISL)
  - cislunar_nohalo_results.json          (Moon, 3-hop: polar+ELFO, with ISL)
  - Paper reference values                (Moon, 4-hop: polar+ELFO+halo)

Outputs:
  - Table to stdout
  - s_vs_relay_depth.pdf / .png
  - s_vs_hops_results.json
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parent

# ── Helpers ────────────────────────────────────────────────────


def load_json(name):
    p = RUNS / name
    if not p.exists():
        print(f"WARNING: {p} not found, skipping")
        return None
    with open(p) as f:
        return json.load(f)


def compute_s(eta):
    """s = η / (1 - η).  Returns inf if η ≥ 1."""
    if eta >= 1.0:
        return float("inf")
    if eta <= 0.0:
        return 0.0
    return eta / (1.0 - eta)


# ── Collect all configs ────────────────────────────────────────


def main():
    configs = []  # list of dicts

    # --- Mars (2-hop: station → orbiter → Earth, NO inter-satellite links) ---

    mars = load_json("mars_eta_universality_results.json")
    if mars:
        seen_mars = set()
        for exp_name in ["exp1_architecture", "exp2_dsn_sweep", "exp3_latitude"]:
            for entry in mars["experiments"].get(exp_name, []):
                label = entry["label"]
                if label in seen_mars:
                    continue
                seen_mars.add(label)
                eta = entry["estimates"]["NORMAL"]["eta_mean"]
                st = entry["oracle"]["S_T_full"]
                if st < 0.95:
                    continue
                configs.append(
                    {
                        "label": f"Mars {label}",
                        "body": "Mars",
                        "eta_norm": eta,
                        "s": compute_s(eta),
                        "S_T_full": st,
                        "relay_layers": 1,
                        "category": "Mars 2-hop\n(no ISL)",
                        "cat_order": 0,
                    }
                )

    # --- Cislunar pure-polar thinning (2-hop + ISL between polars) ---

    purepolar = load_json("cislunar_purepolar_results.json")
    if purepolar:
        for entry in purepolar.get("exp2_thinning", []):
            eta = entry["eta_norm"]
            st = entry["s_t_full"]
            n = entry["n_polars"]
            if st < 0.95:
                continue
            configs.append(
                {
                    "label": f"Pure-polar n={n}",
                    "body": "Moon",
                    "eta_norm": eta,
                    "s": compute_s(eta),
                    "S_T_full": st,
                    "relay_layers": 1,
                    "category": "Cislunar 2-hop\n(polar only, ISL)",
                    "cat_order": 1,
                }
            )

    # --- Cislunar nohalo thinning (3-hop: polar + ELFO relay) ---

    nohalo = load_json("cislunar_nohalo_results.json")
    if nohalo:
        for entry in nohalo.get("exp2_thinning", []):
            eta = entry["eta_norm"]
            st = entry["s_t_full"]
            n = entry["n_polars"]
            if st < 0.95:
                continue
            configs.append(
                {
                    "label": f"Nohalo n={n}",
                    "body": "Moon",
                    "eta_norm": eta,
                    "s": compute_s(eta),
                    "S_T_full": st,
                    "relay_layers": 2,
                    "category": "Cislunar 3-hop\n(polar+ELFO, ISL)",
                    "cat_order": 2,
                }
            )

    # --- Paper reference: full relay (4-hop: polar + ELFO + halo) ---

    paper_refs = [
        ("8-polar+relay", 0.681, 1.0),
        ("1-polar+relay", 0.706, 1.0),
    ]
    for name, eta, st in paper_refs:
        configs.append(
            {
                "label": f"Full relay {name}",
                "body": "Moon",
                "eta_norm": eta,
                "s": compute_s(eta),
                "S_T_full": st,
                "relay_layers": 3,
                "category": "Cislunar 4-hop\n(polar+ELFO+halo)",
                "cat_order": 3,
            }
        )

    # --- Cislunar baseline (DSN=16h, 8 polar+ELFO+halo, S_T=0.667) ---
    # NOT included: sub-critical (S_T_full = 0.667), η is unreliable
    # The paper references above are the canonical full-relay values

    # ── Sort by category, then s descending ────────────────────────

    configs.sort(key=lambda c: (c["cat_order"], -c["s"]))

    # ── Print table ────────────────────────────────────────────────

    print()
    print("=" * 95)
    print(
        f"{'Config':<35} {'η_norm':>8} {'s=η/(1-η)':>12} {'S_T_full':>8} {'Layers':>7} {'Category':>20}"
    )
    print("=" * 95)

    for c in configs:
        s_str = f"{c['s']:.1f}" if c["s"] < 10000 else "∞"
        cat_short = c["category"].split("\n")[0]
        print(
            f"{c['label']:<35} {c['eta_norm']:>8.4f} {s_str:>12} "
            f"{c['S_T_full']:>8.3f} {c['relay_layers']:>7d} {cat_short:>20}"
        )

    print("=" * 95)

    # ── Category summaries ─────────────────────────────────────────

    categories = {}
    for c in configs:
        cat = c["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(c)

    print()
    print("-" * 80)
    print(
        f"{'Category':<40} {'n':>3} {'s_mean':>9} {'s_std':>9} {'s_min':>9} {'s_max':>9} {'η_mean':>8}"
    )
    print("-" * 80)

    cat_stats = {}
    for cat in sorted(categories, key=lambda k: categories[k][0]["cat_order"]):
        entries = categories[cat]
        s_finite = [e["s"] for e in entries if e["s"] < float("inf")]
        eta_vals = [e["eta_norm"] for e in entries]
        n = len(entries)
        s_mean = float(np.mean(s_finite)) if s_finite else float("inf")
        s_std = float(np.std(s_finite)) if len(s_finite) > 1 else 0.0
        s_min = min(s_finite) if s_finite else float("inf")
        s_max = max(s_finite) if s_finite else float("inf")
        eta_mean = float(np.mean(eta_vals))

        cat_label = cat.replace("\n", " ")
        print(
            f"{cat_label:<40} {n:>3} {s_mean:>9.1f} {s_std:>9.1f} "
            f"{s_min:>9.1f} {s_max:>9.1f} {eta_mean:>8.4f}"
        )

        hop_count = {0: 2, 1: 2, 2: 3, 3: 4}[entries[0]["cat_order"]]
        cat_stats[cat] = {
            "relay_depth": hop_count,
            "n": n,
            "s_mean": s_mean,
            "s_std": s_std,
            "s_min": s_min,
            "s_max": s_max,
            "eta_mean": eta_mean,
        }

    print("-" * 80)

    # ── Key diagnostic: δ per hop ──────────────────────────────────

    print()
    print("CTMC per-hop overhead diagnostic")
    print("-" * 60)
    print("If s = γ/(n_hops · δ₁), then δ₁ = γ/(n_hops · s)")
    print("Setting γ = 1 (arbitrary scale):")
    print()

    for cat in sorted(cat_stats, key=lambda k: cat_stats[k]["relay_depth"]):
        stats = cat_stats[cat]
        n_hops = stats["relay_depth"]
        s_m = stats["s_mean"]
        if s_m > 0:
            delta_1 = 1.0 / (n_hops * s_m)
            cat_label = cat.replace("\n", " ")
            print(
                f"  {cat_label:<40}  hops={n_hops}  s_mean={s_m:>7.1f}  "
                f"δ₁·n={1 / s_m:.4f}  δ₁={delta_1:.4f}"
            )

    # ── Plot ───────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax1, ax2 = axes

    # Colors per category
    COLORS = {
        0: "#2196F3",  # Mars: blue
        1: "#FF9800",  # Pure-polar: orange
        2: "#4CAF50",  # Nohalo: green
        3: "#E91E63",  # Full relay: pink
    }

    # Hop counts for x-axis positioning
    HOP_MAP = {0: 2, 1: 2, 2: 3, 3: 4}

    # Jitter for visibility
    rng = np.random.default_rng(42)

    # ── Left panel: s vs relay depth (log scale) ──

    for c in configs:
        if c["s"] >= 10000:
            continue
        hops = HOP_MAP[c["cat_order"]]
        jitter = rng.uniform(-0.12, 0.12)
        color = COLORS[c["cat_order"]]
        ax1.scatter(hops + jitter, c["s"], c=color, s=50, alpha=0.6, edgecolors="k", linewidth=0.4)

    # Category means with diamonds
    for cat, entries in categories.items():
        s_finite = [e["s"] for e in entries if e["s"] < 10000]
        if not s_finite:
            continue
        hops = HOP_MAP[entries[0]["cat_order"]]
        s_mean = np.mean(s_finite)
        color = COLORS[entries[0]["cat_order"]]
        cat_short = cat.replace("\n", " ")
        ax1.scatter(
            hops,
            s_mean,
            c=color,
            s=220,
            marker="D",
            edgecolors="k",
            linewidth=1.5,
            zorder=10,
            label=f"{cat_short}\n  mean s = {s_mean:.1f}",
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("Relay depth (hops)", fontsize=12)
    ax1.set_ylabel("s = η / (1 − η)   [log scale]", fontsize=12)
    ax1.set_title("CTMC Rate Ratio vs Relay Depth", fontsize=13, fontweight="bold")
    ax1.set_xticks([2, 3, 4])
    ax1.set_xlim(1.5, 4.5)
    ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, which="both")

    # Annotate: Mars at hop=2 is separate from cislunar at hop=2
    ax1.annotate(
        "Mars (no ISL) →", xy=(1.7, 150), fontsize=8, color="#2196F3", ha="right", style="italic"
    )
    ax1.annotate(
        "← Cislunar (with ISL)",
        xy=(2.3, 25),
        fontsize=8,
        color="#FF9800",
        ha="left",
        style="italic",
    )

    # ── Right panel: η vs relay depth ──

    for c in configs:
        hops = HOP_MAP[c["cat_order"]]
        jitter = rng.uniform(-0.12, 0.12)
        color = COLORS[c["cat_order"]]
        ax2.scatter(
            hops + jitter, c["eta_norm"], c=color, s=50, alpha=0.6, edgecolors="k", linewidth=0.4
        )

    for cat, entries in categories.items():
        hops = HOP_MAP[entries[0]["cat_order"]]
        eta_mean = np.mean([e["eta_norm"] for e in entries])
        color = COLORS[entries[0]["cat_order"]]
        cat_short = cat.replace("\n", " ")
        ax2.scatter(
            hops,
            eta_mean,
            c=color,
            s=220,
            marker="D",
            edgecolors="k",
            linewidth=1.5,
            zorder=10,
            label=f"mean η = {eta_mean:.3f}",
        )

    ax2.set_xlabel("Relay depth (hops)", fontsize=12)
    ax2.set_ylabel("η (conditional efficiency)", fontsize=12)
    ax2.set_title("Efficiency vs Relay Depth", fontsize=13, fontweight="bold")
    ax2.set_xticks([2, 3, 4])
    ax2.set_xlim(1.5, 4.5)
    ax2.set_ylim(0.55, 1.05)
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Add η₀ = s/(s+1) theoretical curve on right panel
    # using fitted s = A / n_hops model
    # Fit A from the category means:
    # At hops=4, s_mean≈2.16, so A = 4 * 2.16 = 8.65
    # At hops=3, s_mean≈? (from nohalo stats)
    # Let's just draw a smooth guide curve
    hops_smooth = np.linspace(1.5, 4.5, 200)
    # Simple model: s = C / hops^alpha
    # Using 2 anchor points: (hops=2, s=100) and (hops=4, s=2.16)
    # log(s) = log(C) - alpha * log(hops)
    # log(100) = log(C) - alpha * log(2)
    # log(2.16) = log(C) - alpha * log(4)
    # Subtracting: log(100/2.16) = alpha * (log(4) - log(2)) = alpha * log(2)
    # alpha = log(100/2.16) / log(2) = log(46.3) / log(2) ≈ 5.53
    # C = 100 * 2^5.53 ≈ 100 * 46.3 = 4630
    alpha_fit = np.log(100 / 2.16) / np.log(2)
    C_fit = 100.0 * (2.0**alpha_fit)
    s_model = C_fit / (hops_smooth**alpha_fit)
    eta_model = s_model / (s_model + 1.0)
    ax2.plot(
        hops_smooth,
        eta_model,
        "k--",
        alpha=0.4,
        linewidth=1.5,
        label=f"Guide: s ∝ 1/h^{alpha_fit:.1f}",
    )
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)

    plt.tight_layout()

    # Save
    out_pdf = RUNS / "s_vs_relay_depth.pdf"
    out_png = RUNS / "s_vs_relay_depth.png"
    fig.savefig(str(out_pdf), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_pdf}")
    print(f"Figure saved: {out_png}")

    # ── JSON output ────────────────────────────────────────────────

    out_json = RUNS / "s_vs_hops_results.json"
    results = {
        "configs": configs,
        "category_summary": {cat.replace("\n", " "): cat_stats[cat] for cat in cat_stats},
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"JSON saved: {out_json}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
