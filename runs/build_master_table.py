"""build_master_table.py — Consolidate all experimental results into one table.

Reads every *_results.json in runs/ and extracts key metrics into a unified
comparison table. This is the "data backbone" for the companion note.
"""

import json
from pathlib import Path

_HERE = Path(__file__).parent


def _scalar(v, default=0):
    """Extract a scalar from a value that might be a dict with 'mean'/'norm' key."""
    if v is None:
        return float(default)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for key in ("mean", "norm", "value"):
            if key in v:
                return float(v[key])
        return float(default)
    return float(default)


def _load(name):
    p = _HERE / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def main():
    rows = []

    # --- Mars ISL ablation ---
    d = _load("isl_ablation_results.json")
    if d:
        for key in ["with_isl", "no_isl"]:
            entry = d.get(key, {})
            if "eta_norm" in entry:
                rows.append(
                    {
                        "body": "Mars",
                        "label": entry.get("label", key),
                        "K": 1,
                        "n_sats": 6,
                        "ISL": "with" if entry.get("n_isl", 0) > 0 else "without",
                        "hops_mean": entry["hops_norm"]["mean"]
                        if isinstance(entry.get("hops_norm"), dict)
                        else entry.get("hops_mean", 0),
                        "eta_norm": entry["eta_norm"],
                        "eta_emerg": entry.get("eta_emerg"),
                        "s": entry.get("s_norm", 0),
                        "no_route": entry.get("failures_norm", {}).get("no_route", 0)
                        if isinstance(entry.get("failures_norm"), dict)
                        else 0,
                        "filter_L": 2,
                        "source": "isl_ablation",
                    }
                )

    # --- Mars K=1→K=2 heterogeneous ---
    d = _load("mars_hetero_results.json")
    if d:
        for entry in d:
            k = 1 if "K=1" in entry["label"] else 2
            rows.append(
                {
                    "body": "Mars",
                    "label": entry["label"],
                    "K": k,
                    "n_sats": entry["n_sats"],
                    "ISL": "with",
                    "hops_mean": entry.get("hops_mean", 0),
                    "eta_norm": entry["eta"],
                    "s": entry["s"],
                    "no_route": entry["failures"].get("no_route", 0),
                    "filter_L": 2,
                    "source": "mars_hetero",
                }
            )

    # --- Mars distance sweep ---
    d = _load("distance_sweep_results.json")
    if d:
        for entry in d:
            rows.append(
                {
                    "body": f"Mars@{entry['distance_au']}AU",
                    "label": f"n=6, {entry['distance_au']}AU",
                    "K": 1,
                    "n_sats": 6,
                    "ISL": "with",
                    "hops_mean": 7.0,  # from instrumented
                    "eta_norm": entry["eta_norm"],
                    "eta_emerg": entry.get("eta_emerg"),
                    "s": entry["eta_norm"] / (1 - entry["eta_norm"])
                    if entry["eta_norm"] < 1.0
                    else float("inf"),
                    "S_T_full": entry["S_T_full"],
                    "S_T_tau": entry.get("S_T_tau"),
                    "filter_L": 2,
                    "source": "distance_sweep",
                }
            )

    # --- Mars universality sweep ---
    d = _load("mars_eta_universality_results.json")
    if d:
        experiments = d.get("experiments", {})
        for exp_name, exp_data in experiments.items():
            if not isinstance(exp_data, list):
                continue
            for entry in exp_data:
                est = entry.get("estimates", {})
                norm = est.get("NORMAL", {})
                emerg = est.get("EMERGENCY", {})
                cfg = entry.get("config", {})
                oracle = entry.get("oracle", {})
                eta_n = norm.get("eta_mean")
                eta_e = emerg.get("eta_mean")
                if eta_n is not None:
                    rows.append(
                        {
                            "body": "Mars",
                            "label": entry.get("label", exp_name),
                            "K": 1,
                            "n_sats": cfg.get("n_orbiters", 6),
                            "ISL": "with",
                            "eta_norm": eta_n,
                            "eta_emerg": eta_e,
                            "s": eta_n / (1 - eta_n) if eta_n < 1.0 else float("inf"),
                            "S_T_full": oracle.get("S_T_full"),
                            "filter_L": 2,
                            "source": "mars_universality",
                        }
                    )

    # --- Moon ISL ablation ---
    d = _load("moon_isl_ablation_results.json")
    if d:
        for key in ["with_isl", "no_isl"]:
            entry = d.get(key, {})
            if "eta" in entry:
                rows.append(
                    {
                        "body": "Moon",
                        "label": entry.get("label", key),
                        "K": 1,
                        "n_sats": 8,
                        "ISL": "with" if key == "with_isl" else "without",
                        "hops_mean": entry.get("hops_mean", 0),
                        "eta_norm": entry["eta"],
                        "s": entry["s"],
                        "no_route": entry.get("failures", {}).get("no_route", 0),
                        "filter_L": 2,
                        "source": "moon_isl_ablation",
                    }
                )

    # --- Cislunar instrumented ---
    d = _load("cislunar_instrumented_results.json")
    if d:
        k_map = {"pure_polar_8": 1, "nohalo_8": 2, "full_relay_8": 3}
        for entry in d:
            lbl = entry.get("label", "")
            hops_dict = entry.get("hops", {})
            hops_mean = hops_dict.get("mean", 0) if isinstance(hops_dict, dict) else 0
            fails = entry.get("failures", {})
            nr = fails.get("no_route", 0) if isinstance(fails, dict) else 0
            rows.append(
                {
                    "body": "Moon",
                    "label": lbl,
                    "K": k_map.get(lbl, 0),
                    "n_sats": entry.get("n_sats", 0),
                    "ISL": "with",
                    "hops_mean": hops_mean,
                    "eta_norm": entry["eta_norm"],
                    "eta_emerg": entry.get("eta_emerg"),
                    "s": entry.get("s", entry.get("s_norm", 0)),
                    "no_route": nr,
                    "S_T_full": entry.get("S_T_full"),
                    "filter_L": 2,
                    "source": "cislunar_instrumented",
                }
            )

    # --- Filter depth sweep ---
    d = _load("filter_depth_sweep_results.json")
    if d:
        for config_label, results_list in d.items():
            for entry in results_list:
                k = 1 if "K=1" in config_label else 2
                rows.append(
                    {
                        "body": "Moon",
                        "label": f"{config_label} L={entry['depth']}",
                        "K": k,
                        "n_sats": 8 if k == 1 else 9,
                        "ISL": "with",
                        "hops_mean": entry.get("hops_mean", 0),
                        "eta_norm": entry["eta"],
                        "s": entry["s"],
                        "no_route": entry.get("no_route", 0),
                        "filter_L": entry["depth"],
                        "source": "filter_depth_sweep",
                    }
                )

    # --- Emergency η validation ---
    d = _load("eta_emerg_validation.json")
    if d and isinstance(d, dict):
        for entry in d.get("all_configs", []):
            rows.append(
                {
                    "body": "Mars",
                    "label": f"emerg_{entry.get('label', '?')}",
                    "K": 1,
                    "eta_emerg": entry.get("eta_emerg"),
                    "eta_tau": entry.get("eta_tau"),
                    "S_T_full": entry.get("S_T_full"),
                    "S_T_tau": entry.get("S_T_tau"),
                    "source": "eta_emerg_validation",
                }
            )

    # Print summary
    print("=" * 120)
    print("  MASTER COMPARISON TABLE")
    print("=" * 120)
    print(
        f"  {'Body':<12} {'Label':<35} {'K':>2} {'n':>3} {'ISL':>5} "
        f"{'hops':>5} {'η_norm':>7} {'s':>8} {'no_rt':>6} {'L':>3} {'source':<20}"
    )
    print("  " + "─" * 116)

    for r in rows:
        eta = r.get("eta_norm", 0)
        s = r.get("s", 0)
        body = r.get("body", "?")
        label = r.get("label", "?")[:34]
        k = r.get("K", "?")
        n = r.get("n_sats", "?")
        isl = r.get("ISL", "?")
        hops = r.get("hops_mean", "")
        nr = r.get("no_route", "")
        fl = r.get("filter_L", "")
        src = r.get("source", "")[:19]

        hops_s = f"{hops:>5.1f}" if isinstance(hops, (int, float)) and hops else f"{'':>5}"
        nr_s = f"{nr:>6}" if nr != "" else f"{'':>6}"
        fl_s = f"{fl:>3}" if fl != "" else f"{'':>3}"

        if isinstance(eta, (int, float)) and eta:
            print(
                f"  {body:<12} {label:<35} {k:>2} {n:>3} {isl:>5} "
                f"{hops_s} {eta:>7.4f} {s:>8.1f} {nr_s} {fl_s} {src:<20}"
            )

    # Key cross-body comparison
    print("\n" + "=" * 80)
    print("  KEY CROSS-BODY COMPARISON")
    print("=" * 80)
    highlights = [
        ("Mars K=1 (6 polar, with ISL)", "Mars", 1, "with", "isl_ablation"),
        ("Mars K=1 (6 polar, no ISL)", "Mars", 1, "without", "isl_ablation"),
        ("Mars K=2 (6+1 hetero)", "Mars", 2, "with", "mars_hetero"),
        ("Moon K=1 (8 polar, with ISL)", "Moon", 1, "with", "moon_isl_ablation"),
        ("Moon K=1 (8 polar, no ISL)", "Moon", 1, "without", "moon_isl_ablation"),
        ("Moon K=2 (nohalo)", "Moon", 2, "with", "cislunar_instrumented"),
        ("Moon K=3 (full relay)", "Moon", 3, "with", "cislunar_instrumented"),
    ]
    print(f"  {'Config':<35} {'K':>2} {'η':>7} {'s':>8} {'hops':>5} {'no_rt':>6}")
    print("  " + "─" * 65)
    for desc, body, k, isl, src in highlights:
        match = [
            r
            for r in rows
            if r.get("body") == body
            and r.get("K") == k
            and r.get("ISL", "") == isl
            and r.get("source") == src
            and r.get("filter_L", 2) == 2
        ]
        if match:
            r = match[0]
            eta = _scalar(r.get("eta_norm"), 0)
            s = _scalar(r.get("s"), 0)
            hops = _scalar(r.get("hops_mean"), 0)
            nr = int(_scalar(r.get("no_route"), 0))
            print(f"  {desc:<35} {k:>2} {eta:>7.4f} {s:>8.1f} {hops:>5.1f} {nr:>6}")

    print("\n  KEY FINDINGS:")
    print("  1. K=1 → s ≫ 1 (Mars 294, Moon 50, Moon no-ISL 159)")
    print("  2. K=2 → s collapses (Moon 8×: 50→6.1, Mars 1.9×: 294→157)")
    print("  3. ISL removal at K=1 → s INCREASES (Moon 50→159)")
    print("  4. Filter L=1 recovers ~60% of K=2 penalty (Moon s: 6.1→17)")
    print("  5. η_norm distance-invariant (0.5-2.5 AU)")

    # Save
    out = _HERE / "master_comparison.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n  Saved {len(rows)} rows → {out.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
