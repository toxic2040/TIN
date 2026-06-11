#!/usr/bin/env python3
"""Build a provenance manifest linking every result JSON to its producing script.

Output: runs/PROVENANCE.json — machine-readable manifest
        runs/PROVENANCE.md  — human-readable summary

For each result JSON:
  - sha256 of the file
  - file size, modification time
  - matched producing script (by naming convention)
  - sha256 of the producing script (if found)
  - git commit of the script at time of last modification (if tracked)

Usage:
    python runs/build_provenance_manifest.py
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR = Path(__file__).parent
REPO_ROOT = RUNS_DIR.parent
ARCHIVED_ARTIFACTS = {
    "load_sweep_v2_results.json": "archived artifact",
    "period_sweep_results.json": "archived artifact",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_mtime_iso(path: Path) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def git_head_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def git_file_commit(path: Path) -> str:
    """Last commit that touched this file, or 'untracked'."""
    try:
        out = (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%h", "--", str(path)],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return out if out else "untracked"
    except Exception:
        return "untracked"


def match_script(json_path: Path, all_scripts: dict[str, Path]) -> Path | None:
    """Try to find the script that produced a given JSON.

    Naming conventions:
      run_X.py          -> X_results.json
      run_X.py          -> X.json  (rare)
      epyc_phaseN.py    -> production_*/campaign_*
      plot_X.py         -> X_data.json (rare)
    """
    name = json_path.name

    # Strip common suffixes to get the stem
    for suffix in ("_results.json", ".json"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break
    else:
        return None

    # Direct match: run_{stem}.py
    if f"run_{stem}" in all_scripts:
        return all_scripts[f"run_{stem}"]

    # Shard match: phi_sweep_shard_mercury -> run_phi_sweep.py
    for sep in ("_shard_", "_LOCAL_SMOKE_TEST"):
        if sep in stem:
            base_stem = stem[: stem.index(sep)]
            if f"run_{base_stem}" in all_scripts:
                return all_scripts[f"run_{base_stem}"]

    # CRAWDAD pattern: crawdad_contacts.Exp1_results -> run_crawdad_validation.py
    if stem.startswith("crawdad_contacts"):
        if "run_crawdad_validation" in all_scripts:
            return all_scripts["run_crawdad_validation"]

    # EPYC production/campaign patterns
    if stem.startswith("production_P"):
        if "run_production" in all_scripts:
            return all_scripts["run_production"]
    if stem.startswith("campaign_bucket"):
        if "run_campaign" in all_scripts:
            return all_scripts["run_campaign"]
    if stem.startswith("followup_"):
        if "run_followup_abc" in all_scripts:
            return all_scripts["run_followup_abc"]

    # routing_independence pattern
    if stem.startswith("routing_independence"):
        if "run_routing_independence" in all_scripts:
            return all_scripts["run_routing_independence"]

    # Summary files -> epyc orchestrator scripts
    if stem.endswith("_summary"):
        phase = stem.replace("_summary", "")
        for candidate in (f"epyc_{phase}", f"run_{phase}", "epyc_phase5"):
            if candidate in all_scripts:
                return all_scripts[candidate]

    # Holodeck data
    if stem == "holodeck_dtn_data":
        if "export_holodeck_dtn" in all_scripts:
            return all_scripts["export_holodeck_dtn"]

    # Reprocessed variants
    if stem.endswith("_reprocessed"):
        base = stem[: -len("_reprocessed")]
        if f"run_{base}" in all_scripts:
            return all_scripts[f"run_{base}"]
        if "reprocess_epyc_bugs" in all_scripts:
            return all_scripts["reprocess_epyc_bugs"]

    # Partial/checkpoint and phase-tagged variants
    for tag in ("_partial", "_phase6", "_phase5", "_phase4"):
        if stem.endswith(tag):
            base = stem[: -len(tag)]
            # strip trailing _results if present (e.g. per_path_cov_results_phase6)
            if base.endswith("_results"):
                base = base[: -len("_results")]
            if f"run_{base}" in all_scripts:
                return all_scripts[f"run_{base}"]

    # Analysis scripts: analysis_{stem}.py or {stem}.py
    # Also handle: competing_risk_v2_analysis -> analysis_competing_risk_v2
    for prefix in ("analysis_", "compute_", "fit_", "scope_", "generate_", ""):
        candidate = f"{prefix}{stem}"
        if candidate in all_scripts:
            return all_scripts[candidate]
    # Reversed: X_analysis -> analysis_X
    if stem.endswith("_analysis"):
        base = stem[: -len("_analysis")]
        if f"analysis_{base}" in all_scripts:
            return all_scripts[f"analysis_{base}"]

    # Specific known mappings
    KNOWN = {
        "bootstrap_ci": "epyc_phase1",
        "cislunar_baseline": "run_cislunar",
        "eta_baseline": "run_cislunar",
        "eta_dual_station": "run_eta",
        "eta_emerg_validation": "build_master_table",
        "master_comparison": "build_master_table",
        "kpz_scaling": "run_kpz_scaling_test",
        "conjecture_figure_data": "run_conjecture_figure",
        "conjunction_geometry": "scope_conjunction_geometry",
        "extended_seed_sweep": "epyc_phase2",
        "moon_oracle_verification": "epyc_phase2",
        "pair_gamma_power_analysis": "epyc_phase2",
        "fixed_grid_braess_epyc": "run_fixed_grid_braess",
        "crawdad_cross_trace_analysis": "run_crawdad_cross_trace",
        "vehicular_gamma_results_LOCAL_SMOKE_TEST": "run_vehicular_gamma",
        # Phase 6 outputs
        "hardening_tests_4_7": "run_hardening_tests",
        "hull_structure": "run_hull_structure_test",
        "crawdad_neff": "run_hardening_tests",
        "ablation": "run_remaining_analysis",
        "neff": "run_remaining_analysis",
        # Canonical recomputation scripts (no run_ prefix)
        "var_log_p_canonical": "recompute_var_log_p_r2",
        "gamma_oracle_canonical": "recompute_gamma_oracle",
        # Whitespace analysis (no run_ prefix)
        "whitespace_analysis": "run_whitespace_analysis",
        # Script name ≠ result name (expanded script names)
        "architecture_b": "run_architecture_b_isru_direct",
        "cold_chain": "run_cold_chain_worked_example",
        "entropy_duality": "run_entropy_duality_test",
        "independence_test": "run_independence_tests",
        "tau_sensitivity": "run_tau_sensitivity_sweep",
        # Renamed / backed-up outputs
        "itn_delta_screen": "run_itn_delta_screen_analysis",
        "strike2_commodity_oracle": "run_commodity_oracle_v2",
        # Formerly inline analysis, reproducer written 2026-03-22
        "forensic_coupling_sweep": "run_forensic_coupling_sweep",
    }
    if stem in KNOWN and KNOWN[stem] in all_scripts:
        return all_scripts[KNOWN[stem]]

    # phase_transition_T* -> run_phase_transition_atlas.py
    if stem.startswith("phase_transition_T"):
        if "run_phase_transition_atlas" in all_scripts:
            return all_scripts["run_phase_transition_atlas"]

    return None


def collect_jsons(runs_dir: Path) -> list[Path]:
    """Find tracked JSON result files.

    Public provenance should not depend on local ignored result sprawl,
    symlinks into private dataset directories, or generated scratch outputs.
    In a git checkout, use the tracked file list. Outside git, fall back to
    a recursive scan so the script remains usable from an unpacked archive.
    """
    skip = {"PROVENANCE.json"}
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "runs"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        paths = []
        for row in out.splitlines():
            p = REPO_ROOT / row
            if p.suffix == ".json" and p.name not in skip and p.exists():
                paths.append(p)
        if paths:
            return sorted(paths)
    except Exception:
        pass
    return sorted(p for p in runs_dir.rglob("*.json") if p.name not in skip)


def collect_scripts(runs_dir: Path) -> dict[str, Path]:
    """Find all Python scripts, keyed by stem (no .py)."""
    scripts = {}
    for p in sorted(runs_dir.glob("*.py")):
        if p.name == "build_provenance_manifest.py":
            continue
        scripts[p.stem] = p
    return scripts


def build_manifest(runs_dir: Path) -> dict:
    jsons = collect_jsons(runs_dir)
    scripts = collect_scripts(runs_dir)
    head = git_head_short()

    entries = []
    matched = 0
    unmatched_files = []

    for jp in jsons:
        rel = jp.relative_to(runs_dir)
        rel_str = str(rel)
        script = match_script(jp, scripts)
        script_rel = script.relative_to(runs_dir) if script else None
        is_archived = rel_str in ARCHIVED_ARTIFACTS

        entry = {
            "result_file": rel_str,
            "result_sha256": sha256(jp),
            "result_bytes": jp.stat().st_size,
            "result_mtime": file_mtime_iso(jp),
            "script": str(script_rel) if script_rel else ARCHIVED_ARTIFACTS.get(rel_str),
            "script_sha256": sha256(script) if script else None,
            "script_last_commit": git_file_commit(script) if script else None,
            "archived": is_archived,
        }
        entries.append(entry)

        if script:
            matched += 1
        elif not is_archived:
            unmatched_files.append(rel_str)

    manifest = {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "git_head": head,
        "runs_dir": str(runs_dir),
        "total_jsons": len(jsons),
        "matched": matched,
        "unmatched": len(unmatched_files),
        "unmatched_files": unmatched_files,
        "entries": entries,
    }
    return manifest


def write_markdown(manifest: dict, out_path: Path):
    archived_entries = [e for e in manifest["entries"] if e.get("archived")]
    lines = [
        "# Data Provenance Manifest",
        "",
        f"Generated: {manifest['generated']}",
        f"Git HEAD: `{manifest['git_head']}`",
        f"Tracked result JSONs scanned: {manifest['total_jsons']}",
        f"Matched to current public scripts: {manifest['matched']}/{manifest['total_jsons']}",
        "",
        "---",
        "",
        "## Scope",
        "",
        "This file summarizes result provenance for the public checkout. The",
        "machine-readable hash and count manifest is `runs/CONFIG_MANIFEST.json`.",
        "`runs/PROVENANCE.json` is generated locally by this script and is ignored",
        "because it records checkout-specific mtimes and local match state.",
        "",
        "The historical runner table is not used as a source of truth here. Several",
        "tracked results are retained as public data artifacts even when their",
        "original producing runners are not present in current public `main`.",
        "",
        "---",
        "",
        "## Archived Artifacts",
        "",
        "| Result | Status | Result SHA256 (first 12) |",
        "|--------|--------|--------------------------|",
    ]

    if archived_entries:
        for e in archived_entries:
            lines.append(
                f"| `{e['result_file']}` | archived artifact | `{e['result_sha256'][:12]}` |"
            )
    else:
        lines.append("| none | | |")

    lines += [
        "",
        "---",
        "",
        "## Current Match Summary",
        "",
        f"- Current-script matches: {manifest['matched']}",
        f"- Archived artifacts: {len(archived_entries)}",
        f"- Unmatched tracked JSONs: {manifest['unmatched']}",
        "",
        "Unmatched does not mean invalid. It means this checkout does not contain a",
        "current public runner that the naming matcher can bind to that result. Use",
        "`runs/CONFIG_MANIFEST.json` for hash verification.",
    ]

    lines += [
        "",
        "---",
        "",
        "## Verification",
        "",
        "To verify any result file has not been modified since this manifest was built:",
        "```bash",
        "sha256sum runs/<result_file>",
        "# Compare to result_sha256 in PROVENANCE.json",
        "```",
        "",
        "To regenerate:",
        "```bash",
        "python runs/build_provenance_manifest.py",
        "```",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def main():
    runs_dir = RUNS_DIR
    manifest = build_manifest(runs_dir)

    json_path = runs_dir / "PROVENANCE.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Written: {json_path} ({len(manifest['entries'])} entries)")

    md_path = runs_dir / "PROVENANCE.md"
    write_markdown(manifest, md_path)
    print(f"Written: {md_path}")

    print(f"\nMatched: {manifest['matched']}/{manifest['total_jsons']}")
    if manifest["unmatched_files"]:
        print(f"Unmatched ({manifest['unmatched']}): see {runs_dir / 'PROVENANCE.json'}")


if __name__ == "__main__":
    main()
