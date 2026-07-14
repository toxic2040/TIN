#!/usr/bin/env python3
"""Build a current-checkout manifest for present tracked result JSONs.

By default this writes ignored `runs/CONFIG_MANIFEST.current.json`. The
committed `runs/CONFIG_MANIFEST.json` is a historical snapshot and is protected
from accidental overwrite unless both an explicit output path and
`--overwrite-historical` are supplied. Per-file count fields come from
heterogeneous JSON structures; their sum is not a globally unique configuration
count. In a git checkout, the exact file scope is every present tracked JSON
recursively below `runs/`, excluding the three manifest metadata outputs.

For each JSON: filename, SHA-256 hash, file size, config count,
and top-level structure description.

Usage:
    python runs/build_config_manifest.py
    python runs/build_config_manifest.py --output /tmp/tin-config-manifest.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_RUNS = _HERE
_HISTORICAL_MANIFEST = _RUNS / "CONFIG_MANIFEST.json"
_DEFAULT_CURRENT_MANIFEST = _RUNS / "CONFIG_MANIFEST.current.json"
_MANIFEST_METADATA = {
    "CONFIG_MANIFEST.current.json",
    "CONFIG_MANIFEST.json",
    "PROVENANCE.json",
}


def _include_result_json(path: Path, runs_dir: Path = _RUNS) -> bool:
    """Return whether path is a result JSON within the declared scan scope."""
    try:
        relative_path = path.relative_to(runs_dir).as_posix()
    except ValueError:
        return False
    return path.suffix == ".json" and relative_path not in _MANIFEST_METADATA


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_configs(data) -> int:
    """Return a best-effort per-file count from varied JSON structures.

    The result may count configurations, result rows, list elements, or
    top-level keys. It is useful for file-level accounting only.
    """
    if isinstance(data, list):
        return len(data)

    if not isinstance(data, dict):
        return 1

    # Check for explicit count keys
    for key in ("n_configs", "total_configs", "num_configs", "count"):
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                return int(val)

    # Check for 'results' key containing a list or dict
    if "results" in data:
        r = data["results"]
        if isinstance(r, list):
            return len(r)
        if isinstance(r, dict):
            return len(r)

    # Find the longest list value (likely the main data)
    max_list_len = 0
    max_list_key = None
    for key, val in data.items():
        if isinstance(val, list) and len(val) > max_list_len:
            max_list_len = len(val)
            max_list_key = key

    if max_list_len > 0:
        return max_list_len

    # Flat dict — count top-level entries as configs
    # (conservative: may undercount)
    return len(data)


def collect_jsons(runs_dir: Path = _RUNS) -> list[Path]:
    """Collect the exact current-checkout result population.

    In a git checkout, the population is every present tracked JSON below
    ``runs/``, recursively, except the three manifest metadata files. In an
    unpacked archive without git metadata, use the equivalent recursive on-disk
    scan. Returning immediately after a successful ``git ls-files`` call keeps
    untracked local result sprawl out of the checkout manifest.
    """
    repo_root = runs_dir.parent
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", runs_dir.name],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path
            for path in runs_dir.rglob("*.json")
            if path.exists() and _include_result_json(path, runs_dir)
        )

    paths = []
    for row in tracked.splitlines():
        path = repo_root / row
        if path.exists() and _include_result_json(path, runs_dir):
            paths.append(path)
    return sorted(paths)


def _manifest_entry(path: Path, runs_dir: Path) -> dict:
    """Build one hash/count entry, preserving parse errors as manifest rows."""
    rel = path.relative_to(runs_dir).as_posix()
    try:
        sha = _sha256(path)
        size_bytes = path.stat().st_size
        with open(path) as f:
            data = json.load(f)
        n_configs = _count_configs(data)
        if isinstance(data, list):
            structure = f"list[{len(data)}]"
        elif isinstance(data, dict):
            structure = f"dict[{len(data)} keys]"
        else:
            structure = type(data).__name__
    except (json.JSONDecodeError, OSError) as exc:
        return {"file": rel, "error": str(exc)}

    return {
        "file": rel,
        "sha256": sha,
        "size_bytes": size_bytes,
        "n_configs": n_configs,
        "structure": structure,
    }


def build_manifest(runs_dir: Path = _RUNS) -> dict:
    """Build the current-checkout manifest without writing it."""
    entries = [_manifest_entry(path, runs_dir) for path in collect_jsons(runs_dir)]

    heuristic_count_sum = sum(e.get("n_configs", 0) for e in entries)
    total_files = len([e for e in entries if "error" not in e])
    total_bytes = sum(e.get("size_bytes", 0) for e in entries)
    errors = [e for e in entries if "error" in e]

    return {
        "generated_by": "runs/build_config_manifest.py",
        "description": (
            "Current-checkout manifest of present git-tracked result JSONs recursively "
            "under runs/. In an unpacked archive without git metadata, the same recursive "
            "on-disk scope is used. Manifest metadata outputs are excluded. It records "
            "SHA-256 hashes and heterogeneous per-file counts; it does not establish that "
            "every file is regenerable or globally configuration-distinct."
        ),
        "count_semantics": (
            "n_configs may be an explicit count, results length, longest list "
            "length, or top-level dict size. summary.total_configs is the "
            "non-additive sum of those heterogeneous values."
        ),
        "data_license": "CC-BY-4.0 (see data/LICENSE_DATA)",
        "summary": {
            "total_files": total_files,
            "total_configs": heuristic_count_sum,
            "total_size_mb": round(total_bytes / 1_048_576, 1),
            "errors": len(errors),
        },
        "files": entries,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_CURRENT_MANIFEST,
        help="Output JSON path (default: runs/CONFIG_MANIFEST.current.json)",
    )
    parser.add_argument(
        "--overwrite-historical",
        action="store_true",
        help="Permit an explicit --output targeting the committed historical snapshot",
    )
    args = parser.parse_args(argv)

    if args.output.resolve() == _HISTORICAL_MANIFEST.resolve() and not args.overwrite_historical:
        parser.error(
            "refusing to overwrite runs/CONFIG_MANIFEST.json; use the default current-checkout "
            "output or add --overwrite-historical explicitly"
        )
    return args


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    manifest = build_manifest()
    summary = manifest["summary"]
    errors = [entry for entry in manifest["files"] if "error" in entry]

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {out_path}")
    print(f"  Files: {summary['total_files']}")
    print(f"  Heuristic count sum: {summary['total_configs']:,}")
    print(f"  Total size: {summary['total_size_mb']:.1f} MB")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    {e['file']}: {e['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
