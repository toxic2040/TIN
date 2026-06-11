#!/usr/bin/env python3
"""Build a public manifest of all experiment result JSONs.

Produces runs/CONFIG_MANIFEST.json — a committed summary artifact
that lets anyone verify the 290,000+ configuration claim without
needing the gitignored result JSONs.

For each JSON: filename, SHA-256 hash, file size, config count,
and top-level structure description.

Usage:
    python runs/build_config_manifest.py
"""

import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_RUNS = _HERE
_EPYC = _HERE / "epyc_results"
_RETIRED_RESULT_PREFIXES = ("rgen_",)


def _include_result_json(path: Path) -> bool:
    if path.name in ("CONFIG_MANIFEST.json", "PROVENANCE.json"):
        return False
    return not path.name.startswith(_RETIRED_RESULT_PREFIXES)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_configs(data) -> int:
    """Best-effort config count from varied JSON structures."""
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


def _scan_dir(directory: Path, prefix: str = "") -> list[dict]:
    """Scan a directory for JSON files and build manifest entries."""
    entries = []
    if not directory.exists():
        return entries

    for path in sorted(directory.glob("**/*.json")):
        if not _include_result_json(path):
            continue

        rel = str(path.relative_to(_RUNS))
        if prefix:
            rel = f"{prefix}/{rel}"

        try:
            sha = _sha256(path)
            size_bytes = path.stat().st_size

            with open(path) as f:
                data = json.load(f)

            n_configs = _count_configs(data)

            # Structure description
            if isinstance(data, list):
                structure = f"list[{len(data)}]"
            elif isinstance(data, dict):
                structure = f"dict[{len(data)} keys]"
            else:
                structure = type(data).__name__

        except (json.JSONDecodeError, OSError) as exc:
            entries.append(
                {
                    "file": rel,
                    "error": str(exc),
                }
            )
            continue

        entries.append(
            {
                "file": rel,
                "sha256": sha,
                "size_bytes": size_bytes,
                "n_configs": n_configs,
                "structure": structure,
            }
        )

    return entries


def main():
    entries = []

    # Scan runs/*.json (top-level)
    for path in sorted(_RUNS.glob("*.json")):
        if not _include_result_json(path):
            continue
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
            entries.append({"file": path.name, "error": str(exc)})
            continue

        entries.append(
            {
                "file": path.name,
                "sha256": sha,
                "size_bytes": size_bytes,
                "n_configs": n_configs,
                "structure": structure,
            }
        )

    # Scan epyc_results/**/*.json
    if _EPYC.exists():
        for path in sorted(_EPYC.glob("**/*.json")):
            rel = str(path.relative_to(_RUNS))
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
                entries.append({"file": rel, "error": str(exc)})
                continue

            entries.append(
                {
                    "file": rel,
                    "sha256": sha,
                    "size_bytes": size_bytes,
                    "n_configs": n_configs,
                    "structure": structure,
                }
            )

    # Aggregate
    total_configs = sum(e.get("n_configs", 0) for e in entries)
    total_files = len([e for e in entries if "error" not in e])
    total_bytes = sum(e.get("size_bytes", 0) for e in entries)
    errors = [e for e in entries if "error" in e]

    manifest = {
        "generated_by": "runs/build_config_manifest.py",
        "description": (
            "Public manifest of all experiment result JSONs. "
            "Result JSONs are gitignored (regenerable from runner scripts). "
            "This manifest provides SHA-256 hashes and config counts "
            "so the 290,000+ claim can be verified by regenerating any "
            "subset and comparing hashes."
        ),
        "data_license": "CC-BY-4.0 (see data/LICENSE_DATA)",
        "summary": {
            "total_files": total_files,
            "total_configs": total_configs,
            "total_size_mb": round(total_bytes / 1_048_576, 1),
            "errors": len(errors),
        },
        "files": entries,
    }

    out_path = _RUNS / "CONFIG_MANIFEST.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {out_path}")
    print(f"  Files: {total_files}")
    print(f"  Total configs: {total_configs:,}")
    print(f"  Total size: {total_bytes / 1_048_576:.1f} MB")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    {e['file']}: {e['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
