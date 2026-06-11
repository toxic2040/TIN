# Data Provenance Manifest

Generated: 2026-05-12T18:19:41.383998+00:00
Git HEAD: `dae79b3`
Tracked result JSONs scanned: 307
Matched to current public scripts: 5/307

---

## Scope

This file summarizes result provenance for the public checkout. The
machine-readable hash and count manifest is `runs/CONFIG_MANIFEST.json`.
`runs/PROVENANCE.json` is generated locally by this script and is ignored
because it records checkout-specific mtimes and local match state.

The historical runner table is not used as a source of truth here. Several
tracked results are retained as public data artifacts even when their
original producing runners are not present in current public `main`.

---

## Archived Artifacts

| Result | Status | Result SHA256 (first 12) |
|--------|--------|--------------------------|
| `load_sweep_v2_results.json` | archived artifact | `7df7dae8d986` |
| `period_sweep_results.json` | archived artifact | `f3e23d3b9fc2` |

---

## Current Match Summary

- Current-script matches: 5
- Archived artifacts: 2
- Unmatched tracked JSONs: 300

Unmatched does not mean invalid. It means this checkout does not contain a
current public runner that the naming matcher can bind to that result. Use
`runs/CONFIG_MANIFEST.json` for hash verification.

---

## Verification

To verify any result file has not been modified since this manifest was built:
```bash
sha256sum runs/<result_file>
# Compare to result_sha256 in PROVENANCE.json
```

To regenerate:
```bash
python runs/build_provenance_manifest.py
```
