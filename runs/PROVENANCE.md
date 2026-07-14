# Data Provenance Manifest

Generated: 2026-07-13T22:11:24.841884+00:00
Git HEAD: `1e15012`
Tracked result JSONs scanned: 308
Matched to current public scripts: 3/308

---

## Scope

This file summarizes result provenance for the public checkout.
`runs/CONFIG_MANIFEST.json` is the committed historical hash and
heuristic-count snapshot. `runs/PROVENANCE.json` is generated locally
by this script, contains current per-entry hashes and producer-match state,
and is ignored because it also records checkout-specific mtimes.

The historical runner table is not used as a source of truth here. Several
tracked results are retained as public data artifacts even when their
original producing runners are not present in the current public checkout.

---

## Archived Artifacts

| Result | Status | Result SHA256 (first 12) |
|--------|--------|--------------------------|
| `load_sweep_v2_results.json` | archived artifact | `7df7dae8d986` |
| `period_sweep_results.json` | archived artifact | `f3e23d3b9fc2` |

---

## Current Match Summary

- Current-script matches: 3
- Archived artifacts: 2
- Unmatched tracked JSONs: 303

Unmatched alone establishes neither validity nor reproducibility. It means
this checkout does not contain a current public runner that the naming
matcher can bind to that result. Use the per-entry `result_sha256` in
generated `runs/PROVENANCE.json` or the tracked git blob for integrity checks.

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
