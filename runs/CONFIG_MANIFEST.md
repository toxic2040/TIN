# Result Artifact Manifest

This is a historical hash and accounting snapshot, not proof of a globally
unique configuration total. Result artifacts have mixed availability and
producer status in the current public checkout.

## Machine-Readable Manifest

**`runs/CONFIG_MANIFEST.json`** contains 300 historical entries with:
- stored SHA-256 hash for historical artifact-integrity checks
- a best-effort per-file `n_configs` field
- File size
- Structure description

The count heuristic uses, in order, an explicit count field, `results`
length, the longest list, or top-level dictionary size. Depending on the file,
that value can mean configurations, result rows, list elements, or keys.

The stored heuristic sum is **963,456** across 300 entries / 472.9 MiB. It is
neither a uniform raw-row count nor an additive count of distinct
configurations. The earlier `290,000+` and `425,000–483,000 unique`
framings are not established by this manifest and are retired.

At the 2026-07-13 reconciliation checkout, 34 of the 300 listed paths exist
under `runs/`; all 34 match their stored SHA-256 values. `PROVENANCE.md`
separately summarizes tracked JSON result artifacts under `runs/`. Path
absence here does not invalidate an archived artifact, but it does preclude
claiming that every listed file is shipped or currently regenerable.

The safe current-checkout generator recursively inventories all 308 present
Git-tracked result JSONs below `runs/`, including 273 under `runs/results/`.
It excludes `CONFIG_MANIFEST.json`, `CONFIG_MANIFEST.current.json`, and
`PROVENANCE.json` so metadata is not counted as experiment output. The current
heterogeneous heuristic sum is 200,255 across 97.1 MiB. Of these files, 34
overlap the historical snapshot and 274 belong only to the current population.

## Historical source groups

These rows are retained for historical navigation. They overlap and must not
be summed as a distinct-configuration census.

| Source | Historical row/key count | Runner / Location |
|--------|--------:|-------------------|
| Paper 1 revalidation (phi, CRAWDAD, vehicular) | 460,800 | `run_phi_decompose_*.py`, `run_crawdad_*.py`, `run_vehicular_*.py` |
| Production EPYC sweep (11 Mar) | 89,178 | `runs/epyc_results/production_2026_03_11/` |
| Forensic coupling sweep (21 Mar) | 82,515 | `run_forensic_coupling_sweep.py` |
| Local sweeps (Moon + Mars) | ~35,000 | Moon 5,670 + Mars 29,376 configs |
| Phase transition sweeps | ~32,000 | `run_phase_transition_*.py`, `run_cloud_sweep.py` |
| Campaign EPYC (11 Mar) | 7,026 | `runs/epyc_results/campaign_2026_03_11/` |
| Follow-up A/B/C | 3,120 | `run_followup_*.py` |
| Phase 5 EPYC (12 Mar) | ~1,900 | `runs/epyc_results/phase5_2026_03_12/` |
| UQ coupling v1-v4 (21 Mar) | ~1,600 | `run_uq_congestion*.py`, `run_uq_architectural_v3.py` |
| All other experiments | ~5,000 | See CONFIG_MANIFEST.json for file-level entries |

## Verification

1. For a listed file that is present, compare its SHA-256 hash against
   `CONFIG_MANIFEST.json`.
2. Build a current-checkout snapshot with `python runs/build_config_manifest.py`;
   the ignored output is `runs/CONFIG_MANIFEST.current.json`. The committed
   historical snapshot is not overwritten by default. Without git metadata,
   the builder falls back to the same recursive on-disk scope.
3. Regenerate a result only where `PROVENANCE.md` or a named runner supplies an exact
   producer contract; filename similarity alone is not provenance.
4. `runs/PROVENANCE.md` separates current-script matches, archived artifacts,
   and unmatched tracked JSONs. `runs/PROVENANCE.json` is generated locally
   by `runs/build_provenance_manifest.py`.

Archived artifact example: `load_sweep_v2_results.json` and
`period_sweep_results.json` are retained for public figure reproduction, but
their producing runners are not present in this checkout. Verify these two
files by SHA-256.

## Current check boundaries

- The package test suite is run with `pytest tests/ -x -q`; no fixed test
  count is treated as an invariant.
- `verify_paper_claims.py` is a historical manuscript-value reproducer.
  Aggregate checks that require complete input families fail closed as
  `SKIP`; wholly unavailable claim families are omitted and listed as not
  checkable. A `PASS` is not independent scientific validation.
- The factorization residual is an accounting-identity/counter-consistency
  check, not empirical evidence.
- The historical self-averaging result is scoped to its tested uniform-channel,
  sparse/unique-oracle-path regime; it is not a global invariant.

### Retired claim: gamma gap

The former "Gamma gap >= 1.95" invariant is retired. The historical calculation
mixed normalization conventions, and four published orbital values were
hardcoded without recoverable source rows. Under a port of the documented
orbital method, Titan re-derives as **+0.113** rather than the published
**−0.10**; the sign-based classification audit also used sign-derived labels
and was circular. No replacement cross-domain scalar or global threshold is
claimed.

`verify_paper_claims.py` claim `T1-004` reproduces only the historical
arithmetic when a local `bootstrap_ci_results.json` is supplied; that file
is absent from the stock public checkout, so the row is omitted and listed as
not checkable there. The TIN research program closed on 2026-06-26; this
repository is maintained as a simulator and historical reproducibility surface.

## Data License

Result JSONs are licensed under CC-BY-4.0 (see `data/LICENSE_DATA`).
Code is MIT (see `LICENSE`).
