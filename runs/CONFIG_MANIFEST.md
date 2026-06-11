# Configuration Manifest

How we arrive at "290,000+ configurations." Result JSONs are gitignored
(regenerable from runners), so this manifest provides the public accounting.

## Machine-Readable Manifest

**`runs/CONFIG_MANIFEST.json`** — 300 result files, each with:
- SHA-256 hash (verify after regeneration)
- Config count
- File size
- Structure description

Built with `python runs/build_config_manifest.py`.

**Totals:** 963,456 raw records across 300 files, 473 MB.

**After deduplication:** ~425,000–483,000 unique configurations.
The raw total double-counts through summary/metadata files (~202K),
shard-parent duplicates (~35K), phi-family containment (~240K), and
re-run duplicates (~6K). The "290,000+" claim is a conservative lower
bound, well supported after deduplication.

## By Source

| Source | Configs | Runner / Location |
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
| All other experiments | ~5,000 | See CONFIG_MANIFEST.json for full breakdown |

## Verification

1. Regenerate non-archived result JSONs by running the corresponding `run_*.py` script.
2. Compare SHA-256 hash against `CONFIG_MANIFEST.json`.
3. Provenance tracking: `runs/PROVENANCE.md` summarizes current-script matches,
   archived artifacts, and unmatched tracked JSONs. `runs/PROVENANCE.json` is
   generated locally by `runs/build_provenance_manifest.py`.

Archived artifact exception: `load_sweep_v2_results.json` and
`period_sweep_results.json` are committed for public figure reproduction, but
their producing runners are not present in current public `main`. Verify these
two files by SHA-256 unless source runners are deliberately restored.

## Key Invariants

| Claim | Value | How to verify |
|-------|-------|---------------|
| Factorization residual | <= 1.11e-16 | `verify_paper_claims.py` check #2 (per-config records only; aggregated figure-data files may exceed this due to Jensen's inequality on averaged S_T and η) |
| Self-averaging CV | < 0.4% | `verify_paper_claims.py` check #4 |
| Gamma gap | >= 1.95 | `verify_paper_claims.py` check #7 |
| Test suite | 73 passed | `pytest tests/ -x -q` |

## Data License

Result JSONs are licensed under CC-BY-4.0 (see `data/LICENSE_DATA`).
Code is MIT (see `LICENSE`).
