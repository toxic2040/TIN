# Experiment Index

**Last updated:** 2026-06-12 (repro/verify baseline rows added with measured run contract; historical table still needs full row-level regeneration)
**Total:** 111 Python scripts under `runs/` (counted: `find runs/ -maxdepth 1 -name "*.py" | wc -l`), JSON result files as per `runs/CONFIG_MANIFEST.md`

Status: **yes** = results exist | **--** = no results found | **lib** = library/not a runner

---

## Paper 1 — Classification Framework (Zenodo working-paper deposit)

**Topic:** DR factorization, trap/cluster classification via gamma, and related figure/result scripts.
**Data:** 290,000+ accounted configurations; see `runs/CONFIG_MANIFEST.md` for hash and count provenance.

### Runners

| Script | Results | Status | Description |
|--------|---------|--------|-------------|
| `conjecture_figure.py` | `conjecture_figure_data.json` | -- | Ergodic vs non-ergodic contact-density bound figure data |
| `eta_tau_fit.py` | `eta_tau_fit_results.json` | -- | CTMC and stretched-exponential fits to TTL x Distance surface |
| `helio_s_derivation.py` | `helio_s_derivation_results.json` | -- | Analytical s(p) model fitting from 780-epoch synodic sweep |
| `phi_sweep_merge.py` | `phi_sweep_results.json` | -- | Merges Phi shards; Phi_time/Phi_rel distributions, Braess boundary |

### Analysis & Verification

| Script | Results | Description |
|--------|---------|-------------|
| `analysis_pair_eta.py` | `analysis_pair_eta_results.json` | Per-pair eta ansatz test on CRAWDAD traces |
| `s_vs_decisions.py` | `s_vs_decisions_results.json` | s = eta/(1-eta) vs routing decisions power law test |
| `validate_eta_emerg.py` | `eta_emerg_validation.json` | Emergency-mode CTMC factorization validation |
| `verify_paper_claims.py` | — | Automated verification of ~35 quantitative claims in .tex |
| `independence_tests.py` | `independence_test_results.json` | Pooled + body-stratified MI(S_T;eta) and heteroscedasticity; 0.5452 nats on 91,540 configs, permutation null, no Simpson, U-dip in conditional mean |
| `independence_mi_attribution.py` | `independence_mi_attribution.json` | Within-bin shuffle attribution of the MI; 20-bin convention reproduces the 2026-03-15 anchor (0.421, 77.2% retained), decile variant 65.2%, bin-count sensitivity 56-85% |

### Plot Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `plot_fig1_classification.py` | `fig1_gamma_vs_p.*` | Gamma vs p_eff + gamma bar chart (8 bodies) |
| `plot_fig2_classification.py` | `fig2_gamma_saturation.*` | Mean gamma vs contact density for CRAWDAD traces |
| `plot_fig3_classification.py` | `fig3_mars_distance_law.*` | eta vs Earth-Mars distance with log-linear fits |
| `plot_fig4_phase_space.py` | `fig4_phase_space.*` | E[H] vs ln(Phi) phase space (trap vs cluster) |
| `plot_math_primer.py` | `fig_three_factor_decomposition.*` etc | 4 math figures: decomposition, Lyapunov, Phi range, Wald |
| `plot_finite_size_scaling.py` | FSS figures | 4-panel FSS suite: DR vs p, p_crit, collapse, Braess heatmap |

### Animations

| Script | Output | Description |
|--------|--------|-------------|
| `anim_t3_classification_reveal.py` | `t3_classification_reveal.gif` | Animated trap/cluster cloud reveal |

---

## Paper 2 — Mechanism Taxonomy (historical planning index)

**Topic:** Link heterogeneity, trap classification, and topological/routing/mixed trap mechanisms.
**Historical data row:** 89,178 production + 7,026 campaign + 3,120 follow-up configs.

### Runners

| Script | Results | Status | Description |
|--------|---------|--------|-------------|
| `decompose_analysis.py` | `decompose_analysis_results.json` | -- | Phi ≈ Phi_myopic x Phi_retry factorization test |
| `dr_fluctuation_spectrum.py` | `dr_fluctuation_spectrum_results.json` | -- | DR fluctuation spectrum (Var, skewness, kurtosis) |
| `eh_disentangle.py` | `eh_disentangle_results.json` | -- | E[H] confound in ln(Phi) via partial correlation |
| `ffwd_gap_analysis.py` | `ffwd_gap_analysis_results.json` | -- | f_fwd distribution and gap at f_fwd=1 boundary |
| `functional_forms.py` | `functional_forms_results.json` | -- | beta_myopic/beta_retry functional forms from 129k-point dataset |
| `functional_forms_v2.py` | `functional_forms_v2_results.json` | -- | Two-branch fitting strategy (v2) |
| `jupiter_boundary.py` | `jupiter_boundary_results.json` | -- | Jupiter as trap/cluster phase-transition boundary body |
| `kpz_scaling_test.py` | `kpz_scaling_results.json` | -- | KPZ universality: Var(ln DR) ~ E[H]^{2/3} vs E[H]^1 |
| `pair_gamma_mosaic.py` | `pair_gamma_mosaic_results.json` | -- | Global invariant vs per-pair mosaic test |
| `r2_collapse.py` | `r2_collapse_results.json` | -- | Universal R^2 collapse: R^2(p,config) = f(p*exp(gamma)) |
| `skewness_classifier.py` | `skewness_classifier_results.json` | -- | DR skewness as trap/cluster classifier without gamma fitting |
| `venus_epoch_decomposition.py` | `venus_epoch_decomposition_results.json` | -- | Venus trap-like (conjunction) vs cluster-like (opposition) within synodic |
| `beta_eff_survey.py` | `beta_eff_survey_results.json` | -- | Effective Boltzmann temperature beta_eff across multi-family configs |
| `diamond_beta_eff.py` | `diamond_beta_eff_results.json` | -- | Beta_eff on 3 dwell-diamond configs (two-path utility gap) |
| `dwell_diamond.py` | `dwell_diamond_results.json` | -- | Synthetic diamond: oracle path-switching under dwell decay |
| `remaining_analysis.py` | `remaining_analysis_results.json` | -- | Layer 4 decomposition under dwell decay + delta_screen closed form |
| `recompute_var_log_p_r2.py` | `var_log_p_canonical_results.json` | -- | Canonical R²(var_log_p, gamma) from 89K production (R²=0.903) |
| `recompute_gamma_oracle.py` | `gamma_oracle_canonical_results.json` | yes | Canonical gamma decomposition (normal/myopic/retry) per body |
| `whitespace_analysis.py` | `whitespace_analysis_results.json` | -- | Layer -1 (alive/dead orthogonality) + R² structural relationship check |

### Analysis Scripts

| Script | Results | Description |
|--------|---------|-------------|
| `analysis_beta_myopic.py` | `analysis_beta_myopic_results.json` | Orbital beta_myopic formula on CRAWDAD; slope of ln(Phi) vs E[H] |
| `analysis_braess_variance.py` | `analysis_braess_variance_results.json` | Braess-epoch variance structure in Mars 4-tier data |
| `analysis_competing_risk.py` | `competing_risk_analysis.json` | Competing-risk Phi formula: myopic rate vs inter-contact hazard |
| `analysis_competing_risk_v2.py` | `competing_risk_v2_analysis.json` | Revised: survival CDF for xi, three variants |
| `analysis_competing_risk_v3.py` | `competing_risk_v3_analysis.json` | Pareto-model h(t)=alpha/t; far-tail exponent alpha test |
| `analysis_competing_risk_v4.py` | `competing_risk_v4_analysis.json` | Lorentzian balance f(xi)=1/(1+xi) test |
| `analysis_running_coupling.py` | `running_coupling_results.json` | alpha_eff collapse vs commitment horizon t* |
| `analysis_shape_correction.py` | `shape_correction_results.json` | First correction to Lorentzian from non-Pareto shape |

### Plot Scripts

| Script | Description |
|--------|-------------|
| `plot_braess_paper.py` | DR vs synodic epoch for n=2,4,8,12 (Braess architecture dependence) |
| `plot_gamma_chi_duality.py` | Gamma-chi duality: Var(chi_DR) vs gamma separating trap/cluster |
| `plot_phi_paper.py` | 4 Phi figures: Phi_time scatter, histogram by n, landscape, Braess heatmap |

---

## Paper 3 — Relay Architectures (historical planning index)

**Topic:** Topology-vs-band effects, non-monotonic DR(n), and mixed-architecture Braess mitigation.
**Historical data row:** 134K+ configs, 17K atlas, 50 Braess experiments.
**Historical manuscript pointer:** `papers/relay_architectures.tex` (756 lines, Sec 1-2 & 7-9 placeholder)

### Runners

No runner scripts for Paper 3 are present in the current tree. The EPYC results under `runs/epyc_results/campaign_2026_03_11/` are the stored outputs from the historical campaign runs.

### Analysis & Plot Scripts

| Script | Description |
|--------|-------------|
| `analysis_mars_architecture.py` | Deep dive: synodic profiles, Braess, conjunction, factored delivery ratio |
| `compute_s_vs_hops.py` | s vs relay depth across Moon/Mars architectures |
| `plot_helio_primer.py` | 4 heliocentric figures: synodic DR, phase collapse, multi-body, CGR bars |

---

## Paper 4 — TASEP / Load Universality (historical planning index)

**Topic:** J_beta saturation and TASEP-style phase structure in DTN.
**Historical manuscript pointer:** `papers/tasep_dtn.tex` (443 lines)

### Archived Artifacts

No runner scripts for Paper 4 are present in the current tree. The following result files exist as archived data with no producing runner in this checkout:

| File | Status | Description |
|------|--------|-------------|
| `load_sweep_v2_results.json` | hash-pinned | Multi-bundle with persistent capacity state; DR=S_T*eta under load |
| `period_sweep_results.json` | hash-pinned | Relay period ratio sweep: golden mean → K_eff prediction |

---

## Cross-cutting Theory (Q1-Q5 + Statistical Mechanics)

**Core questions:** Five open questions from the classification conjecture + stat mech correspondence.

### Runners

| Script | Results | Status | Description |
|--------|---------|--------|-------------|
| `q3_neff_threshold.py` | `q3_neff_threshold_results.json` | -- | Q3: N_eff threshold — absolute count vs coverage fraction |

---

## ITN Whitepaper — Interplanetary Transport Networks (historical planning index)

**Topic:** Possible transfers from DTN analysis to physical transport: percolation threshold, Braess effects, and fleet phasing.

### Runners

| Script | Results | Status | Description |
|--------|---------|--------|-------------|
| `itn_delta_screen_analysis.py` | `itn_delta_screen_results.json` | -- | Closed-form Delta_screen ≈ 0.085 × (σ²_D/H) × λ², R²=0.919 |

### Plot Scripts

| Script | Description |
|--------|-------------|
| `plot_itn_fig1_legendre_hull.py` | Legendre hull + affine parametric shortest-path structure |
| `plot_itn_fig2_cascade.py` | Three-cascade: contact → oracle → routing pipeline |
| `plot_itn_fig3_one_tau.py` | Single-tau slice: S_T × eta factorization at one half-life |

---

## Exploratory

No exploratory runner scripts are present in the current tree.

---

## Infrastructure

| Script | Description |
|--------|-------------|
| `_chunked_base.py` | **lib:** Chunked parallel oracle engine (adjacency, Dijkstra, batch metrics) |
| `build_master_table.py` | Consolidates all result JSONs into `master_comparison.json` |
| `build_provenance_manifest.py` | Generates PROVENANCE.json/.md linking every result to its producing script |
| `paper_sims.py` | Original TIN conference paper quantitative results |
| `validation.py` | Standalone TIN DTN core validation (custody FSM, routing, fragments) |
| `repro_v0_3_8.py` | Deterministic v0.3.8 lunar-baseline entrypoint; seed 42 is the canonical baseline. Single 28-day sim + coverage grid; `--coverage_workers N` parallelizes the coverage loop with byte-identical output (integer reduction) |
| `verify_repro_v0_3_8_baseline.py` | C1 contract: regenerates the seed-42 baseline in a temp dir and field-compares against `results/repro_v0_3_8_baseline.json` (only `timestamp_utc` excluded); prints pass/fail, exit 0/1. Measured 2026-06-12: 6m47s wall, 45 MB peak RSS at `--coverage-workers 2` on a 16-core box — PASSED |
| `epyc_phase1.py` | Phase 1 EPYC batch orchestrator |
| `epyc_phase3.py` | Phase 3 EPYC batch orchestrator |
| `epyc_phase5.py` | Phase 5 EPYC batch orchestrator |
| `epyc_phase6.py` | Phase 6 EPYC batch orchestrator |
| `epyc_setup.sh` | EPYC server provisioning script |
| `epyc_setup_v2.sh` | EPYC v2 provisioning (+ vehicular GPS data) |

### Presentation / Docs Plot Scripts

| Script | Description |
|--------|-------------|
| `plot_d1_pipeline_architecture.py` | Five-stage pipeline architecture diagram |
| `plot_d2_protocol_mapping.py` | TIN simulation → DTN protocol stack mapping |
| `plot_d4_yaml_to_result.py` | YAML config → perc CLI → engine → output flowchart |

---

## EPYC Results (`runs/epyc_results/`) — ~95 MB total

### production_2026_03_11/ (79 MB) — Papers 1+2 production

| File | Description |
|------|-------------|
| `production_P{1,3,4,5P6,7,8,12,14a,14b,14c,14d}_results.json` | 11 production panel files, 89,178 configs total |
| `production_summary.json` | Summary statistics |

### campaign_2026_03_11/ — Paper 3 campaign

| File | Description |
|------|-------------|
| `campaign_bucket_{1-8,6b}_results.json` | 9 campaign buckets (7,026 configs) |
| `campaign_summary.json` | Campaign summary statistics |
| `followup_{A,B,C}_results.json` | 3 follow-up experiments (3,120 configs) |
| `followup_summary.json` | Follow-up summary statistics |

---

## UQ & Coupling Experiments (2026-03-20 – 2026-03-21)

**Core question:** Does the S_T × η factorization's independence assumption hold under congestion?
**Evidence base:** v1–v4 experiments + forensic sweep across 82,515 production configs.
**Result:** Independence was never exact — the coupling has a non-monotonic three-regime structure.
Factored form remains operationally optimal because no single-parameter correction improves all regimes.

No UQ runner scripts are present in the current tree. The forensic sweep script is in the repo:

### Forensic Analysis

| Script | Results | Status | Description |
|--------|---------|--------|-------------|
| `forensic_coupling_sweep.py` | `forensic_coupling_sweep_results.json` | -- | 82,515-config production sweep: per-body rho, three-regime breakdown, phase diagram |

---

## Historical Summary Counts

These counts are from the 2026-05-07 audit and reflect the original experiment plan; many runner scripts were not committed to this checkout. For the current tree, use `runs/CONFIG_MANIFEST.md` for result-file accounting and the header total above for script count.

| Category | Historical plan count |
|----------|-----------------------|
| Paper 1 scripts | 46 (36 runners + 3 analysis + 7 plotters) |
| Paper 2 scripts | 62 (53 runners + 9 analysis + 4 plotters) |
| Paper 3 scripts | 25 (22 runners + 2 analysis + 1 plotter) |
| Paper 4 scripts | 5 runners |
| ITN Whitepaper scripts | 7 (4 runners + 3 plotters) |
| Exploratory | 5 scripts |
| Infrastructure | 20 (17 scripts + 3 doc plotters) |
| UQ/Coupling scripts | 10 (9 runners + 1 forensic analysis) |
| Total configs | ~290,000+ (plus 82,515-config forensic sweep) |
