# Contributing to TIN

Thanks for your interest in contributing. This document covers what you
need to know.

## License

TIN is licensed under the MIT License (see [LICENSE](LICENSE)). By
submitting a pull request you agree that your contribution is licensed
under the same terms.

If you are contributing on behalf of an employer, confirm that you have
authorization to contribute under these terms before submitting.

## Getting Started

```bash
git clone https://github.com/toxic2040/TIN.git
cd TIN
pip install -e ".[dev]"
pytest tests/ -x -v
```

Python 3.12+ required. The only core dependency is NumPy. For
visualization and SPICE support: `pip install -e ".[dev,viz,spice]"`.

## What We Welcome

- **Bug reports** — open an issue with a minimal reproducer.
- **Test additions** — especially edge cases on oracle, efficiency, or
  contact generation.
- **New body/trace integrations** — additional CRAWDAD traces, SpaceNet
  scenarios, or orbital targets.
- **Documentation fixes** — typos, broken links, unclear docstrings.
- **Performance improvements** — especially parallelism in runners.
  See the parallel feasibility API in `tin/core/oracle.py`
  (`parallel_feasibility_sweep`, `build_adjacency`).

## What Needs Discussion First

Open an issue before writing code for:

- Changes to the public core package (`tin/core/`) — these modules serve the
  maintained simulator and historical experiment surfaces. Breaking changes
  need careful thought and explicit before/after evidence.
- New theory or mechanism claims — the epistemic framework
  (definition / consequence / discovery) is load-bearing.
- Anything that changes numerical output of existing experiments.

## Code Style

- **No sys.path hacks.** The repo uses a single editable install.
  Import `tin` directly.
- **Parallel runners use ProcessPoolExecutor** with module-level
  workers (required for pickle). See `tin/core/oracle.py` for the
  initializer pattern that avoids per-task contact pickling.
- **Seeds:** `seed=42` for single runs, `[42 + i*7 for i in range(n)]`
  for sweeps.
- **Runner output:** JSON to `runs/<name>_results.json`, gitignored.
  Register in `runs/INDEX.md` and regenerate `runs/PROVENANCE.json`
  via `python runs/build_provenance_manifest.py`.

## Pull Request Process

1. Branch from `main`.
2. Run the full test suite: `pytest tests/ -x -v` — all tests must pass.
3. If you add a runner, register it in `runs/INDEX.md`.
4. Keep PRs focused — one logical change per PR.
5. Describe what changed and why. If it affects numerical results,
   show before/after.

## Experiment Results

Generated result JSONs should not be committed unless they are curated
reproducibility artifacts with provenance.

## Questions

Open an issue or email jcouncilman2040@gmail.com.
