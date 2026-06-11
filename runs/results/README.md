# Run Results Licensing

This directory holds reproducibility baselines, seed ensembles, and derived
analysis outputs produced by TIN runner scripts.

Checked-in JSON and JSONL result files are experimental data, not source code.
They follow the repository data license:

- License: CC-BY-4.0
- License file: `../../data/LICENSE_DATA`
- Attribution: J. Councilman, "TIN Project - Experimental Data Corpus," 2026.

The runner scripts that generate these files remain source code and follow the
repository source-code license described in `../../LICENSING.md`.

Before adding new result files to a public commit, verify that they contain no
local absolute paths, private dependency references, restricted trace material,
or non-public engine details. Untracked local run outputs should be treated as
draft artifacts until that check is complete.
