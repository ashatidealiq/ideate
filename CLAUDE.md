# CLAUDE.md

Read DESIGN.md fully before writing any code. Follow BUILD.md phase order; do not skip ahead. When DESIGN.md and this file conflict, DESIGN.md wins.

## What this project is

A pipeline that turns a written idea into a falsifiable backtest with an immutable, hashed evidence trail. It does not trade. It does not optimise. Its output is an evidence package for a human committee.

## Non-negotiable constraints

- Point-in-time correctness is enforced in `common/pit.py` and nowhere else. No code outside `common/pit.py` reads `panel/` parquet directly.
- Stage directories are write-once. `manifest.begin()` fails if the directory exists. Never add an overwrite flag.
- Every stage reads prior-stage outputs only via `manifest.read_input()`, which verifies hashes. Never read a prior stage's parquet with `pd.read_parquet` directly.
- The signal vocabulary in DESIGN §6 is closed. Do not add primitives unless DESIGN.md is updated first.
- Gate thresholds live in `gates.py` as constants. They are never read from a spec, an environment variable, or an LLM output.
- `execution_lag` cannot be below 1 outside a replication run.
- LLM calls happen only in `common/llm.py`. Model id, temperature, and seed are constants there. Every prompt is a file in `prompts/` and its hash goes in the manifest.
- Agents never receive gate results or thresholds in any prompt.
- Agent-authored `signal.py` may import only `pandas`, `numpy`, and `common`. The compiled spec is what runs; the agent code is for audit.

## Code conventions

- Python ≥3.11, pandas + pyarrow, pydantic for every artifact schema.
- Long-form panels: columns `date, asset_id, ...`. Never wide-form across module boundaries.
- Notebooks are jupytext-paired `.py` in `percent` format. Edit the `.py`; never edit `.ipynb` directly.
- Every function in `common/` has a docstring stating its contract. No function in `common/` calls an LLM except those in `llm.py`.
- Tests in `tests/` mirror `common/` module names. Every primitive, every gate, every construction type has a unit test.
- No global mutable state. Config comes from `common/config.py` only.
- Raise on any ambiguity. Silent fallbacks and default-to-empty behaviour are bugs.

## Halting

A stage that cannot proceed writes its manifest with `status: halted` and a `halt_reason`, appends to the registry, and exits cleanly. A stage that crashes writes `status: failed`. There is no third outcome.

## Do not

- Do not build orchestration, scheduling, a UI, a database, or anything listed as out of scope in BUILD.md.
- Do not "improve" a gate threshold, a primitive, or the spec schema without a corresponding DESIGN.md change.
- Do not write code that reads from vendor files at run time. `panel/` is built once by `scripts/build_panel.py`.
- Do not use any data structure or library not listed in `pyproject.toml` without adding it there and stating why in the commit.
