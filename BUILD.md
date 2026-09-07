# BUILD.md — Phased build plan

Each phase ends with its acceptance tests passing. Do not start the next phase until they do. Do not build anything not listed in the current phase.

## Phase 0 — Scaffold

Build:
- `pyproject.toml` (python ≥3.11; pandas, pyarrow, numpy, pydantic, pyyaml, papermill, jupytext, pytest, statsmodels, matplotlib)
- `common/config.py` reading `PIPELINE_ROOT` from env, defaulting to `./data`
- `common/schemas.py` with pydantic models: `Manifest, Claim, FieldDef, Resolution, Spec, Assumption, BacktestResult, GateRow, Decision`
- Empty modules for everything else in `common/` with docstrings stating their contract from DESIGN.md
- `fixtures/build_fixture.py` producing the synthetic panel described in DESIGN §12

Accept:
- `pytest tests/test_schemas.py` — every schema round-trips through JSON and parquet
- `python fixtures/build_fixture.py` writes `fixtures/panel/*.parquet` with `date, asset_id, knowledge_date, value` columns

## Phase 1 — Manifest, registry, PIT loader

Build: `manifest.py`, `catalogue.py`, `pit.py`; `catalogue/fields.yaml` for the fixture fields.

Accept:
- `manifest.begin()` on an existing stage directory raises
- `manifest.read_input()` on a tampered file raises with the path and both hashes
- `load_panel` on the fixture never returns the lookahead-trap field before `date + 30d`
- `load_panel(as_of=X)` returns no rows with `knowledge_date > X`
- Delisted fixture names appear until their delist date and not after
- Registry append is atomic and idempotent per `(claim_id, run_id, stage)`

## Phase 2 — Spec, compile, English

Build: `spec.py`, `compile.py`, all primitives in DESIGN §6.2.

Accept:
- `validate_spec` rejects: unknown op, unknown field, >12 nodes, inline numeric constant, cycle, undeclared param
- Every primitive has a unit test against a hand-computed result on a 3-asset, 10-date frame
- `compile_signal(spec)(fixture_panel)` returns a Series indexed `(date, asset_id)` with no NaN where inputs are complete
- `describe_signal` on three example specs produces the exact strings stored in `tests/expected_english/`

## Phase 3 — Portfolio, costs, backtest, metrics

Build: `portfolio.py`, `costs.py`, `backtest.py`, `metrics.py`; `catalogue/costs.yaml` with placeholder buckets.

Accept:
- Each construction type produces weights summing to `gross_leverage` and netting to `net_exposure`
- `max_position` is never exceeded
- Overlapping portfolios with `holding_period=k` hold exactly `k` cohorts
- `execution_lag=0` raises unless `context.replication=True`
- Planted fixture signals recover their known net Sharpe within ±0.1 at fixed costs
- A signal equal to next-day return (lookahead) produces the expected impossible Sharpe under `replication=True` and cannot be constructed at all otherwise

## Phase 4 — Gates and factor decomposition

Build: `gates.py`; `fixtures/factors.parquet`.

Accept:
- Each gate has a unit test with a constructed `BacktestResult` on both sides of fail and warn thresholds
- Planted "good" fixture signal passes all gates; planted "factor-proxy" signal fails G9/G10; planted "unstable" signal fails G6
- Gate runner stops at first fail and records all prior rows
- DSR uses `trial_count` from registry, not a function argument

## Phase 5 — LLM layer and prompts

Build: `llm.py`, all five prompt files, notebooks 01, 03, 04.

Accept:
- Every call appends to `llm_raw.jsonl` and records prompt sha256 in the manifest
- Stage 1 on three test PDFs in `fixtures/papers/` produces claims that pass `Claim` schema; `required_fields` are all catalogue names
- Stage 3 on those claims produces specs passing `validate_spec`; assumptions non-empty
- Stage 4 round-trip check passes on those specs; a deliberately corrupted `signal.py` is caught
- Parse failure retries once then halts with `status: halted`

## Phase 6 — Deterministic notebooks and end-to-end

Build: notebooks 02, 05, 06, 07; `report.py`; papermill runner script `run_claim.py`.

Accept:
- `python run_claim.py --source fixtures/papers/p1.pdf` executes all seven stages on the fixture panel and produces a full run directory with every manifest chain verifying
- Halting stage 2 by removing a catalogue field produces `status: halted` and an `unmet_requirements.parquet` row
- `ic_pack.md` renders with all eleven sections; `ic_pack.pdf` builds
- Second run of the same claim gets a new `run_id`, `parent_run_id` set, `trial_count` carried forward
- Fourth run of the same claim is refused

## Phase 7 — Real data

Build: `panel/` from real vendor files via `scripts/build_panel.py`; real `catalogue/fields.yaml`, `universes.yaml`, `costs.yaml`, `factors.parquet`.

Accept:
- All Phase 1–4 tests pass against the real panel where applicable
- Three hand-written claims for known published anomalies produce headline results in the expected range
- Registry shows the correct trial counts after the above

## Out of scope for this build

Orchestration beyond papermill, scheduling, a UI, a database, any v2 construction types, live trading integration.
