# DESIGN.md — Idea-to-Evidence Pipeline

## 0. Purpose

An idea enters as text. The framework turns it into a falsifiable test with a complete evidentiary record, then reports. It does not trade and it does not optimise.

Ideas may originate from an academic paper, a dataset, or a human. The ingestion adapter differs; everything downstream is identical.

Every result traces to: the idea text → an extracted claim → a strategy spec → generated signal code → a data snapshot → a backtest → a gate table → a critique → a committee decision. Each link is hashed and immutable.

## 1. Principles

1. **Agents specify; the harness executes.** LLM agents produce structured documents (claims, specs, critiques) and one small piece of code (the signal function). Data access, portfolio construction, costs, backtesting and gates are fixed infrastructure the agents call but cannot modify.
2. **Point-in-time is an infrastructure property.** The data loader cannot return information that was not knowable on the as-of date. Lookahead is prevented by construction, not by review.
3. **Closed vocabulary for signals.** A signal is a shallow expression tree of named primitives. Its plain-English description is generated mechanically from the tree. Free-form signal logic is not permitted.
4. **Every stage output is a write-once, hashed artifact.** Re-runs are new runs. Nothing is overwritten.
5. **Rejections are recorded with the same fidelity as successes.** The registry of failed claims is what makes the multiple-testing burden computable.
6. **Deterministic gates precede human judgment.** Cheap kills happen first. LLM critique and committee review see only survivors.

## 2. Repository and storage layout

```
repo/
  CLAUDE.md
  DESIGN.md
  BUILD.md
  pyproject.toml
  common/
    __init__.py
    config.py          # PIPELINE_ROOT, paths, constants
    schemas.py         # pydantic models for every artifact
    manifest.py        # write/read/verify manifests
    catalogue.py       # field catalogue loader and resolver
    pit.py             # point-in-time panel loader
    spec.py            # spec parsing and validation
    compile.py         # spec -> signal function; spec -> English
    portfolio.py       # construction, neutralisation, weighting
    costs.py           # spread and impact models
    backtest.py        # deterministic backtester
    metrics.py         # performance and risk statistics
    gates.py           # gate definitions and runner
    llm.py             # pinned model calls with prompt hashing
    report.py          # evidence package renderer
  prompts/
    s1_extract.md
    s1_tradability.md
    s3_spec.md
    s4_code.md
    s6_critique.md
  notebooks/
    01_claim.ipynb / 01_claim.py
    02_data.ipynb / 02_data.py
    03_spec.ipynb / 03_spec.py
    04_code.ipynb / 04_code.py
    05_backtest.ipynb / 05_backtest.py
    06_evaluate.ipynb / 06_evaluate.py
    07_ic_pack.ipynb / 07_ic_pack.py
  catalogue/
    fields.yaml
    universes.yaml
    costs.yaml
  fixtures/            # small synthetic datasets for tests
  tests/
```

Notebooks are paired with `.py` via jupytext (`percent` format). The `.py` is the source of truth. Notebooks are executed headless with papermill; the executed `.ipynb` is stored in the run directory as an artifact.

Data lives outside the repo at `PIPELINE_ROOT` (Google Drive mount or local path):

```
$PIPELINE_ROOT/
  raw/                 # vendor files, never modified
  panel/               # PIT parquet, one file per field
  catalogue/           # frozen copies of catalogue yaml with hashes
  runs/
    {claim_id}/
      {run_id}/
        00_source/
        01_claim/
        02_data/
        03_spec/
        04_code/
        05_backtest/
        06_evaluation/
        07_ic/
  registry.parquet     # one row per (claim_id, run_id, stage, status)
```

## 3. Artifact contract

### 3.1 Stage directory contents

Each stage directory contains:

- One or more `.parquet` files: the typed, machine-readable output. The only thing the next stage may read.
- One `.md` file: human-readable narrative, generated from the parquet at the end of the stage. Never hand-edited.
- `manifest.json`: provenance.
- The executed notebook `NN_stage.executed.ipynb`.
- For LLM stages: `llm_raw.jsonl` containing every request and raw completion.

### 3.2 Manifest schema

```json
{
  "stage": "03_spec",
  "claim_id": "c_20260907_0001",
  "run_id": "r_20260907_143012",
  "parent_run_id": null,
  "started_utc": "...",
  "completed_utc": "...",
  "status": "complete | halted | failed",
  "halt_reason": null,
  "inputs": [{"path": "01_claim/claim.parquet", "sha256": "..."}],
  "outputs": [{"path": "03_spec/spec.parquet", "sha256": "..."}],
  "code_commit": "...",
  "common_version": "...",
  "catalogue_sha256": "...",
  "llm": {"model": "...", "prompt_sha256": "...", "temperature": 0, "seed": 0}
}
```

`inputs` hashes are verified on read by `manifest.read_input()`. A mismatch raises and the stage does not run.

### 3.3 Write-once

`manifest.begin(stage, claim_id, run_id)` fails if the stage directory already exists. A re-run is a new `run_id` with `parent_run_id` set.

### 3.4 Registry

`registry.parquet` is append-only. One row per stage completion or halt. Columns: `claim_id, run_id, parent_run_id, stage, status, halt_reason, completed_utc, trial_count`. `trial_count` is the cumulative number of backtests executed under this `claim_id` across all runs.

## 4. Field catalogue

`catalogue/fields.yaml`:

```yaml
fields:
  - name: close
    source: panel/close.parquet
    dtype: float64
    pit: {event_col: date, knowledge_col: date, lag_days: 0}
    coverage: {start: "2000-01-03", universes: [us_eod, stoxx600]}
    description: "Adjusted close"
  - name: cds_spread_5y
    source: panel/cds_spread_5y.parquet
    dtype: float64
    pit: {event_col: date, knowledge_col: date, lag_days: 1}
    coverage: {start: "2006-01-02", universes: [cds_names]}
    known_gaps: ["sparse before 2008 for EU names"]
    description: "5y senior CDS mid spread, bps"
```

`catalogue/universes.yaml` defines base universes as membership rules over fields, evaluated per date. `catalogue/costs.yaml` holds spread-by-ADV-bucket tables per region.

Every panel parquet has columns `date, asset_id, knowledge_date, value`. `knowledge_date >= date` always.

Initial fields: `close, open, high, low, volume, adv_20, mktcap, ret_1d, sector, country, region, cds_spread_5y`. Derived fields (`adv_20`, `ret_1d`) are materialised into `panel/` by a build script, not computed on the fly.

## 5. Harness API

All in `common/`. Agents and notebooks call these; nothing else touches data.

```python
# pit.py
def load_panel(fields: list[str], universe: str, start: str, end: str,
               as_of: str | None = None) -> pd.DataFrame:
    """Long-form: date, asset_id, <field>... 
    Rows with knowledge_date > date + lag are never returned.
    If as_of is given, rows with knowledge_date > as_of are never returned."""

# spec.py
def load_spec(path) -> Spec            # pydantic; raises on any unknown key
def validate_spec(spec: Spec) -> None  # vocabulary, node count, param declarations

# compile.py
def compile_signal(spec: Spec) -> Callable[[pd.DataFrame], pd.Series]
def describe_signal(spec: Spec) -> str  # mechanical English

# backtest.py
def run_backtest(spec: Spec, panel: pd.DataFrame) -> BacktestResult
    # BacktestResult: positions (date, asset_id, weight), 
    #                 returns (date, gross, net, cost), 
    #                 trades (date, asset_id, delta_weight, cost_bps)

# gates.py
def run_gates(result: BacktestResult, spec: Spec, claim: Claim,
              context: GateContext) -> pd.DataFrame
    # one row per gate: name, value, threshold, status(pass|warn|fail)
```

Signal functions produced at stage 4 have exactly this shape and nothing else:

```python
def compute_signal(panel: pd.DataFrame) -> pd.Series:
    """Index (date, asset_id). Higher = more attractive long."""
```

Stage 4 output is validated by round-trip: `compile_signal(spec)` and the agent's `compute_signal` must produce identical output on the fixture panel. If they differ, stage 4 halts. The agent-authored code exists for readability and audit; the compiled version is what runs.

## 6. Strategy spec vocabulary

### 6.1 Signal expression tree

```yaml
signal:
  nodes:
    - {id: n1, op: pct_change, inputs: [cds_spread_5y], params: {n: 5}}
    - {id: n2, op: cs_zscore, inputs: [n1]}
    - {id: n3, op: cs_winsorize, inputs: [n2], params: {k: 3.0}}
  output: n3
  sign: -1
```

Rules: max 12 nodes; every numeric constant in `params`; inputs are catalogue fields or prior node ids; DAG, no cycles.

### 6.2 Primitives

Time-series (per asset): `lag(n)`, `diff(n)`, `pct_change(n)`, `log`, `rolling_mean(n)`, `rolling_std(n)`, `rolling_sum(n)`, `rolling_max(n)`, `rolling_min(n)`, `rolling_zscore(n)`, `rolling_rank(n)`, `ewm(halflife)`, `rolling_beta(n)` [y, x], `rolling_corr(n)` [a, b], `rolling_vol(n)`.

Cross-sectional (per date): `cs_rank` (uniform on [−0.5, 0.5]), `cs_zscore`, `cs_demean`, `cs_winsorize(k)`, `cs_neutralize(group)` for group ∈ {sector, country, region}, `cs_scale` (unit L1).

Arithmetic: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sign`, `clip(lo, hi)`, `where` [cond, a, b].

Combination: `combine(weights)` — weighted sum; inputs must be cross-sectionally standardised.

Excluded from v1: optimisers, fitted models, any stateful estimator other than rolling windows, intraday data, derivatives, portfolio-level optimisation. These are v2 construction types with their own primitive sets.

### 6.3 Universe

```yaml
universe:
  base: cds_names                 # from universes.yaml
  filters:
    min_mktcap: 1.0e9
    min_adv_20: 5.0e6
    min_price: 5.0
    min_history_days: 252
    require_fields: [cds_spread_5y]
    exclude_sectors: []
```

Membership is per-date, PIT. Delisted names remain until delisting; terminal return is included.

### 6.4 Portfolio

```yaml
portfolio:
  construction: long_short_quantile   # long_short_quantile | long_only_quantile | top_n_bottom_n | rank_weighted | threshold
  quantile: 0.1
  n: null                             # for top_n_bottom_n
  threshold: {lo: null, hi: null}     # for threshold
  weighting: equal                    # equal | rank | signal | vol_scaled
  neutralize: [sector]                # subset of {sector, country, beta}
  gross_leverage: 1.0
  net_exposure: 0.0
  max_position: 0.02
  rebalance: weekly                   # daily | weekly | monthly
  holding_period: 1                   # rebalance units; >1 = overlapping portfolios
  execution_lag: 1                    # minimum 1; signal at t close, trade at t+lag close
```

### 6.5 Costs

```yaml
costs:
  spread_model: adv_based             # fixed_bps | adv_based
  spread_bps: null
  impact_model: sqrt                  # none | sqrt
  impact_coeff: 0.1
  borrow_bps_annual: 50
```

`adv_based` uses `catalogue/costs.yaml`. Not agent-adjustable.

### 6.6 Assumptions

Mandatory, non-empty. Every choice the source did not specify.

```yaml
assumptions:
  - {id: A1, choice: "cs_winsorize k=3.0", source_says: "outliers handled", alternatives: [2.5, 5.0, none]}
  - {id: A2, choice: "stale quotes carried 5 days then excluded", source_says: nothing, alternatives: [0, 10]}
```

### 6.7 Parameters

```yaml
params:
  n_lookback: {value: 5, sweep: [3, 10, 20]}
  k_winsor:   {value: 3.0, sweep: [2.5, 5.0]}
```

Node params reference these by name. Stage 6 sweeps `sweep` lists.

## 7. Stage definitions

Each stage: reads only prior-stage parquet via `manifest.read_input()`; writes parquet + md + manifest; halts with a reason rather than failing silently.

### Stage 1 — Claim

Input: `00_source/` (PDF, or `idea.md` for human/dataset-originated ideas).
Two LLM calls, separate prompts:
1. **Extraction** (`s1_extract.md`): transcription task. Output `claim.parquet` with columns: `claim_id, source_type, source_title, source_authors, source_year, source_doi, edge_statement, mechanism, predicted_sign, universe_description, frequency, horizon_days_lo, horizon_days_hi, formula_latex, formula_location, required_fields (list, catalogue vocabulary only), reported_sharpe, reported_period_start, reported_period_end, reported_gross, author_caveats (list)`.
2. **Tradability** (`s1_tradability.md`): judgment task. Output columns appended: `tradable (bool), tradability_reason`.
Halt if: `tradable == False`; or `required_fields` contains names outside the catalogue vocabulary.
Human-originated ideas skip extraction and are written directly in the claim schema.

### Stage 2 — Data resolution

Deterministic. Joins `required_fields` against `catalogue/fields.yaml`. Output `resolution.parquet`: `field, resolved (bool), source, coverage_start, coverage_universes, pit_lag_days, known_gaps`. Also writes `coverage.parquet`: per-date count of universe members with all required fields non-null.
Halt if: any field unresolved; or coverage < 30 names on more than 10% of dates in the proposed test period. Unresolved fields are appended to `$PIPELINE_ROOT/unmet_requirements.parquet`.

### Stage 3 — Spec

LLM call (`s3_spec.md`) with claim + resolution + full vocabulary in context. Output `spec.yaml`, `spec.parquet` (flattened), `assumptions.parquet`.
`validate_spec()` must pass. `describe_signal()` is generated and stored as `english.md`. A second LLM call compares `english.md` to the agent's own rationale in the claim; if the *what* diverges, halt.
Halt if: spec validation fails after 2 attempts; or assumptions block empty; or English/rationale mismatch.

### Stage 4 — Code

LLM call (`s4_code.md`) with spec + harness signature. Output `signal.py`, `signal.sha256`. Round-trip check against `compile_signal(spec)` on fixture panel.
Halt if: mismatch after 2 attempts; or code imports anything outside `pandas, numpy, common`.

### Stage 5 — Backtest

Deterministic. Three runs of `run_backtest`:
- `headline`: spec as written, full available period, `execution_lag ≥ 1`.
- `replication`: spec restricted to source's reported period and construction, `execution_lag` as the source used (may be 0). This run is for the replication gate only and is labelled as such everywhere.
- `oos`: post-source-period only.
Output `positions.parquet, returns.parquet, trades.parquet` per run, `metrics.parquet` with one row per run, `backtest.md`.
Halt if: universe empty on any rebalance date; or fewer than 24 rebalance periods in headline.

### Stage 6 — Evaluation

Deterministic gates first (section 8), in order, stopping at first `fail`. Then param and assumption sweeps for survivors: every `sweep` value and every `alternatives` value, one backtest each, appended to `sweeps.parquet`. Every backtest increments `trial_count` in the registry. Then factor decomposition against `catalogue/factors.parquet` (market, size, value, momentum, quality, low-vol; region-appropriate). Then LLM critique (`s6_critique.md`) with everything above in context, output `critique.md` with a mandatory `verdict: advance | reject` line and reasons.
Output `gates.parquet, sweeps.parquet, factor_decomp.parquet, critique.md, evaluation.md`.
Halt if: any hard gate fails; or critique verdict is `reject`.

### Stage 7 — IC pack

Deterministic render (`report.py`) of the evidence package (section 10). Output `ic_pack.md`, `ic_pack.pdf`, and `decision.parquet` with schema `decided_utc, decision (approve|reject|return), conditions, reviewer` — written by a human, not the pipeline.

## 8. Gates

Run in this order. `fail` halts; `warn` is recorded and shown in the IC pack.

| # | Gate | Definition | Fail | Warn |
|---|---|---|---|---|
| G1 | Turnover/cost | net Sharpe / gross Sharpe on headline | < 0.5 | < 0.7 |
| G2 | Replication | replication-run Sharpe vs source reported Sharpe | < 50% of reported, or wrong sign | < 75% |
| G3 | Headline significance | headline net t-stat, Newey-West, lag = holding period | < 2.0 | < 3.0 |
| G4 | Deflated Sharpe | Bailey-López de Prado DSR using registry `trial_count` for this claim | < 0.90 | < 0.95 |
| G5 | Out-of-sample | oos net Sharpe | < 0 | < 50% of headline |
| G6 | Sub-period stability | net Sharpe positive in each of 3 equal sub-periods of headline | < 2 of 3 | — |
| G7 | Drawdown | max drawdown / annualised vol on headline | > 3.0 | > 2.0 |
| G8 | Capacity | net Sharpe at 5× target AUM (impact rescaled) | < 50% of headline | < 75% |
| G9 | Factor residual | alpha t-stat after regression on catalogue factors | < 1.5 | < 2.0 |
| G10 | Factor loading | max |loading| on any single factor | > 0.5 | > 0.3 |
| G11 | Assumption robustness | fraction of assumption alternatives with net Sharpe > 50% of headline | < 0.5 | < 0.75 |
| G12 | Parameter robustness | fraction of sweep values with net Sharpe > 50% of headline | < 0.5 | < 0.75 |
| G13 | Book orthogonality | max correlation of daily returns with any approved claim's returns | > 0.7 | > 0.5 |

Target AUM and factor panel are in `common/config.py`. Thresholds are constants in `gates.py`, not spec-adjustable.

## 9. Iteration and trial accounting

- A claim has one spec per run. The headline result of a claim is the headline result of its base spec. A claim cannot advance on a sweep or alternative result.
- Sweeps and alternatives inform G11 and G12 only. They are not candidates.
- Every backtest under a `claim_id`, across all runs, increments `trial_count`. G4 uses it.
- A halted claim may be re-run with a modified spec as a new `run_id` with `parent_run_id` set. The modification must be recorded in `03_spec/change_reason.md`. `trial_count` carries forward.
- Maximum 3 runs per claim. After that the claim is `dead`. A materially different idea from the same source is a new claim citing the old `claim_id` in `source_notes`.
- Agents never see gate results when regenerating a spec. Re-runs are human-initiated.

## 10. Evidence package

`ic_pack.md` in this order, first page only for §1–3:

1. **Headline**: claim id, source citation, edge statement (one sentence), mechanism (one paragraph), mechanical English of the signal.
2. **Verdict table**: net Sharpe, t-stat, DSR, oos Sharpe, max DD, turnover, capacity — headline run only.
3. **Gate table**: all 13 rows, status coloured.
4. **Cumulative return chart**: headline net, replication, oos, on one axis, source period shaded.
5. **Assumptions and alternatives**: table of each assumption, chosen value, source text, alternative results.
6. **Parameter sweep heatmap**.
7. **Factor decomposition**: loadings, alpha, R².
8. **Sub-period table**.
9. **Critique**: verbatim `critique.md`.
10. **Provenance**: every manifest hash in the chain, `trial_count`, registry history for this claim.
11. **Decision block**: blank, for the reviewer.

## 11. LLM calls

`common/llm.py` is the only module that calls a model. Model id, temperature (0), and seed are constants. Every prompt is a file in `prompts/`; its sha256 goes in the manifest. Every request and raw completion is appended to the stage's `llm_raw.jsonl`. Structured outputs are requested as JSON, parsed with pydantic, and parse failures retry once with the error message appended; a second failure halts the stage.

Agents receive: the prior-stage parquet rendered as JSON, the catalogue vocabulary, and the spec vocabulary. Agents never receive gate thresholds or results.

## 12. Testing

`fixtures/` contains a synthetic panel (200 assets, 2010–2020, two regions, CDS on 80 names) with three planted signals of known Sharpe and one deliberate lookahead trap (a field whose `knowledge_date` lags `date` by 30 days). Tests assert the loader never returns the trap early, the backtester recovers the planted Sharpes within tolerance, gates classify the planted signals correctly, and the round-trip compile check catches a hand-corrupted `signal.py`.
