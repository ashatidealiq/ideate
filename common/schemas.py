"""Pydantic models for every artifact in the pipeline (DESIGN.md §3, §4, §6, §7).

Each top-level model here is a stage artifact schema: the typed contract that
a stage's parquet output must satisfy. Field names and shapes are taken
directly from the DESIGN.md sections cited in each model's docstring. Nested
models exist only where DESIGN.md itself nests structure (e.g. a spec's
`portfolio` block); they are not an invitation to add fields DESIGN.md does
not mention.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every schema: unknown keys raise, matching `load_spec`'s
    contract (DESIGN §5) and the general "raise on any ambiguity" rule
    (CLAUDE.md)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Manifest (DESIGN §3.2)
# --------------------------------------------------------------------------


class FileRef(StrictModel):
    """A hashed reference to one stage-output file, as it appears in a
    manifest's `inputs` / `outputs` lists."""

    path: str
    sha256: str


class LLMCallInfo(StrictModel):
    """Provenance for the LLM call(s) made during a stage, if any (DESIGN §11)."""

    model: str
    prompt_sha256: str
    temperature: float
    seed: int


class Manifest(StrictModel):
    """Provenance record written by every stage (DESIGN §3.2, §3.3, §3.4).

    One manifest per stage directory. `status` is one of `complete`,
    `halted`, `failed` — there is no fourth outcome (CLAUDE.md "Halting").
    `inputs` hashes are verified by `manifest.read_input()` on read.
    """

    stage: str
    claim_id: str
    run_id: str
    parent_run_id: str | None = None
    started_utc: datetime
    completed_utc: datetime | None = None
    status: Literal["complete", "halted", "failed"]
    halt_reason: str | None = None
    inputs: list[FileRef] = []
    outputs: list[FileRef] = []
    code_commit: str
    common_version: str
    catalogue_sha256: str | None = None
    llm: LLMCallInfo | None = None


# --------------------------------------------------------------------------
# Claim (DESIGN §7 Stage 1)
# --------------------------------------------------------------------------


class Claim(StrictModel):
    """Stage 1 output: `claim.parquet`. Extraction columns plus the
    tradability judgment columns appended by the second LLM call.
    """

    claim_id: str
    source_type: str
    source_title: str
    source_authors: list[str]
    source_year: int
    source_doi: str | None = None
    edge_statement: str
    mechanism: str
    predicted_sign: Literal[-1, 1]
    universe_description: str
    frequency: str
    horizon_days_lo: int
    horizon_days_hi: int
    formula_latex: str | None = None
    formula_location: str | None = None
    required_fields: list[str]
    reported_sharpe: float | None = None
    reported_period_start: date | None = None
    reported_period_end: date | None = None
    reported_gross: bool | None = None
    author_caveats: list[str] = []
    tradable: bool
    tradability_reason: str


# --------------------------------------------------------------------------
# FieldDef (DESIGN §4 catalogue/fields.yaml)
# --------------------------------------------------------------------------


class PitInfo(StrictModel):
    event_col: str
    knowledge_col: str
    lag_days: int


class Coverage(StrictModel):
    start: date
    universes: list[str]


class FieldDef(StrictModel):
    """One entry in `catalogue/fields.yaml`."""

    name: str
    source: str
    dtype: str
    pit: PitInfo
    coverage: Coverage
    known_gaps: list[str] = []
    description: str


class UniverseDef(StrictModel):
    """One entry in `catalogue/universes.yaml`: a base universe defined as a
    per-date membership rule over a catalogue field (DESIGN §4, §6.3). An
    asset is a member on a date iff `member_if_present` has a value for that
    (date, asset_id) — which, since fields are write-once and delisted
    assets simply stop appearing in their rows, also gives delisting for
    free. Spec-level `universe.filters` (min_mktcap etc., DESIGN §6.3) are
    layered on top of this base universe later; they are not part of it."""

    name: str
    member_if_present: str


# --------------------------------------------------------------------------
# Resolution (DESIGN §7 Stage 2)
# --------------------------------------------------------------------------


class Resolution(StrictModel):
    """One row of Stage 2's `resolution.parquet`: a claim's `required_fields`
    joined against the catalogue."""

    field: str
    resolved: bool
    source: str | None = None
    coverage_start: date | None = None
    coverage_universes: list[str] = []
    pit_lag_days: int | None = None
    known_gaps: list[str] = []


# --------------------------------------------------------------------------
# Spec (DESIGN §6)
# --------------------------------------------------------------------------


class SignalNode(StrictModel):
    """One node of a signal expression tree (DESIGN §6.1)."""

    id: str
    op: str
    inputs: list[str]
    params: dict[str, float | str | list[str | float]] = {}
    """Values are usually a name in the spec's top-level `params` block
    (DESIGN §6.7) -- `spec.validate_spec` rejects an inline float (or, for
    `combine`'s `weights`, an inline list of floats) here, since every
    numeric constant must be declared and named there (DESIGN §6.1). Both
    are still accepted by this schema so that rejecting them, with a clear
    message, is validate_spec's job rather than a generic parse error."""


class Signal(StrictModel):
    """The signal block of a spec. Max 12 nodes, DAG, no cycles (DESIGN §6.1);
    those rules are enforced by `spec.validate_spec`, not by this schema."""

    nodes: list[SignalNode]
    output: str
    sign: Literal[-1, 1]


class UniverseFilters(StrictModel):
    min_mktcap: float
    min_adv_20: float
    min_price: float
    min_history_days: int
    require_fields: list[str] = []
    exclude_sectors: list[str] = []


class Universe(StrictModel):
    base: str
    filters: UniverseFilters


class PortfolioThreshold(StrictModel):
    lo: float | None = None
    hi: float | None = None


class Portfolio(StrictModel):
    """DESIGN §6.4."""

    construction: Literal[
        "long_short_quantile",
        "long_only_quantile",
        "top_n_bottom_n",
        "rank_weighted",
        "threshold",
    ]
    quantile: float | None = None
    n: int | None = None
    threshold: PortfolioThreshold = PortfolioThreshold()
    weighting: Literal["equal", "rank", "signal", "vol_scaled"]
    neutralize: list[Literal["sector", "country", "beta"]] = []
    gross_leverage: float
    net_exposure: float
    max_position: float
    rebalance: Literal["daily", "weekly", "monthly"]
    holding_period: int
    execution_lag: int


class Costs(StrictModel):
    """DESIGN §6.5. `adv_based` reads `catalogue/costs.yaml`; costs are never
    agent-adjustable regardless of what a spec sets here."""

    spread_model: Literal["fixed_bps", "adv_based"]
    spread_bps: float | None = None
    impact_model: Literal["none", "sqrt"]
    impact_coeff: float
    borrow_bps_annual: float


class Assumption(StrictModel):
    """DESIGN §6.6. Mandatory, non-empty on every spec."""

    id: str
    choice: str
    source_says: str
    alternatives: list[str | float | int | None] = []


class ParamSpec(StrictModel):
    """DESIGN §6.7. `value` is what the base spec uses; `sweep` is the list
    of alternative values Stage 6 backtests for G12."""

    value: float
    sweep: list[float] = []


class Spec(StrictModel):
    """A full strategy spec (DESIGN §6): `spec.yaml` / the in-memory object
    `load_spec` returns and `validate_spec` checks."""

    signal: Signal
    universe: Universe
    portfolio: Portfolio
    costs: Costs
    assumptions: list[Assumption]
    params: dict[str, ParamSpec] = {}


# --------------------------------------------------------------------------
# BacktestResult (DESIGN §5 harness API)
# --------------------------------------------------------------------------


class Position(StrictModel):
    date: date
    asset_id: str
    weight: float


class ReturnRow(StrictModel):
    date: date
    gross: float
    net: float
    cost: float


class Trade(StrictModel):
    date: date
    asset_id: str
    delta_weight: float
    cost_bps: float


class BacktestResult(StrictModel):
    """Return type of `backtest.run_backtest` (DESIGN §5): the three tables
    it produces for one run (headline, replication, or oos)."""

    positions: list[Position]
    returns: list[ReturnRow]
    trades: list[Trade]


# --------------------------------------------------------------------------
# GateRow (DESIGN §5 harness API, §8)
# --------------------------------------------------------------------------


class GateRow(StrictModel):
    """One row of `gates.parquet`, as returned by `gates.run_gates`
    (DESIGN §5): "one row per gate: name, value, threshold, status"."""

    name: str
    value: float
    threshold: float
    status: Literal["pass", "warn", "fail"]


# --------------------------------------------------------------------------
# Decision (DESIGN §7 Stage 7)
# --------------------------------------------------------------------------


class Decision(StrictModel):
    """`decision.parquet` from Stage 7 — written by a human reviewer, not
    the pipeline."""

    decided_utc: datetime
    decision: Literal["approve", "reject", "return"]
    conditions: str | None = None
    reviewer: str
