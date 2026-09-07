"""Stage 5 -- Backtest.

Deterministic (no LLM call): three `backtest.run_backtest` runs (DESIGN §7
Stage 5) --

- `headline`: the spec as written, over the full available period,
  `execution_lag >= 1` (already enforced by `run_backtest` itself).
- `replication`: the spec's construction restricted to the claim's own
  reported period, with `execution_lag=0` under `context.replication=True`
  -- the closest reproduction of what the source itself likely did,
  for the replication gate only (DESIGN is explicit this run is never the
  headline result).
- `oos`: the spec as written, restricted to the period after the claim's
  reported period ends.

Halts if the headline run's universe was empty on any of its expected
rebalance dates (fewer trade dates materialized than
`backtest.rebalance_dates` says should exist), or if headline has fewer
than 24 rebalance periods.
"""

from __future__ import annotations

# %% tags=["parameters"]
claim_id = ""
run_id = ""
parent_run_id = None
code_commit = ""
common_version = "0.0.1"
headline_start = "2010-01-04"
headline_end = "2020-12-31"
min_headline_periods = 24

# %%
import hashlib
from pathlib import Path

import pandas as pd

from common import backtest, manifest, metrics, pit
from common.schemas import Claim, FileRef, Spec

STANDARD_FIELDS = ("ret_1d", "mktcap", "adv_20", "close", "sector", "country", "region")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_fields(spec: Spec) -> list[str]:
    node_ids = {n.id for n in spec.signal.nodes}
    fields = {inp for node in spec.signal.nodes for inp in node.inputs if inp not in node_ids}
    for node in spec.signal.nodes:
        if node.op == "cs_neutralize" and node.params.get("group") in ("sector", "country"):
            fields.add(node.params["group"])
    fields.update(STANDARD_FIELDS)
    return sorted(fields)


def _run_metrics_row(run: str, result) -> dict:
    net = pd.Series([r.net for r in result.returns], index=pd.to_datetime([r.date for r in result.returns]))
    gross = pd.Series([r.gross for r in result.returns], index=pd.to_datetime([r.date for r in result.returns]))
    return {
        "run": run,
        "net_sharpe": metrics.sharpe(net),
        "gross_sharpe": metrics.sharpe(gross),
        "total_return": metrics.total_return(net),
        "annualized_vol": metrics.annualized_vol(net),
        "max_drawdown": metrics.max_drawdown(net),
        "turnover": metrics.turnover(pd.DataFrame([p.model_dump() for p in result.positions])) if result.positions else 0.0,
        "n_periods": len({p.date for p in result.positions}),
    }


def _write_run_tables(stage_dir: Path, run: str, result) -> list[FileRef]:
    refs = []
    for name, rows in [("positions", result.positions), ("returns", result.returns), ("trades", result.trades)]:
        path = stage_dir / f"{name}_{run}.parquet"
        pd.DataFrame([r.model_dump(mode="json") for r in rows]).to_parquet(path, index=False)
        refs.append(FileRef(path=path.name, sha256=_sha256_file(path)))
    return refs


def run_stage5(
    spec: Spec,
    claim: Claim,
    claim_id: str,
    run_id: str,
    parent_run_id: str | None = None,
    code_commit: str = "",
    common_version: str = "0.0.1",
    headline_start: str = "2010-01-04",
    headline_end: str = "2020-12-31",
    min_headline_periods: int = 24,
) -> tuple[manifest.Manifest, dict | None]:
    """Returns (manifest, {"headline": BacktestResult, "replication": ..., "oos": ...})
    -- the dict is None if the stage halted."""
    ctx = manifest.begin("05_backtest", claim_id, run_id, parent_run_id=parent_run_id)
    common_kwargs = dict(code_commit=code_commit, common_version=common_version)

    fields = spec_fields(spec)
    headline_panel = pit.load_panel(fields, spec.universe.base, headline_start, headline_end)
    headline_result = backtest.run_backtest(spec, headline_panel)

    all_dates = pd.DatetimeIndex(sorted(headline_panel["date"].unique()))
    signal_dates = backtest.rebalance_dates(all_dates, spec.portfolio.rebalance)
    # expected *trade* dates: each signal date shifted by execution_lag, dropping
    # any that fall past the panel's end (a boundary truncation, not an empty universe)
    expected_trade_dates = [
        d.date()
        for d in (backtest.advance_trading_days(all_dates, s, spec.portfolio.execution_lag) for s in signal_dates)
        if d is not None
    ]
    actual_dates = sorted({p.date for p in headline_result.positions})

    # A gap before the first trade is normal warmup lag (e.g. adv_20 needs
    # 20 days before adv_based costs can price a trade at all) -- not a
    # broken universe. A gap *after* trading has started is the real
    # signal DESIGN's halt condition means to catch: the universe (or the
    # data a chosen cost model needs) unexpectedly evaporating mid-backtest.
    if actual_dates:
        expected_after_start = [d for d in expected_trade_dates if d >= actual_dates[0]]
        if len(actual_dates) < len(expected_after_start):
            return manifest.halt(ctx, halt_reason="universe empty on at least one rebalance date", **common_kwargs), None
    elif expected_trade_dates:
        return manifest.halt(ctx, halt_reason="universe empty on at least one rebalance date", **common_kwargs), None
    if len(actual_dates) < min_headline_periods:
        return (
            manifest.halt(ctx, halt_reason=f"only {len(actual_dates)} rebalance periods, need >= {min_headline_periods}", **common_kwargs),
            None,
        )

    rep_start = str(claim.reported_period_start) if claim.reported_period_start else headline_start
    rep_end = str(claim.reported_period_end) if claim.reported_period_end else headline_end
    replication_spec = spec.model_copy(deep=True)
    replication_spec.portfolio.execution_lag = 0
    replication_panel = pit.load_panel(fields, spec.universe.base, rep_start, rep_end)
    replication_result = backtest.run_backtest(replication_spec, replication_panel, context=backtest.BacktestContext(replication=True))

    oos_start_ts = pd.Timestamp(rep_end) + pd.Timedelta(days=1)
    if oos_start_ts <= pd.Timestamp(headline_end):
        oos_panel = pit.load_panel(fields, spec.universe.base, str(oos_start_ts.date()), headline_end)
        oos_result = backtest.run_backtest(spec, oos_panel) if len(oos_panel) else backtest.run_backtest(spec, headline_panel.iloc[:0])
    else:
        oos_panel = headline_panel.iloc[:0]
        oos_result = backtest.run_backtest(spec, oos_panel)

    results = {"headline": headline_result, "replication": replication_result, "oos": oos_result}

    outputs: list[FileRef] = []
    metrics_rows = []
    for run, result in results.items():
        outputs.extend(_write_run_tables(ctx.stage_dir, run, result))
        metrics_rows.append(_run_metrics_row(run, result))

    metrics_path = ctx.stage_dir / "metrics.parquet"
    pd.DataFrame(metrics_rows).to_parquet(metrics_path, index=False)
    outputs.append(FileRef(path="metrics.parquet", sha256=_sha256_file(metrics_path)))

    md_path = ctx.stage_dir / "backtest.md"
    md_path.write_text(pd.DataFrame(metrics_rows).to_markdown(index=False), encoding="utf-8")
    outputs.append(FileRef(path="backtest.md", sha256=_sha256_file(md_path)))

    result_manifest = manifest.complete(ctx, outputs=outputs, **common_kwargs)
    return result_manifest, results
