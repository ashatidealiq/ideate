"""Deterministic backtester (DESIGN.md §5, §7 Stage 5).

`run_backtest(spec, panel, context=None) -> BacktestResult` combines
`compile.compile_signal`, `portfolio.construct_weights`, and `costs.py`
into positions, trades, and a daily return series for a single run.

Contract on `panel`: it must already be the caller's `pit.load_panel(...,
universe=spec.universe.base, ...)` output, with `ret_1d` included (this is
the only module that turns weights into P&L, so it is the one that needs
returns) plus whatever else the signal and `spec.universe.filters` need
(`mktcap`, `adv_20`, `close`, `sector`, `country`, `region`). A column this
needs but the caller didn't load is simply not filtered/priced on --
`spec.universe.filters.min_history_days` and `.require_fields` are not
enforced here for the same reason (Phase 3 scope; not exercised by any
accept test).

Timing: for each `spec.portfolio.rebalance` date, the signal at that date's
close selects and sizes a "cohort" (a full `portfolio.construct_weights`
call). With `holding_period=k`, the position actually held is the average
of the last (up to) `k` cohorts -- exactly `k` once warmed up -- which
keeps `gross_leverage`/`net_exposure` exact, since averaging any number of
vectors that each already sum to a target preserves that sum. A cohort's
weights become tradeable `execution_lag` trading days after its signal
date (0 only permitted when `context.replication` is True -- DESIGN's
"signal at t close, trade at t+lag close", and the mechanism that makes a
lookahead signal (e.g. one built from the same day's own return) produce
an "impossible" Sharpe only under a replication run, and impossible to
even run otherwise).

Stage 5's three named runs (headline/replication/oos) are Stage-5-notebook
orchestration (Phase 6), not this function's job: this is the single-run
engine, called once per run.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import catalogue
from common import compile as compiler
from common import costs as cost_model
from common import portfolio as port
from common.schemas import Position, ReturnRow, Spec, Trade
from common.schemas import BacktestResult as BacktestResultSchema

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestContext:
    """Execution context for one run. `replication=True` is the only way
    `execution_lag` may be 0 -- reserved for reproducing a source's own
    (possibly lookahead-tainted) construction for the replication gate."""

    replication: bool = False


def _rebalance_dates(all_dates: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    if rebalance == "daily":
        return all_dates
    if rebalance == "weekly":
        iso = all_dates.isocalendar()
        key = [f"{y}-W{w:02d}" for y, w in zip(iso["year"], iso["week"])]
    elif rebalance == "monthly":
        key = [f"{y}-{m:02d}" for y, m in zip(all_dates.year, all_dates.month)]
    else:
        raise ValueError(f"unknown rebalance: {rebalance!r}")
    first_per_key = pd.Series(all_dates, index=all_dates).groupby(key).first()
    return pd.DatetimeIndex(sorted(first_per_key.values))


def _advance_trading_days(all_dates: pd.DatetimeIndex, date: pd.Timestamp, n: int) -> pd.Timestamp | None:
    pos = all_dates.get_loc(date)
    target = pos + n
    if target >= len(all_dates):
        return None
    return all_dates[target]


def _eligible_assets(candidate_index, rdate, filters, mktcap_wide, adv_wide, close_wide, sector_wide):
    eligible = pd.Series(True, index=candidate_index)
    if mktcap_wide is not None and rdate in mktcap_wide.index:
        vals = mktcap_wide.loc[rdate].reindex(candidate_index)
        eligible &= vals.notna() & (vals >= filters.min_mktcap)
    if adv_wide is not None and rdate in adv_wide.index:
        vals = adv_wide.loc[rdate].reindex(candidate_index)
        eligible &= vals.notna() & (vals >= filters.min_adv_20)
    if close_wide is not None and rdate in close_wide.index:
        vals = close_wide.loc[rdate].reindex(candidate_index)
        eligible &= vals.notna() & (vals >= filters.min_price)
    if sector_wide is not None and rdate in sector_wide.index and filters.exclude_sectors:
        vals = sector_wide.loc[rdate].reindex(candidate_index)
        eligible &= ~vals.isin(filters.exclude_sectors)
    return eligible.index[eligible]


def _pivot(panel: pd.DataFrame, field: str) -> pd.DataFrame | None:
    if field not in panel.columns:
        return None
    return panel.pivot(index="date", columns="asset_id", values=field)


def run_backtest(spec: Spec, panel: pd.DataFrame, context: BacktestContext | None = None) -> BacktestResultSchema:
    context = context or BacktestContext()

    if spec.portfolio.execution_lag < 1 and not context.replication:
        raise ValueError("execution_lag < 1 is only allowed when context.replication=True")
    if "ret_1d" not in panel.columns:
        raise ValueError("run_backtest requires 'ret_1d' in the panel")

    signal = compiler.compile_signal(spec)(panel)
    signal_dates = set(signal.index.get_level_values("date"))

    all_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    ret_wide = _pivot(panel, "ret_1d")
    mktcap_wide = _pivot(panel, "mktcap")
    adv_wide = _pivot(panel, "adv_20")
    close_wide = _pivot(panel, "close")
    sector_wide = _pivot(panel, "sector")
    country_wide = _pivot(panel, "country")
    region_wide = _pivot(panel, "region")

    costs_catalogue = catalogue.load_costs() if spec.costs.spread_model == "adv_based" else {}

    rebal_dates = _rebalance_dates(all_dates, spec.portfolio.rebalance)
    cohorts: deque[pd.Series] = deque(maxlen=spec.portfolio.holding_period)
    scheduled: dict[pd.Timestamp, pd.Series] = {}

    for rdate in rebal_dates:
        if rdate not in signal_dates:
            continue
        sig_slice = signal.xs(rdate, level="date").dropna()
        if sig_slice.empty:
            continue

        eligible = _eligible_assets(sig_slice.index, rdate, spec.universe.filters, mktcap_wide, adv_wide, close_wide, sector_wide)
        sig_slice = sig_slice.loc[sig_slice.index.intersection(eligible)]
        if sig_slice.empty:
            continue

        group_data = {}
        if "sector" in spec.portfolio.neutralize and sector_wide is not None and rdate in sector_wide.index:
            group_data["sector"] = sector_wide.loc[rdate]
        if "country" in spec.portfolio.neutralize and country_wide is not None and rdate in country_wide.index:
            group_data["country"] = country_wide.loc[rdate]

        cohort_weights = port.construct_weights(sig_slice, spec.portfolio, group_data=group_data or None)
        if cohort_weights.empty:
            continue
        cohorts.append(cohort_weights)
        held = pd.concat(list(cohorts), axis=1).fillna(0.0).sum(axis=1) / len(cohorts)

        trade_date = _advance_trading_days(all_dates, rdate, spec.portfolio.execution_lag)
        if trade_date is None:
            continue
        scheduled[trade_date] = held

    positions_rows: list[Position] = []
    trades_rows: list[Trade] = []
    returns_rows: list[ReturnRow] = []

    active_weights = pd.Series(dtype=float)
    trade_dates_sorted = sorted(scheduled)
    next_idx = 0

    for date in all_dates:
        daily_cost_frac = 0.0

        if next_idx < len(trade_dates_sorted) and date == trade_dates_sorted[next_idx]:
            new_weights = scheduled[date]
            delta = new_weights.add(-active_weights, fill_value=0.0)
            changed = delta[delta.abs() > 1e-12]

            for asset_id, dw in changed.items():
                adv_dollars = None
                if adv_wide is not None and close_wide is not None and date in adv_wide.index and date in close_wide.index:
                    adv_shares = adv_wide.at[date, asset_id] if asset_id in adv_wide.columns else np.nan
                    price = close_wide.at[date, asset_id] if asset_id in close_wide.columns else np.nan
                    if pd.notna(adv_shares) and pd.notna(price):
                        adv_dollars = float(adv_shares * price)
                region = None
                if region_wide is not None and date in region_wide.index and asset_id in region_wide.columns:
                    r = region_wide.at[date, asset_id]
                    region = r if pd.notna(r) else None
                region_costs = costs_catalogue.get(region) if region else None

                cost_bps = cost_model.trade_cost_bps(spec.costs, float(dw), adv_dollars, region_costs)
                trades_rows.append(Trade(date=date.date(), asset_id=asset_id, delta_weight=float(dw), cost_bps=float(cost_bps)))
                daily_cost_frac += abs(float(dw)) * (cost_bps / 10_000.0)

            active_weights = new_weights
            for asset_id, w in active_weights.items():
                positions_rows.append(Position(date=date.date(), asset_id=asset_id, weight=float(w)))
            next_idx += 1

        if active_weights.empty:
            gross = 0.0
        elif ret_wide is not None and date in ret_wide.index:
            rets = ret_wide.loc[date].reindex(active_weights.index).fillna(0.0)
            gross = float((active_weights * rets).sum())
        else:
            gross = 0.0

        short_exposure = float(-active_weights[active_weights < 0].sum()) if not active_weights.empty else 0.0
        borrow_drag = short_exposure * (spec.costs.borrow_bps_annual / 10_000.0) / TRADING_DAYS_PER_YEAR

        net = gross - daily_cost_frac - borrow_drag
        returns_rows.append(
            ReturnRow(date=date.date(), gross=gross, net=net, cost=daily_cost_frac + borrow_drag)
        )

    return BacktestResultSchema(positions=positions_rows, returns=returns_rows, trades=trades_rows)
