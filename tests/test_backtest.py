"""Tests for common/backtest.py (BUILD.md Phase 3 accept criteria)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common import backtest as bt
from common import config
from common import metrics as m
from common import pit
from common.schemas import (
    Assumption,
    Costs,
    Portfolio,
    PortfolioThreshold,
    Signal,
    SignalNode,
    Spec,
    Universe,
    UniverseFilters,
)


def _tiny_panel(n_assets: int = 6, n_days: int = 12) -> pd.DataFrame:
    """A hand-constructed panel where, on day t, asset (t % n_assets) has
    the single highest 'sig' value and asset ((t + n_assets//2) % n_assets)
    has the single lowest -- a clean, predictable rotation for verifying
    exactly which cohort is active on which date."""
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for t, d in enumerate(dates):
        top = t % n_assets
        bottom = (t + n_assets // 2) % n_assets
        for i, a in enumerate(assets):
            sig = 10.0 if i == top else (-10.0 if i == bottom else 0.0)
            rows.append({"date": d, "asset_id": a, "sig": sig, "ret_1d": 0.0})
    return pd.DataFrame(rows)


def _spec(**portfolio_overrides) -> Spec:
    base = dict(
        construction="top_n_bottom_n",
        n=1,
        threshold=PortfolioThreshold(),
        weighting="equal",
        gross_leverage=1.0,
        net_exposure=0.0,
        max_position=1.0,
        rebalance="daily",
        holding_period=1,
        execution_lag=1,
    )
    base.update(portfolio_overrides)
    return Spec(
        signal=Signal(nodes=[SignalNode(id="n1", op="cs_rank", inputs=["sig"])], output="n1", sign=1),
        universe=Universe(
            base="all", filters=UniverseFilters(min_mktcap=0, min_adv_20=0, min_price=0, min_history_days=0)
        ),
        portfolio=Portfolio(**base),
        costs=Costs(spread_model="fixed_bps", spread_bps=0, impact_model="none", impact_coeff=0, borrow_bps_annual=0),
        assumptions=[Assumption(id="A1", choice="x", source_says="y", alternatives=[])],
    )


def test_execution_lag_zero_raises_without_replication():
    spec = _spec(execution_lag=0)
    with pytest.raises(ValueError, match="execution_lag"):
        bt.run_backtest(spec, _tiny_panel())


def test_execution_lag_zero_allowed_under_replication():
    spec = _spec(execution_lag=0)
    result = bt.run_backtest(spec, _tiny_panel(), context=bt.BacktestContext(replication=True))
    assert len(result.positions) > 0


def test_holding_period_holds_exactly_k_cohorts():
    k = 3
    n_assets = 6
    panel = _tiny_panel(n_assets=n_assets, n_days=12)
    spec = _spec(holding_period=k)

    result = bt.run_backtest(spec, panel)
    pos = pd.DataFrame([r.model_dump() for r in result.positions])
    wide = pos.pivot(index="date", columns="asset_id", values="weight").fillna(0.0)

    # Warms up by 2 names per period (1 long + 1 short per cohort), then
    # stays flat at exactly 2*k -- proving old cohorts are evicted, not
    # accumulated indefinitely.
    nonzero_counts = (wide != 0).sum(axis=1)
    expected = [min(i + 1, k) * 2 for i in range(len(wide))]
    assert list(nonzero_counts) == expected

    # Once warmed, every active name's magnitude is exactly 1/(2k) (equal
    # weighting, gross_leverage=1.0 spread over exactly 2*k names: 2 per
    # cohort -- one long, one short -- times k cohorts).
    warmed = wide.iloc[k - 1 :]
    nonzero_vals = warmed.to_numpy()[warmed.to_numpy() != 0]
    assert np.allclose(np.abs(nonzero_vals), 1.0 / (2 * k))

    # Gross leverage is exact at every single rebalance, warmup included.
    assert np.allclose(wide.abs().sum(axis=1), 1.0)


def test_lookahead_signal_impossible_sharpe_only_under_replication():
    """A signal built from the *same day's own* ret_1d, combined with
    execution_lag=0, uses information that would not exist at trade time --
    the degenerate case DESIGN's PIT principle rules out. It is refused
    outright unless context.replication=True, and produces an obviously
    impossible Sharpe when allowed."""
    n_assets = 10
    dates = pd.bdate_range("2021-01-04", periods=60)
    rng = np.random.default_rng(1)
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for d in dates:
        rets = rng.normal(0, 0.02, size=n_assets)
        for a, r in zip(assets, rets):
            rows.append({"date": d, "asset_id": a, "ret_1d": r})
    panel = pd.DataFrame(rows)

    spec = Spec(
        signal=Signal(nodes=[SignalNode(id="n1", op="cs_rank", inputs=["ret_1d"])], output="n1", sign=1),
        universe=Universe(
            base="all", filters=UniverseFilters(min_mktcap=0, min_adv_20=0, min_price=0, min_history_days=0)
        ),
        portfolio=Portfolio(
            construction="long_short_quantile", quantile=0.3, threshold=PortfolioThreshold(), weighting="signal",
            gross_leverage=1.0, net_exposure=0.0, max_position=0.2, rebalance="daily", holding_period=1,
            execution_lag=0,
        ),
        costs=Costs(spread_model="fixed_bps", spread_bps=0, impact_model="none", impact_coeff=0, borrow_bps_annual=0),
        assumptions=[Assumption(id="A1", choice="x", source_says="y", alternatives=[])],
    )

    with pytest.raises(ValueError, match="execution_lag"):
        bt.run_backtest(spec, panel)

    result = bt.run_backtest(spec, panel, context=bt.BacktestContext(replication=True))
    returns = pd.Series([r.net for r in result.returns])
    sharpe = m.sharpe(returns)
    assert sharpe > 10.0  # no real strategy gets near this; it's a smoking gun for lookahead
    assert (returns > 0).mean() > 0.95  # wins almost every single day, not just on average


PLANTED_RECIPE = dict(
    construction="long_short_quantile", quantile=0.1, threshold=PortfolioThreshold(), weighting="equal",
    gross_leverage=1.0, net_exposure=0.0, max_position=0.05, rebalance="daily", holding_period=1,
    execution_lag=1,
)


@pytest.fixture
def use_fixture_panel(monkeypatch, fixture_panel_ready):
    monkeypatch.setattr(config, "PIPELINE_ROOT", fixture_panel_ready)
    return fixture_panel_ready


@pytest.mark.parametrize("field", ["planted_signal_1", "planted_signal_2", "planted_signal_3"])
def test_planted_signals_recover_known_net_sharpe(field, use_fixture_panel):
    from fixtures.build_fixture import PLANTED_SIGNALS

    spec = Spec(
        signal=Signal(nodes=[SignalNode(id="n1", op="cs_zscore", inputs=[field])], output="n1", sign=1),
        universe=Universe(
            base="all", filters=UniverseFilters(min_mktcap=0, min_adv_20=0, min_price=0, min_history_days=0)
        ),
        portfolio=Portfolio(**PLANTED_RECIPE),
        costs=Costs(spread_model="fixed_bps", spread_bps=1.0, impact_model="none", impact_coeff=0, borrow_bps_annual=5),
        assumptions=[Assumption(id="A1", choice="x", source_says="y", alternatives=[])],
    )
    panel = pit.load_panel([field, "ret_1d"], "all", "2010-01-04", "2020-12-31")
    result = bt.run_backtest(spec, panel)
    net_returns = pd.Series([r.net for r in result.returns])

    realized_sharpe = m.sharpe(net_returns)
    target = PLANTED_SIGNALS[field]["target_sharpe"]
    assert realized_sharpe == pytest.approx(target, abs=0.1)
