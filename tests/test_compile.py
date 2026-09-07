"""Tests for common/compile.py (BUILD.md Phase 2 accept criteria).

Every primitive is checked against a value hand-computed from a small,
3-asset, 10-date frame:

    date index:  0  1  2  3  4  5  6  7  8  9
    A:           1  2  3  4  5  6  7  8  9 10
    B:          10 20 30 40 50 60 70 80 90 100   (= 10 * A)
    C:         100 90 80 70 60 50 40 30 20 10    (perfectly anti-correlated with A)

so that time-series ops (per asset, grouped by asset_id) and cross-sectional
ops (per date, grouped across A/B/C) both have an unambiguous, independently
computable expected value.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common import compile as c
from common import config, pit
from common import spec as sp
from common.schemas import (
    Assumption,
    Costs,
    ParamSpec,
    Portfolio,
    PortfolioThreshold,
    Signal,
    SignalNode,
    Spec,
    Universe,
    UniverseFilters,
)

EXPECTED_ENGLISH_DIR = Path(__file__).parent / "expected_english"

DATES = pd.date_range("2021-01-01", periods=10, freq="D")
ASSETS = ["A", "B", "C"]

A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
B = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
C = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]


def _series(values: dict[str, list], dtype=float) -> pd.Series:
    idx = pd.MultiIndex.from_product([DATES, ASSETS], names=["date", "asset_id"])
    data = [values[asset][i] for i in range(len(DATES)) for asset in ASSETS]
    return pd.Series(data, index=idx, dtype=dtype)


X = _series({"A": A, "B": B, "C": C})


def _at(s: pd.Series, date_idx: int, asset: str) -> float:
    return s.loc[(DATES[date_idx], asset)]


# --------------------------------------------------------------------------
# Time-series primitives (asset A, window ending at date index 5: values 4,5,6)
# --------------------------------------------------------------------------


def test_op_lag():
    assert _at(c.op_lag(X, 2), 5, "A") == 4.0  # A[3]


def test_op_diff():
    assert _at(c.op_diff(X, 2), 5, "A") == 2.0  # A[5]-A[3] = 6-4


def test_op_pct_change():
    assert _at(c.op_pct_change(X, 2), 5, "A") == pytest.approx(0.5)  # (6-4)/4


def test_op_log():
    assert _at(c.op_log(X), 5, "A") == pytest.approx(math.log(6))


def test_op_rolling_mean():
    assert _at(c.op_rolling_mean(X, 3), 5, "A") == pytest.approx(5.0)  # mean(4,5,6)


def test_op_rolling_std():
    assert _at(c.op_rolling_std(X, 3), 5, "A") == pytest.approx(1.0)  # std(4,5,6, ddof=1)


def test_op_rolling_sum():
    assert _at(c.op_rolling_sum(X, 3), 5, "A") == pytest.approx(15.0)  # 4+5+6


def test_op_rolling_max():
    assert _at(c.op_rolling_max(X, 3), 5, "A") == pytest.approx(6.0)


def test_op_rolling_min():
    assert _at(c.op_rolling_min(X, 3), 5, "A") == pytest.approx(4.0)


def test_op_rolling_zscore():
    assert _at(c.op_rolling_zscore(X, 3), 5, "A") == pytest.approx(1.0)  # (6-5)/1


def test_op_rolling_rank():
    # window [4,5,6]; current value 6 is the max -> percentile rank 3/3 = 1.0
    assert _at(c.op_rolling_rank(X, 3), 5, "A") == pytest.approx(1.0)


def test_op_ewm():
    # halflife=1 -> alpha=0.5; weighted average of x0,x1,x2=1,2,3 (newest-first weights 1, .5, .25)
    expected = (1 * 3 + 0.5 * 2 + 0.25 * 1) / (1 + 0.5 + 0.25)
    assert _at(c.op_ewm(X, 1), 2, "A") == pytest.approx(expected)


def test_op_rolling_beta():
    # B = 10 * A exactly, so the rolling slope of B on A is 10 in any window.
    y = _relabel(X, "B", "T")
    x = _relabel(X, "A", "T")
    assert _at(c.op_rolling_beta(y, x, 3), 5, "T") == pytest.approx(10.0)


def test_op_rolling_corr():
    # A and C are perfectly, linearly anti-correlated -> corr = -1.0
    a = _relabel(X, "A", "T")
    c_ = _relabel(X, "C", "T")
    assert _at(c.op_rolling_corr(a, c_, 3), 5, "T") == pytest.approx(-1.0)


def test_op_rolling_vol():
    expected = 1.0 * math.sqrt(252)  # rolling_std(3) at A[5] is 1.0
    assert _at(c.op_rolling_vol(X, 3), 5, "A") == pytest.approx(expected)


def _relabel(s: pd.Series, asset: str, new_asset: str) -> pd.Series:
    """Extracts one asset's series and relabels it under a new asset_id, so
    two single-asset series (e.g. B and A) can be fed into a 2-input
    primitive as if they were aligned columns for the same asset."""
    sub = s.xs(asset, level="asset_id")
    return pd.Series(sub.values, index=pd.MultiIndex.from_arrays([sub.index, [new_asset] * len(sub)], names=["date", "asset_id"]))


# --------------------------------------------------------------------------
# Cross-sectional primitives (date index 0: A=1, B=10, C=100)
# --------------------------------------------------------------------------


def test_op_cs_rank():
    # ranks 1,2,3 of 3 -> pct 1/3, 2/3, 3/3, minus 0.5
    assert _at(c.op_cs_rank(X), 0, "A") == pytest.approx(1 / 3 - 0.5)
    assert _at(c.op_cs_rank(X), 0, "C") == pytest.approx(1.0 - 0.5)


def test_op_cs_zscore():
    mean, std = 37.0, math.sqrt(2997.0)  # ddof=1 over [1,10,100]
    assert _at(c.op_cs_zscore(X), 0, "A") == pytest.approx((1 - mean) / std)


def test_op_cs_demean():
    assert _at(c.op_cs_demean(X), 0, "A") == pytest.approx(1 - 37.0)


def test_op_cs_winsorize():
    mean, std = 37.0, math.sqrt(2997.0)
    hi = mean + 1.0 * std
    assert _at(c.op_cs_winsorize(X, 1.0), 0, "C") == pytest.approx(hi)  # 100 clipped down
    assert _at(c.op_cs_winsorize(X, 1.0), 0, "A") == pytest.approx(1.0)  # unaffected


def test_op_cs_neutralize():
    group = _series({"A": ["g1"] * 10, "B": ["g1"] * 10, "C": ["g2"] * 10}, dtype=object)
    result = c.op_cs_neutralize(X, group)
    assert _at(result, 0, "A") == pytest.approx(1 - 5.5)  # g1 mean of (1,10)
    assert _at(result, 0, "B") == pytest.approx(10 - 5.5)
    assert _at(result, 0, "C") == pytest.approx(0.0)  # only member of g2


def test_op_cs_scale():
    total = 1 + 10 + 100
    assert _at(c.op_cs_scale(X), 0, "A") == pytest.approx(1 / total)
    assert _at(c.op_cs_scale(X), 0, "C") == pytest.approx(100 / total)


# --------------------------------------------------------------------------
# Arithmetic and combination primitives (date index 0: A=1, B=10)
# --------------------------------------------------------------------------


def test_op_add():
    assert _at(c.op_add(X, X), 0, "B") == pytest.approx(20.0)


def test_op_sub():
    diff = c.op_sub(X, _relabel(X, "B", "A"))
    assert _at(diff, 0, "A") == pytest.approx(1 - 10)


def test_op_mul():
    assert _at(c.op_mul(X, X), 0, "A") == pytest.approx(1.0)


def test_op_div():
    doubled = c.op_add(X, X)
    assert _at(c.op_div(doubled, X), 0, "A") == pytest.approx(2.0)


def test_op_neg():
    assert _at(c.op_neg(X), 0, "A") == pytest.approx(-1.0)


def test_op_abs():
    shifted = c.op_sub(X, X.groupby(level="date").transform("mean"))
    assert (c.op_abs(shifted) >= 0).all()
    assert _at(c.op_abs(shifted), 0, "A") == pytest.approx(abs(1 - 37.0))


def test_op_sign():
    shifted = X - 5.5  # A[4]=5 -> -0.5 (neg); A[5]=6 -> 0.5 (pos)
    assert _at(c.op_sign(shifted), 4, "A") == -1.0
    assert _at(c.op_sign(shifted), 5, "A") == 1.0


def test_op_clip():
    clipped = c.op_clip(X, 3.0, 7.0)
    assert _at(clipped, 0, "A") == pytest.approx(3.0)  # 1 -> clipped up to lo
    assert _at(clipped, 9, "A") == pytest.approx(7.0)  # 10 -> clipped down to hi
    assert _at(clipped, 4, "A") == pytest.approx(5.0)  # unaffected


def test_op_where():
    a_t = _relabel(X, "A", "T")
    b_t = _relabel(X, "B", "T")
    cond_t = a_t - 5.0  # zero exactly at A[4] == 5
    result = c.op_where(cond_t, a_t, b_t)
    assert _at(result, 4, "T") == pytest.approx(50.0)  # cond==0 -> b (B[4])
    assert _at(result, 0, "T") == pytest.approx(1.0)  # cond!=0 -> a (A[0])


def test_op_combine():
    a = _relabel(X, "A", "T")
    b = _relabel(X, "B", "T")
    result = c.op_combine([a, b], [2.0, 3.0])
    assert _at(result, 0, "T") == pytest.approx(2 * 1 + 3 * 10)


# --------------------------------------------------------------------------
# compile_signal integration test (BUILD.md Phase 2 accept criterion)
# --------------------------------------------------------------------------


def _boilerplate_spec_kwargs() -> dict:
    return dict(
        universe=Universe(
            base="all", filters=UniverseFilters(min_mktcap=0, min_adv_20=0, min_price=0, min_history_days=0)
        ),
        portfolio=Portfolio(
            construction="long_short_quantile",
            quantile=0.1,
            threshold=PortfolioThreshold(),
            weighting="equal",
            gross_leverage=1.0,
            net_exposure=0.0,
            max_position=0.02,
            rebalance="weekly",
            holding_period=1,
            execution_lag=1,
        ),
        costs=Costs(spread_model="adv_based", impact_model="sqrt", impact_coeff=0.1, borrow_bps_annual=50),
        assumptions=[Assumption(id="A1", choice="x", source_says="y", alternatives=[])],
    )


def _example_spec_momentum() -> Spec:
    """DESIGN §6.1's own example: CDS spread momentum, cross-sectionally
    standardised and winsorized, taken short (sign=-1)."""
    return Spec(
        signal=Signal(
            nodes=[
                SignalNode(id="n1", op="pct_change", inputs=["cds_spread_5y"], params={"n": "n_lookback"}),
                SignalNode(id="n2", op="cs_zscore", inputs=["n1"]),
                SignalNode(id="n3", op="cs_winsorize", inputs=["n2"], params={"k": "k_winsor"}),
            ],
            output="n3",
            sign=-1,
        ),
        params={
            "n_lookback": ParamSpec(value=5, sweep=[3, 10, 20]),
            "k_winsor": ParamSpec(value=3.0, sweep=[2.5, 5.0]),
        },
        **_boilerplate_spec_kwargs(),
    )


def _example_spec_simple_rank() -> Spec:
    """A trivial single-node signal: plain cross-sectional rank of close."""
    return Spec(
        signal=Signal(nodes=[SignalNode(id="n1", op="cs_rank", inputs=["close"])], output="n1", sign=1),
        params={},
        **_boilerplate_spec_kwargs(),
    )


def _example_spec_combo() -> Spec:
    """Exercises combine(): a weighted blend of two independently
    standardised fields."""
    return Spec(
        signal=Signal(
            nodes=[
                SignalNode(id="n1", op="cs_zscore", inputs=["close"]),
                SignalNode(id="n2", op="cs_zscore", inputs=["volume"]),
                SignalNode(id="n3", op="combine", inputs=["n1", "n2"], params={"weights": ["w1", "w2"]}),
            ],
            output="n3",
            sign=1,
        ),
        params={"w1": ParamSpec(value=0.6), "w2": ParamSpec(value=0.4)},
        **_boilerplate_spec_kwargs(),
    )


EXAMPLE_SPECS = {
    "momentum": _example_spec_momentum,
    "simple_rank": _example_spec_simple_rank,
    "combo": _example_spec_combo,
}


@pytest.mark.parametrize("name", EXAMPLE_SPECS)
def test_example_specs_are_valid(name):
    sp.validate_spec(EXAMPLE_SPECS[name]())


@pytest.mark.parametrize("name", EXAMPLE_SPECS)
def test_describe_signal_matches_expected_english(name):
    expected = (EXPECTED_ENGLISH_DIR / f"{name}.txt").read_text()
    assert c.describe_signal(EXAMPLE_SPECS[name]()) == expected


@pytest.fixture
def use_fixture_panel(monkeypatch, fixture_panel_ready):
    monkeypatch.setattr(config, "PIPELINE_ROOT", fixture_panel_ready)
    return fixture_panel_ready


def test_compile_signal_returns_series_with_no_nan_where_inputs_complete(use_fixture_panel):
    spec = _example_spec_momentum()
    sp.validate_spec(spec)

    start, end = "2010-01-04", "2010-06-30"
    panel = pit.load_panel(["cds_spread_5y"], "cds_names", start, end)
    fn = c.compile_signal(spec)
    result = fn(panel)

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["date", "asset_id"]

    # pct_change(n=5) needs 5 prior observations per asset; those rows are
    # genuinely incomplete inputs, not a bug. Every row after warmup is not.
    warmup_dates = set(sorted(panel["date"].unique())[:5])
    complete = result[~result.index.get_level_values("date").isin(warmup_dates)]
    assert complete.notna().all()
    assert len(complete) > 0
