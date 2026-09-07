"""Tests for common/metrics.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from common import metrics as m


def test_sharpe_hand_computed():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    mean, std = returns.mean(), returns.std(ddof=1)
    expected = mean / std * math.sqrt(252)
    assert m.sharpe(returns) == pytest.approx(expected)


def test_sharpe_zero_vol_is_zero():
    assert m.sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_total_return_compounds():
    returns = pd.Series([0.1, 0.1])  # 1.1 * 1.1 - 1
    assert m.total_return(returns) == pytest.approx(0.21)


def test_max_drawdown_hand_computed():
    # growth path: 1 -> 1.1 -> 0.99 -> 1.155 ; trough 0.99 is 10% below peak 1.1
    returns = pd.Series([0.1, -0.1, 0.16666667])
    assert m.max_drawdown(returns) == pytest.approx(0.10, abs=1e-6)


def test_annualized_vol_hand_computed():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01])
    expected = returns.std(ddof=1) * math.sqrt(252)
    assert m.annualized_vol(returns) == pytest.approx(expected)


def test_turnover_hand_computed():
    positions = pd.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d2"],
            "asset_id": ["A", "B", "A", "B"],
            "weight": [0.5, -0.5, 0.2, -0.8],
        }
    )
    # |0.2-0.5| + |-0.8-(-0.5)| = 0.3 + 0.3 = 0.6, over 1 transition
    assert m.turnover(positions) == pytest.approx(0.6)


def test_empty_series_are_handled():
    empty = pd.Series(dtype=float)
    assert m.sharpe(empty) == 0.0
    assert m.total_return(empty) == 0.0
    assert m.max_drawdown(empty) == 0.0
    assert m.turnover(pd.DataFrame(columns=["date", "asset_id", "weight"])) == 0.0
