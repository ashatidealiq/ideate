"""Tests for common/portfolio.py (BUILD.md Phase 3 accept criteria)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common import portfolio as p
from common.schemas import Portfolio, PortfolioThreshold

N_ASSETS = 100


def _signal() -> pd.Series:
    rng = np.random.default_rng(0)
    ids = [f"A{i:03d}" for i in range(N_ASSETS)]
    return pd.Series(rng.normal(size=N_ASSETS), index=ids)


def _portfolio(**overrides) -> Portfolio:
    base = dict(
        construction="long_short_quantile",
        quantile=0.1,
        n=None,
        threshold=PortfolioThreshold(),
        weighting="equal",
        neutralize=[],
        gross_leverage=1.0,
        net_exposure=0.0,
        max_position=0.05,
        rebalance="weekly",
        holding_period=1,
        execution_lag=1,
    )
    base.update(overrides)
    return Portfolio(**base)


CONSTRUCTIONS = [
    dict(construction="long_short_quantile", quantile=0.1, max_position=0.05),
    # long_only_quantile puts the *full* gross_leverage on one leg (10 names), so it
    # needs more headroom per name than a balanced long/short split does.
    dict(construction="long_only_quantile", quantile=0.1, net_exposure=1.0, max_position=0.15),
    dict(construction="top_n_bottom_n", n=8, max_position=0.15),
    dict(construction="rank_weighted", max_position=0.05),
    dict(construction="threshold", threshold=PortfolioThreshold(lo=-0.5, hi=0.5), max_position=0.05),
]


@pytest.mark.parametrize("kwargs", CONSTRUCTIONS, ids=[k["construction"] for k in CONSTRUCTIONS])
@pytest.mark.parametrize("weighting", ["equal", "rank", "signal"])
def test_weights_sum_to_gross_and_net_leverage(kwargs, weighting):
    signal = _signal()
    portfolio = _portfolio(weighting=weighting, **kwargs)
    weights = p.construct_weights(signal, portfolio)

    assert weights.abs().sum() == pytest.approx(portfolio.gross_leverage, abs=1e-6)
    assert weights.sum() == pytest.approx(portfolio.net_exposure, abs=1e-6)


@pytest.mark.parametrize("kwargs", CONSTRUCTIONS, ids=[k["construction"] for k in CONSTRUCTIONS])
def test_max_position_is_never_exceeded(kwargs):
    signal = _signal()
    portfolio = _portfolio(weighting="signal", **kwargs)  # signal weighting skews sizes the most
    weights = p.construct_weights(signal, portfolio)
    assert weights.abs().max() <= portfolio.max_position + 1e-9


def test_vol_scaled_weighting_uses_inverse_vol():
    signal = _signal()
    vol = pd.Series(1.0, index=signal.index)
    vol.loc["A000"] = 10.0  # this name should end up with much smaller weight
    # quantile=0.5 over 100 names selects everyone, so A000 is included regardless of sign.
    portfolio = _portfolio(weighting="vol_scaled", quantile=0.5, max_position=1.0)
    weights = p.construct_weights(signal, portfolio, vol=vol)
    assert abs(weights.loc["A000"]) < weights.abs().median()


def test_vol_scaled_without_vol_raises():
    signal = _signal()
    portfolio = _portfolio(weighting="vol_scaled")
    with pytest.raises(ValueError, match="vol_scaled"):
        p.construct_weights(signal, portfolio)


def test_sector_neutralize_nets_zero_per_sector():
    signal = pd.Series({f"A{i}": float(i) for i in range(10)})
    sector = pd.Series({f"A{i}": ("tech" if i % 2 == 0 else "fin") for i in range(10)})
    portfolio = _portfolio(quantile=0.5, neutralize=["sector"], max_position=0.3)

    weights = p.construct_weights(signal, portfolio, group_data={"sector": sector})

    per_sector = pd.concat([weights.rename("w"), sector.rename("sector")], axis=1).groupby("sector")["w"].sum()
    assert (per_sector.abs() < 1e-6).all()
    # neutralize runs after leverage/cap, so exact neutrality (just checked) trades
    # off against exact leverage -- it should still land in the right ballpark.
    assert weights.abs().sum() == pytest.approx(portfolio.gross_leverage, abs=0.1)


def test_beta_neutralize_zeroes_net_beta_exposure():
    signal = pd.Series({f"A{i}": float(i) for i in range(10)})
    beta = pd.Series({f"A{i}": 1.0 + 0.1 * i for i in range(10)})
    portfolio = _portfolio(quantile=0.5, neutralize=["beta"], max_position=0.3)

    weights = p.construct_weights(signal, portfolio, group_data={"beta": beta})

    net_beta = float((weights * beta.reindex(weights.index)).sum())
    assert net_beta == pytest.approx(0.0, abs=1e-6)


def test_neutralize_without_group_data_raises():
    signal = _signal()
    portfolio = _portfolio(neutralize=["sector"])
    with pytest.raises(ValueError, match="neutralize"):
        p.construct_weights(signal, portfolio)


def test_capping_redistributes_excess_to_uncapped_names():
    signal = pd.Series({"BIG": 1000.0, "B": 2.0, "C": 1.5, "D": 1.0, "E": 0.5,
                         "big_s": -1000.0, "F": -2.0, "G": -1.5, "H": -1.0, "I": -0.5})
    portfolio = _portfolio(construction="top_n_bottom_n", n=5, weighting="signal", max_position=0.15)

    weights = p.construct_weights(signal, portfolio)

    assert weights["BIG"] == pytest.approx(0.15)
    assert weights["big_s"] == pytest.approx(-0.15)
    assert weights.abs().sum() == pytest.approx(1.0, abs=1e-6)
    assert weights.sum() == pytest.approx(0.0, abs=1e-6)


def test_infeasible_max_position_raises():
    signal = _signal()
    # 10 names per leg, gross=1.0, but max_position way too small to reach it.
    portfolio = _portfolio(quantile=0.1, max_position=0.001)
    with pytest.raises(ValueError, match="cannot reach"):
        p.construct_weights(signal, portfolio)
