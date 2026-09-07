"""Performance and risk statistics computed from a return series.

Phase 3 implements the building blocks `backtest.py`'s own tests need
(`sharpe`, `annualized_return`, `annualized_vol`, `max_drawdown`,
`turnover`) plus the ones later phases are already known to need
(`total_return`, `cumulative_returns`). Newey-West t-stats and the
deflated Sharpe ratio (DESIGN §8 G3, G4) are gate-specific statistics that
belong to `gates.py` (Phase 4), not here, since they need the registry's
`trial_count` and a gate-specific lag choice that this module has no
business knowing about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Growth of 1 unit of capital compounded through `returns`."""
    return (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float(cumulative_returns(returns).iloc[-1] - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    growth = float(cumulative_returns(returns).iloc[-1])
    years = len(returns) / periods_per_year
    if years <= 0 or growth <= 0:
        return 0.0
    return growth ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (no risk-free adjustment: returns are
    already excess-of-cash net returns in this pipeline)."""
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline of cumulative growth, as a positive
    fraction (0.2 == a 20% drawdown)."""
    if len(returns) == 0:
        return 0.0
    growth = cumulative_returns(returns)
    running_max = growth.cummax()
    drawdown = growth / running_max - 1.0
    return float(-drawdown.min())


def turnover(positions: pd.DataFrame) -> float:
    """Average per-rebalance one-way turnover: mean absolute weight change
    per rebalance date, summed across assets. `positions` has columns
    `date, asset_id, weight`; consecutive dates are compared per asset,
    with a missing weight on either side of a transition treated as 0."""
    if positions.empty:
        return 0.0
    wide = positions.pivot(index="date", columns="asset_id", values="weight").fillna(0.0)
    if len(wide) < 2:
        return 0.0
    changes = wide.diff().iloc[1:].abs().sum(axis=1)
    return float(changes.mean())
