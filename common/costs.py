"""Spread and impact cost models (DESIGN.md §6.5).

Prices one side of a trade in bps, as the sum of two pieces:

- **Spread**: half of the round-trip spread. `fixed_bps` uses
  `costs.spread_bps` directly; `adv_based` looks the region's ADV bucket up
  in `catalogue/costs.yaml` (via `common.catalogue`) using the trade's
  dollar ADV (`adv_20 shares * close`). Not agent-adjustable: the spec
  picks a *mode*, never a number.
- **Impact**: a square-root model, `impact_coeff * sqrt(participation)`
  where participation is the trade's dollar size over dollar ADV, scaled by
  `TARGET_AUM` (`common/config.py`) to turn a position *weight* into a
  dollar amount. `impact_model: none` disables it.

Borrow cost (`borrow_bps_annual`) is not priced here: it is a continuous
holding-period drag on short positions, not a per-trade cost, so
`backtest.py` applies it directly when computing net returns.
"""

from __future__ import annotations

import math

from common import config
from common.schemas import Costs, RegionCosts


def spread_cost_bps(costs: Costs, adv_dollars: float | None, region_costs: RegionCosts | None) -> float:
    """Half the round-trip spread, in bps, for one trade."""
    if costs.spread_model == "fixed_bps":
        return costs.spread_bps / 2.0

    if region_costs is None:
        raise ValueError("adv_based spread model requires region_costs (catalogue/costs.yaml has no entry for this asset's region)")
    if adv_dollars is None:
        raise ValueError("adv_based spread model requires adv_dollars")

    for bucket in region_costs.adv_buckets:
        if bucket.max_adv is None or adv_dollars <= bucket.max_adv:
            return bucket.spread_bps / 2.0
    raise AssertionError("unreachable: catalogue.load_costs guarantees an unbounded last bucket")


def impact_cost_bps(costs: Costs, trade_dollars: float, adv_dollars: float | None) -> float:
    """Square-root market impact, in bps, for one trade of `trade_dollars`
    (signed or unsigned; only magnitude matters) against `adv_dollars`."""
    if costs.impact_model == "none":
        return 0.0
    if not adv_dollars:
        raise ValueError("sqrt impact model requires a positive adv_dollars")
    participation = abs(trade_dollars) / adv_dollars
    return costs.impact_coeff * math.sqrt(participation) * 10_000.0


def trade_cost_bps(
    costs: Costs,
    delta_weight: float,
    adv_dollars: float | None,
    region_costs: RegionCosts | None,
    aum: float = config.TARGET_AUM,
) -> float:
    """Total cost, in bps of the *trade's* notional, for a position change
    of `delta_weight` (a fraction of the book)."""
    trade_dollars = abs(delta_weight) * aum
    return spread_cost_bps(costs, adv_dollars, region_costs) + impact_cost_bps(costs, trade_dollars, adv_dollars)
