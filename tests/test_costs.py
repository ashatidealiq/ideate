"""Tests for common/costs.py."""

from __future__ import annotations

import math

import pytest

from common import costs as c
from common.schemas import AdvBucket, Costs, RegionCosts

REGION = RegionCosts(
    region="US",
    adv_buckets=[
        AdvBucket(max_adv=1_000_000, spread_bps=20),
        AdvBucket(max_adv=None, spread_bps=4),
    ],
)


def test_fixed_bps_spread_is_half_the_round_trip():
    costs = Costs(spread_model="fixed_bps", spread_bps=10.0, impact_model="none", impact_coeff=0, borrow_bps_annual=0)
    assert c.spread_cost_bps(costs, adv_dollars=None, region_costs=None) == pytest.approx(5.0)


def test_adv_based_spread_picks_correct_bucket():
    costs = Costs(spread_model="adv_based", impact_model="none", impact_coeff=0, borrow_bps_annual=0)
    assert c.spread_cost_bps(costs, adv_dollars=500_000, region_costs=REGION) == pytest.approx(10.0)  # 20/2
    assert c.spread_cost_bps(costs, adv_dollars=5_000_000, region_costs=REGION) == pytest.approx(2.0)  # 4/2


def test_adv_based_spread_without_region_raises():
    costs = Costs(spread_model="adv_based", impact_model="none", impact_coeff=0, borrow_bps_annual=0)
    with pytest.raises(ValueError, match="region_costs"):
        c.spread_cost_bps(costs, adv_dollars=1.0, region_costs=None)


def test_impact_none_is_zero():
    costs = Costs(spread_model="fixed_bps", spread_bps=0, impact_model="none", impact_coeff=0.5, borrow_bps_annual=0)
    assert c.impact_cost_bps(costs, trade_dollars=1_000_000, adv_dollars=1_000_000) == 0.0


def test_impact_sqrt_matches_formula():
    costs = Costs(spread_model="fixed_bps", spread_bps=0, impact_model="sqrt", impact_coeff=0.2, borrow_bps_annual=0)
    trade_dollars, adv_dollars = 250_000.0, 1_000_000.0
    expected = 0.2 * math.sqrt(trade_dollars / adv_dollars) * 10_000.0
    assert c.impact_cost_bps(costs, trade_dollars, adv_dollars) == pytest.approx(expected)


def test_trade_cost_bps_sums_spread_and_impact():
    costs = Costs(spread_model="fixed_bps", spread_bps=10.0, impact_model="sqrt", impact_coeff=0.1, borrow_bps_annual=0)
    aum = 1_000_000.0
    delta_weight = 0.1
    adv_dollars = 2_000_000.0
    expected_spread = 5.0
    expected_impact = 0.1 * math.sqrt((0.1 * aum) / adv_dollars) * 10_000.0
    result = c.trade_cost_bps(costs, delta_weight, adv_dollars, None, aum=aum)
    assert result == pytest.approx(expected_spread + expected_impact)
