"""Tests for common/spec.py (BUILD.md Phase 2 accept criteria)."""

from __future__ import annotations

import pytest

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


def _spec(nodes: list[SignalNode], output: str = "n1", sign: int = 1, params: dict | None = None) -> Spec:
    return Spec(
        signal=Signal(nodes=nodes, output=output, sign=sign),
        universe=Universe(
            base="all",
            filters=UniverseFilters(min_mktcap=0, min_adv_20=0, min_price=0, min_history_days=0),
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
        params=params or {},
    )


def test_valid_spec_passes():
    valid = _spec(
        [
            SignalNode(id="n1", op="pct_change", inputs=["close"], params={"n": "n_lookback"}),
        ],
        params={"n_lookback": ParamSpec(value=5, sweep=[3, 10])},
    )
    sp.validate_spec(valid)  # should not raise


def test_rejects_unknown_op():
    bad = _spec([SignalNode(id="n1", op="teleport", inputs=["close"])])
    with pytest.raises(ValueError, match="unknown op"):
        sp.validate_spec(bad)


def test_rejects_unknown_field():
    bad = _spec([SignalNode(id="n1", op="cs_rank", inputs=["not_a_real_field"])])
    with pytest.raises(ValueError, match="unknown field"):
        sp.validate_spec(bad)


def test_rejects_more_than_12_nodes():
    nodes = [SignalNode(id="n1", op="cs_rank", inputs=["close"])]
    for i in range(2, 14):
        nodes.append(SignalNode(id=f"n{i}", op="cs_rank", inputs=[f"n{i - 1}"]))
    bad = _spec(nodes, output="n13")
    with pytest.raises(ValueError, match="max is 12"):
        sp.validate_spec(bad)


def test_rejects_inline_numeric_constant():
    bad = _spec([SignalNode(id="n1", op="pct_change", inputs=["close"], params={"n": 5})])
    with pytest.raises(ValueError, match="inline numeric constant"):
        sp.validate_spec(bad)


def test_rejects_inline_numeric_constant_in_combine_weights():
    bad = _spec(
        [
            SignalNode(id="n1", op="cs_zscore", inputs=["close"]),
            SignalNode(id="n2", op="cs_zscore", inputs=["volume"]),
            SignalNode(id="n3", op="combine", inputs=["n1", "n2"], params={"weights": [0.5, 0.5]}),
        ],
        output="n3",
    )
    with pytest.raises(ValueError, match="inline numeric constant"):
        sp.validate_spec(bad)


def test_rejects_cycle():
    bad = _spec(
        [
            SignalNode(id="n1", op="add", inputs=["n2", "close"]),
            SignalNode(id="n2", op="add", inputs=["n1", "close"]),
        ],
        output="n1",
    )
    with pytest.raises(ValueError, match="cycle"):
        sp.validate_spec(bad)


def test_rejects_undeclared_param():
    bad = _spec(
        [SignalNode(id="n1", op="pct_change", inputs=["close"], params={"n": "not_declared"})],
        params={"n_lookback": ParamSpec(value=5)},
    )
    with pytest.raises(ValueError, match="undeclared param"):
        sp.validate_spec(bad)


def test_rejects_wrong_input_count():
    bad = _spec([SignalNode(id="n1", op="add", inputs=["close"])])  # add needs 2 inputs
    with pytest.raises(ValueError, match="takes 2 input"):
        sp.validate_spec(bad)


def test_rejects_missing_required_param():
    bad = _spec([SignalNode(id="n1", op="pct_change", inputs=["close"], params={})])
    with pytest.raises(ValueError, match="missing required param"):
        sp.validate_spec(bad)


def test_rejects_bad_enum_param():
    bad = _spec(
        [SignalNode(id="n1", op="cs_neutralize", inputs=["close"], params={"group": "not_a_group"})]
    )
    with pytest.raises(ValueError, match="must be one of"):
        sp.validate_spec(bad)


def test_rejects_output_not_a_node_id():
    bad = _spec(
        [SignalNode(id="n1", op="cs_rank", inputs=["close"])],
        output="does_not_exist",
    )
    with pytest.raises(ValueError, match="not a defined node id"):
        sp.validate_spec(bad)
