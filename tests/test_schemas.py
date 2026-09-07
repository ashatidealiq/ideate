"""Schema round-trip tests (BUILD.md Phase 0 accept: pytest tests/test_schemas.py).

Every artifact schema in common/schemas.py must round-trip losslessly
through JSON and through parquet, since write-once stage outputs must stay
exactly readable by later stages and by manifest verification.
"""

import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from common import schemas as s


def _roundtrip_json(instance):
    dumped = instance.model_dump(mode="json")
    reloaded = json.loads(json.dumps(dumped))
    restored = type(instance).model_validate(reloaded)
    assert restored == instance


def _roundtrip_parquet(instance, tmp_path):
    """Round-trips a model through an actual parquet file.

    Stored as a single JSON-string column so the test does not depend on
    pyarrow's schema inference for arbitrarily nested/optional structures —
    the same reason a stage's nested spec.yaml gets a separately
    hand-flattened spec.parquet rather than a raw dump of the Spec model
    (DESIGN §7 Stage 3).
    """
    path = tmp_path / "artifact.parquet"
    df = pd.DataFrame({"data": [json.dumps(instance.model_dump(mode="json"))]})
    df.to_parquet(path)
    reloaded = pd.read_parquet(path)
    restored = type(instance).model_validate(json.loads(reloaded["data"].iloc[0]))
    assert restored == instance


def _example_manifest():
    return s.Manifest(
        stage="03_spec",
        claim_id="c_20260907_0001",
        run_id="r_20260907_143012",
        parent_run_id=None,
        started_utc=datetime(2026, 9, 7, 14, 30, 12, tzinfo=timezone.utc),
        completed_utc=datetime(2026, 9, 7, 14, 31, 0, tzinfo=timezone.utc),
        status="complete",
        halt_reason=None,
        inputs=[s.FileRef(path="01_claim/claim.parquet", sha256="a" * 64)],
        outputs=[s.FileRef(path="03_spec/spec.parquet", sha256="b" * 64)],
        code_commit="deadbeef",
        common_version="0.0.1",
        catalogue_sha256="c" * 64,
        llm=s.LLMCallInfo(model="claude-x", prompt_sha256="d" * 64, temperature=0.0, seed=0),
    )


def _example_claim():
    return s.Claim(
        claim_id="c_20260907_0001",
        source_type="paper",
        source_title="CDS Momentum",
        source_authors=["A. Author", "B. Author"],
        source_year=2019,
        source_doi="10.1000/example",
        edge_statement="CDS spread momentum predicts equity returns.",
        mechanism="Credit markets price information before equity markets.",
        predicted_sign=-1,
        universe_description="Liquid single-name CDS, US and EU.",
        frequency="weekly",
        horizon_days_lo=5,
        horizon_days_hi=20,
        formula_latex=r"z(\Delta s_{5d})",
        formula_location="Table 3",
        required_fields=["cds_spread_5y", "close"],
        reported_sharpe=1.2,
        reported_period_start=date(2006, 1, 2),
        reported_period_end=date(2016, 12, 31),
        reported_gross=True,
        author_caveats=["Sample tilted toward large names"],
        tradable=True,
        tradability_reason="Fields and universe resolve against the catalogue.",
    )


def _example_field_def():
    return s.FieldDef(
        name="cds_spread_5y",
        source="panel/cds_spread_5y.parquet",
        dtype="float64",
        pit=s.PitInfo(event_col="date", knowledge_col="date", lag_days=1),
        coverage=s.Coverage(start=date(2006, 1, 2), universes=["cds_names"]),
        known_gaps=["sparse before 2008 for EU names"],
        description="5y senior CDS mid spread, bps",
    )


def _example_resolution():
    return s.Resolution(
        field="cds_spread_5y",
        resolved=True,
        source="panel/cds_spread_5y.parquet",
        coverage_start=date(2006, 1, 2),
        coverage_universes=["cds_names"],
        pit_lag_days=1,
        known_gaps=["sparse before 2008 for EU names"],
    )


def _example_assumption():
    return s.Assumption(
        id="A1",
        choice="cs_winsorize k=3.0",
        source_says="outliers handled",
        alternatives=[2.5, 5.0, None],
    )


def _example_spec():
    return s.Spec(
        signal=s.Signal(
            nodes=[
                s.SignalNode(id="n1", op="pct_change", inputs=["cds_spread_5y"], params={"n": 5}),
                s.SignalNode(id="n2", op="cs_zscore", inputs=["n1"]),
                s.SignalNode(id="n3", op="cs_winsorize", inputs=["n2"], params={"k": 3.0}),
            ],
            output="n3",
            sign=-1,
        ),
        universe=s.Universe(
            base="cds_names",
            filters=s.UniverseFilters(
                min_mktcap=1.0e9,
                min_adv_20=5.0e6,
                min_price=5.0,
                min_history_days=252,
                require_fields=["cds_spread_5y"],
                exclude_sectors=[],
            ),
        ),
        portfolio=s.Portfolio(
            construction="long_short_quantile",
            quantile=0.1,
            n=None,
            threshold=s.PortfolioThreshold(lo=None, hi=None),
            weighting="equal",
            neutralize=["sector"],
            gross_leverage=1.0,
            net_exposure=0.0,
            max_position=0.02,
            rebalance="weekly",
            holding_period=1,
            execution_lag=1,
        ),
        costs=s.Costs(
            spread_model="adv_based",
            spread_bps=None,
            impact_model="sqrt",
            impact_coeff=0.1,
            borrow_bps_annual=50,
        ),
        assumptions=[_example_assumption()],
        params={"n_lookback": s.ParamSpec(value=5, sweep=[3, 10, 20])},
    )


def _example_backtest_result():
    return s.BacktestResult(
        positions=[s.Position(date=date(2020, 1, 3), asset_id="A0001", weight=0.02)],
        returns=[s.ReturnRow(date=date(2020, 1, 3), gross=0.001, net=0.0008, cost=0.0002)],
        trades=[s.Trade(date=date(2020, 1, 3), asset_id="A0001", delta_weight=0.02, cost_bps=1.5)],
    )


def _example_gate_row():
    return s.GateRow(name="G3_headline_significance", value=2.4, threshold=2.0, status="pass")


def _example_decision():
    return s.Decision(
        decided_utc=datetime(2026, 9, 7, 16, 0, 0, tzinfo=timezone.utc),
        decision="approve",
        conditions=None,
        reviewer="ash",
    )


EXAMPLES = {
    "manifest": _example_manifest,
    "claim": _example_claim,
    "field_def": _example_field_def,
    "resolution": _example_resolution,
    "spec": _example_spec,
    "assumption": _example_assumption,
    "backtest_result": _example_backtest_result,
    "gate_row": _example_gate_row,
    "decision": _example_decision,
}


@pytest.mark.parametrize("name", EXAMPLES)
def test_json_roundtrip(name):
    _roundtrip_json(EXAMPLES[name]())


@pytest.mark.parametrize("name", EXAMPLES)
def test_parquet_roundtrip(name, tmp_path):
    _roundtrip_parquet(EXAMPLES[name](), tmp_path)


def test_strict_models_reject_unknown_keys():
    with pytest.raises(Exception):
        s.Decision(
            decided_utc=datetime.now(timezone.utc),
            decision="approve",
            reviewer="ash",
            extra_field="not allowed",
        )
