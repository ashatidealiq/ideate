"""Tests for common/pit.py (BUILD.md Phase 1 accept criteria)."""

from __future__ import annotations

import pandas as pd
import pytest

from common import config, pit


@pytest.fixture(autouse=True)
def use_fixture_panel(monkeypatch, fixture_panel_ready):
    monkeypatch.setattr(config, "PIPELINE_ROOT", fixture_panel_ready)


def test_load_panel_shape_and_columns():
    df = pit.load_panel(["close"], "all", "2010-01-04", "2010-01-08")
    assert list(df.columns) == ["date", "asset_id", "close"]
    assert df["close"].notna().all()


def test_load_panel_multiple_fields_merge_on_date_asset():
    df = pit.load_panel(["close", "volume"], "all", "2010-01-04", "2010-01-08")
    assert list(df.columns) == ["date", "asset_id", "close", "volume"]


def test_load_panel_unknown_field_raises():
    with pytest.raises(KeyError):
        pit.load_panel(["not_a_real_field"], "all", "2010-01-04", "2010-01-08")


def test_load_panel_unknown_universe_raises():
    with pytest.raises(KeyError):
        pit.load_panel(["close"], "not_a_real_universe", "2010-01-04", "2010-01-08")


def test_cds_universe_only_includes_cds_names():
    df = pit.load_panel(["cds_spread_5y"], "cds_names", "2010-01-04", "2010-01-08")
    assert df["asset_id"].nunique() <= 80
    assert df["cds_spread_5y"].notna().all()


def test_lookahead_trap_never_returned_before_date_plus_30d(fixture_panel_ready):
    d0 = pd.Timestamp("2010-03-01")

    as_of_early = d0 + pd.Timedelta(days=29)
    df_early = pit.load_panel(
        ["lookahead_trap"], "all", str(d0.date()), str(d0.date()), as_of=str(as_of_early.date())
    )
    assert df_early["lookahead_trap"].isna().all()

    as_of_ready = d0 + pd.Timedelta(days=30)
    df_ready = pit.load_panel(
        ["lookahead_trap"], "all", str(d0.date()), str(d0.date()), as_of=str(as_of_ready.date())
    )
    assert df_ready["lookahead_trap"].notna().any()


def test_as_of_matches_raw_knowledge_date_filter_exactly(fixture_panel_ready):
    as_of = pd.Timestamp("2010-06-15")
    start, end = "2010-01-04", "2010-12-31"

    raw = pd.read_parquet(fixture_panel_ready / "panel" / "cds_spread_5y.parquet")
    raw = raw[(raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(end))]
    visible = raw[raw["knowledge_date"] <= as_of]
    expected = set(zip(visible["date"], visible["asset_id"]))

    df = pit.load_panel(["cds_spread_5y"], "cds_names", start, end, as_of=str(as_of.date()))
    non_null = df[df["cds_spread_5y"].notna()]
    actual = set(zip(non_null["date"], non_null["asset_id"]))

    assert actual == expected


def test_delisted_names_disappear_after_delist_date(fixture_panel_ready):
    close = pd.read_parquet(fixture_panel_ready / "panel" / "close.parquet")
    last_seen = close.groupby("asset_id")["date"].max()
    panel_end = close["date"].max()
    delisted = last_seen[last_seen < panel_end]
    assert len(delisted) > 0  # sanity: the fixture does plant delisted names

    asset_id, delist_date = next(iter(delisted.items()))
    df = pit.load_panel(["close"], "all", "2010-01-04", "2020-12-31")
    asset_rows = df[df["asset_id"] == asset_id]

    assert asset_rows["date"].max() == delist_date
    assert (asset_rows["date"] <= delist_date).all()
