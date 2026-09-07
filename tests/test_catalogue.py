"""Tests for common/catalogue.py."""

from __future__ import annotations

import pytest

from common import catalogue


def test_load_fields_returns_all_fixture_fields():
    fields = catalogue.load_fields()
    assert "close" in fields
    assert "lookahead_trap" in fields
    assert fields["lookahead_trap"].pit.lag_days == 30
    assert fields["cds_spread_5y"].pit.lag_days == 1


def test_load_fields_rejects_unknown_key(tmp_path):
    bad = tmp_path / "fields.yaml"
    bad.write_text(
        "fields:\n"
        "  - name: x\n"
        "    source: panel/x.parquet\n"
        "    dtype: float64\n"
        "    pit: {event_col: date, knowledge_col: date, lag_days: 0}\n"
        "    coverage: {start: '2010-01-01', universes: [all]}\n"
        "    description: x\n"
        "    not_a_real_key: true\n"
    )
    with pytest.raises(Exception):
        catalogue.load_fields(bad)


def test_load_fields_rejects_duplicate_name(tmp_path):
    dup = tmp_path / "fields.yaml"
    entry = (
        "  - name: x\n"
        "    source: panel/x.parquet\n"
        "    dtype: float64\n"
        "    pit: {event_col: date, knowledge_col: date, lag_days: 0}\n"
        "    coverage: {start: '2010-01-01', universes: [all]}\n"
        "    description: x\n"
    )
    dup.write_text("fields:\n" + entry + entry)
    with pytest.raises(ValueError):
        catalogue.load_fields(dup)


def test_get_field_raises_on_unknown_name():
    with pytest.raises(KeyError):
        catalogue.get_field("not_a_real_field")


def test_load_universes_returns_expected_names():
    universes = catalogue.load_universes()
    assert set(universes) == {"all", "cds_names"}
    assert universes["cds_names"].member_if_present == "cds_spread_5y"


def test_get_universe_raises_on_unknown_name():
    with pytest.raises(KeyError):
        catalogue.get_universe("not_a_real_universe")


def test_load_costs_returns_expected_regions():
    regions = catalogue.load_costs()
    assert set(regions) == {"US", "EU"}
    assert regions["US"].adv_buckets[-1].max_adv is None


def test_load_costs_rejects_missing_unbounded_bucket(tmp_path):
    bad = tmp_path / "costs.yaml"
    bad.write_text(
        "regions:\n"
        "  - region: US\n"
        "    adv_buckets:\n"
        "      - {max_adv: 1000000, spread_bps: 10}\n"
    )
    with pytest.raises(ValueError, match="max_adv: null"):
        catalogue.load_costs(bad)


def test_get_region_costs_raises_on_unknown_region():
    with pytest.raises(KeyError):
        catalogue.get_region_costs("not_a_real_region")
