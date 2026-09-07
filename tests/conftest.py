"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_panel_ready() -> Path:
    """Builds fixtures/panel/*.parquet once per test session if it isn't
    already there (it's gitignored generated data, see fixtures/build_fixture.py)."""
    if not (FIXTURES_DIR / "panel" / "close.parquet").exists():
        from fixtures.build_fixture import build

        build()
    return FIXTURES_DIR
