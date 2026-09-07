"""Tests for notebooks/01_claim.py, 03_spec.py, 04_code.py (BUILD.md Phase 5
accept criteria). Notebook filenames start with a digit (matching DESIGN
§2's `01_claim.py` convention), so they aren't importable via a normal
`import` statement -- loaded here via importlib.util by file path, the
same way papermill executes them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from common import catalogue, config, llm, pit
from common import spec as spec_module
from common.schemas import Claim

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
PAPERS_DIR = Path(__file__).parent.parent / "fixtures" / "papers"


def _load_notebook(filename: str):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), NOTEBOOKS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


notebook_01 = _load_notebook("01_claim.py")
notebook_03 = _load_notebook("03_spec.py")
notebook_04 = _load_notebook("04_code.py")


# --------------------------------------------------------------------------
# Canned responses for the three fixture papers -- one dict per paper,
# covering Stage 1 extraction, Stage 1 tradability, Stage 3 spec, and
# Stage 4 code. Kept together so it's obvious each paper's spec really
# does express its own paper's thesis in the closed vocabulary.
# --------------------------------------------------------------------------

PAPERS = {
    "reversal.pdf": {
        "extraction": {
            "source_type": "paper",
            "source_title": "Short-Horizon Reversal in Daily Equity Returns",
            "source_authors": ["Ada Lin", "Marcus Ward"],
            "source_year": 2018,
            "source_doi": None,
            "edge_statement": "Stocks with the lowest one-day return outperform over the next five days.",
            "mechanism": "Short-term liquidity provision causes transient price moves to partially reverse.",
            "predicted_sign": -1,
            "universe_description": "broad, liquid, exchange-listed equities",
            "frequency": "daily",
            "horizon_days_lo": 1,
            "horizon_days_hi": 5,
            "formula_latex": "signal_t = -zscore(ret_{1d,t})",
            "formula_location": "Formula section",
            "required_fields": ["ret_1d", "close"],
            "reported_sharpe": 0.9,
            "reported_period_start": "2005-01-01",
            "reported_period_end": "2015-12-31",
            "reported_gross": True,
            "author_caveats": ["sensitive to exclusion of illiquid names"],
        },
        "tradability": {"tradable": True, "tradability_reason": "ret_1d and close both resolve against the catalogue."},
        "spec": {
            "signal": {"nodes": [{"id": "n1", "op": "cs_zscore", "inputs": ["ret_1d"], "params": {}}], "output": "n1", "sign": -1},
            "universe": {"base": "all", "filters": {"min_mktcap": 0, "min_adv_20": 0, "min_price": 0, "min_history_days": 0, "require_fields": [], "exclude_sectors": []}},
            "portfolio": {
                "construction": "long_short_quantile", "quantile": 0.1, "n": None, "threshold": {"lo": None, "hi": None},
                "weighting": "equal", "neutralize": [], "gross_leverage": 1.0, "net_exposure": 0.0, "max_position": 0.05,
                "rebalance": "daily", "holding_period": 1, "execution_lag": 1,
            },
            "costs": {"spread_model": "adv_based", "spread_bps": None, "impact_model": "sqrt", "impact_coeff": 0.1, "borrow_bps_annual": 50},
            "assumptions": [{"id": "A1", "choice": "no winsorization", "source_says": "nothing", "alternatives": ["cs_winsorize k=3"]}],
            "params": {},
        },
        "code": (
            "import pandas as pd\n\n"
            "def compute_signal(panel: pd.DataFrame) -> pd.Series:\n"
            '    s = panel.set_index(["date", "asset_id"])["ret_1d"].sort_index()\n'
            '    z = s.groupby(level="date").transform(lambda x: (x - x.mean()) / x.std())\n'
            "    return -1 * z\n"
        ),
    },
    "credit_momentum.pdf": {
        "extraction": {
            "source_type": "paper",
            "source_title": "Credit Leads Equity: CDS Spread Momentum and Cross-Asset Return Predictability",
            "source_authors": ["Elena Petrova", "Sam O'Rourke"],
            "source_year": 2016,
            "source_doi": None,
            "edge_statement": "Widening 5y CDS spreads over 5 days predict negative equity returns over the next 2-3 weeks.",
            "mechanism": "Credit investors are more attentive to downside risk and price new information first.",
            "predicted_sign": -1,
            "universe_description": "issuers with continuous single-name CDS coverage",
            "frequency": "weekly",
            "horizon_days_lo": 10,
            "horizon_days_hi": 15,
            "formula_latex": "signal_t = -zscore(cds\\_spread\\_5y_t - cds\\_spread\\_5y_{t-5})",
            "formula_location": "Formula section",
            "required_fields": ["cds_spread_5y"],
            "reported_sharpe": 1.3,
            "reported_period_start": "2008-01-01",
            "reported_period_end": "2014-12-31",
            "reported_gross": True,
            "author_caveats": ["sample spans the 2008 credit crisis", "no stale-quote handling specified"],
        },
        "tradability": {"tradable": True, "tradability_reason": "cds_spread_5y resolves against the catalogue for the cds_names universe."},
        "spec": {
            "signal": {
                "nodes": [
                    {"id": "n1", "op": "diff", "inputs": ["cds_spread_5y"], "params": {"n": "n_lookback"}},
                    {"id": "n2", "op": "cs_zscore", "inputs": ["n1"], "params": {}},
                ],
                "output": "n2", "sign": -1,
            },
            "universe": {"base": "cds_names", "filters": {"min_mktcap": 0, "min_adv_20": 0, "min_price": 0, "min_history_days": 0, "require_fields": [], "exclude_sectors": []}},
            "portfolio": {
                "construction": "long_short_quantile", "quantile": 0.1, "n": None, "threshold": {"lo": None, "hi": None},
                "weighting": "equal", "neutralize": [], "gross_leverage": 1.0, "net_exposure": 0.0, "max_position": 0.05,
                "rebalance": "weekly", "holding_period": 1, "execution_lag": 1,
            },
            "costs": {"spread_model": "adv_based", "spread_bps": None, "impact_model": "sqrt", "impact_coeff": 0.1, "borrow_bps_annual": 50},
            "assumptions": [{"id": "A1", "choice": "5-day CDS change window", "source_says": "5 trading days", "alternatives": [3, 10]}],
            "params": {"n_lookback": {"value": 5, "sweep": [3, 10]}},
        },
        "code": (
            "import pandas as pd\n\n"
            "def compute_signal(panel: pd.DataFrame) -> pd.Series:\n"
            '    s = panel.set_index(["date", "asset_id"])["cds_spread_5y"]\n'
            '    by_asset = s.reorder_levels(["asset_id", "date"]).sort_index()\n'
            "    diffed = by_asset.groupby(level=\"asset_id\").diff(5)\n"
            '    diffed = diffed.reorder_levels(["date", "asset_id"]).sort_index()\n'
            '    z = diffed.groupby(level="date").transform(lambda x: (x - x.mean()) / x.std())\n'
            "    return -1 * z\n"
        ),
    },
    "volume_momentum.pdf": {
        "extraction": {
            "source_type": "paper",
            "source_title": "Volume-Confirmed Momentum: High-Turnover Winners Continue to Win",
            "source_authors": ["Priya Nair"],
            "source_year": 2020,
            "source_doi": None,
            "edge_statement": "Stocks with strong 20-day momentum AND high trading volume continue to outperform over the next month.",
            "mechanism": "High volume proxies for broad information diffusion, making the move more likely to persist.",
            "predicted_sign": 1,
            "universe_description": "broad, exchange-listed equities",
            "frequency": "monthly",
            "horizon_days_lo": 20,
            "horizon_days_hi": 20,
            "formula_latex": "signal_t = 0.5 \\cdot zscore(pct\\_change\\_20d) + 0.5 \\cdot zscore(adv\\_20)",
            "formula_location": "Formula section",
            "required_fields": ["close", "adv_20"],
            "reported_sharpe": 1.1,
            "reported_period_start": "2010-01-01",
            "reported_period_end": "2019-12-31",
            "reported_gross": True,
            "author_caveats": ["combination weight chosen for simplicity, not fit"],
        },
        "tradability": {"tradable": True, "tradability_reason": "close and adv_20 both resolve against the catalogue."},
        "spec": {
            "signal": {
                "nodes": [
                    {"id": "n1", "op": "pct_change", "inputs": ["close"], "params": {"n": "n_lookback"}},
                    {"id": "n2", "op": "cs_zscore", "inputs": ["n1"], "params": {}},
                    {"id": "n3", "op": "cs_zscore", "inputs": ["adv_20"], "params": {}},
                    {"id": "n4", "op": "combine", "inputs": ["n2", "n3"], "params": {"weights": ["w1", "w2"]}},
                ],
                "output": "n4", "sign": 1,
            },
            "universe": {"base": "all", "filters": {"min_mktcap": 0, "min_adv_20": 0, "min_price": 0, "min_history_days": 0, "require_fields": [], "exclude_sectors": []}},
            "portfolio": {
                "construction": "long_short_quantile", "quantile": 0.1, "n": None, "threshold": {"lo": None, "hi": None},
                "weighting": "equal", "neutralize": [], "gross_leverage": 1.0, "net_exposure": 0.0, "max_position": 0.05,
                "rebalance": "monthly", "holding_period": 1, "execution_lag": 1,
            },
            "costs": {"spread_model": "adv_based", "spread_bps": None, "impact_model": "sqrt", "impact_coeff": 0.1, "borrow_bps_annual": 50},
            "assumptions": [{"id": "A1", "choice": "w1=0.5 (equal-weight combination)", "source_says": "equal blend chosen for simplicity", "alternatives": [0.3, 0.7]}],
            "params": {"n_lookback": {"value": 20, "sweep": [10, 40]}, "w1": {"value": 0.5}, "w2": {"value": 0.5}},
        },
        "code": (
            "import pandas as pd\n\n"
            "def compute_signal(panel: pd.DataFrame) -> pd.Series:\n"
            '    close = panel.set_index(["date", "asset_id"])["close"]\n'
            '    adv = panel.set_index(["date", "asset_id"])["adv_20"]\n'
            '    by_asset = close.reorder_levels(["asset_id", "date"]).sort_index()\n'
            "    mom = by_asset.groupby(level=\"asset_id\").pct_change(20)\n"
            '    mom = mom.reorder_levels(["date", "asset_id"]).sort_index()\n'
            '    mom_z = mom.groupby(level="date").transform(lambda x: (x - x.mean()) / x.std())\n'
            '    adv_z = adv.groupby(level="date").transform(lambda x: (x - x.mean()) / x.std())\n'
            "    return 0.5 * mom_z + 0.5 * adv_z\n"
        ),
    },
}


def _make_responder(paper_key: str):
    canned = PAPERS[paper_key]

    def responder(system: str, user: str) -> str:
        context = json.loads(user.split("\n\nYour previous")[0])
        if "source_text" in context:
            return json.dumps(canned["extraction"])
        if "claim" in context and "resolution" not in context:
            return json.dumps(canned["tradability"])
        if "resolution" in context:
            return json.dumps(canned["spec"])
        if "spec" in context:
            return json.dumps({"code": canned["code"]})
        raise AssertionError(f"unrecognized prompt context: {list(context)}")

    return responder


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.parquet")


@pytest.fixture
def use_fixture_panel(monkeypatch, fixture_panel_ready):
    monkeypatch.setattr(config, "PIPELINE_ROOT", fixture_panel_ready)
    return fixture_panel_ready


# --------------------------------------------------------------------------
# Stage 1: claims pass the Claim schema; required_fields are catalogue names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("paper", PAPERS)
def test_stage1_produces_valid_claims(paper, isolated_registry):
    client = llm.FakeLLMClient(responder=_make_responder(paper))
    catalogue_fields = set(catalogue.load_fields())

    result_manifest, claim = notebook_01.run_stage1(
        source_path=str(PAPERS_DIR / paper), source_type="paper", claim_id=f"c_{paper}", run_id="r1", client=client,
    )

    assert result_manifest.status == "complete"
    assert isinstance(claim, Claim)
    assert set(claim.required_fields) <= catalogue_fields


# --------------------------------------------------------------------------
# Stage 3: specs pass validate_spec; assumptions non-empty
# --------------------------------------------------------------------------


@pytest.mark.parametrize("paper", PAPERS)
def test_stage3_produces_valid_specs(paper, isolated_registry):
    client = llm.FakeLLMClient(responder=_make_responder(paper))
    claim = Claim(claim_id=f"c_{paper}", **PAPERS[paper]["extraction"], **PAPERS[paper]["tradability"])
    resolution = catalogue.resolve_required_fields(claim.required_fields)

    result_manifest, result_spec = notebook_03.run_stage3(claim, resolution, client, f"c_{paper}", "r1")

    assert result_manifest.status == "complete"
    spec_module.validate_spec(result_spec)  # should not raise
    assert len(result_spec.assumptions) > 0


# --------------------------------------------------------------------------
# Stage 4: round-trip passes on those specs; a corrupted signal.py is caught
# --------------------------------------------------------------------------


@pytest.mark.parametrize("paper", PAPERS)
def test_stage4_round_trip_passes(paper, isolated_registry, use_fixture_panel):
    from common.schemas import Spec

    claim = Claim(claim_id=f"c_{paper}", **PAPERS[paper]["extraction"], **PAPERS[paper]["tradability"])
    spec_obj = Spec(**PAPERS[paper]["spec"])

    required = claim.required_fields
    fields = required if "ret_1d" in required else [*required, "ret_1d"]
    panel = pit.load_panel(fields, spec_obj.universe.base, "2010-01-04", "2010-06-30")

    client = llm.FakeLLMClient(responder=_make_responder(paper))
    result_manifest, code = notebook_04.run_stage4(spec_obj, client, f"c_{paper}", "r1", panel)

    assert result_manifest.status == "complete", result_manifest.halt_reason
    assert code is not None


def test_stage4_catches_deliberately_corrupted_code(isolated_registry, use_fixture_panel):
    from common.schemas import Spec

    paper = "reversal.pdf"
    spec_obj = Spec(**PAPERS[paper]["spec"])
    panel = pit.load_panel(["ret_1d"], "all", "2010-01-04", "2010-06-30")

    corrupted_code = PAPERS[paper]["code"].replace("return -1 * z", "return z")  # sign dropped -> wrong output
    client = llm.FakeLLMClient(responder=lambda system, user: json.dumps({"code": corrupted_code}))

    result_manifest, code = notebook_04.run_stage4(spec_obj, client, "c_corrupted", "r1", panel)

    assert result_manifest.status == "halted"
    assert "round-trip" in result_manifest.halt_reason
    assert code is None


# --------------------------------------------------------------------------
# Parse failure retries once then halts (also covered at the llm.py level
# in tests/test_llm.py; this confirms the halt actually propagates through
# a real stage, not just complete_structured in isolation).
# --------------------------------------------------------------------------


def test_stage1_halts_after_repeated_parse_failure(isolated_registry):
    calls = []

    def responder(system, user):
        calls.append(user)
        return "this is not json"

    client = llm.FakeLLMClient(responder=responder)

    result_manifest, claim = notebook_01.run_stage1(
        source_path=str(PAPERS_DIR / "reversal.pdf"), source_type="paper", claim_id="c_badparse", run_id="r1", client=client,
    )

    assert result_manifest.status == "halted"
    assert "extraction" in result_manifest.halt_reason
    assert claim is None
    assert len(calls) == 2  # complete_structured's own one retry, then LLMParseError propagates up to a halt
