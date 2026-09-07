"""Tests for run_claim.py (BUILD.md Phase 6 accept criteria)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from common import config, llm, manifest
from run_claim import ClaimRefused, MAX_RUNS_PER_CLAIM, run_claim

EXTRACTION = {
    "source_type": "paper",
    "source_title": "Short-Horizon Reversal in Daily Equity Returns",
    "source_authors": ["Ada Lin", "Marcus Ward"],
    "source_year": 2018,
    "source_doi": None,
    "edge_statement": "A fixture-only planted signal predicts returns (integration test).",
    "mechanism": "Test mechanism for end-to-end pipeline verification.",
    "predicted_sign": 1,
    "universe_description": "all",
    "frequency": "weekly",
    "horizon_days_lo": 1,
    "horizon_days_hi": 5,
    "formula_latex": None,
    "formula_location": None,
    "required_fields": ["planted_signal_1"],
    "reported_sharpe": 1.0,
    "reported_period_start": "2010-01-04",
    "reported_period_end": "2015-12-31",
    "reported_gross": True,
    "author_caveats": [],
}
TRADABILITY = {"tradable": True, "tradability_reason": "planted_signal_1 resolves against the catalogue."}
SPEC_JSON = {
    "signal": {
        "nodes": [
            {"id": "n1", "op": "clip", "inputs": ["planted_signal_1"], "params": {"lo": "clip_lo", "hi": "clip_hi"}},
            {"id": "n2", "op": "cs_zscore", "inputs": ["n1"], "params": {}},
        ],
        "output": "n2", "sign": 1,
    },
    "universe": {"base": "all", "filters": {"min_mktcap": 0, "min_adv_20": 0, "min_price": 0, "min_history_days": 0, "require_fields": [], "exclude_sectors": []}},
    "portfolio": {
        "construction": "long_short_quantile", "quantile": 0.1, "n": None, "threshold": {"lo": None, "hi": None},
        "weighting": "equal", "neutralize": [], "gross_leverage": 1.0, "net_exposure": 0.0, "max_position": 0.05,
        "rebalance": "weekly", "holding_period": 1, "execution_lag": 1,
    },
    "costs": {"spread_model": "fixed_bps", "spread_bps": 1.0, "impact_model": "none", "impact_coeff": 0, "borrow_bps_annual": 5},
    "assumptions": [{"id": "A1", "choice": "clip bounds generous, should not bind", "source_says": "nothing", "alternatives": []}],
    "params": {"clip_lo": {"value": -100.0, "sweep": [-50.0, -200.0]}, "clip_hi": {"value": 100.0, "sweep": [50.0, 200.0]}},
}
CODE = (
    "import pandas as pd\n\n"
    "def compute_signal(panel: pd.DataFrame) -> pd.Series:\n"
    '    s = panel.set_index(["date", "asset_id"])["planted_signal_1"].sort_index()\n'
    "    clipped = s.clip(lower=-100.0, upper=100.0)\n"
    '    z = clipped.groupby(level="date").transform(lambda x: (x - x.mean()) / x.std())\n'
    "    return z\n"
)
CRITIQUE = {"critique": "Strong, low-turnover signal; clean factor decomposition; stable across sweeps.", "verdict": "advance"}


def _responder(system: str, user: str) -> str:
    context = json.loads(user.split("\n\nYour previous")[0])
    if "source_text" in context:
        return json.dumps(EXTRACTION)
    if "catalogue_fields" in context and "claim" in context and "resolution" not in context:
        return json.dumps(TRADABILITY)
    if "resolution" in context:
        return json.dumps(SPEC_JSON)
    if "spec" in context and "sweeps" not in context:
        return json.dumps({"code": CODE})
    if "sweeps" in context:
        return json.dumps(CRITIQUE)
    raise AssertionError(f"unrecognized prompt context: {list(context)}")


@pytest.fixture
def client() -> llm.LLMClient:
    return llm.FakeLLMClient(responder=_responder)


@pytest.fixture
def isolated_run(tmp_path, monkeypatch, fixture_panel_ready):
    monkeypatch.setattr(config, "PIPELINE_ROOT", fixture_panel_ready)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.parquet")
    return tmp_path


SOURCE = "fixtures/papers/reversal.pdf"


def test_run_claim_executes_all_seven_stages_with_verifying_manifest_chain(client, isolated_run):
    result = run_claim(SOURCE, client)

    assert result["status"] == "complete"
    assert list(result["manifests"]) == [
        "01_claim", "02_data", "03_spec", "04_code", "05_backtest", "06_evaluation", "07_ic_pack",
    ]
    assert all(m.status == "complete" for m in result["manifests"].values())

    claim_id, run_id = result["claim_id"], result["run_id"]
    # manifest.read_input() is specifically for parquet stage outputs (DESIGN
    # §3.1: "the only thing the next stage may read"); every *other* declared
    # output (spec.yaml, english.md, signal.py, ic_pack.pdf, ...) is verified
    # here the same way read_input verifies parquet: recompute its sha256 and
    # compare against what the manifest recorded.
    for stage, m in result["manifests"].items():
        stage_dir = manifest.stage_dir(claim_id, run_id, stage)
        for out in m.outputs:
            if out.path.endswith(".parquet"):
                manifest.read_input(claim_id, run_id, stage, out.path)  # raises on any hash mismatch
            else:
                actual_sha256 = manifest._sha256_file(stage_dir / out.path)
                assert actual_sha256 == out.sha256, f"{stage}/{out.path}: hash mismatch"

    ic_pack_dir = config.RUNS_DIR / claim_id / run_id / "07_ic_pack"
    assert (ic_pack_dir / "ic_pack.md").exists()
    assert (ic_pack_dir / "ic_pack.pdf").exists()


def test_ic_pack_renders_all_eleven_sections(client, isolated_run):
    import re

    result = run_claim(SOURCE, client)
    assert result["status"] == "complete"

    md_path = config.RUNS_DIR / result["claim_id"] / result["run_id"] / "07_ic_pack" / "ic_pack.md"
    content = md_path.read_text(encoding="utf-8")
    sections = re.findall(r"^## (\d+)\.", content, re.MULTILINE)
    assert sections == [str(i) for i in range(1, 12)]

    pdf_path = config.RUNS_DIR / result["claim_id"] / result["run_id"] / "07_ic_pack" / "ic_pack.pdf"
    assert pdf_path.stat().st_size > 0


def test_second_run_gets_new_run_id_parent_set_trial_count_carried_forward(client, isolated_run):
    first = run_claim(SOURCE, client)
    assert first["status"] == "complete"
    claim_id = first["claim_id"]
    trial_count_after_first = manifest.get_trial_count(claim_id)
    assert trial_count_after_first > 0

    second = run_claim(SOURCE, client, claim_id=claim_id)
    assert second["status"] == "complete"

    assert second["run_id"] != first["run_id"]
    second_manifest = second["manifests"]["01_claim"]
    assert second_manifest.parent_run_id == first["run_id"]

    trial_count_after_second = manifest.get_trial_count(claim_id)
    assert trial_count_after_second > trial_count_after_first  # carried forward and grown, not reset


def test_fourth_run_of_same_claim_is_refused(client, isolated_run):
    claim_id = None
    for _ in range(MAX_RUNS_PER_CLAIM):
        result = run_claim(SOURCE, client, claim_id=claim_id)
        assert result["status"] == "complete"
        claim_id = result["claim_id"]

    with pytest.raises(ClaimRefused):
        run_claim(SOURCE, client, claim_id=claim_id)


def test_stage2_halts_on_unresolvable_field_and_logs_unmet_requirements(isolated_run):
    """DESIGN §7 Stage 1 *also* halts on a required_field outside the
    catalogue vocabulary, so a claim built through the full LLM pipeline
    can never actually reach Stage 2 with one -- Stage 1 catches it first.
    BUILD.md's scenario ("removing a catalogue field") is Stage 2's own
    condition: a field that was resolvable when the claim was written is
    no longer in the catalogue by the time data resolution runs. Testing
    that means calling Stage 2 directly with such a claim, the same way
    Phase 4's gate tests construct inputs directly rather than routing
    through stages that would filter them out first.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("notebook_02", "notebooks/02_data.py")
    notebook_02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook_02)

    from common.schemas import Claim

    claim = Claim(claim_id="c_stage2_test", **{**EXTRACTION, "required_fields": ["not_a_real_catalogue_field"]}, **TRADABILITY)

    result_manifest, resolution, coverage = notebook_02.run_stage2(
        claim, "c_stage2_test", "r1", "all", "2010-01-04", "2020-12-31"
    )

    assert result_manifest.status == "halted"
    assert resolution is None

    unmet_path = config.PIPELINE_ROOT / "unmet_requirements.parquet"
    assert unmet_path.exists()
    unmet = pd.read_parquet(unmet_path)
    assert (unmet["field"] == "not_a_real_catalogue_field").any()
    assert (unmet["claim_id"] == "c_stage2_test").any()
