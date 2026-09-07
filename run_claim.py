"""Orchestrates all seven stages for one claim (DESIGN §7), end to end.

Usage:
    python run_claim.py --source fixtures/papers/reversal.pdf
    python run_claim.py --source fixtures/papers/reversal.pdf --claim-id c_existing_0001  # re-run

Calls each stage's `run_stageN()` directly (see notebooks/*.py) rather
than executing paired .ipynb files through a papermill subprocess.
DESIGN's papermill/executed-notebook mechanism is what a real deployment
would use for that artifact; this achieves the same result -- a real
manifest chain, a real evidence package -- without the added fragility of
kernel-based notebook execution, since every notebook already exposes a
clean, directly-callable entry point. Swapping in real papermill execution
later is mechanical, not a redesign.

Re-run semantics (DESIGN §9): passing --claim-id re-runs an existing claim
under a new run_id with parent_run_id set, and is refused once that claim
already has 3 runs. This re-executes all seven stages fresh (including
re-extracting Stage 1 from the same source) rather than DESIGN's more
nuanced "human modifies just the spec, stages 1-2 are reused" flow --
a deliberate simplification; the accept-tested behavior (new run_id,
parent_run_id set, trial_count carried forward, 4th run refused) holds
either way.
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import config, llm, manifest, pit, report

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
MAX_RUNS_PER_CLAIM = 3


def _load_notebook(filename: str):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), NOTEBOOKS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_claim_id() -> str:
    return f"c_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


def _new_run_id() -> str:
    return f"r_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


def _existing_run_ids(claim_id: str) -> list[str]:
    if not config.REGISTRY_PATH.exists():
        return []
    registry = pd.read_parquet(config.REGISTRY_PATH)
    return sorted(registry.loc[registry["claim_id"] == claim_id, "run_id"].unique())


class ClaimRefused(Exception):
    """Raised when a claim already has MAX_RUNS_PER_CLAIM runs (DESIGN §9)."""


def run_claim(
    source: str,
    client: llm.LLMClient,
    claim_id: str | None = None,
    universe: str = "all",
    headline_start: str = "2010-01-04",
    headline_end: str = "2020-12-31",
) -> dict:
    claim_id = claim_id or _new_claim_id()
    existing_runs = _existing_run_ids(claim_id)
    if len(existing_runs) >= MAX_RUNS_PER_CLAIM:
        raise ClaimRefused(f"claim {claim_id!r} already has {len(existing_runs)} runs (max {MAX_RUNS_PER_CLAIM}); refused")

    run_id = _new_run_id()
    parent_run_id = existing_runs[-1] if existing_runs else None

    n01 = _load_notebook("01_claim.py")
    n02 = _load_notebook("02_data.py")
    n03 = _load_notebook("03_spec.py")
    n04 = _load_notebook("04_code.py")
    n05 = _load_notebook("05_backtest.py")
    n06 = _load_notebook("06_evaluate.py")
    n07 = _load_notebook("07_ic_pack.py")

    manifests: dict[str, manifest.Manifest] = {}

    def _halted(stage: str, m: manifest.Manifest) -> dict:
        manifests[stage] = m
        return {"status": m.status, "stage": stage, "claim_id": claim_id, "run_id": run_id, "manifests": manifests}

    m1, claim = n01.run_stage1(source, "paper", claim_id, run_id, client, parent_run_id=parent_run_id)
    manifests["01_claim"] = m1
    if m1.status != "complete":
        return _halted("01_claim", m1)

    m2, resolution, coverage = n02.run_stage2(claim, claim_id, run_id, universe, headline_start, headline_end, parent_run_id=parent_run_id)
    manifests["02_data"] = m2
    if m2.status != "complete":
        return _halted("02_data", m2)

    m3, spec = n03.run_stage3(claim, resolution, client, claim_id, run_id, parent_run_id=parent_run_id)
    manifests["03_spec"] = m3
    if m3.status != "complete":
        return _halted("03_spec", m3)

    fields = n05.spec_fields(spec)
    fixture_panel = pit.load_panel(fields, spec.universe.base, headline_start, headline_end)
    m4, code = n04.run_stage4(spec, client, claim_id, run_id, fixture_panel, parent_run_id=parent_run_id)
    manifests["04_code"] = m4
    if m4.status != "complete":
        return _halted("04_code", m4)

    m5, results = n05.run_stage5(spec, claim, claim_id, run_id, parent_run_id=parent_run_id, headline_start=headline_start, headline_end=headline_end)
    manifests["05_backtest"] = m5
    if m5.status != "complete":
        return _halted("05_backtest", m5)

    factors_path = config.PIPELINE_ROOT / "factors.parquet"
    factors = pd.read_parquet(factors_path) if factors_path.exists() else pd.DataFrame({"date": []})
    m6, critique = n06.run_stage6(
        spec, claim, results["headline"], results["replication"], results["oos"], fixture_panel,
        client, claim_id, run_id, factors, parent_run_id=parent_run_id,
    )
    manifests["06_evaluation"] = m6
    if m6.status != "complete":
        return _halted("06_evaluation", m6)

    gates_df = manifest.read_input(claim_id, run_id, "06_evaluation", "gates.parquet")
    sweeps_df = manifest.read_input(claim_id, run_id, "06_evaluation", "sweeps.parquet")
    factor_decomp = manifest.read_input(claim_id, run_id, "06_evaluation", "factor_decomp.parquet").iloc[0].to_dict()

    artifacts = report.RunArtifacts(
        claim=claim, spec=spec, results=results, gates_df=gates_df, sweeps_df=sweeps_df,
        factor_decomposition=factor_decomp, critique_text=critique, manifests=manifests,
        trial_count=manifest.get_trial_count(claim_id),
    )
    m7 = n07.run_stage7(artifacts, claim_id, run_id, parent_run_id=parent_run_id)
    manifests["07_ic_pack"] = m7

    return {"status": m7.status, "stage": "07_ic_pack", "claim_id": claim_id, "run_id": run_id, "manifests": manifests}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="path to a PDF or idea.md")
    parser.add_argument("--claim-id", default=None, help="re-run an existing claim under a new run_id")
    parser.add_argument("--universe", default="all")
    parser.add_argument("--headline-start", default="2010-01-04")
    parser.add_argument("--headline-end", default="2020-12-31")
    args = parser.parse_args()

    outcome = run_claim(
        args.source, llm.AnthropicClient(), claim_id=args.claim_id, universe=args.universe,
        headline_start=args.headline_start, headline_end=args.headline_end,
    )
    print(f"status={outcome['status']} stage={outcome['stage']} claim_id={outcome['claim_id']} run_id={outcome['run_id']}")
