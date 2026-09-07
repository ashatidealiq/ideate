"""Stage 2 -- Data resolution.

Deterministic (no LLM call): joins `claim.required_fields` against the
catalogue (`catalogue.resolve_required_fields`), then checks coverage --
how many universe members actually have every required field, per date,
over the proposed test period (DESIGN §7 Stage 2).

Halts if: any field is unresolved (and appends a row per unresolved field
to `$PIPELINE_ROOT/unmet_requirements.parquet` -- a cross-claim log, not a
per-run artifact, so failed data requirements accumulate across the whole
pipeline's history); or coverage drops below 30 names on more than 10% of
dates in the period.
"""

from __future__ import annotations

# %% tags=["parameters"]
claim_id = ""
run_id = ""
parent_run_id = None
code_commit = ""
common_version = "0.0.1"
start = "2010-01-04"
end = "2020-12-31"
min_coverage_names = 30
max_fraction_dates_below_coverage = 0.10

# %%
import hashlib
from pathlib import Path

import pandas as pd

from common import catalogue, config, manifest, pit
from common.schemas import Claim, FileRef, Resolution


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_unmet_requirements(claim: Claim, unresolved: list[Resolution]) -> None:
    path = config.PIPELINE_ROOT / "unmet_requirements.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        [{"claim_id": claim.claim_id, "field": r.field, "recorded_utc": pd.Timestamp.now(tz="UTC")} for r in unresolved]
    )
    if path.exists():
        rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
    rows.to_parquet(path, index=False)


def run_stage2(
    claim: Claim,
    claim_id: str,
    run_id: str,
    universe: str,
    start: str,
    end: str,
    parent_run_id: str | None = None,
    code_commit: str = "",
    common_version: str = "0.0.1",
    min_coverage_names: int = 30,
    max_fraction_dates_below_coverage: float = 0.10,
) -> tuple[manifest.Manifest, list[Resolution] | None, pd.DataFrame | None]:
    """Returns (manifest, resolution, coverage) -- the last two are None
    if the stage halted."""
    ctx = manifest.begin("02_data", claim_id, run_id, parent_run_id=parent_run_id)
    common_kwargs = dict(code_commit=code_commit, common_version=common_version)

    resolution = catalogue.resolve_required_fields(claim.required_fields)
    unresolved = [r for r in resolution if not r.resolved]

    if unresolved:
        _append_unmet_requirements(claim, unresolved)
        return (
            manifest.halt(ctx, halt_reason=f"unresolved field(s): {[r.field for r in unresolved]}", **common_kwargs),
            None,
            None,
        )

    resolved_fields = [r.field for r in resolution]
    panel = pit.load_panel(resolved_fields, universe, start, end)
    complete_mask = panel[resolved_fields].notna().all(axis=1)
    coverage = panel.loc[complete_mask].groupby("date").size().rename("n_names").reset_index()

    all_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    coverage = coverage.set_index("date").reindex(all_dates, fill_value=0).rename_axis("date").reset_index()

    frac_below = float((coverage["n_names"] < min_coverage_names).mean())
    if frac_below > max_fraction_dates_below_coverage:
        return (
            manifest.halt(
                ctx,
                halt_reason=(
                    f"coverage below {min_coverage_names} names on {frac_below:.1%} of dates "
                    f"(max allowed {max_fraction_dates_below_coverage:.0%})"
                ),
                **common_kwargs,
            ),
            None,
            None,
        )

    resolution_path = ctx.stage_dir / "resolution.parquet"
    pd.DataFrame([r.model_dump(mode="json") for r in resolution]).to_parquet(resolution_path, index=False)
    coverage_path = ctx.stage_dir / "coverage.parquet"
    coverage.to_parquet(coverage_path, index=False)

    outputs = [
        FileRef(path="resolution.parquet", sha256=_sha256_file(resolution_path)),
        FileRef(path="coverage.parquet", sha256=_sha256_file(coverage_path)),
    ]
    result_manifest = manifest.complete(ctx, outputs=outputs, **common_kwargs)
    return result_manifest, resolution, coverage
