"""Stage 7 -- IC pack.

Deterministic (no LLM call): renders the evidence package (DESIGN §10, all
eleven sections) via `report.render_ic_pack`, and writes `decision.parquet`
-- blank, to be filled in by a human reviewer, not the pipeline.
"""

from __future__ import annotations

# %% tags=["parameters"]
claim_id = ""
run_id = ""
parent_run_id = None
code_commit = ""
common_version = "0.0.1"

# %%
import hashlib
from pathlib import Path

import pandas as pd

from common import manifest, report
from common.schemas import FileRef


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage7(
    artifacts: report.RunArtifacts,
    claim_id: str,
    run_id: str,
    parent_run_id: str | None = None,
    code_commit: str = "",
    common_version: str = "0.0.1",
) -> manifest.Manifest:
    ctx = manifest.begin("07_ic_pack", claim_id, run_id, parent_run_id=parent_run_id)

    md_path, pdf_path = report.render_ic_pack(artifacts, ctx.stage_dir)

    decision_path = ctx.stage_dir / "decision.parquet"
    pd.DataFrame(
        [{"decided_utc": pd.NaT, "decision": None, "conditions": None, "reviewer": None}]
    ).to_parquet(decision_path, index=False)

    outputs = [
        FileRef(path="ic_pack.md", sha256=_sha256_file(md_path)),
        FileRef(path="ic_pack.pdf", sha256=_sha256_file(pdf_path)),
        FileRef(path="decision.parquet", sha256=_sha256_file(decision_path)),
    ]
    return manifest.complete(ctx, outputs=outputs, code_commit=code_commit, common_version=common_version)
