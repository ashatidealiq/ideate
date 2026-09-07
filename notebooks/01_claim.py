"""Stage 1 -- Claim.

Extracts a claim from a source (PDF or `idea.md`) via two LLM calls --
extraction (`prompts/s1_extract.md`), then a tradability judgment
(`prompts/s1_tradability.md`) -- and halts if the claim isn't tradable or
any `required_fields` fall outside the catalogue vocabulary (DESIGN §7
Stage 1). Human-originated ideas (`source_type: human`) still go through
extraction: the source text is just the idea's own prose rather than a
paper, so the same transcription discipline applies.

Jupytext percent format: this file is plain, directly runnable Python.
`run_stage1` is what notebooks/tests actually call; the parameters cell
below is what papermill (Phase 6's `run_claim.py`) injects into.
"""

from __future__ import annotations

# %% tags=["parameters"]
source_path = ""
source_type = "paper"  # paper | human | dataset
claim_id = ""
run_id = ""
parent_run_id = None
code_commit = ""
common_version = "0.0.1"

# %%
import hashlib
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from common import catalogue, llm, manifest
from common.schemas import Claim, ClaimExtraction, FileRef, TradabilityJudgment

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
EXTRACT_PROMPT = PROMPTS_DIR / "s1_extract.md"
TRADABILITY_PROMPT = PROMPTS_DIR / "s1_tradability.md"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source_text(path: Path) -> str:
    """PDFs are extracted page-by-page with pypdf; anything else (idea.md)
    is read as plain text (DESIGN §7: "PDF, or idea.md for human/dataset-
    originated ideas")."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _write_claim_md(claim: Claim) -> str:
    lines = [
        f"# {claim.source_title}",
        "",
        f"*{', '.join(claim.source_authors)} ({claim.source_year})*",
        "",
        f"**Edge statement:** {claim.edge_statement}",
        "",
        f"**Mechanism:** {claim.mechanism}",
        "",
        f"**Predicted sign:** {claim.predicted_sign}  ",
        f"**Universe:** {claim.universe_description}  ",
        f"**Frequency:** {claim.frequency}  ",
        f"**Required fields:** {', '.join(claim.required_fields)}",
        "",
        f"**Tradable:** {claim.tradable} -- {claim.tradability_reason}",
    ]
    if claim.author_caveats:
        lines += ["", "**Author caveats:**"] + [f"- {c}" for c in claim.author_caveats]
    return "\n".join(lines)


def run_stage1(
    source_path: str,
    source_type: str,
    claim_id: str,
    run_id: str,
    client: llm.LLMClient,
    parent_run_id: str | None = None,
    code_commit: str = "",
    common_version: str = "0.0.1",
) -> tuple[manifest.Manifest, Claim | None]:
    """Runs Stage 1 end to end. Returns (manifest, claim) -- claim is None
    if the stage halted."""
    ctx = manifest.begin("01_claim", claim_id, run_id, parent_run_id=parent_run_id)

    catalogue_fields = sorted(catalogue.load_fields())
    llm_raw_path = ctx.stage_dir / "llm_raw.jsonl"
    source_text = read_source_text(Path(source_path))

    halt_common = dict(
        inputs=[FileRef(path=str(source_path), sha256=_sha256_file(Path(source_path)))],
        code_commit=code_commit,
        common_version=common_version,
    )
    try:
        extraction, extract_sha = llm.complete_structured(
            client,
            EXTRACT_PROMPT,
            {"source_text": source_text, "source_type": source_type, "catalogue_fields": catalogue_fields},
            ClaimExtraction,
            llm_raw_path,
        )
    except llm.LLMParseError as e:
        return manifest.halt(ctx, halt_reason=f"extraction: {e}", **halt_common), None

    try:
        tradability, tradability_sha = llm.complete_structured(
            client,
            TRADABILITY_PROMPT,
            {"claim": extraction.model_dump(mode="json"), "catalogue_fields": catalogue_fields},
            TradabilityJudgment,
            llm_raw_path,
        )
    except llm.LLMParseError as e:
        return manifest.halt(ctx, halt_reason=f"tradability: {e}", **halt_common), None

    claim = Claim(claim_id=claim_id, **extraction.model_dump(), **tradability.model_dump())
    unknown_fields = sorted(set(claim.required_fields) - set(catalogue_fields))

    common_kwargs = dict(
        inputs=[FileRef(path=str(source_path), sha256=_sha256_file(Path(source_path)))],
        code_commit=code_commit,
        common_version=common_version,
        llm=manifest.LLMCallInfo(model=llm.MODEL_ID, prompt_sha256=extract_sha, temperature=llm.TEMPERATURE, seed=llm.SEED),
    )

    if not claim.tradable:
        return manifest.halt(ctx, halt_reason=f"not tradable: {claim.tradability_reason}", **common_kwargs), None
    if unknown_fields:
        return (
            manifest.halt(ctx, halt_reason=f"required_fields outside catalogue vocabulary: {unknown_fields}", **common_kwargs),
            None,
        )

    claim_path = ctx.stage_dir / "claim.parquet"
    pd.DataFrame([claim.model_dump(mode="json")]).to_parquet(claim_path, index=False)
    md_path = ctx.stage_dir / "claim.md"
    md_path.write_text(_write_claim_md(claim), encoding="utf-8")

    outputs = [
        FileRef(path="claim.parquet", sha256=_sha256_file(claim_path)),
        FileRef(path="claim.md", sha256=_sha256_file(md_path)),
    ]
    result_manifest = manifest.complete(ctx, outputs=outputs, **common_kwargs)
    return result_manifest, claim


# %%
if __name__ == "__main__":
    run_stage1(source_path, source_type, claim_id, run_id, llm.AnthropicClient(), parent_run_id, code_commit, common_version)
