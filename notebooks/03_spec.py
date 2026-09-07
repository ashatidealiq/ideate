"""Stage 3 -- Spec.

One LLM call (`prompts/s3_spec.md`), given the claim and its field
resolution, proposes a full spec: the signal DAG, universe, portfolio
construction, costs, and assumptions -- this is the "how would we actually
trade or test this thesis" step. `spec.validate_spec` then checks it
against the closed vocabulary deterministically; on failure, the notebook
re-prompts with the specific error once before halting (DESIGN §7 Stage 3:
"Halt if: spec validation fails after 2 attempts; or assumptions block
empty").

Not implemented here: DESIGN §7 also describes a second LLM call comparing
`english.md` to the claim's own rationale, halting on a mismatch. That
would need a sixth prompt file; BUILD.md Phase 5 specifies exactly five
("s1_extract, s1_tradability, s3_spec, s4_code, s6_critique"), and its
accept criteria for Stage 3 don't exercise this check either. Deferred,
not silently dropped -- flagging it here so it isn't forgotten.
"""

from __future__ import annotations

# %% tags=["parameters"]
claim_id = ""
run_id = ""
parent_run_id = None
code_commit = ""
common_version = "0.0.1"
max_attempts = 2

# %%
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from common import compile as compiler
from common import llm, manifest
from common import spec as spec_module
from common.schemas import Assumption, Claim, FileRef, Resolution, Spec

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SPEC_PROMPT = PROMPTS_DIR / "s3_spec.md"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage3(
    claim: Claim,
    resolution: list[Resolution],
    client: llm.LLMClient,
    claim_id: str,
    run_id: str,
    parent_run_id: str | None = None,
    code_commit: str = "",
    common_version: str = "0.0.1",
    max_attempts: int = 2,
) -> tuple[manifest.Manifest, Spec | None]:
    ctx = manifest.begin("03_spec", claim_id, run_id, parent_run_id=parent_run_id)
    llm_raw_path = ctx.stage_dir / "llm_raw.jsonl"

    base_context = {
        "claim": claim.model_dump(mode="json"),
        "resolution": [r.model_dump(mode="json") for r in resolution],
    }
    common_kwargs = dict(code_commit=code_commit, common_version=common_version)

    last_error: str | None = None
    prompt_sha: str | None = None
    candidate: Spec | None = None

    for _attempt in range(max_attempts):
        context = base_context if last_error is None else {**base_context, "previous_attempt_error": last_error}
        try:
            candidate, prompt_sha = llm.complete_structured(client, SPEC_PROMPT, context, Spec, llm_raw_path)
        except llm.LLMParseError as e:
            last_error = str(e)
            candidate = None
            continue
        try:
            spec_module.validate_spec(candidate)
        except ValueError as e:
            last_error = f"validate_spec: {e}"
            candidate = None
            continue
        break  # candidate parsed and passed validate_spec

    llm_info = manifest.LLMCallInfo(model=llm.MODEL_ID, prompt_sha256=prompt_sha or "", temperature=llm.TEMPERATURE, seed=llm.SEED)

    if candidate is None:
        return (
            manifest.halt(ctx, halt_reason=f"spec validation failed after {max_attempts} attempts: {last_error}", **common_kwargs),
            None,
        )
    if not candidate.assumptions:
        return manifest.halt(ctx, halt_reason="assumptions block empty", **common_kwargs), None

    english = compiler.describe_signal(candidate)

    spec_yaml_path = ctx.stage_dir / "spec.yaml"
    spec_yaml_path.write_text(yaml.safe_dump(candidate.model_dump(mode="json"), sort_keys=False))

    spec_parquet_path = ctx.stage_dir / "spec.parquet"
    pd.DataFrame({"spec_json": [candidate.model_dump_json()]}).to_parquet(spec_parquet_path, index=False)

    assumptions_path = ctx.stage_dir / "assumptions.parquet"
    pd.DataFrame([a.model_dump(mode="json") for a in candidate.assumptions]).to_parquet(assumptions_path, index=False)

    english_path = ctx.stage_dir / "english.md"
    english_path.write_text(english, encoding="utf-8")

    outputs = [
        FileRef(path="spec.yaml", sha256=_sha256_file(spec_yaml_path)),
        FileRef(path="spec.parquet", sha256=_sha256_file(spec_parquet_path)),
        FileRef(path="assumptions.parquet", sha256=_sha256_file(assumptions_path)),
        FileRef(path="english.md", sha256=_sha256_file(english_path)),
    ]
    result_manifest = manifest.complete(ctx, outputs=outputs, llm=llm_info, **common_kwargs)
    return result_manifest, candidate
