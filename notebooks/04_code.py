"""Stage 4 -- Code.

One LLM call (`prompts/s4_code.md`) writes `compute_signal(panel)` for
human audit. It is validated by round-trip: `compile_signal(spec)` (the
version that actually runs) and the agent's own code must produce
identical output on the fixture panel. A mismatch, a disallowed import, or
a runtime error all count as a failed attempt; the notebook re-prompts
with the specific error once before halting (DESIGN §5, §7 Stage 4).

Executing LLM-authored code is a real trust boundary -- this module does
not sandbox it. That is acceptable here only because this executed code
is never what actually runs in production (the compiled version is); this
run is solely the audit round-trip check, against a fixture panel, in a
controlled environment.
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
import ast
import hashlib
from pathlib import Path

import pandas as pd

from common import compile as compiler
from common import llm, manifest
from common.schemas import FileRef, SignalCodeOutput, Spec

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CODE_PROMPT = PROMPTS_DIR / "s4_code.md"
ALLOWED_IMPORTS = {"pandas", "numpy", "common"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _disallowed_imports(code: str) -> list[str]:
    tree = ast.parse(code)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return sorted(found - ALLOWED_IMPORTS)


def _exec_compute_signal(code: str, panel: pd.DataFrame) -> pd.Series:
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 -- audit round-trip only, see module docstring
    if "compute_signal" not in namespace:
        raise ValueError("code does not define compute_signal")
    return namespace["compute_signal"](panel)


def _series_match(expected: pd.Series, actual: pd.Series) -> bool:
    try:
        pd.testing.assert_series_equal(
            expected.sort_index(), actual.sort_index(), check_exact=False, rtol=1e-9, check_names=False
        )
        return True
    except AssertionError:
        return False


def run_stage4(
    spec: Spec,
    client: llm.LLMClient,
    claim_id: str,
    run_id: str,
    fixture_panel: pd.DataFrame,
    parent_run_id: str | None = None,
    code_commit: str = "",
    common_version: str = "0.0.1",
    max_attempts: int = 2,
) -> tuple[manifest.Manifest, str | None]:
    ctx = manifest.begin("04_code", claim_id, run_id, parent_run_id=parent_run_id)
    llm_raw_path = ctx.stage_dir / "llm_raw.jsonl"
    context = {"spec": spec.model_dump(mode="json")}
    common_kwargs = dict(code_commit=code_commit, common_version=common_version)

    expected = compiler.compile_signal(spec)(fixture_panel)

    last_error: str | None = None
    code_text: str | None = None
    prompt_sha: str | None = None

    for _attempt in range(max_attempts):
        this_context = context if last_error is None else {**context, "previous_attempt_error": last_error}
        try:
            output, prompt_sha = llm.complete_structured(client, CODE_PROMPT, this_context, SignalCodeOutput, llm_raw_path)
        except llm.LLMParseError as e:
            last_error = str(e)
            continue

        bad_imports = _disallowed_imports(output.code)
        if bad_imports:
            last_error = f"disallowed import(s): {bad_imports}"
            continue

        try:
            actual = _exec_compute_signal(output.code, fixture_panel)
        except Exception as e:  # noqa: BLE001 -- any failure here is a failed attempt, not a crash
            last_error = f"agent code raised: {e!r}"
            continue

        if not _series_match(expected, actual):
            last_error = "round-trip mismatch: agent code output differs from compile_signal(spec)"
            continue

        code_text = output.code
        break

    llm_info = manifest.LLMCallInfo(model=llm.MODEL_ID, prompt_sha256=prompt_sha or "", temperature=llm.TEMPERATURE, seed=llm.SEED)

    if code_text is None:
        return (
            manifest.halt(ctx, halt_reason=f"round-trip check failed after {max_attempts} attempts: {last_error}", **common_kwargs),
            None,
        )

    signal_py_path = ctx.stage_dir / "signal.py"
    signal_py_path.write_text(code_text, encoding="utf-8")
    code_sha256 = hashlib.sha256(code_text.encode("utf-8")).hexdigest()
    sha_path = ctx.stage_dir / "signal.sha256"
    sha_path.write_text(code_sha256)

    outputs = [
        FileRef(path="signal.py", sha256=_sha256_file(signal_py_path)),
        FileRef(path="signal.sha256", sha256=_sha256_file(sha_path)),
    ]
    result_manifest = manifest.complete(ctx, outputs=outputs, llm=llm_info, **common_kwargs)
    return result_manifest, code_text
