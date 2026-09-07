"""The only module that calls an LLM (CLAUDE.md, DESIGN.md §11).

`MODEL_ID`, `TEMPERATURE`, `SEED` are constants here, never per-call
parameters. Every prompt is a file in `prompts/`; its sha256 is what the
calling stage records in its manifest under `llm.prompt_sha256`. Every
request and raw completion is appended to the calling stage's
`llm_raw.jsonl`.

Structured outputs are requested as JSON and parsed with the caller's
pydantic schema; a parse failure retries once with the error message
appended to the prompt, and a second failure raises `LLMParseError` --
callers catch this and call `manifest.halt()`, per CLAUDE.md's "Halting"
(a stage that cannot proceed halts; it does not crash).

Two clients implement `LLMClient`:

- `AnthropicClient`: the real, production path (DESIGN's actual intent).
  Lazily imports the `anthropic` package (an optional dependency -- see
  pyproject.toml's `llm` extra) and reads `ANTHROPIC_API_KEY` from the
  environment. Nothing in this repo's test suite uses it: no API key is
  configured in this environment, and tests must be deterministic and
  free to run, so they use `FakeLLMClient` instead.
- `FakeLLMClient`: a deterministic, canned-response client for tests and
  offline development -- no network, no key, no cost. Its `responder`
  callable receives the same (system, user) the real client would and
  returns a canned string.

Agents receive the prior-stage parquet rendered as JSON, the catalogue
vocabulary, and the spec vocabulary. Agents never receive gate thresholds
or gate results.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

MODEL_ID = "claude-fake-dev"
TEMPERATURE = 0.0
SEED = 0

T = TypeVar("T", bound=BaseModel)


class LLMParseError(Exception):
    """Raised when a structured completion fails schema validation twice
    (once, then once more after a retry with the error appended)."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass
class FakeLLMClient:
    """Deterministic canned-response client -- see module docstring."""

    responder: Callable[[str, str], str]

    def complete(self, system: str, user: str) -> str:
        return self.responder(system, user)


class AnthropicClient:
    """Thin wrapper around the Anthropic Messages API. Requires
    `pip install anthropic` (the `llm` extra) and `ANTHROPIC_API_KEY` in
    the environment. Not exercised by this repo's tests."""

    def __init__(self, model: str = MODEL_ID):
        import anthropic  # lazy: only needed if this client is actually used

        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


def _extract_json(text: str) -> str:
    """Strips a ```json ... ``` (or bare ```...```) fence if present;
    otherwise returns text unchanged. LLMs reliably wrap JSON in fences
    even when told not to."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def _prompt_sha256(prompt_path: Path) -> str:
    return hashlib.sha256(prompt_path.read_bytes()).hexdigest()


def complete_structured(
    client: LLMClient,
    prompt_path: Path,
    context: dict,
    response_model: type[T],
    llm_raw_path: Path,
) -> tuple[T, str]:
    """Renders `prompt_path` as the system prompt, `context` (as JSON) as
    the user message, calls `client`, and parses the response as
    `response_model`. Retries once, with the validation error appended, on
    a parse failure; raises `LLMParseError` if the retry also fails.
    Appends every request/response to `llm_raw_path`. Returns
    `(parsed, prompt_sha256)` -- the caller records the hash in its own
    manifest.
    """
    system = prompt_path.read_text()
    prompt_sha256 = _prompt_sha256(prompt_path)
    user = json.dumps(context, indent=2, default=str)

    def _call_and_log(user_content: str) -> str:
        raw = client.complete(system, user_content)
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_sha256,
            "request": user_content,
            "response": raw,
        }
        llm_raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(llm_raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return raw

    raw = _call_and_log(user)
    try:
        return response_model.model_validate_json(_extract_json(raw)), prompt_sha256
    except ValidationError as first_error:
        retry_user = (
            f"{user}\n\n"
            f"Your previous response failed schema validation:\n{first_error}\n\n"
            "Return ONLY corrected JSON matching the required schema -- no commentary, no markdown fences."
        )
        raw2 = _call_and_log(retry_user)
        try:
            return response_model.model_validate_json(_extract_json(raw2)), prompt_sha256
        except ValidationError as second_error:
            raise LLMParseError(
                f"structured output for {prompt_path.name} failed validation twice: {second_error}"
            ) from second_error
