"""Tests for common/llm.py (BUILD.md Phase 5 accept criteria)."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import BaseModel

from common import llm


class _Thing(BaseModel):
    value: int


@pytest.fixture
def prompt_file(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text("You are a test prompt.")
    return p


def test_complete_structured_parses_valid_json_on_first_try(tmp_path, prompt_file):
    client = llm.FakeLLMClient(responder=lambda system, user: json.dumps({"value": 42}))
    llm_raw_path = tmp_path / "llm_raw.jsonl"

    parsed, prompt_sha256 = llm.complete_structured(client, prompt_file, {"x": 1}, _Thing, llm_raw_path)

    assert parsed == _Thing(value=42)
    assert prompt_sha256 == hashlib.sha256(prompt_file.read_bytes()).hexdigest()


def test_complete_structured_strips_markdown_fence():
    client = llm.FakeLLMClient(responder=lambda s, u: '```json\n{"value": 7}\n```')
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        prompt_path = Path(d) / "p.md"
        prompt_path.write_text("x")
        parsed, _ = llm.complete_structured(client, prompt_path, {}, _Thing, Path(d) / "llm_raw.jsonl")
    assert parsed.value == 7


def test_every_call_appends_to_llm_raw_jsonl(tmp_path, prompt_file):
    client = llm.FakeLLMClient(responder=lambda system, user: json.dumps({"value": 1}))
    llm_raw_path = tmp_path / "llm_raw.jsonl"

    llm.complete_structured(client, prompt_file, {"x": 1}, _Thing, llm_raw_path)

    lines = llm_raw_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["prompt_sha256"] == hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    assert entry["model"] == llm.MODEL_ID
    assert entry["temperature"] == llm.TEMPERATURE
    assert entry["response"] == json.dumps({"value": 1})


def test_parse_failure_retries_once_then_succeeds(tmp_path, prompt_file):
    calls = []

    def responder(system, user):
        calls.append(user)
        if len(calls) == 1:
            return "not json at all"
        return json.dumps({"value": 99})

    client = llm.FakeLLMClient(responder=responder)
    llm_raw_path = tmp_path / "llm_raw.jsonl"

    parsed, _ = llm.complete_structured(client, prompt_file, {"x": 1}, _Thing, llm_raw_path)

    assert parsed.value == 99
    assert len(calls) == 2
    assert "failed schema validation" in calls[1]  # retry prompt includes the error
    assert len(llm_raw_path.read_text().strip().splitlines()) == 2  # both attempts logged


def test_parse_failure_twice_raises_llm_parse_error(tmp_path, prompt_file):
    client = llm.FakeLLMClient(responder=lambda system, user: "still not json")
    llm_raw_path = tmp_path / "llm_raw.jsonl"

    with pytest.raises(llm.LLMParseError):
        llm.complete_structured(client, prompt_file, {"x": 1}, _Thing, llm_raw_path)

    assert len(llm_raw_path.read_text().strip().splitlines()) == 2  # both failed attempts still logged
