"""The only module that calls an LLM (CLAUDE.md, DESIGN.md §11).

Model id, temperature (0), and seed are module-level constants here — never
per-call parameters. Every prompt is a file in `prompts/`; its sha256 is
recorded in the calling stage's manifest under `llm.prompt_sha256`. Every
request and raw completion is appended to that stage's `llm_raw.jsonl`.

Structured outputs are requested as JSON and parsed with the caller's
pydantic schema; a parse failure retries once with the error message
appended to the prompt, and a second failure halts the stage (does not
raise past it). Agents receive the prior-stage parquet as JSON, the
catalogue vocabulary, and the spec vocabulary — never gate thresholds or
gate results.
"""
