"""Compiles a validated spec into an executable signal and its English
description (DESIGN.md §5, §6).

`compile_signal(spec: Spec) -> Callable[[pd.DataFrame], pd.Series]` builds
`compute_signal(panel) -> pd.Series` indexed `(date, asset_id)` by evaluating
the spec's node DAG with the DESIGN §6.2 primitives. This is what actually
runs at Stage 4/5 — the agent's own `signal.py` exists only for audit and
must round-trip against this compiled version on the fixture panel.

`describe_signal(spec: Spec) -> str` generates the plain-English rendering
of the node tree mechanically (no LLM call), used for the Stage 3
English/rationale consistency check and the IC pack.
"""
