"""Spec parsing and validation (DESIGN.md §6).

`load_spec(path) -> Spec` parses `spec.yaml` into `schemas.Spec` and raises
on any unknown key.

`validate_spec(spec: Spec) -> None` enforces what the pydantic schema alone
cannot: the signal vocabulary is closed to the primitives in DESIGN §6.2;
at most 12 nodes; every numeric constant is declared in `params`, never
inlined in a node; node inputs are catalogue fields or prior node ids
forming a DAG with no cycles; every param referenced by a node is declared.
Raises, naming the specific violation, rather than silently dropping or
coercing.
"""
