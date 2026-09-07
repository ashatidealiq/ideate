"""Spec parsing and validation (DESIGN.md §6).

`load_spec(path) -> Spec` parses `spec.yaml` into `schemas.Spec`, raising on
any unknown key (via `Spec`'s strict schema).

`validate_spec(spec) -> None` enforces what the pydantic schema alone
cannot, checking every node against `compile.PRIMITIVES` -- the same table
`compile.py` executes against, so validation and execution can't drift
apart:

- the op is in the closed vocabulary (DESIGN §6.2)
- at most `MAX_NODES` nodes (DESIGN §6.1)
- each input is either a prior node id or a real catalogue field
- each node has exactly the inputs and params its op requires, no more,
  no fewer
- every numeric param is a *name* in `spec.params`, never an inline
  constant (DESIGN §6.1), and that name is actually declared
- the node graph has no cycles, and `signal.output` names a real node

Raises immediately on the first violation found, naming it, rather than
silently dropping or coercing anything.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from common import catalogue
from common.compile import PRIMITIVES
from common.schemas import Spec

MAX_NODES = 12


def load_spec(path: Path) -> Spec:
    raw = yaml.safe_load(Path(path).read_text())
    return Spec.model_validate(raw)


def _validate_node_shape(node, catalogue_fields: dict, node_ids: set[str]) -> None:
    if node.op not in PRIMITIVES:
        raise ValueError(f"node {node.id!r}: unknown op {node.op!r}")
    primitive = PRIMITIVES[node.op]

    n_inputs = primitive["n_inputs"]
    if n_inputs is not None and len(node.inputs) != n_inputs:
        raise ValueError(
            f"node {node.id!r}: op {node.op!r} takes {n_inputs} input(s), got {len(node.inputs)}"
        )
    if n_inputs is None and len(node.inputs) < 2:
        raise ValueError(f"node {node.id!r}: op {node.op!r} takes at least 2 inputs, got {len(node.inputs)}")

    for ref in node.inputs:
        if ref in node_ids:
            continue
        if ref not in catalogue_fields:
            raise ValueError(
                f"node {node.id!r}: unknown field {ref!r} (not a prior node id or a catalogue field)"
            )


def _validate_node_params(node, spec: Spec) -> None:
    expected = PRIMITIVES[node.op]["params"]

    extra = set(node.params) - set(expected)
    if extra:
        raise ValueError(f"node {node.id!r}: unexpected param(s) {sorted(extra)} for op {node.op!r}")

    for name, kind in expected.items():
        if name not in node.params:
            raise ValueError(f"node {node.id!r}: missing required param {name!r} for op {node.op!r}")
        value = node.params[name]

        if kind == "numeric":
            if not isinstance(value, str):
                raise ValueError(
                    f"node {node.id!r}: param {name!r} must reference a name in spec.params, "
                    f"not an inline numeric constant ({value!r})"
                )
            if value not in spec.params:
                raise ValueError(f"node {node.id!r}: param {name!r} references undeclared param {value!r}")

        elif kind == "numeric_list":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(
                    f"node {node.id!r}: param {name!r} must be a list of names in spec.params, "
                    f"not inline numeric constants ({value!r})"
                )
            if len(value) != len(node.inputs):
                raise ValueError(f"node {node.id!r}: param {name!r} must have one weight per input")
            for v in value:
                if v not in spec.params:
                    raise ValueError(f"node {node.id!r}: param {name!r} references undeclared param {v!r}")

        elif kind.startswith("enum:"):
            allowed = kind.split(":", 1)[1].split(",")
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"node {node.id!r}: param {name!r} must be one of {allowed}, got {value!r}")


def _check_no_cycles(node_by_id: dict) -> None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node_id: WHITE for node_id in node_by_id}

    def visit(node_id: str, path: list[str]) -> None:
        color[node_id] = GRAY
        for ref in node_by_id[node_id].inputs:
            if ref not in node_by_id:
                continue
            if color[ref] == GRAY:
                raise ValueError(f"cycle in signal.nodes: {' -> '.join(path + [ref])}")
            if color[ref] == WHITE:
                visit(ref, path + [ref])
        color[node_id] = BLACK

    for node_id in node_by_id:
        if color[node_id] == WHITE:
            visit(node_id, [node_id])


def validate_spec(spec: Spec) -> None:
    nodes = spec.signal.nodes

    if len(nodes) > MAX_NODES:
        raise ValueError(f"signal has {len(nodes)} nodes, max is {MAX_NODES}")

    node_ids = [n.id for n in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate node id in signal.nodes")
    node_by_id = {n.id: n for n in nodes}

    catalogue_fields = catalogue.load_fields()

    for node in nodes:
        _validate_node_shape(node, catalogue_fields, set(node_ids))
        _validate_node_params(node, spec)

    _check_no_cycles(node_by_id)

    if spec.signal.output not in node_by_id:
        raise ValueError(f"signal.output {spec.signal.output!r} is not a defined node id")
