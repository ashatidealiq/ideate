"""Field catalogue loader and resolver (DESIGN.md §4).

Loads and validates `catalogue/fields.yaml` and `catalogue/universes.yaml`
into `schemas.FieldDef` / `schemas.UniverseDef` rows. `pit.py` is the only
other module that calls this one; it uses `load_fields` to turn a field
name into a parquet path and PIT lag, and `load_universes` to turn a
universe name into the field whose per-date presence defines membership.

Stage 2's resolution of a claim's `required_fields` against this catalogue
is deterministic joins over the same `load_fields` output and belongs to
`spec.py`/the Stage 2 notebook, not to this module.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from common import config
from common.schemas import FieldDef, UniverseDef


def load_fields(path: Path | None = None) -> dict[str, FieldDef]:
    """Parses `catalogue/fields.yaml` into `{field_name: FieldDef}`.

    Raises on any unknown key (via `FieldDef`'s strict schema) and on a
    duplicate field name.
    """
    path = path or (config.CATALOGUE_SOURCE_DIR / "fields.yaml")
    raw = yaml.safe_load(path.read_text())
    fields: dict[str, FieldDef] = {}
    for entry in raw["fields"]:
        field = FieldDef.model_validate(entry)
        if field.name in fields:
            raise ValueError(f"duplicate field name {field.name!r} in {path}")
        fields[field.name] = field
    return fields


def get_field(name: str, fields: dict[str, FieldDef] | None = None) -> FieldDef:
    """Looks up one field by name, raising `KeyError` if it is not in the
    catalogue vocabulary."""
    fields = fields if fields is not None else load_fields()
    if name not in fields:
        raise KeyError(f"unknown catalogue field: {name!r}")
    return fields[name]


def load_universes(path: Path | None = None) -> dict[str, UniverseDef]:
    """Parses `catalogue/universes.yaml` into `{universe_name: UniverseDef}`.

    Raises on any unknown key and on a duplicate universe name.
    """
    path = path or (config.CATALOGUE_SOURCE_DIR / "universes.yaml")
    raw = yaml.safe_load(path.read_text())
    universes: dict[str, UniverseDef] = {}
    for entry in raw["universes"]:
        universe = UniverseDef.model_validate(entry)
        if universe.name in universes:
            raise ValueError(f"duplicate universe name {universe.name!r} in {path}")
        universes[universe.name] = universe
    return universes


def get_universe(name: str, universes: dict[str, UniverseDef] | None = None) -> UniverseDef:
    """Looks up one universe by name, raising `KeyError` if it is not in the
    catalogue vocabulary."""
    universes = universes if universes is not None else load_universes()
    if name not in universes:
        raise KeyError(f"unknown catalogue universe: {name!r}")
    return universes[name]
