"""Field catalogue loader and resolver (DESIGN.md §4).

Loads and validates `catalogue/fields.yaml`, `catalogue/universes.yaml`, and
`catalogue/costs.yaml` into `schemas.FieldDef` / `schemas.UniverseDef` /
`schemas.RegionCosts` rows. `pit.py` uses `load_fields` to turn a field name
into a parquet path and PIT lag, and `load_universes` to turn a universe
name into the field whose per-date presence defines membership.
`common/costs.py` uses `load_costs` for its ADV-bucketed spread table.

Stage 2's resolution of a claim's `required_fields` against this catalogue
is deterministic joins over the same `load_fields` output and belongs to
`spec.py`/the Stage 2 notebook, not to this module.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from common import config
from common.schemas import FieldDef, RegionCosts, UniverseDef


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


def load_costs(path: Path | None = None) -> dict[str, RegionCosts]:
    """Parses `catalogue/costs.yaml` into `{region: RegionCosts}`.

    Raises on any unknown key, a duplicate region, and if any region's
    bucket list doesn't end with an unbounded (`max_adv: null`) bucket.
    """
    path = path or (config.CATALOGUE_SOURCE_DIR / "costs.yaml")
    raw = yaml.safe_load(path.read_text())
    regions: dict[str, RegionCosts] = {}
    for entry in raw["regions"]:
        region = RegionCosts.model_validate(entry)
        if region.region in regions:
            raise ValueError(f"duplicate region {region.region!r} in {path}")
        if not region.adv_buckets or region.adv_buckets[-1].max_adv is not None:
            raise ValueError(f"region {region.region!r} in {path} must end with a max_adv: null bucket")
        regions[region.region] = region
    return regions


def get_region_costs(region: str, regions: dict[str, RegionCosts] | None = None) -> RegionCosts:
    """Looks up one region's cost table, raising `KeyError` if it is not in
    the catalogue vocabulary."""
    regions = regions if regions is not None else load_costs()
    if region not in regions:
        raise KeyError(f"unknown catalogue region: {region!r}")
    return regions[region]
