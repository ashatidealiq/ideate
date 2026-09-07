"""Point-in-time panel loader (DESIGN.md §1.2, §5).

The only module that reads `panel/` parquet directly — no other module may
call `pd.read_parquet` on a panel file (CLAUDE.md).

`load_panel(fields, universe, start, end, as_of=None)` returns a long-form
frame (`date, asset_id, <field>...`). Each requested field is read from its
own catalogue-declared source under `PIPELINE_ROOT`, restricted to
`[start, end]`; when `as_of` is given, any row whose `knowledge_date`
exceeds it is dropped before the field ever reaches the output — that row's
value is simply absent for that snapshot, not carried forward or zeroed.
Lookahead is prevented by construction this way, not by review elsewhere
(DESIGN principle 2).

`universe` selects a `schemas.UniverseDef` from `catalogue.load_universes`:
an asset is a member of a date iff it has a row in the universe's
`member_if_present` field on that date. Delisted fixture/vendor names drop
out of membership for free, since their rows simply stop.
"""

from __future__ import annotations

import pandas as pd

from common import catalogue, config
from common.schemas import FieldDef


def _read_field(field: FieldDef, start_ts: pd.Timestamp, end_ts: pd.Timestamp, as_of_ts: pd.Timestamp | None) -> pd.DataFrame:
    path = config.PIPELINE_ROOT / field.source
    df = pd.read_parquet(path)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    if as_of_ts is not None:
        df = df[df["knowledge_date"] <= as_of_ts]
    return df


def load_panel(
    fields: list[str],
    universe: str,
    start: str,
    end: str,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Long-form: date, asset_id, <field>...

    Rows with knowledge_date > date + lag are never returned. If as_of is
    given, rows with knowledge_date > as_of are never returned.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else None

    catalogue_fields = catalogue.load_fields()
    unknown = sorted(set(fields) - set(catalogue_fields))
    if unknown:
        raise KeyError(f"unknown catalogue field(s): {unknown}")

    universe_def = catalogue.get_universe(universe)
    membership_field = universe_def.member_if_present
    if membership_field not in catalogue_fields:
        raise KeyError(
            f"universe {universe!r} requires field {membership_field!r}, "
            "which is not in the catalogue"
        )

    fields_to_read = list(dict.fromkeys([membership_field, *fields]))
    read = {
        name: _read_field(catalogue_fields[name], start_ts, end_ts, as_of_ts)
        for name in fields_to_read
    }

    result = read[membership_field][["date", "asset_id"]].drop_duplicates()
    for name in fields:
        field_frame = read[name][["date", "asset_id", "value"]].rename(columns={"value": name})
        result = result.merge(field_frame, on=["date", "asset_id"], how="left")

    return result.sort_values(["date", "asset_id"]).reset_index(drop=True)
