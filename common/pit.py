"""Point-in-time panel loader (DESIGN.md §1.2, §5).

The only module that reads `panel/` parquet directly — no other module may
call `pd.read_parquet` on a panel file (CLAUDE.md).

`load_panel(fields, universe, start, end, as_of=None) -> pd.DataFrame`
returns a long-form frame (`date, asset_id, <field>...`). A row is never
returned if its `knowledge_date` exceeds `date + lag_days` for that field,
or exceeds `as_of` when `as_of` is given. Lookahead is prevented by
construction here, not by review elsewhere (DESIGN principle 2).
"""
