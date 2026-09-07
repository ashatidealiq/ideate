"""Field catalogue loader and resolver (DESIGN.md §4).

Loads and validates `catalogue/fields.yaml` (into `schemas.FieldDef` rows),
`catalogue/universes.yaml`, and `catalogue/costs.yaml`. Provides the lookup
`pit.py` uses to turn a field name into its parquet source path and PIT lag,
and the resolution `catalogue.py` performs for Stage 2: joining a claim's
`required_fields` against the catalogue vocabulary and reporting, per field,
whether it resolves and with what coverage.
"""
