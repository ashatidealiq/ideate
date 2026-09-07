"""Spread and impact cost models (DESIGN.md §6.5).

Prices each trade using either a flat `spread_bps` (`fixed_bps`) or the
ADV-bucketed spread table in `catalogue/costs.yaml` (`adv_based`), plus an
optional square-root impact model (`impact_model: sqrt`, scaled by
`impact_coeff`) and an annualised borrow cost for short positions
(`borrow_bps_annual`). These models are fixed infrastructure: costs are
never spec- or agent-adjustable beyond selecting among the modes DESIGN
defines.
"""
