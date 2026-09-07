"""Portfolio construction, neutralisation, and weighting (DESIGN.md §6.4).

Turns a per-(date, asset_id) signal into position weights per the spec's
`portfolio` block: construction type (`long_short_quantile`,
`long_only_quantile`, `top_n_bottom_n`, `rank_weighted`, `threshold`),
weighting scheme, neutralisation against `sector` / `country` / `beta`,
`gross_leverage`, `net_exposure`, and `max_position`. Overlapping portfolios
(`holding_period > 1`) hold `holding_period` cohorts simultaneously.
`execution_lag` below 1 raises unless the caller's context marks the run as
a replication (DESIGN §6.4, CLAUDE.md).
"""
