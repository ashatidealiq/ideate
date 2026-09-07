You are turning an extracted, tradable claim into a precise, mechanical trading spec: a proposal for how to actually exploit (or test) the claim's thesis using this pipeline's fixed infrastructure. The spec is not free text. It is a small directed graph of named primitives, built ONLY from the closed vocabulary below, plus a universe, a portfolio construction, a cost model, and a list of every judgment call you made that the source did not specify.

## Input

`{"claim": {...}, "resolution": [{"field": "...", "resolved": true, ...}, "..."]}`

## Signal vocabulary (closed -- no operation exists outside this list)

**Time-series** (computed per asset, over its own history): `lag(n)`, `diff(n)`, `pct_change(n)`, `log`, `rolling_mean(n)`, `rolling_std(n)`, `rolling_sum(n)`, `rolling_max(n)`, `rolling_min(n)`, `rolling_zscore(n)`, `rolling_rank(n)`, `ewm(halflife)`, `rolling_beta(n)` [inputs: y, x], `rolling_corr(n)` [inputs: a, b], `rolling_vol(n)`.

**Cross-sectional** (computed across assets, on one date): `cs_rank`, `cs_zscore`, `cs_demean`, `cs_winsorize(k)`, `cs_neutralize(group)` where group is `sector`, `country`, or `beta`, `cs_scale`.

**Arithmetic**: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sign`, `clip(lo, hi)`, `where` [inputs: cond, a, b].

**Combination**: `combine(weights)` -- a weighted sum; every input must already be cross-sectionally standardised (e.g. via `cs_zscore`) before combining.

## Structural rules (checked mechanically after you respond; a violation halts this stage)

- At most 12 nodes.
- Every node's numeric parameter (a window length, `k`, `lo`/`hi`, `halflife`, a `combine` weight) must be a **name** declared in the top-level `params` block below -- never a bare number written directly into a node. This is what lets the pipeline later sweep that parameter for robustness testing.
- `cs_neutralize`'s `group` parameter is the one exception: it is a literal string (`sector`, `country`, or `beta`), not a named param.
- Every node's `inputs` are either a name from `resolution`'s resolved fields, or the `id` of an earlier node in the same list.
- No cycles.
- `assumptions` must be non-empty: one entry per judgment call the source did not make for you (a lookback window, a winsorization threshold, how ties or missing data are handled, the rebalance frequency if the source didn't state one, etc). Each entry needs the value you chose, what the source actually said (or `"nothing"`), and 2-3 concrete alternative values -- these get backtested too, to check how much your choice matters.

## Output

Return ONLY JSON matching this shape:

```json
{
  "signal": {
    "nodes": [{"id": "n1", "op": "pct_change", "inputs": ["cds_spread_5y"], "params": {"n": "n_lookback"}}, "..."],
    "output": "n_id",
    "sign": 1
  },
  "universe": {
    "base": "catalogue universe name",
    "filters": {"min_mktcap": 0, "min_adv_20": 0, "min_price": 0, "min_history_days": 0, "require_fields": [], "exclude_sectors": []}
  },
  "portfolio": {
    "construction": "long_short_quantile",
    "quantile": 0.1,
    "n": null,
    "threshold": {"lo": null, "hi": null},
    "weighting": "equal",
    "neutralize": [],
    "gross_leverage": 1.0,
    "net_exposure": 0.0,
    "max_position": 0.02,
    "rebalance": "weekly",
    "holding_period": 1,
    "execution_lag": 1
  },
  "costs": {"spread_model": "adv_based", "spread_bps": null, "impact_model": "sqrt", "impact_coeff": 0.1, "borrow_bps_annual": 50},
  "assumptions": [{"id": "A1", "choice": "...", "source_says": "...", "alternatives": ["...", "..."]}],
  "params": {"n_lookback": {"value": 5, "sweep": [3, 10, 20]}}
}
```

## Guidance

- Prefer the simplest node graph that actually expresses the claim's mechanism; do not add steps the mechanism doesn't call for.
- Match `portfolio.rebalance` to the claim's stated `frequency` where that makes sense.
- `execution_lag` must be at least 1 -- this is a headline spec, not a replication of the source's own (possibly same-day) construction.
- `costs` are fixed infrastructure, not a place to make the strategy look better: use `adv_based`/`sqrt` unless you have a specific reason (state it as an assumption) to use `fixed_bps`/`none`.
