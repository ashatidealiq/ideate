"""Compiles a validated spec into an executable signal and its English
description (DESIGN.md §5, §6).

`PRIMITIVES` is the single source of truth for the closed signal vocabulary
(DESIGN §6.2): for each op, how many series inputs it takes (`n_inputs`,
`None` for `combine`'s variadic case) and what named params it requires,
tagged by kind:

- `"numeric"`: the node's `params[name]` must be a string naming an entry
  in the spec's top-level `params` block (DESIGN §6.7); `spec.validate_spec`
  is what rejects an inline float here (DESIGN §6.1's "every numeric
  constant in params").
- `"numeric_list"`: same, but a list of such names, one per input --
  `combine`'s `weights`.
- `"enum:a,b,c"`: a literal string from the given set -- `cs_neutralize`'s
  `group`, which is a structural choice, not a tunable constant, so it is
  not routed through the named-params/sweep machinery.

`spec.py` imports this table to validate specs against the same vocabulary
this module executes; there is exactly one place either of them is
authoritative, so the two can't drift apart.

`compile_signal(spec) -> Callable[[pd.DataFrame], pd.Series]` evaluates the
node DAG on a panel (the same long-form shape `pit.load_panel` returns) by
recursive, memoized evaluation of `spec.signal.output`, treating any input
string that isn't a node id as a catalogue field to read straight from the
panel.

`describe_signal(spec) -> str` mechanically (no LLM) renders the same DAG
as one line per node, e.g. `n1 = pct_change(cds_spread_5y, n=5)`, with
every param resolved to its actual numeric value.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from common.schemas import Signal, SignalNode, Spec

PRIMITIVES: dict[str, dict] = {
    # time-series (per asset)
    "lag": {"n_inputs": 1, "params": {"n": "numeric"}},
    "diff": {"n_inputs": 1, "params": {"n": "numeric"}},
    "pct_change": {"n_inputs": 1, "params": {"n": "numeric"}},
    "log": {"n_inputs": 1, "params": {}},
    "rolling_mean": {"n_inputs": 1, "params": {"n": "numeric"}},
    "rolling_std": {"n_inputs": 1, "params": {"n": "numeric"}},
    "rolling_sum": {"n_inputs": 1, "params": {"n": "numeric"}},
    "rolling_max": {"n_inputs": 1, "params": {"n": "numeric"}},
    "rolling_min": {"n_inputs": 1, "params": {"n": "numeric"}},
    "rolling_zscore": {"n_inputs": 1, "params": {"n": "numeric"}},
    "rolling_rank": {"n_inputs": 1, "params": {"n": "numeric"}},
    "ewm": {"n_inputs": 1, "params": {"halflife": "numeric"}},
    "rolling_beta": {"n_inputs": 2, "params": {"n": "numeric"}},  # [y, x]
    "rolling_corr": {"n_inputs": 2, "params": {"n": "numeric"}},  # [a, b]
    "rolling_vol": {"n_inputs": 1, "params": {"n": "numeric"}},
    # cross-sectional (per date)
    "cs_rank": {"n_inputs": 1, "params": {}},
    "cs_zscore": {"n_inputs": 1, "params": {}},
    "cs_demean": {"n_inputs": 1, "params": {}},
    "cs_winsorize": {"n_inputs": 1, "params": {"k": "numeric"}},
    "cs_neutralize": {"n_inputs": 1, "params": {"group": "enum:sector,country,region"}},
    "cs_scale": {"n_inputs": 1, "params": {}},
    # arithmetic
    "add": {"n_inputs": 2, "params": {}},
    "sub": {"n_inputs": 2, "params": {}},
    "mul": {"n_inputs": 2, "params": {}},
    "div": {"n_inputs": 2, "params": {}},
    "neg": {"n_inputs": 1, "params": {}},
    "abs": {"n_inputs": 1, "params": {}},
    "sign": {"n_inputs": 1, "params": {}},
    "clip": {"n_inputs": 1, "params": {"lo": "numeric", "hi": "numeric"}},
    "where": {"n_inputs": 3, "params": {}},  # [cond, a, b]
    # combination
    "combine": {"n_inputs": None, "params": {"weights": "numeric_list"}},
}

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------
# Per-asset / per-date grouping helpers. Every op operates on pd.Series
# indexed by a (date, asset_id) MultiIndex, matching what pit.load_panel's
# columns become once read into that shape.
# --------------------------------------------------------------------------


def _map_by_asset(obj, fn) -> pd.Series:
    """Applies fn(per_asset_series_or_frame) -> Series, one asset at a time,
    via an explicit loop + concat rather than groupby().apply(). apply()
    tries to be clever about reshaping its result when there is exactly one
    group, which silently produces a wide, mis-shaped frame instead of the
    expected long Series -- a real pandas footgun, not a hypothetical one."""
    reordered = obj.reorder_levels(["asset_id", "date"]).sort_index()
    parts = [fn(g) for _, g in reordered.groupby(level="asset_id")]
    result = pd.concat(parts)
    return result.reorder_levels(["date", "asset_id"]).sort_index()


def _map_by_date(obj, fn) -> pd.Series:
    """Same as `_map_by_asset`, grouped by date instead of asset_id."""
    reordered = obj.reorder_levels(["date", "asset_id"]).sort_index()
    parts = [fn(g) for _, g in reordered.groupby(level="date")]
    return pd.concat(parts).sort_index()


# --------------------------------------------------------------------------
# Time-series primitives (DESIGN §6.2)
# --------------------------------------------------------------------------


def op_lag(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.shift(n))


def op_diff(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.diff(n))


def op_pct_change(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.pct_change(n))


def op_log(x: pd.Series) -> pd.Series:
    return np.log(x)


def op_rolling_mean(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.rolling(n).mean())


def op_rolling_std(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.rolling(n).std())


def op_rolling_sum(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.rolling(n).sum())


def op_rolling_max(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.rolling(n).max())


def op_rolling_min(x: pd.Series, n: int) -> pd.Series:
    return _map_by_asset(x, lambda s: s.rolling(n).min())


def op_rolling_zscore(x: pd.Series, n: int) -> pd.Series:
    def _f(s: pd.Series) -> pd.Series:
        mean = s.rolling(n).mean()
        std = s.rolling(n).std()
        return (s - mean) / std

    return _map_by_asset(x, _f)


def op_rolling_rank(x: pd.Series, n: int) -> pd.Series:
    """Percentile rank (in (0, 1]) of the current value within its trailing
    n-observation window, per asset."""

    def _f(s: pd.Series) -> pd.Series:
        return s.rolling(n).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False)

    return _map_by_asset(x, _f)


def op_ewm(x: pd.Series, halflife: float) -> pd.Series:
    return _map_by_asset(x, lambda s: s.ewm(halflife=halflife).mean())


def op_rolling_beta(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    """Rolling n-window OLS slope of y on x, per asset: cov(y, x) / var(x)."""
    df = pd.DataFrame({"y": y, "x": x})

    def _f(g: pd.DataFrame) -> pd.Series:
        return g["y"].rolling(n).cov(g["x"]) / g["x"].rolling(n).var()

    return _map_by_asset(df, _f)


def op_rolling_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    df = pd.DataFrame({"a": a, "b": b})

    def _f(g: pd.DataFrame) -> pd.Series:
        return g["a"].rolling(n).corr(g["b"])

    return _map_by_asset(df, _f)


def op_rolling_vol(x: pd.Series, n: int) -> pd.Series:
    """Annualised rolling volatility: rolling_std(n) * sqrt(252 trading days).
    Distinct from rolling_std, which is the raw, unannualised window std."""
    return op_rolling_std(x, n) * float(np.sqrt(TRADING_DAYS_PER_YEAR))


# --------------------------------------------------------------------------
# Cross-sectional primitives (DESIGN §6.2)
# --------------------------------------------------------------------------


def op_cs_rank(x: pd.Series) -> pd.Series:
    """Uniform on (-0.5, 0.5): percentile rank across the cross-section,
    recentred at 0."""
    return _map_by_date(x, lambda s: s.rank(pct=True) - 0.5)


def op_cs_zscore(x: pd.Series) -> pd.Series:
    return _map_by_date(x, lambda s: (s - s.mean()) / s.std())


def op_cs_demean(x: pd.Series) -> pd.Series:
    return _map_by_date(x, lambda s: s - s.mean())


def op_cs_winsorize(x: pd.Series, k: float) -> pd.Series:
    def _f(s: pd.Series) -> pd.Series:
        mean, std = s.mean(), s.std()
        return s.clip(lower=mean - k * std, upper=mean + k * std)

    return _map_by_date(x, _f)


def op_cs_neutralize(x: pd.Series, group_values: pd.Series) -> pd.Series:
    """Demeans x within each (date, group) bucket, e.g. sector-neutral."""
    df = pd.DataFrame({"x": x, "g": group_values})

    def _f(g: pd.DataFrame) -> pd.Series:
        return g["x"] - g.groupby("g")["x"].transform("mean")

    return _map_by_date(df, _f)


def op_cs_scale(x: pd.Series) -> pd.Series:
    """Unit L1: rescales so sum(|x|) == 1 across the cross-section on each date."""

    def _f(s: pd.Series) -> pd.Series:
        total = s.abs().sum()
        return s / total if total else s * 0.0

    return _map_by_date(x, _f)


# --------------------------------------------------------------------------
# Arithmetic and combination primitives (DESIGN §6.2)
# --------------------------------------------------------------------------


def op_add(a: pd.Series, b: pd.Series) -> pd.Series:
    return a + b


def op_sub(a: pd.Series, b: pd.Series) -> pd.Series:
    return a - b


def op_mul(a: pd.Series, b: pd.Series) -> pd.Series:
    return a * b


def op_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b


def op_neg(a: pd.Series) -> pd.Series:
    return -a


def op_abs(a: pd.Series) -> pd.Series:
    return a.abs()


def op_sign(a: pd.Series) -> pd.Series:
    return np.sign(a)


def op_clip(a: pd.Series, lo: float, hi: float) -> pd.Series:
    return a.clip(lower=lo, upper=hi)


def op_where(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    mask = cond.fillna(0) != 0
    return pd.Series(np.where(mask, a, b), index=cond.index)


def op_combine(inputs: list[pd.Series], weights: list[float]) -> pd.Series:
    """Weighted sum. Inputs are expected to already be cross-sectionally
    standardised (DESIGN §6.2); that precondition is a spec-authoring
    convention, not something this function checks."""
    result = None
    for series, weight in zip(inputs, weights):
        term = series * weight
        result = term if result is None else result + term
    return result


# --------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------


def _field_series(panel: pd.DataFrame, field: str) -> pd.Series:
    if field not in panel.columns:
        raise KeyError(f"field {field!r} is not a column of the panel passed to compute_signal")
    return panel.set_index(["date", "asset_id"])[field].sort_index()


def _resolve_numeric(spec: Spec, ref: float | str) -> float:
    if isinstance(ref, str):
        return spec.params[ref].value
    return float(ref)


def _apply_op(spec: Spec, node: SignalNode, inputs: list[pd.Series], panel: pd.DataFrame) -> pd.Series:
    op = node.op

    if op == "lag":
        return op_lag(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "diff":
        return op_diff(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "pct_change":
        return op_pct_change(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "log":
        return op_log(inputs[0])
    if op == "rolling_mean":
        return op_rolling_mean(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_std":
        return op_rolling_std(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_sum":
        return op_rolling_sum(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_max":
        return op_rolling_max(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_min":
        return op_rolling_min(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_zscore":
        return op_rolling_zscore(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_rank":
        return op_rolling_rank(inputs[0], int(_resolve_numeric(spec, node.params["n"])))
    if op == "ewm":
        return op_ewm(inputs[0], _resolve_numeric(spec, node.params["halflife"]))
    if op == "rolling_beta":
        return op_rolling_beta(inputs[0], inputs[1], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_corr":
        return op_rolling_corr(inputs[0], inputs[1], int(_resolve_numeric(spec, node.params["n"])))
    if op == "rolling_vol":
        return op_rolling_vol(inputs[0], int(_resolve_numeric(spec, node.params["n"])))

    if op == "cs_rank":
        return op_cs_rank(inputs[0])
    if op == "cs_zscore":
        return op_cs_zscore(inputs[0])
    if op == "cs_demean":
        return op_cs_demean(inputs[0])
    if op == "cs_winsorize":
        return op_cs_winsorize(inputs[0], _resolve_numeric(spec, node.params["k"]))
    if op == "cs_neutralize":
        group_field = node.params["group"]
        return op_cs_neutralize(inputs[0], _field_series(panel, group_field))
    if op == "cs_scale":
        return op_cs_scale(inputs[0])

    if op == "add":
        return op_add(inputs[0], inputs[1])
    if op == "sub":
        return op_sub(inputs[0], inputs[1])
    if op == "mul":
        return op_mul(inputs[0], inputs[1])
    if op == "div":
        return op_div(inputs[0], inputs[1])
    if op == "neg":
        return op_neg(inputs[0])
    if op == "abs":
        return op_abs(inputs[0])
    if op == "sign":
        return op_sign(inputs[0])
    if op == "clip":
        return op_clip(inputs[0], _resolve_numeric(spec, node.params["lo"]), _resolve_numeric(spec, node.params["hi"]))
    if op == "where":
        return op_where(inputs[0], inputs[1], inputs[2])

    if op == "combine":
        weight_refs = node.params["weights"]
        weights = [_resolve_numeric(spec, w) for w in weight_refs]
        return op_combine(inputs, weights)

    raise ValueError(f"unknown op: {op!r}")


def compile_signal(spec: Spec) -> Callable[[pd.DataFrame], pd.Series]:
    """Builds `compute_signal(panel) -> pd.Series`, indexed (date, asset_id),
    by evaluating `spec.signal`'s node DAG. Assumes `spec` already passed
    `spec.validate_spec`."""
    node_by_id = {n.id: n for n in spec.signal.nodes}

    def compute_signal(panel: pd.DataFrame) -> pd.Series:
        cache: dict[str, pd.Series] = {}

        def eval_ref(ref: str) -> pd.Series:
            if ref not in node_by_id:
                return _field_series(panel, ref)
            if ref not in cache:
                node = node_by_id[ref]
                inputs = [eval_ref(i) for i in node.inputs]
                cache[ref] = _apply_op(spec, node, inputs, panel)
            return cache[ref]

        result = eval_ref(spec.signal.output) * spec.signal.sign
        return result.sort_index()

    return compute_signal


# --------------------------------------------------------------------------
# Mechanical English (DESIGN §5, §7 Stage 3)
# --------------------------------------------------------------------------


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _fmt_param(spec: Spec, node: SignalNode, name: str, kind: str) -> str:
    value = node.params[name]
    if kind == "numeric":
        return f"{name}={_fmt_num(_resolve_numeric(spec, value))}"
    if kind == "numeric_list":
        resolved = ", ".join(_fmt_num(_resolve_numeric(spec, v)) for v in value)
        return f"{name}=[{resolved}]"
    return f"{name}={value}"  # enum: shown as its literal string


def describe_signal(spec: Spec) -> str:
    """Deterministic, mechanical (no LLM) English rendering of the signal
    DAG: one line per node, inputs then params in the primitive's declared
    order, every numeric param resolved to its actual value."""
    lines = [f"Signal (sign={spec.signal.sign}):"]
    for node in spec.signal.nodes:
        primitive = PRIMITIVES[node.op]
        parts = list(node.inputs) + [
            _fmt_param(spec, node, name, kind) for name, kind in primitive["params"].items()
        ]
        lines.append(f"  {node.id} = {node.op}({', '.join(parts)})")
    lines.append(f"Output: {spec.signal.output}")
    return "\n".join(lines)
