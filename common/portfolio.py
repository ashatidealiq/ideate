"""Portfolio construction, neutralisation, and weighting (DESIGN.md §6.4).

`construct_weights(signal, portfolio, vol=None, group_data=None) -> pd.Series`
turns one date's cross-sectional signal (indexed by asset_id) into target
weights for that date, in three stages:

1. **Selection + within-leg sizing**: which assets go long/short and how
   much of each leg's target they get, per `portfolio.construction`
   (`long_short_quantile`, `long_only_quantile`, `top_n_bottom_n`,
   `rank_weighted`, `threshold`) and `portfolio.weighting` (`equal`,
   `rank`, `signal`, `vol_scaled` -- the last needs `vol`). `rank_weighted`
   is self-contained (every eligible asset participates, sign and
   magnitude both from its centred percentile rank), so `weighting` does
   not apply to it.
2. **Leverage + cap**: rescales the long leg and short leg independently so
   the whole book sums to `gross_leverage` (in absolute value) and nets to
   `net_exposure`, then iteratively caps any position exceeding
   `max_position` and redistributes the capped-off amount across the
   still-uncapped names in that leg -- repeating until nothing is left to
   cap, or raising if `max_position` makes the target leverage
   unreachable outright.
3. **Neutralisation**: for each name in `portfolio.neutralize`, `sector`/
   `country` demean weights within each category (via `group_data`);
   `beta` instead projects out the portfolio's net beta exposure
   (`group_data["beta"]`), since beta is continuous, not categorical. This
   runs *last*, after leverage/cap, because it's the only order that makes
   its own guarantee exact: demeaning a group, or projecting out a beta
   exposure, always drives that group's sum (or the book's net beta) to
   exactly zero regardless of what came before -- whereas doing it first
   and then rescaling the long and short legs independently (needed for an
   exact `gross_leverage`/`net_exposure`) generally reintroduces exposure,
   since the two legs get different scale factors. The tradeoff is that
   when `neutralize` is non-empty, `gross_leverage`/`net_exposure`/
   `max_position` are honored only approximately, not exactly -- an
   inherent tension between simultaneous exact constraints that a real
   optimizer would resolve jointly; this module resolves it sequentially,
   favoring the property `neutralize` exists to guarantee.

Time -- which date this is, and `execution_lag`, `holding_period` -- is not
this module's concern; `backtest.py` calls this once per rebalance date (or
per cohort, for overlapping portfolios) and handles timing itself.
"""

from __future__ import annotations

import pandas as pd

from common.schemas import Portfolio


def _select_long_short_quantile(signal: pd.Series, portfolio: Portfolio) -> tuple[pd.Index, pd.Index]:
    k = max(1, round(portfolio.quantile * len(signal)))
    ranked = signal.sort_values(ascending=False)
    return ranked.index[:k], ranked.index[-k:]


def _select_long_only_quantile(signal: pd.Series, portfolio: Portfolio) -> tuple[pd.Index, pd.Index]:
    k = max(1, round(portfolio.quantile * len(signal)))
    ranked = signal.sort_values(ascending=False)
    return ranked.index[:k], pd.Index([])


def _select_top_n_bottom_n(signal: pd.Series, portfolio: Portfolio) -> tuple[pd.Index, pd.Index]:
    if portfolio.n is None:
        raise ValueError("construction=top_n_bottom_n requires portfolio.n")
    ranked = signal.sort_values(ascending=False)
    return ranked.index[: portfolio.n], ranked.index[-portfolio.n :]


def _select_threshold(signal: pd.Series, portfolio: Portfolio) -> tuple[pd.Index, pd.Index]:
    hi, lo = portfolio.threshold.hi, portfolio.threshold.lo
    longs = signal.index[signal > hi] if hi is not None else pd.Index([])
    shorts = signal.index[signal < lo] if lo is not None else pd.Index([])
    return longs, shorts


_SELECTORS = {
    "long_short_quantile": _select_long_short_quantile,
    "long_only_quantile": _select_long_only_quantile,
    "top_n_bottom_n": _select_top_n_bottom_n,
    "threshold": _select_threshold,
}


def _leg_raw_weights(signal: pd.Series, leg_index: pd.Index, weighting: str, sign: float, vol: pd.Series | None) -> pd.Series:
    """Unsigned-magnitude-then-signed weight for one leg; sign is +1 for the
    long leg, -1 for the short leg."""
    if len(leg_index) == 0:
        return pd.Series(dtype=float)
    sub = signal.loc[leg_index]

    if weighting == "equal":
        magnitude = pd.Series(1.0, index=leg_index)
    elif weighting == "rank":
        extremity = sub * sign  # larger = more extreme in this leg's own direction
        magnitude = extremity.rank(ascending=True).astype(float)
    elif weighting == "signal":
        magnitude = sub.abs()
    elif weighting == "vol_scaled":
        if vol is None:
            raise ValueError("weighting=vol_scaled requires a vol series")
        v = vol.reindex(leg_index)
        if v.isna().any():
            raise ValueError("weighting=vol_scaled: missing vol for some selected assets")
        magnitude = 1.0 / v
    else:
        raise ValueError(f"unknown weighting: {weighting!r}")

    return magnitude * sign


def _demean_by_category(weights: pd.Series, categories: pd.Series) -> pd.Series:
    cats = categories.reindex(weights.index)
    if cats.isna().any():
        raise ValueError("neutralize: missing category values for some selected assets")
    return weights - weights.groupby(cats).transform("mean")


def _neutralize_beta(weights: pd.Series, beta: pd.Series) -> pd.Series:
    b = beta.reindex(weights.index)
    if b.isna().any():
        raise ValueError("neutralize=beta: missing beta values for some selected assets")
    denom = float((b * b).sum())
    if denom == 0:
        return weights
    coef = float((weights * b).sum()) / denom
    return weights - coef * b


def _rescale_and_cap_leg(magnitudes: pd.Series, target_sum: float, max_position: float) -> pd.Series:
    """`magnitudes` are one leg's non-negative raw sizes. Scales them to sum
    to `target_sum`, then iteratively clips anything over `max_position`
    and redistributes the clipped amount across the still-free names."""
    if len(magnitudes) == 0 or target_sum <= 0:
        return magnitudes * 0.0
    if max_position * len(magnitudes) < target_sum - 1e-9:
        raise ValueError(
            f"cannot reach leg target {target_sum} with {len(magnitudes)} name(s) "
            f"capped at max_position={max_position}"
        )

    result = pd.Series(0.0, index=magnitudes.index)
    capped = pd.Series(False, index=magnitudes.index)
    remaining_target = target_sum

    for _ in range(len(magnitudes) + 1):
        free = magnitudes.index[~capped]
        if len(free) == 0:
            break
        free_magnitudes = magnitudes.loc[free]
        total = free_magnitudes.sum()
        scaled = (
            pd.Series(remaining_target / len(free), index=free)
            if total <= 0
            else free_magnitudes * (remaining_target / total)
        )
        over = scaled.index[scaled > max_position + 1e-9]
        if len(over) == 0:
            result.loc[free] = scaled
            break
        result.loc[over] = max_position
        capped.loc[over] = True
        remaining_target -= max_position * len(over)

    return result


def _apply_leverage_and_cap(weights: pd.Series, gross_leverage: float, net_exposure: float, max_position: float) -> pd.Series:
    target_long = (gross_leverage + net_exposure) / 2.0
    target_short = (gross_leverage - net_exposure) / 2.0
    if target_long < -1e-9 or target_short < -1e-9:
        raise ValueError("net_exposure cannot exceed gross_leverage in magnitude")

    long_leg = weights[weights > 0]
    short_leg = weights[weights < 0]

    result = pd.Series(0.0, index=weights.index)
    result.loc[long_leg.index] = _rescale_and_cap_leg(long_leg, target_long, max_position)
    result.loc[short_leg.index] = -_rescale_and_cap_leg(-short_leg, target_short, max_position)
    return result


def construct_weights(
    signal: pd.Series,
    portfolio: Portfolio,
    vol: pd.Series | None = None,
    group_data: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """Builds target weights for one rebalance date from that date's signal
    (indexed by asset_id, already restricted to the eligible universe)."""
    signal = signal.dropna()
    if len(signal) == 0:
        return pd.Series(dtype=float)

    if portfolio.construction == "rank_weighted":
        raw = signal.rank(pct=True) - 0.5  # in (-0.5, 0.5]; sign is built in
    elif portfolio.construction in _SELECTORS:
        longs, shorts = _SELECTORS[portfolio.construction](signal, portfolio)
        long_w = _leg_raw_weights(signal, longs, portfolio.weighting, 1.0, vol)
        short_w = _leg_raw_weights(signal, shorts, portfolio.weighting, -1.0, vol)
        raw = pd.concat([long_w, short_w])
        if raw.empty:
            return pd.Series(dtype=float)
    else:
        raise ValueError(f"unknown construction: {portfolio.construction!r}")

    weights = _apply_leverage_and_cap(raw, portfolio.gross_leverage, portfolio.net_exposure, portfolio.max_position)

    for group in portfolio.neutralize:
        if group_data is None or group not in group_data:
            raise ValueError(f"neutralize={group!r} requires group_data[{group!r}]")
        weights = (
            _neutralize_beta(weights, group_data[group])
            if group == "beta"
            else _demean_by_category(weights, group_data[group])
        )

    return weights
