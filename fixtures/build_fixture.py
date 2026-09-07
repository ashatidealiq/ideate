"""Builds the synthetic fixture panel described in DESIGN.md §12.

200 assets across two regions (US, EU), business days 2010-01-04 through
2020-12-31, CDS coverage on 80 of the 200 names, three planted alpha
signals of documented target Sharpe, and one deliberate lookahead-trap
field whose `knowledge_date` lags `date` by 30 days (used by Phase 1's PIT
loader tests: the loader must never return it early). ~15 names delist
partway through the period; their rows simply stop after the delist date,
per DESIGN §6.3.

Every field is written as its own long-form parquet under `fixtures/panel/`
with columns `date, asset_id, knowledge_date, value` (DESIGN §4). Generation
is seeded (SEED) for reproducibility.

The three `planted_signal_N` fields are each linearly combined into the
next day's return with a hand-picked effect size, giving each a real
(injected, not estimated) net Sharpe once actually run through
`backtest.py`. Each `target_sharpe` below is the value tests/test_backtest.py
asserts (+/-0.1, per BUILD.md Phase 3) `run_backtest` recovers under this
exact canonical recipe -- changing either invalidates the other:

    signal: cs_zscore(planted_signal_N), sign=+1
    universe: all, no filters
    portfolio: long_short_quantile, quantile=0.1, weighting=equal,
               gross_leverage=1.0, net_exposure=0.0, max_position=0.05,
               rebalance=daily, holding_period=1, execution_lag=1
    costs: fixed_bps, spread_bps=1.0, impact_model=none, borrow_bps_annual=5
    period: 2010-01-04 .. 2020-12-31

These are measured values (from actually running that recipe against this
seeded fixture), not the closed-form approximation this module used before
`backtest.py` existed to check against -- weekly rebalancing in particular
decays `planted_signal_3` (phi=0.7, the least persistent) far more than the
others, so the recipe's exact settings matter, not just its shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ASSETS = 200
N_CDS_NAMES = 80
N_DELISTED = 15
START = "2010-01-04"
END = "2020-12-31"
OUT_DIR = Path(__file__).parent / "panel"

SECTORS = [
    "Financials", "Industrials", "Technology", "Energy",
    "Healthcare", "Consumer", "Materials", "Utilities",
]
REGIONS = ["US", "EU"]
COUNTRIES = {"US": ["US"], "EU": ["DE", "FR", "GB", "IT"]}

CDS_LAG_DAYS = 1
LOOKAHEAD_LAG_DAYS = 30

DRIFT = 0.0002          # daily log-return drift, ~5%/yr
NOISE_STD = 0.02         # idiosyncratic daily log-return noise std

# Planted signal -> (AR(1) persistence, next-day-return effect size, measured
# net Sharpe under the canonical recipe documented above).
PLANTED_SIGNALS = {
    "planted_signal_1": {"phi": 0.9, "alpha": 0.00017, "target_sharpe": 1.07},
    "planted_signal_2": {"phi": 0.8, "alpha": 0.00011, "target_sharpe": 0.60},
    "planted_signal_3": {"phi": 0.7, "alpha": 0.00006, "target_sharpe": 0.46},
}

# DESIGN §8 / BUILD.md Phase 4: three gate-testing fixtures, each composed
# from data already in this fixture rather than new alpha-injection --
# `factor_proxy_score` and `regime_flip` are static/structural fields;
# `common/gates.py`'s spec.py-vocabulary compositions build the actual
# planted "factor-proxy" and "unstable" signals from them:
#   good           = planted_signal_1 itself (already known-recoverable, Phase 3)
#   factor-proxy   = cs_zscore(factor_proxy_score)      -- IS a sector spread
#   unstable       = mul(planted_signal_1, regime_flip) -- real alpha, sign
#                     flipped for the last two-thirds of history
#
# factor_proxy_score: +1 for Financials, -1 for Technology, 0 otherwise
# (static per asset). FACTORS.PARQUET's "value" factor is defined as
# exactly the Financials-vs-Technology mean return spread, so a portfolio
# built by ranking on this score *is*, by construction, a bet on that
# factor -- guaranteed high loading, near-zero residual alpha (G9/G10).
FACTOR_PROXY_LONG_SECTOR = "Financials"
FACTOR_PROXY_SHORT_SECTOR = "Technology"

# Sector pairs defining each fixture factor's spread return (DESIGN §8:
# market, size, value, momentum, quality, low_vol). Arbitrary pairings --
# these are structural, not economically meaningful -- except "value",
# which must match FACTOR_PROXY_LONG_SECTOR/FACTOR_PROXY_SHORT_SECTOR.
FACTOR_SECTOR_SPREADS = {
    "value": ("Financials", "Technology"),
    "size": ("Technology", "Utilities"),
    "momentum": ("Energy", "Healthcare"),
    "quality": ("Industrials", "Materials"),
    "low_vol": ("Consumer", "Financials"),
}

REGIME_FLIP_SPLIT_FRACTION = 1.0 / 3.0  # positive in the first third of history, negative after


def _asset_ids(n: int) -> list[str]:
    return [f"A{i:04d}" for i in range(1, n + 1)]


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance stationary AR(1) series of length n."""
    eps_std = float(np.sqrt(1 - phi**2))
    eps = rng.normal(0.0, eps_std, size=n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, 1.0)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _assign_static_attributes(rng: np.random.Generator, asset_ids: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(asset_ids)
    region = rng.choice(REGIONS, size=n, p=[0.6, 0.4])
    country = np.array([rng.choice(COUNTRIES[r]) for r in region])
    sector = rng.choice(SECTORS, size=n)

    is_cds_name = np.zeros(n, dtype=bool)
    is_cds_name[:N_CDS_NAMES] = True
    rng.shuffle(is_cds_name)

    shares_out = rng.uniform(5e7, 5e9, size=n)

    delist_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    eligible_dates = dates[(dates >= "2013-01-01") & (dates <= "2020-06-30")]
    delist_idx = rng.choice(n, size=N_DELISTED, replace=False)
    delist_dates = rng.choice(eligible_dates.values, size=N_DELISTED, replace=False)
    delist_date[delist_idx] = delist_dates

    return pd.DataFrame(
        {
            "asset_id": asset_ids,
            "region": region,
            "country": country,
            "sector": sector,
            "is_cds_name": is_cds_name,
            "shares_out": shares_out,
            "delist_date": delist_date,
        }
    )


def _write_field(name: str, frame: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.parquet"
    frame = frame[["date", "asset_id", "knowledge_date", "value"]].reset_index(drop=True)
    frame.to_parquet(path, index=False)
    print(f"wrote {path} ({len(frame)} rows)")


def build() -> None:
    rng = np.random.default_rng(SEED)
    asset_ids = _asset_ids(N_ASSETS)
    all_dates = pd.bdate_range(start=START, end=END)
    static = _assign_static_attributes(rng, asset_ids, all_dates).set_index("asset_id")

    close_rows, volume_rows, mktcap_rows = [], [], []
    ret1d_rows, adv20_rows = [], []
    sector_rows, country_rows, region_rows = [], [], []
    cds_rows = []
    signal_rows = {name: [] for name in PLANTED_SIGNALS}
    lookahead_rows = []
    factor_proxy_rows = []
    regime_flip_rows = []

    regime_boundary = all_dates[int(len(all_dates) * REGIME_FLIP_SPLIT_FRACTION)]

    for asset_id in asset_ids:
        row = static.loc[asset_id]
        delist_date = row["delist_date"]
        dates = all_dates[all_dates <= delist_date] if pd.notna(delist_date) else all_dates
        n = len(dates)
        if n < 2:
            continue

        signals = {name: _ar1(n, cfg["phi"], rng) for name, cfg in PLANTED_SIGNALS.items()}
        noise = rng.normal(0.0, NOISE_STD, size=n)

        alpha_component = np.zeros(n)
        for name, cfg in PLANTED_SIGNALS.items():
            alpha_component += cfg["alpha"] * signals[name]

        log_ret = np.empty(n)
        log_ret[0] = rng.normal(DRIFT, NOISE_STD)
        log_ret[1:] = DRIFT + alpha_component[:-1] + noise[1:]
        close = 100.0 * np.exp(np.cumsum(log_ret))
        volume = rng.lognormal(mean=13.0, sigma=0.6, size=n)
        mktcap = close * row["shares_out"]

        close_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": close}))
        volume_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": volume}))
        mktcap_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": mktcap}))

        ret1d = pd.Series(close).pct_change()
        ret1d_rows.append(
            pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": ret1d}).dropna()
        )
        adv20 = pd.Series(volume).rolling(20).mean()
        adv20_rows.append(
            pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": adv20}).dropna()
        )

        sector_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": row["sector"]}))
        country_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": row["country"]}))
        region_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": row["region"]}))

        if row["is_cds_name"]:
            base_level = rng.uniform(80.0, 400.0)
            spread = base_level + 20.0 * _ar1(n, 0.95, rng)
            spread = np.clip(spread, 10.0, None)
            knowledge = dates + pd.Timedelta(days=CDS_LAG_DAYS)
            cds_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": knowledge, "value": spread}))

        for name in PLANTED_SIGNALS:
            signal_rows[name].append(
                pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": signals[name]})
            )

        trap_value = rng.normal(0.0, 1.0, size=n)
        trap_knowledge = dates + pd.Timedelta(days=LOOKAHEAD_LAG_DAYS)
        lookahead_rows.append(
            pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": trap_knowledge, "value": trap_value})
        )

        if row["sector"] == FACTOR_PROXY_LONG_SECTOR:
            proxy_score = 1.0
        elif row["sector"] == FACTOR_PROXY_SHORT_SECTOR:
            proxy_score = -1.0
        else:
            proxy_score = 0.0
        factor_proxy_rows.append(
            pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": proxy_score})
        )

        regime = np.where(dates < regime_boundary, 1.0, -1.0)
        regime_flip_rows.append(pd.DataFrame({"date": dates, "asset_id": asset_id, "knowledge_date": dates, "value": regime}))

    ret1d_all = pd.concat(ret1d_rows)
    sector_all = pd.concat(sector_rows)

    _write_field("close", pd.concat(close_rows))
    _write_field("volume", pd.concat(volume_rows))
    _write_field("mktcap", pd.concat(mktcap_rows))
    _write_field("ret_1d", ret1d_all)
    _write_field("adv_20", pd.concat(adv20_rows))
    _write_field("sector", sector_all)
    _write_field("country", pd.concat(country_rows))
    _write_field("region", pd.concat(region_rows))
    _write_field("cds_spread_5y", pd.concat(cds_rows))
    for name, frames in signal_rows.items():
        _write_field(name, pd.concat(frames))
    _write_field("lookahead_trap", pd.concat(lookahead_rows))
    _write_field("factor_proxy_score", pd.concat(factor_proxy_rows))
    _write_field("regime_flip", pd.concat(regime_flip_rows))

    _write_factors(ret1d_all, sector_all)

    print(f"assets: {N_ASSETS} (cds names: {N_CDS_NAMES}, delisted: {N_DELISTED})")
    print(f"date range: {all_dates[0].date()} .. {all_dates[-1].date()}")
    print(f"regime_flip boundary: {regime_boundary.date()}")
    print(f"planted sharpe targets (approximate, see module docstring): {PLANTED_SHARPE_TARGETS}")


def _write_factors(ret1d: pd.DataFrame, sector: pd.DataFrame) -> None:
    """Builds fixtures/factors.parquet: daily factor returns, wide-form
    (date, market, size, value, momentum, quality, low_vol), for
    common/gates.py's G9/G10 factor regression. Each style factor is a
    sector-group mean-return spread computed directly from this fixture's
    own ret_1d and sector fields (DESIGN §8's factor list, structurally
    simplified for a synthetic fixture -- not economically meaningful)."""
    merged = ret1d.merge(sector[["date", "asset_id", "value"]], on=["date", "asset_id"], suffixes=("_ret", "_sector"))
    sector_means = merged.groupby(["date", "value_sector"])["value_ret"].mean().unstack("value_sector")

    market = merged.groupby("date")["value_ret"].mean().rename("market")
    factors = {name: sector_means[long] - sector_means[short] for name, (long, short) in FACTOR_SECTOR_SPREADS.items()}
    out = pd.concat([market, pd.DataFrame(factors)], axis=1).reset_index().rename(columns={"date": "date"})

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR.parent / "factors.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


PLANTED_SHARPE_TARGETS = {name: cfg["target_sharpe"] for name, cfg in PLANTED_SIGNALS.items()}


if __name__ == "__main__":
    build()
