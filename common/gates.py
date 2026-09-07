"""Gate definitions and runner (DESIGN.md §8).

Thirteen deterministic gates (G1-G13), each a small pure function taking
exactly the data it needs and returning one `GateRow`. Thresholds are
module-level constants (`G*_FAIL`, `G*_WARN`) -- never read from a spec, an
environment variable, or an LLM output (CLAUDE.md). `GateRow.threshold` is
always the *fail* threshold; the warn tier is a second, narrower band each
gate checks internally and reports only via `status`.

G4 (deflated Sharpe) is the one gate DESIGN singles out: it reads
`trial_count` from the registry itself (`manifest.get_trial_count`), never
as a function argument, so it can't be handed a stale or spoofed value.

`run_gates(result, spec, claim, context) -> pd.DataFrame` evaluates all
thirteen gates in order and stops at the first `fail`, returning every row
computed so far (the failing one included). `warn` does not stop the run.

Gates that need more than the headline `result` (replication, oos,
capacity, sweeps, assumption alternatives, other approved claims' returns,
the factor panel) get that extra data from `GateContext`, which
`run_gates` requires in full -- there is no silent "skip this gate because
its input wasn't provided" (CLAUDE.md: "raise on any ambiguity"). By the
time Stage 6 actually calls this (Phase 6), Stage 5's three runs plus the
sweep/assumption backtests will have produced all of it; Phase 4's own
tests construct it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from common import manifest
from common.schemas import BacktestResult, Claim, GateRow, Spec
from common import metrics

TRADING_DAYS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649015329

# Fail / warn thresholds, DESIGN §8. Never spec-adjustable.
G1_FAIL, G1_WARN = 0.5, 0.7
G2_FAIL, G2_WARN = 0.5, 0.75
G3_FAIL, G3_WARN = 2.0, 3.0
G4_FAIL, G4_WARN = 0.90, 0.95
G5_FAIL, G5_WARN = 0.0, 0.5  # fail: raw oos sharpe < 0; warn: oos/headline ratio < 0.5
G6_MIN_POSITIVE_OF_3 = 2
G7_FAIL, G7_WARN = 3.0, 2.0
G8_FAIL, G8_WARN = 0.5, 0.75
G9_FAIL, G9_WARN = 1.5, 2.0
G10_FAIL, G10_WARN = 0.5, 0.3
G11_FAIL, G11_WARN = 0.5, 0.75
G12_FAIL, G12_WARN = 0.5, 0.75
G13_FAIL, G13_WARN = 0.7, 0.5


@dataclass
class GateContext:
    """Everything `run_gates` needs beyond the headline `result`. All
    fields are required -- there is no default that lets a gate be
    silently skipped."""

    replication_result: BacktestResult
    oos_result: BacktestResult
    capacity_result: BacktestResult
    factors: pd.DataFrame  # wide: date, market, size, value, momentum, quality, low_vol
    assumption_net_sharpes: list[float]
    sweep_net_sharpes: list[float]
    other_claim_returns: list[pd.Series] = field(default_factory=list)


def _net_returns(result: BacktestResult) -> pd.Series:
    return pd.Series(
        [r.net for r in result.returns], index=pd.to_datetime([r.date for r in result.returns]), name="net"
    )


def _gross_returns(result: BacktestResult) -> pd.Series:
    return pd.Series(
        [r.gross for r in result.returns], index=pd.to_datetime([r.date for r in result.returns]), name="gross"
    )


def _status_low_bad(value: float, fail: float, warn: float) -> str:
    """The gate fails when `value` is too LOW (fail threshold < warn threshold)."""
    if value < fail:
        return "fail"
    if value < warn:
        return "warn"
    return "pass"


def _status_high_bad(value: float, fail: float, warn: float) -> str:
    """The gate fails when `value` is too HIGH (fail threshold > warn threshold)."""
    if value > fail:
        return "fail"
    if value > warn:
        return "warn"
    return "pass"


def newey_west_tstat(returns: pd.Series, lags: int) -> float:
    """t-stat of the mean of `returns`, HAC (Newey-West) standard error at
    `lags` lags -- DESIGN §8 G3's "headline net t-stat, Newey-West, lag =
    holding period"."""
    if len(returns) < 2:
        return 0.0
    y = returns.to_numpy()
    x = np.ones_like(y)
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, lags)})
    return float(model.tvalues[0])


def _expected_max_sharpe(trial_count: int, n_obs: int) -> float:
    """Bailey & Lopez de Prado's SR0: the expected maximum per-period
    Sharpe across `trial_count` independent trials under the null,
    approximating each trial's Sharpe estimate as N(0, 1/n_obs)."""
    if trial_count <= 1 or n_obs < 2:
        return 0.0
    sr_std = math.sqrt(1.0 / n_obs)
    return sr_std * (
        (1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / trial_count)
        + EULER_MASCHERONI * norm.ppf(1 - 1.0 / (trial_count * math.e))
    )


def deflated_sharpe_ratio(returns: pd.Series, trial_count: int) -> float:
    """Bailey & Lopez de Prado's DSR: the probability the *true* Sharpe
    exceeds the expected maximum achievable by chance across `trial_count`
    independent trials, given the observed Sharpe, sample size, skew and
    kurtosis. In [0, 1]; DESIGN §8 G4 wants it near 1."""
    n = len(returns)
    if n < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    sr_hat = returns.mean() / std  # per-period, not annualised -- the formula wants this scale
    skew = float(returns.skew())
    kurt = float(returns.kurtosis()) + 3.0  # pandas reports *excess* kurtosis; formula wants raw

    sr0 = _expected_max_sharpe(max(trial_count, 1), n)
    denom = math.sqrt(max(1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2, 1e-12))
    z = (sr_hat - sr0) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def _factor_regression(returns: pd.Series, factors: pd.DataFrame) -> tuple[float, dict[str, float]]:
    aligned = factors.set_index("date").reindex(returns.index).dropna()
    y = returns.reindex(aligned.index)
    x = sm.add_constant(aligned)
    model = sm.OLS(y.to_numpy(), x.to_numpy()).fit()
    alpha_tstat = float(model.tvalues[0])
    loadings = dict(zip(aligned.columns, model.params[1:]))
    return alpha_tstat, loadings


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def gate_g1_turnover_cost(headline: BacktestResult) -> GateRow:
    gross_sharpe = metrics.sharpe(_gross_returns(headline))
    net_sharpe = metrics.sharpe(_net_returns(headline))
    value = (net_sharpe / gross_sharpe) if gross_sharpe != 0 else 0.0
    return GateRow(name="G1_turnover_cost", value=value, threshold=G1_FAIL, status=_status_low_bad(value, G1_FAIL, G1_WARN))


def gate_g2_replication(replication: BacktestResult, claim: Claim) -> GateRow:
    rep_sharpe = metrics.sharpe(_net_returns(replication))
    reported = claim.reported_sharpe or 0.0
    ratio = (rep_sharpe / abs(reported)) if reported != 0 else 0.0
    wrong_sign = reported != 0 and rep_sharpe != 0 and np.sign(rep_sharpe) != np.sign(reported)
    if wrong_sign or ratio < G2_FAIL:
        status = "fail"
    elif ratio < G2_WARN:
        status = "warn"
    else:
        status = "pass"
    return GateRow(name="G2_replication", value=ratio, threshold=G2_FAIL, status=status)


def gate_g3_headline_significance(headline: BacktestResult, holding_period: int) -> GateRow:
    value = newey_west_tstat(_net_returns(headline), lags=holding_period)
    return GateRow(
        name="G3_headline_significance", value=value, threshold=G3_FAIL, status=_status_low_bad(value, G3_FAIL, G3_WARN)
    )


def gate_g4_deflated_sharpe(headline: BacktestResult, claim_id: str) -> GateRow:
    trial_count = manifest.get_trial_count(claim_id)
    value = deflated_sharpe_ratio(_net_returns(headline), trial_count)
    return GateRow(
        name="G4_deflated_sharpe", value=value, threshold=G4_FAIL, status=_status_low_bad(value, G4_FAIL, G4_WARN)
    )


def gate_g5_out_of_sample(oos: BacktestResult, headline_sharpe: float) -> GateRow:
    oos_sharpe = metrics.sharpe(_net_returns(oos))
    if oos_sharpe < G5_FAIL:
        status = "fail"
    elif headline_sharpe != 0 and (oos_sharpe / headline_sharpe) < G5_WARN:
        status = "warn"
    else:
        status = "pass"
    return GateRow(name="G5_out_of_sample", value=oos_sharpe, threshold=G5_FAIL, status=status)


def gate_g6_subperiod_stability(headline: BacktestResult) -> GateRow:
    net = _net_returns(headline).sort_index()
    cut_points = np.array_split(np.arange(len(net)), 3)
    thirds = [net.iloc[idx] for idx in cut_points]
    positive_count = sum(1 for part in thirds if metrics.sharpe(part) > 0)
    status = "fail" if positive_count < G6_MIN_POSITIVE_OF_3 else "pass"
    return GateRow(
        name="G6_subperiod_stability", value=float(positive_count), threshold=float(G6_MIN_POSITIVE_OF_3), status=status
    )


def gate_g7_drawdown(headline: BacktestResult) -> GateRow:
    net = _net_returns(headline)
    vol = metrics.annualized_vol(net)
    value = (metrics.max_drawdown(net) / vol) if vol != 0 else 0.0
    return GateRow(name="G7_drawdown", value=value, threshold=G7_FAIL, status=_status_high_bad(value, G7_FAIL, G7_WARN))


def gate_g8_capacity(capacity: BacktestResult, headline_sharpe: float) -> GateRow:
    cap_sharpe = metrics.sharpe(_net_returns(capacity))
    ratio = (cap_sharpe / headline_sharpe) if headline_sharpe != 0 else 0.0
    return GateRow(name="G8_capacity", value=ratio, threshold=G8_FAIL, status=_status_low_bad(ratio, G8_FAIL, G8_WARN))


def gate_g9_factor_residual(headline: BacktestResult, factors: pd.DataFrame) -> GateRow:
    alpha_tstat, _ = _factor_regression(_net_returns(headline), factors)
    return GateRow(
        name="G9_factor_residual", value=alpha_tstat, threshold=G9_FAIL, status=_status_low_bad(alpha_tstat, G9_FAIL, G9_WARN)
    )


def gate_g10_factor_loading(headline: BacktestResult, factors: pd.DataFrame) -> GateRow:
    _, loadings = _factor_regression(_net_returns(headline), factors)
    max_loading = max((abs(v) for v in loadings.values()), default=0.0)
    return GateRow(
        name="G10_factor_loading", value=max_loading, threshold=G10_FAIL, status=_status_high_bad(max_loading, G10_FAIL, G10_WARN)
    )


def gate_g11_assumption_robustness(assumption_net_sharpes: list[float], headline_sharpe: float) -> GateRow:
    threshold_sharpe = 0.5 * headline_sharpe
    fraction = (
        sum(1 for s in assumption_net_sharpes if s > threshold_sharpe) / len(assumption_net_sharpes)
        if assumption_net_sharpes
        else 0.0
    )
    return GateRow(
        name="G11_assumption_robustness", value=fraction, threshold=G11_FAIL, status=_status_low_bad(fraction, G11_FAIL, G11_WARN)
    )


def gate_g12_parameter_robustness(sweep_net_sharpes: list[float], headline_sharpe: float) -> GateRow:
    threshold_sharpe = 0.5 * headline_sharpe
    fraction = (
        sum(1 for s in sweep_net_sharpes if s > threshold_sharpe) / len(sweep_net_sharpes)
        if sweep_net_sharpes
        else 0.0
    )
    return GateRow(
        name="G12_parameter_robustness", value=fraction, threshold=G12_FAIL, status=_status_low_bad(fraction, G12_FAIL, G12_WARN)
    )


def gate_g13_book_orthogonality(headline: BacktestResult, other_claim_returns: list[pd.Series]) -> GateRow:
    net = _net_returns(headline)
    corrs = []
    for other in other_claim_returns:
        if len(other) < 2:
            continue
        corr = net.reindex(other.index).corr(other)
        if pd.notna(corr):
            corrs.append(abs(corr))
    value = max(corrs) if corrs else 0.0
    return GateRow(name="G13_book_orthogonality", value=value, threshold=G13_FAIL, status=_status_high_bad(value, G13_FAIL, G13_WARN))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_gates(result: BacktestResult, spec: Spec, claim: Claim, context: GateContext) -> pd.DataFrame:
    """Evaluates all 13 gates in DESIGN §8 order, stopping at the first
    `fail` (that row included). Returns a DataFrame with columns
    name, value, threshold, status."""
    rows: list[GateRow] = []
    headline_sharpe = metrics.sharpe(_net_returns(result))

    gate_calls = [
        lambda: gate_g1_turnover_cost(result),
        lambda: gate_g2_replication(context.replication_result, claim),
        lambda: gate_g3_headline_significance(result, spec.portfolio.holding_period),
        lambda: gate_g4_deflated_sharpe(result, claim.claim_id),
        lambda: gate_g5_out_of_sample(context.oos_result, headline_sharpe),
        lambda: gate_g6_subperiod_stability(result),
        lambda: gate_g7_drawdown(result),
        lambda: gate_g8_capacity(context.capacity_result, headline_sharpe),
        lambda: gate_g9_factor_residual(result, context.factors),
        lambda: gate_g10_factor_loading(result, context.factors),
        lambda: gate_g11_assumption_robustness(context.assumption_net_sharpes, headline_sharpe),
        lambda: gate_g12_parameter_robustness(context.sweep_net_sharpes, headline_sharpe),
        lambda: gate_g13_book_orthogonality(result, context.other_claim_returns),
    ]

    for call in gate_calls:
        row = call()
        rows.append(row)
        if row.status == "fail":
            break

    return pd.DataFrame([r.model_dump() for r in rows])
