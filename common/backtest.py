"""Deterministic backtester (DESIGN.md §5, §7 Stage 5).

`run_backtest(spec: Spec, panel: pd.DataFrame) -> BacktestResult` combines
`compile.compile_signal`, `portfolio.py`, and `costs.py` into positions,
returns, and trades tables for a single run.

Stage 5 calls this three times per claim: `headline` (spec as written, full
available period, `execution_lag >= 1`), `replication` (spec restricted to
the source's reported period and construction, `execution_lag` as the
source used — may be 0, and only permitted to be 0 on this run), and `oos`
(post-source-period only). A run halts if the universe is empty on any
rebalance date, or headline has fewer than 24 rebalance periods.
"""
