"""Gate definitions and runner (DESIGN.md §8).

Thirteen deterministic gates (G1 turnover/cost through G13 book
orthogonality) with fixed fail/warn thresholds. Thresholds live here as
module-level constants — they are never read from a spec, an environment
variable, or an LLM output (CLAUDE.md).

`run_gates(result: BacktestResult, spec: Spec, claim: Claim, context: GateContext) -> pd.DataFrame`
evaluates the gates in order and stops at the first `fail`, returning one
`GateRow` per gate evaluated (all rows up to and including the failure).
G4 (deflated Sharpe) reads `trial_count` from the registry, not from an
argument.
"""
