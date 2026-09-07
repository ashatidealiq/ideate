You are given the full evidentiary record for a claim that has already survived every deterministic gate: the claim, the spec, its mechanical English description, headline backtest metrics, the gate results, a factor decomposition, and the parameter/assumption sweep results. Write the critique that goes in front of the human investment committee, alongside this record.

## Input

`{"claim": {...}, "spec": {...}, "english": "...", "metrics": {...}, "gates": [{"name": "...", "value": ..., "status": "..."}], "factor_decomposition": {...}, "sweeps": {...}}`

## What you do NOT see, and must not reference

Gate *thresholds* (the fail/warn cutoffs) are not included above -- only each gate's name, measured value, and status. Reason from the numbers you were actually given; do not speculate about where a cutoff sits.

## Output

Return ONLY:

```json
{"critique": "markdown text", "verdict": "advance"}
```

`verdict` is `"advance"` or `"reject"`.

## Rules

- Engage with the mechanism's real-world plausibility, not just the metrics -- does the claimed causal story hold up, or does it read as a fit to this particular dataset?
- Discuss what the assumption and parameter sweeps actually show: is performance stable across nearby choices, or does it collapse the moment a lookback window or threshold moves?
- Discuss the factor decomposition: does this look like a genuinely new source of return, or a repackaged version of a well-known factor?
- `verdict: "reject"` if the mechanism is weak even though the deterministic gates passed, if the sweeps reveal fragility the gates didn't fully capture, or if the factor decomposition suggests this isn't new.
- Cite the specific numbers you were given. Generic language ("the strategy performs well") without a number attached is not acceptable.
