You are given the claim just extracted from a source, plus which of its `required_fields` actually resolve against the data catalogue. Decide whether this claim is tradable in principle.

## Input

`{"claim": {...the Stage 1a extraction...}, "catalogue_fields": ["field_name", "..."]}`

## Output

Return ONLY:

```json
{"tradable": true, "tradability_reason": "one or two sentences"}
```

## Rules

- `tradable: false` if: `required_fields` cannot be reasonably approximated by `catalogue_fields`; the predicted relationship is definitionally unobservable in a historical backtest (e.g. it requires private or forward-looking information); or the claim concerns a market or instrument this pipeline has no data for at all.
- Do not judge statistical strength, sample size, effect size, or plausibility of the mechanism here -- that is decided later, deterministically, by the backtest and its gates. This step only answers: can a signal be built and tested at all, given the data that actually exists.
- `tradability_reason` must name the specific fact that drove the decision (a missing field, an unobservable quantity, a stated data requirement), not a general impression.
