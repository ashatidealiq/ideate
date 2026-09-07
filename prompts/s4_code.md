Write Python code implementing the spec below as a single function, for human audit. This code is NOT what actually runs: `common/compile.py` compiles the same spec independently, and your code must produce output identical to it on the fixture panel, or this stage halts. Your job is to make the computation legible to a human reader, faithfully -- not to reinterpret, simplify, or "improve" it.

## Input

`{"spec": {...}}`

## Required shape (exactly this signature, nothing else)

```python
def compute_signal(panel: pd.DataFrame) -> pd.Series:
    """Index (date, asset_id). Higher = more attractive long."""
```

## Rules

- May import ONLY `pandas`, `numpy`, and `common` -- nothing else. A different import fails this stage outright.
- Must implement the spec's node DAG exactly: same ops in the same order, same inputs, same parameter values (resolved from the spec's `params` block, not the param names), the same output node, the same sign applied at the end.
- No side effects, no I/O, no randomness, no network calls.
- Write it as ordinary, readable pandas -- a human on the investment committee should be able to follow it without reading `common/compile.py` first.

## Output

Return ONLY JSON:

```json
{"code": "def compute_signal(panel: pd.DataFrame) -> pd.Series:\n    ...\n"}
```
