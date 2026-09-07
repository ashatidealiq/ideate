# ideate

A pipeline that takes a written trading idea (a paper, a dataset, or just a human hunch) and turns it into a falsifiable backtest with a full, tamper-proof paper trail. It does not trade. It does not optimize for a good-looking result. Its only job is to produce an honest evidence package that a human investment committee can review and either approve or reject.

The short version of why this exists: most backtests are easy to fool, including by the person running them. Look-ahead bias, silent data snooping, quietly re-running until something works, cherry-picked periods. This project tries to make those mistakes structurally impossible instead of relying on the researcher to avoid them. Every stage of the pipeline is deterministic and hashed, data access is enforced to only ever see what would have actually been knowable on a given date, and the vocabulary a strategy can be built from is deliberately small and fixed so nothing sneaky can hide inside custom code.

## How it works

An idea goes in one end (a PDF of a research paper, or a plain-text description someone writes themselves) and, if it survives, an evidence package comes out the other end. In between there are seven stages:

1. **Claim** - an LLM reads the source and transcribes it into a structured claim: what's being predicted, by what, over what horizon, what data it needs. This is a transcription step, not a judgment step. If the claim can't be tested with data this pipeline actually has, it gets rejected here.
2. **Data resolution** - a plain deterministic check that the fields the claim needs actually exist and have enough coverage. No LLM involved.
3. **Spec** - an LLM proposes how to actually trade or test the idea, but only using a small, fixed set of building blocks (rolling averages, cross-sectional ranks, z-scores, that kind of thing). It can't write arbitrary code here. Every number it picks (a lookback window, a threshold) has to be named and declared up front so it can be stress-tested later.
4. **Code** - the spec gets turned into both a compiled function (what actually runs) and a human-readable version written by the LLM for audit. The two have to produce identical output on a test dataset or the stage fails outright.
5. **Backtest** - a deterministic backtester runs the strategy three times: the full history, a replication of the original paper's own period and setup, and an out-of-sample period after that. Real trading costs are applied. There's no way to "trade" using information from the future, this is enforced by the code, not by trusting the person who wrote the spec.
6. **Evaluation** - thirteen separate statistical checks have to pass before anything moves forward: is it actually profitable after costs, is it statistically significant, does it hold up out of sample, is it secretly just a repackaged version of a well-known market factor, does it fall apart if you nudge a parameter slightly. Any one hard failure stops the claim right there. An LLM writes a critique at the end and can still reject it even after all the checks pass.
7. **IC pack** - a rendered report (markdown and PDF) with everything a reviewer needs: the claim, the numbers, the gate results, the assumptions and how sensitive the result is to each one, the factor exposure, and a blank decision block for a human to actually sign off on.

Nothing about steps 2, 5, or 6 involves an AI model. They're just code. The only places an LLM touches anything are reading the source, proposing a strategy, writing audit code, and writing the final critique, and every one of those calls is logged and hashed so you can always go back and see exactly what was asked and what came back.

## The rules that don't bend

A few things are enforced everywhere, not just suggested:

- **No look-ahead, ever.** There's exactly one place in the codebase allowed to read historical market data (`common/pit.py`), and it will never hand back a data point before the date it would actually have been known.
- **Nothing gets overwritten.** Every run writes to a brand new folder. If you re-run a claim, you get a new run with a link back to the one before it, not a file that got quietly replaced.
- **Closed vocabulary.** A strategy can only be built out of a fixed list of primitives (things like rolling averages, z-scores, ranks). No custom logic, no arbitrary model code. This is what makes every strategy auditable and comparable.
- **Gate thresholds are fixed in code**, not something a spec or an LLM can adjust to make a result look better.
- **Every LLM call is pinned and logged.** Same model, same temperature, same seed, every prompt is a file with its hash recorded, every raw response is saved.

## What's built so far

This is being built in phases (see `BUILD.md`), each one fully tested before moving to the next:

- Phase 0-1: the scaffolding, schemas, and the point-in-time data loader
- Phase 2: the signal vocabulary and the spec compiler
- Phase 3: portfolio construction, trading costs, and the backtester
- Phase 4: the thirteen statistical gates
- Phase 5: the LLM layer and prompts
- Phase 6: wiring all seven stages together end to end, plus the report

Right now the whole thing runs against a synthetic fixture dataset (fake stocks, fake prices, a few planted "known good" and "known bad" signals so the gates can be tested against something with a predictable answer) rather than real market data. Real data is a later phase. And critically, no real AI model has been called yet anywhere in this pipeline. Everything LLM-shaped has been tested with a scripted fake client that returns canned responses, because there's no API key wired up yet. The plumbing is real and tested. Whether an actual model can read a real paper and propose something sensible is still an open question, and it's the next thing to find out.

## Running it

```bash
pip install -e ".[dev]"
pytest tests/
```

To run the whole pipeline against the fixture data:

```bash
python fixtures/build_fixture.py     # generates the synthetic dataset
python run_claim.py --source fixtures/papers/reversal.pdf
```

This won't do anything useful yet without a real LLM client wired in (defaults to calling the Anthropic API.) Running this second command that takes a paper all the way to a finished report.

## Layout

```
common/       the actual pipeline logic (data loading, signal compiler, backtester, gates, LLM calls)
notebooks/    one script per pipeline stage, each callable directly or meant to run headless
prompts/      the exact prompt text sent to the LLM at each stage
catalogue/    what data fields exist and where they come from
fixtures/     synthetic test data and papers, so the pipeline can be tested without real market data
tests/        a test for basically everything above
```

`DESIGN.md` has the full technical spec. `BUILD.md` has the phase-by-phase build plan. `CLAUDE.md` has the non-negotiable rules for anyone (human or AI) working on this codebase.
