"""Stage 6 -- Evaluation.

Deterministic gates first (DESIGN §8, `gates.run_gates`), stopping at the
first fail. For survivors: parameter sweeps (every value in each
`spec.params[name].sweep`, one backtest each) and, since assumption
alternatives are meant to be expressed the same way DESIGN §6.7 describes
tunable choices (a name in `spec.params`), this reuses the same sweep
backtests for G11 -- there is no separate substitution mechanism for a
non-numeric assumption alternative. A factor-decomposition-capable
`factors` panel and a capacity run (5x `TARGET_AUM`) round out the
`GateContext`. Then one LLM call (`prompts/s6_critique.md`) -> `critique.md`
with a mandatory verdict.

Halts if any hard gate fails, or the critique's verdict is "reject".
Every sweep/alternative backtest increments this claim's `trial_count` in
the registry (DESIGN §3.4, §9) -- G4 (deflated Sharpe) reads it back on
the *next* run of this claim, not this one.
"""

from __future__ import annotations

# %% tags=["parameters"]
claim_id = ""
run_id = ""
parent_run_id = None
code_commit = ""
common_version = "0.0.1"
factors_path = ""
other_claim_returns_paths: list[str] = []

# %%
import hashlib
from pathlib import Path

import pandas as pd

from common import backtest, config, gates, llm, manifest
from common.schemas import Claim, Critique, FileRef, Spec

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CRITIQUE_PROMPT = PROMPTS_DIR / "s6_critique.md"
CAPACITY_MULTIPLIER = 5.0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _net_sharpe(result) -> float:
    from common import metrics

    net = pd.Series([r.net for r in result.returns])
    return metrics.sharpe(net)


def _run_sweeps(spec: Spec, panel: pd.DataFrame) -> list[float]:
    from common import metrics

    sweep_sharpes: list[float] = []
    for name, param in spec.params.items():
        for value in param.sweep:
            swept = spec.model_copy(deep=True)
            swept.params[name].value = value
            result = backtest.run_backtest(swept, panel)
            sweep_sharpes.append(metrics.sharpe(pd.Series([r.net for r in result.returns])))
    return sweep_sharpes


def run_stage6(
    spec: Spec,
    claim: Claim,
    headline_result,
    replication_result,
    oos_result,
    headline_panel: pd.DataFrame,
    client: llm.LLMClient,
    claim_id: str,
    run_id: str,
    factors: pd.DataFrame,
    parent_run_id: str | None = None,
    code_commit: str = "",
    common_version: str = "0.0.1",
    other_claim_returns: list[pd.Series] | None = None,
) -> tuple[manifest.Manifest, str | None]:
    ctx = manifest.begin("06_evaluation", claim_id, run_id, parent_run_id=parent_run_id)
    common_kwargs = dict(code_commit=code_commit, common_version=common_version)

    # G4 (deflated Sharpe) reads trial_count from the registry itself
    # (common.manifest.get_trial_count), not an argument -- so it only
    # ever sees the cumulative total from runs that finished *before* this
    # one. This run's own sweep/capacity/replication/oos backtests are
    # this run's contribution, recorded in the registry only when this
    # stage itself completes, for the *next* run's G4 to see.
    trial_count_before_this_run = manifest.get_trial_count(claim_id)

    capacity_result = backtest.run_backtest(spec, headline_panel, aum=config.TARGET_AUM * CAPACITY_MULTIPLIER)
    sweep_sharpes = _run_sweeps(spec, headline_panel)
    # +1 (headline) +1 (replication) +1 (oos) +1 (capacity) for this run's own backtests.
    trial_count_after_this_run = trial_count_before_this_run + 4 + len(sweep_sharpes)

    context = gates.GateContext(
        replication_result=replication_result,
        oos_result=oos_result,
        capacity_result=capacity_result,
        factors=factors,
        assumption_net_sharpes=sweep_sharpes,
        sweep_net_sharpes=sweep_sharpes,
        other_claim_returns=other_claim_returns or [],
    )

    gates_df = gates.run_gates(headline_result, spec, claim, context)
    gates_path = ctx.stage_dir / "gates.parquet"
    gates_df.to_parquet(gates_path, index=False)

    sweeps_path = ctx.stage_dir / "sweeps.parquet"
    pd.DataFrame({"net_sharpe": sweep_sharpes}).to_parquet(sweeps_path, index=False)

    factor_tstat, factor_loadings = gates._factor_regression(gates._net_returns(headline_result), factors)
    factor_decomp_path = ctx.stage_dir / "factor_decomp.parquet"
    pd.DataFrame([{"alpha_tstat": factor_tstat, **factor_loadings}]).to_parquet(factor_decomp_path, index=False)

    outputs = [
        FileRef(path="gates.parquet", sha256=_sha256_file(gates_path)),
        FileRef(path="sweeps.parquet", sha256=_sha256_file(sweeps_path)),
        FileRef(path="factor_decomp.parquet", sha256=_sha256_file(factor_decomp_path)),
    ]

    if (gates_df["status"] == "fail").any():
        failed = gates_df[gates_df["status"] == "fail"].iloc[0]["name"]
        return (
            manifest.halt(ctx, halt_reason=f"gate {failed} failed", trial_count=trial_count_after_this_run, **common_kwargs),
            None,
        )

    critique_context = {
        "claim": claim.model_dump(mode="json"),
        "spec": spec.model_dump(mode="json"),
        "english": "",
        "metrics": {"headline_net_sharpe": _net_sharpe(headline_result)},
        "gates": gates_df.to_dict(orient="records"),
        "factor_decomposition": {"alpha_tstat": factor_tstat, **factor_loadings},
        "sweeps": {"net_sharpe": sweep_sharpes},
    }

    llm_raw_path = ctx.stage_dir / "llm_raw.jsonl"
    try:
        critique, prompt_sha = llm.complete_structured(client, CRITIQUE_PROMPT, critique_context, Critique, llm_raw_path)
    except llm.LLMParseError as e:
        return manifest.halt(ctx, halt_reason=f"critique: {e}", trial_count=trial_count_after_this_run, **common_kwargs), None

    critique_path = ctx.stage_dir / "critique.md"
    critique_path.write_text(f"# Critique\n\n{critique.critique}\n\n**Verdict:** {critique.verdict}\n", encoding="utf-8")
    outputs.append(FileRef(path="critique.md", sha256=_sha256_file(critique_path)))

    evaluation_path = ctx.stage_dir / "evaluation.md"
    evaluation_path.write_text(gates_df.to_markdown(index=False), encoding="utf-8")
    outputs.append(FileRef(path="evaluation.md", sha256=_sha256_file(evaluation_path)))

    llm_info = manifest.LLMCallInfo(model=llm.MODEL_ID, prompt_sha256=prompt_sha, temperature=llm.TEMPERATURE, seed=llm.SEED)

    if critique.verdict == "reject":
        return (
            manifest.halt(ctx, halt_reason="critique verdict: reject", llm=llm_info, trial_count=trial_count_after_this_run, **common_kwargs),
            None,
        )

    result_manifest = manifest.complete(ctx, outputs=outputs, llm=llm_info, trial_count=trial_count_after_this_run, **common_kwargs)
    return result_manifest, critique.critique
