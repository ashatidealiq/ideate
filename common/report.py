"""Evidence package renderer (DESIGN.md §10).

Deterministic (no LLM call) render of `ic_pack.md` (all eleven DESIGN §10
sections) and `ic_pack.pdf` from a completed run's artifacts. Charts
(cumulative return, parameter sweep) are real matplotlib plots, not text
approximations -- rendered via `matplotlib.backends.backend_pdf.PdfPages`,
the same no-new-dependency approach `fixtures/build_papers.py` uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this module only ever renders to a PDF file, never a screen

import matplotlib.pyplot as plt  # noqa: E402 -- must follow matplotlib.use()
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from common import compile as compiler
from common import metrics
from common.schemas import BacktestResult, Claim, Manifest, Spec


@dataclass
class RunArtifacts:
    claim: Claim
    spec: Spec
    results: dict[str, BacktestResult]  # "headline", "replication", "oos"
    gates_df: pd.DataFrame
    sweeps_df: pd.DataFrame
    factor_decomposition: dict
    critique_text: str
    manifests: dict[str, Manifest]  # stage -> its manifest
    trial_count: int


def _net_returns(result: BacktestResult) -> pd.Series:
    return pd.Series([r.net for r in result.returns], index=pd.to_datetime([r.date for r in result.returns])).sort_index()


def _section_headline(a: RunArtifacts) -> str:
    citation = f"{', '.join(a.claim.source_authors)} ({a.claim.source_year}). {a.claim.source_title}."
    return (
        "## 1. Headline\n\n"
        f"**Claim ID:** {a.claim.claim_id}\n\n"
        f"**Source:** {citation}\n\n"
        f"**Edge statement:** {a.claim.edge_statement}\n\n"
        f"**Mechanism:** {a.claim.mechanism}\n\n"
        f"**Signal (mechanical English):**\n\n```\n{compiler.describe_signal(a.spec)}\n```\n"
    )


def _section_verdict(a: RunArtifacts) -> str:
    headline = _net_returns(a.results["headline"])
    oos = _net_returns(a.results["oos"])
    row = {
        "Net Sharpe": round(metrics.sharpe(headline), 3),
        "OOS Sharpe": round(metrics.sharpe(oos), 3),
        "Max drawdown": round(metrics.max_drawdown(headline), 3),
        "Turnover": round(metrics.turnover(pd.DataFrame([p.model_dump() for p in a.results["headline"].positions])), 4)
        if a.results["headline"].positions
        else 0.0,
        "Trial count": a.trial_count,
    }
    table = "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in row.items())
    return f"## 2. Verdict table\n\n{table}\n"


def _section_gates(a: RunArtifacts) -> str:
    return "## 3. Gate table\n\n" + a.gates_df.to_markdown(index=False) + "\n"


def _section_return_chart_note() -> str:
    return "## 4. Cumulative return chart\n\nSee `ic_pack.pdf` (headline, replication, oos on one axis).\n"


def _section_assumptions(a: RunArtifacts) -> str:
    rows = [
        f"| {x.id} | {x.choice} | {x.source_says} | {', '.join(str(v) for v in x.alternatives)} |"
        for x in a.spec.assumptions
    ]
    header = "| ID | Choice | Source says | Alternatives |\n|---|---|---|---|"
    return "## 5. Assumptions and alternatives\n\n" + "\n".join([header, *rows]) + "\n"


def _section_sweep_note() -> str:
    return "## 6. Parameter sweep heatmap\n\nSee `ic_pack.pdf`.\n"


def _section_factor_decomposition(a: RunArtifacts) -> str:
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in a.factor_decomposition.items())
    return "## 7. Factor decomposition\n\n| Factor | Value |\n|---|---|\n" + rows + "\n"


def _section_subperiods(a: RunArtifacts) -> str:
    net = _net_returns(a.results["headline"])
    thirds = [net.iloc[i] for i in [slice(0, len(net) // 3), slice(len(net) // 3, 2 * len(net) // 3), slice(2 * len(net) // 3, None)]]
    rows = "\n".join(f"| {i + 1} | {metrics.sharpe(part):.3f} |" for i, part in enumerate(thirds))
    return "## 8. Sub-period table\n\n| Sub-period | Net Sharpe |\n|---|---|\n" + rows + "\n"


def _section_critique(a: RunArtifacts) -> str:
    return f"## 9. Critique\n\n{a.critique_text}\n"


def _section_provenance(a: RunArtifacts) -> str:
    rows = []
    for stage, m in sorted(a.manifests.items()):
        for out in m.outputs:
            rows.append(f"| {stage} | {out.path} | {out.sha256} |")
    table = "| Stage | Output | sha256 |\n|---|---|---|\n" + "\n".join(rows)
    return f"## 10. Provenance\n\n**Trial count:** {a.trial_count}\n\n{table}\n"


def _section_decision() -> str:
    return (
        "## 11. Decision\n\n"
        "**Decision:** _____________\n\n"
        "**Conditions:** _____________\n\n"
        "**Reviewer:** _____________\n\n"
        "**Decided (UTC):** _____________\n"
    )


def render_ic_pack_markdown(a: RunArtifacts) -> str:
    sections = [
        _section_headline(a),
        _section_verdict(a),
        _section_gates(a),
        _section_return_chart_note(),
        _section_assumptions(a),
        _section_sweep_note(),
        _section_factor_decomposition(a),
        _section_subperiods(a),
        _section_critique(a),
        _section_provenance(a),
        _section_decision(),
    ]
    return f"# IC Pack -- {a.claim.claim_id}\n\n" + "\n".join(sections)


def _render_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.text(0.06, 0.95, title, fontsize=13, fontweight="bold", va="top")
    ax.text(0.06, 0.90, body, fontsize=8, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def render_ic_pack_pdf(a: RunArtifacts, path: Path) -> None:
    with PdfPages(path) as pdf:
        _render_text_page(pdf, f"IC Pack -- {a.claim.claim_id}", _section_headline(a) + "\n" + _section_verdict(a))
        _render_text_page(pdf, "Gate table", a.gates_df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(8.5, 5))
        for run, result in a.results.items():
            net = _net_returns(result)
            metrics.cumulative_returns(net).plot(ax=ax, label=run)
        ax.set_title("Cumulative net return")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        _render_text_page(pdf, "Assumptions and alternatives", _section_assumptions(a))

        if not a.sweeps_df.empty:
            fig, ax = plt.subplots(figsize=(8.5, 3))
            ax.bar(range(len(a.sweeps_df)), a.sweeps_df["net_sharpe"])
            ax.set_title("Parameter sweep: net Sharpe per sweep run")
            ax.set_xlabel("sweep index")
            pdf.savefig(fig)
            plt.close(fig)

        _render_text_page(pdf, "Factor decomposition", _section_factor_decomposition(a))
        _render_text_page(pdf, "Sub-period stability", _section_subperiods(a))
        _render_text_page(pdf, "Critique", a.critique_text)
        _render_text_page(pdf, "Provenance", _section_provenance(a))
        _render_text_page(pdf, "Decision", _section_decision())


def render_ic_pack(a: RunArtifacts, out_dir: Path) -> tuple[Path, Path]:
    md_path = out_dir / "ic_pack.md"
    md_path.write_text(render_ic_pack_markdown(a), encoding="utf-8")
    pdf_path = out_dir / "ic_pack.pdf"
    render_ic_pack_pdf(a, pdf_path)
    return md_path, pdf_path
