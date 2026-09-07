"""Builds fixtures/papers/*.pdf: three short, entirely fictional synthetic
research papers (fictional authors, fictional findings), for Stage 1
extraction tests (BUILD.md Phase 5). Each states a clean, simple thesis
expressible in DESIGN §6.2's closed signal vocabulary, so Stage 3's spec
generation has something concrete to work with.

Rendered as single-page PDFs via matplotlib (already a dependency) rather
than adding a PDF-authoring library -- these are plain text pages, not
scanned documents, so pypdf's text extraction (used by notebooks/01_claim.py)
round-trips them cleanly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "papers"

PAPERS = [
    {
        "filename": "reversal.pdf",
        "title": "Short-Horizon Reversal in Daily Equity Returns",
        "authors": "Ada Lin, Marcus Ward (2018)",
        "body": """\
Abstract

We document that stocks with the lowest one-day return tend to
outperform over the following five trading days. We attribute this to
short-term liquidity provision: large one-day price moves are frequently
driven by transient order-flow imbalances rather than new information,
and partially reverse as liquidity providers step in and absorb the
imbalance.

Methodology

For each stock and date, we compute the one-day close-to-close return
and cross-sectionally standardise it (z-score) within the trading
universe on that date. Our signal is the negative of this z-score:
stocks with the most negative recent return receive the highest signal
value. We form equal-weighted decile portfolios, long the top decile
and short the bottom decile, rebalanced daily.

Formula

signal_t = -1 * zscore_cross_sectional( one_day_return_t )

Data

Daily close-to-close returns for a broad universe of liquid, exchange-
listed equities. We exclude the smallest, least liquid names.

Results

The long-short decile portfolio earns a Sharpe ratio of approximately
0.9 (gross of costs) over the sample period 2005-01-01 through
2015-12-31.

Caveats

Results are sensitive to the exclusion of illiquid names; we do not
report performance including transaction costs, and leave the choice
of exact winsorization/outlier treatment to future implementers.
""",
    },
    {
        "filename": "credit_momentum.pdf",
        "title": "Credit Leads Equity: CDS Spread Momentum and Cross-Asset Return Predictability",
        "authors": "Elena Petrova, Sam O'Rourke (2016)",
        "body": """\
Abstract

We show that widening five-year credit default swap (CDS) spreads over
the trailing five trading days predict negative equity returns for the
same issuer over the following two to three weeks. We argue this
reflects a difference in information diffusion speed: credit market
participants, who are more concentrated and more attentive to downside
risk, incorporate new information about issuer distress before the
broader equity market does.

Methodology

For each issuer with CDS coverage, we compute the five-day change in
the five-year CDS mid spread and cross-sectionally standardise it
across all CDS-covered names on that date. Our signal is the negative
of this standardised change: issuers whose CDS spreads have widened
most receive the most negative (least attractive) signal.

Formula

signal_t = -1 * zscore_cross_sectional( cds_spread_5y_t - cds_spread_5y_{t-5} )

Data

Five-year single-name CDS mid spreads, matched to the issuer's listed
equity. Restricted to names with continuous CDS coverage.

Results

A weekly-rebalanced long-short portfolio on this signal earns a Sharpe
ratio of approximately 1.3 (gross of costs) over 2008-01-01 through
2014-12-31, a period spanning the 2008 credit crisis.

Caveats

The sample period includes an unusually volatile credit environment;
we do not test an out-of-sample period after 2014. We do not specify a
precise treatment for stale or missing CDS quotes.
""",
    },
    {
        "filename": "volume_momentum.pdf",
        "title": "Volume-Confirmed Momentum: High-Turnover Winners Continue to Win",
        "authors": "Priya Nair (2020)",
        "body": """\
Abstract

We find that stocks exhibiting both strong trailing 20-day returns and
elevated trading volume continue to outperform over the following
month, more so than momentum alone would predict. We interpret high
volume as a proxy for the pace of information diffusion: price moves
accompanied by heavy trading are more likely to reflect genuine new
information being absorbed by a broad set of investors, rather than a
single large trader, and are therefore more likely to persist.

Methodology

We combine two cross-sectionally standardised components with equal
weight: trailing 20-day price momentum, and the current level of
20-day average daily trading volume. Both are z-scored across the
universe on each rebalance date before being combined.

Formula

signal_t = 0.5 * zscore_cross_sectional( pct_change_20d(close_t) )
         + 0.5 * zscore_cross_sectional( adv_20_t )

Data

Daily close prices (for momentum) and 20-day average daily volume (for
the volume component), for a broad universe of exchange-listed
equities.

Results

A monthly-rebalanced long-short decile portfolio on the combined signal
earns a Sharpe ratio of approximately 1.1 (gross of costs) over
2010-01-01 through 2019-12-31.

Caveats

We do not report results separately for the momentum-only or
volume-only components, and do not specify the exact combination weight
as anything other than an equal blend chosen for simplicity.
""",
    },
]


def _render_pdf(path: Path, title: str, authors: str, body: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")

    y = 0.96
    ax.text(0.5, y, title, ha="center", va="top", fontsize=14, fontweight="bold", wrap=True)
    y -= 0.06
    ax.text(0.5, y, authors, ha="center", va="top", fontsize=10, style="italic")
    y -= 0.05

    wrapped_lines = []
    for paragraph in body.split("\n\n"):
        wrapped_lines.extend(textwrap.wrap(paragraph.replace("\n", " "), width=95))
        wrapped_lines.append("")

    ax.text(0.06, y, "\n".join(wrapped_lines), ha="left", va="top", fontsize=9, family="monospace")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"wrote {path}")


def build() -> None:
    for paper in PAPERS:
        _render_pdf(OUT_DIR / paper["filename"], paper["title"], paper["authors"], paper["body"])


if __name__ == "__main__":
    build()
