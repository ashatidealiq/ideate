"""Evidence package renderer (DESIGN.md §10).

Deterministic (no LLM call) render of `ic_pack.md` / `ic_pack.pdf` from a
completed run's artifacts, in order: headline, verdict table, 13-row gate
table, cumulative return chart (headline/replication/oos, source period
shaded), assumptions and alternatives, parameter sweep heatmap, factor
decomposition, sub-period table, the verbatim Stage 6 critique, full
provenance (every manifest hash in the chain, `trial_count`, registry
history for the claim), and a blank decision block for the human reviewer.
"""
