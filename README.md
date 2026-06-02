# Formal matrix analysis for portfolio optimization

**Positive semidefinite covariance, quadratic risk, and Markowitz mean–variance optimization**

Survey quadratic forms and PSD matrices; formalize key matrix facts in **Lean 4**; compute the **efficient frontier** on real ETF return data in **Python** (Polars + httpx + CVXPY). Course project for applied linear algebra — not investment advice.

See [docs/PLANNING_BRIEF.md](docs/PLANNING_BRIEF.md) for the full spec.

## Repository layout

| Path | Purpose |
|------|---------|
| `python/` | Data fetch, covariance, frontier QP, plots |
| `lean/` | Lean 4 + Mathlib PSD lemmas |
| `docs/` | Planning brief, proposal |
| `scripts/run_all.py` | Regenerate data + figures |

## Python (uv)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
uv run python scripts/run_all.py
```

Outputs:

- `python/data/returns.parquet`, `returns.csv`, `metadata.json`
- `python/figures/*.png`
- `python/figures/comparison/*.png` (shrinkage frontiers, spectrum, KKT sensitivity, eigenportfolios)

**Application notebook (Part 3 narrative):** open `python/notebooks/application.ipynb` — loads or fetches data, tables, CVXPY frontier, inline figures. Quick stats: `uv run python scripts/summarize.py`.

**Data:** daily closes via `httpx` from the Yahoo Finance chart API. Citation in `python/data/metadata.json`.

## Lean

```bash
cd lean
lake update
lake build
```

See [lean/README.md](lean/README.md).

## Status

- **Topic approval:** done (Lean + Python mean–variance matrix analysis)
- **Data:** Yahoo Finance chart API via `httpx` only

## Scope (honest)

Single-period sample Markowitz; no transaction costs; not competing with industrial QP/MIP solvers.
