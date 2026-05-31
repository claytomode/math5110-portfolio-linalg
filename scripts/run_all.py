#!/usr/bin/env python3
"""Regenerate cached returns and all figures."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from portfolio_linalg.config import load_config
from portfolio_linalg.covariance import build_covariance
from portfolio_linalg.fetch_data import fetch_and_cache
from portfolio_linalg.frontier import compute_frontier, min_variance_portfolio
from portfolio_linalg.plots import generate_all


def main() -> None:
    cfg = load_config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching / caching returns (httpx + Yahoo chart API)...")
    returns = fetch_and_cache(cfg)
    print(f"  {returns.height} rows x {len(cfg.tickers)} assets")

    print("Estimating mu, Sigma; eigenanalysis...")
    cov = build_covariance(returns)
    print(f"  PSD check: {cov.is_psd} (min eigenvalue = {cov.min_eigenvalue:.2e})")
    print(f"  condition number kappa(Sigma) ~ {cov.condition_number:.2e}")

    print("Solving efficient frontier (CVXPY)...")
    frontier = compute_frontier(cov, cfg)
    mvp = min_variance_portfolio(cov, cfg)
    print(f"  min-variance portfolio: mu={mvp['mu']:.4f}, sigma={mvp['sigma']:.4f}")

    print("Writing figures...")
    paths = generate_all(cov, frontier, cfg.figures_dir)
    for p in paths:
        print(f"  {p}")

    print("Done.")


if __name__ == "__main__":
    main()
