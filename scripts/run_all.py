#!/usr/bin/env python3
"""Regenerate cached returns and all figures."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from portfolio_linalg.comparison import build_comparison_bundle
from portfolio_linalg.config import load_config
from portfolio_linalg.covariance import build_covariance
from portfolio_linalg.fetch_data import fetch_and_cache
from portfolio_linalg.frontier import compute_frontier, min_variance_portfolio
from portfolio_linalg.interpret import print_comparison_summary
from portfolio_linalg.kkt import format_kkt_report
from portfolio_linalg.plots import generate_all, generate_comparison_figures


def main() -> None:
    cfg = load_config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_comparison_dir.mkdir(parents=True, exist_ok=True)

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

    print("Writing baseline figures...")
    paths = generate_all(cov, frontier, cfg.figures_dir)
    for p in paths:
        print(f"  {p}")

    print("\nShrinkage, spectrum, KKT comparison...")
    bundle = build_comparison_bundle(returns, cfg, cfg.comparison)
    print_comparison_summary(bundle, cfg)
    for kkt in bundle.kkt_results:
        print()
        print(format_kkt_report(kkt, bundle.kkt_tickers))

    print("\nWriting comparison figures...")
    comp_paths = generate_comparison_figures(
        bundle, cfg.figures_comparison_dir, solver=cfg.solver
    )
    for p in comp_paths:
        print(f"  {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
