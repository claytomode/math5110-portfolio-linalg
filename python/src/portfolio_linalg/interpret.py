"""Summaries for reports, CLI, and the application notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from portfolio_linalg.config import ProjectConfig
from portfolio_linalg.covariance import CovarianceResult
from portfolio_linalg.frontier import compute_frontier, min_variance_portfolio


@dataclass(frozen=True)
class FrontierSummary:
    mu_low: float
    sigma_low: float
    mu_high: float
    sigma_high: float
    best_mu: float
    best_sigma: float
    best_sharpe: float
    high_return_weights: pl.DataFrame


def asset_summary_table(cov: CovarianceResult) -> pl.DataFrame:
    vol = np.sqrt(np.diag(cov.sigma))
    sharpe = np.where(vol > 0, cov.mu / vol, np.nan)
    return (
        pl.DataFrame(
            {
                "ticker": cov.tickers,
                "mu_daily": cov.mu,
                "sigma_daily": vol,
                "mu_over_sigma": sharpe,
            }
        )
        .sort("mu_over_sigma", descending=True)
    )


def min_variance_weights_table(
    cov: CovarianceResult, cfg: ProjectConfig, *, min_weight: float = 0.001
) -> pl.DataFrame:
    mvp = min_variance_portfolio(cov, cfg)
    rows = [
        {"ticker": t, "weight": float(w)}
        for t, w in zip(cov.tickers, mvp["weights"], strict=True)
        if w >= min_weight
    ]
    return pl.DataFrame(rows).sort("weight", descending=True)


def frontier_summary_table(cov: CovarianceResult, frontier: pl.DataFrame) -> FrontierSummary:
    pts = frontier.select(["mu", "sigma", "r_min_target"]).unique().sort("mu")
    lo, hi = pts.row(0, named=True), pts.row(-1, named=True)
    best = pts.with_columns((pl.col("mu") / pl.col("sigma")).alias("sh")).sort(
        "sh", descending=True
    ).row(0, named=True)
    r_hi = hi["r_min_target"]
    w_hi = (
        frontier.filter(pl.col("r_min_target") == r_hi)
        .select(["ticker", "weight"])
        .filter(pl.col("weight") > 0.001)
        .sort("weight", descending=True)
    )
    return FrontierSummary(
        mu_low=float(lo["mu"]),
        sigma_low=float(lo["sigma"]),
        mu_high=float(hi["mu"]),
        sigma_high=float(hi["sigma"]),
        best_mu=float(best["mu"]),
        best_sigma=float(best["sigma"]),
        best_sharpe=float(best["sh"]),
        high_return_weights=w_hi,
    )


def _print_df(df: pl.DataFrame) -> None:
    """ASCII-safe table for Windows consoles."""
    for row in df.iter_rows(named=True):
        parts = "  ".join(
            f"{k}={row[k]!r}" if isinstance(row[k], str) else f"{k}={row[k]:.6g}"
            for k in df.columns
        )
        print(f"  {parts}")


def print_summary(cov: CovarianceResult, cfg: ProjectConfig, frontier: pl.DataFrame) -> None:
    """CLI-friendly text summary."""
    print("=== Single assets (daily) ===")
    _print_df(asset_summary_table(cov))
    print("\n=== Min-variance portfolio ===")
    mvp = min_variance_portfolio(cov, cfg)
    print(f"mu={mvp['mu']:.6f}  sigma={mvp['sigma']:.6f}")
    _print_df(min_variance_weights_table(cov, cfg))
    fs = frontier_summary_table(cov, frontier)
    print("\n=== Frontier (daily) ===")
    print(f"  low:  mu={fs.mu_low:.6f}  sigma={fs.sigma_low:.6f}")
    print(f"  high: mu={fs.mu_high:.6f}  sigma={fs.sigma_high:.6f}")
    print(f"  best mu/sigma (rf=0): {fs.best_sharpe:.2f} at mu={fs.best_mu:.6f}")
    print("\n  weights at high-return end:")
    _print_df(fs.high_return_weights)
