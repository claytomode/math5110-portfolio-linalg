"""Summaries for reports, CLI, and the application notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from portfolio_linalg.comparison import ComparisonBundle, min_variance_comparison
from portfolio_linalg.config import ProjectConfig
from portfolio_linalg.covariance import CovarianceResult
from portfolio_linalg.frontier import compute_frontier, min_variance_portfolio
from portfolio_linalg.spectral import eigenportfolio_table

# Simple calendar scaling for report figures (i.i.d. daily returns approximation).
TRADING_DAYS_PER_YEAR = 252


def annualize_mu(mu_daily: float) -> float:
    return mu_daily * TRADING_DAYS_PER_YEAR


def annualize_sigma(sigma_daily: float) -> float:
    return sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)


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


def asset_summary_annualized(cov: CovarianceResult) -> pl.DataFrame:
    """Per-asset mu and sigma scaled to ~252 trading days per year."""
    daily = asset_summary_table(cov)
    return (
        daily.with_columns(
            (pl.col("mu_daily") * TRADING_DAYS_PER_YEAR).alias("mu_annual"),
            (pl.col("sigma_daily") * np.sqrt(TRADING_DAYS_PER_YEAR)).alias("sigma_annual"),
        )
        .with_columns(
            (pl.col("mu_annual") / pl.col("sigma_annual")).alias("mu_over_sigma_annual")
        )
        .sort("mu_over_sigma_annual", descending=True)
    )


def frontier_points_annualized(frontier: pl.DataFrame) -> pl.DataFrame:
    """Unique frontier points with annualized mu and sigma."""
    scale = np.sqrt(TRADING_DAYS_PER_YEAR)
    return (
        frontier.select(["mu", "sigma", "r_min_target"])
        .unique()
        .sort("mu")
        .with_columns(
            (pl.col("mu") * TRADING_DAYS_PER_YEAR).alias("mu_annual"),
            (pl.col("sigma") * scale).alias("sigma_annual"),
        )
    )


def min_variance_annualized(cov: CovarianceResult, cfg: ProjectConfig) -> dict[str, float]:
    mvp = min_variance_portfolio(cov, cfg)
    mu_d = float(mvp["mu"])
    sig_d = float(mvp["sigma"])
    return {
        "mu_daily": mu_d,
        "sigma_daily": sig_d,
        "mu_annual": annualize_mu(mu_d),
        "sigma_annual": float(annualize_sigma(sig_d)),
    }


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


def comparison_summary_table(bundle: ComparisonBundle, cfg: ProjectConfig) -> pl.DataFrame:
    """Min-variance metrics across covariance estimators."""
    return min_variance_comparison(bundle, cfg)


def kkt_summary_table(bundle: ComparisonBundle) -> pl.DataFrame:
    rows = []
    for k in bundle.kkt_results:
        rows.append(
            {
                "r_min": k.r_min,
                "max_weight_diff": k.max_weight_diff,
                "stationarity_inf": k.stationarity_residual,
                "comp_slack_max": k.comp_slack_max,
                "return_slack": k.return_slack,
                "lambda_sum": k.lambda_sum,
                "nu_return": k.nu_return,
                "return_active": k.return_constraint_active,
                "zero_weights": ",".join(k.zero_weight_tickers) or "-",
            }
        )
    return pl.DataFrame(rows)


def kkt_stationarity_table(bundle: ComparisonBundle) -> pl.DataFrame:
    rows = []
    for k in bundle.kkt_results:
        for t, res in zip(bundle.kkt_tickers, k.stationarity, strict=True):
            rows.append({"r_min": k.r_min, "ticker": t, "stationarity": float(res)})
    return pl.DataFrame(rows)


def print_comparison_summary(bundle: ComparisonBundle, cfg: ProjectConfig) -> None:
    print("=== Covariance comparison (min-variance) ===")
    print(f"  Ledoit-Wolf shrinkage: {bundle.lw_shrinkage:.4f}")
    _print_df(comparison_summary_table(bundle, cfg))
    print("\n=== KKT verification (3-asset subset) ===")
    _print_df(kkt_summary_table(bundle))
    print("\n=== Top eigenportfolio loadings ===")
    _print_df(
        bundle.eigenportfolios.filter(pl.col("mode_rank") == 1).select(
            ["ticker", "loading", "eigenvalue", "variance_share"]
        )
    )


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
