"""Shrinkage, spectral covariance, and frontier comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from portfolio_linalg.config import ComparisonConfig, ProjectConfig
from portfolio_linalg.covariance import (
    CovarianceResult,
    build_covariance,
    covariance_from,
    eigenvalue_floor_sigma,
    ledoit_wolf_sigma,
    rank_k_sigma,
    variance_explained,
)
from portfolio_linalg.frontier import compute_frontier, min_variance_portfolio
from portfolio_linalg.kkt import KKTVerification, verify_markowitz_kkt
from portfolio_linalg.spectral import eigenportfolio_table


@dataclass(frozen=True)
class ComparisonBundle:
    sample: CovarianceResult
    ledoit_wolf: CovarianceResult
    rank_k: CovarianceResult | None
    eigen_floor: CovarianceResult | None
    lw_shrinkage: float
    frontier_sample: pl.DataFrame
    frontier_lw: pl.DataFrame
    frontier_rank_k: pl.DataFrame | None
    frontier_floor: pl.DataFrame | None
    kkt_results: list[KKTVerification]
    kkt_tickers: list[str]
    cumulative_variance: np.ndarray
    eigenportfolios: pl.DataFrame


def _subset_returns(returns: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    cols = ["date", *tickers]
    missing = [t for t in tickers if t not in returns.columns]
    if missing:
        raise ValueError(f"tickers not in returns: {missing}")
    return returns.select(cols)


def build_comparison_bundle(
    returns: pl.DataFrame,
    cfg: ProjectConfig,
    comp: ComparisonConfig,
) -> ComparisonBundle:
    cov_sample = build_covariance(returns)
    sigma_lw, shrink = ledoit_wolf_sigma(returns)
    cov_lw = covariance_from(cov_sample.tickers, cov_sample.mu, sigma_lw, label="ledoit_wolf")

    cov_rank: CovarianceResult | None = None
    cov_floor: CovarianceResult | None = None
    if comp.rank_k is not None:
        sigma_k = rank_k_sigma(cov_sample.sigma, comp.rank_k)
        cov_rank = covariance_from(
            cov_sample.tickers, cov_sample.mu, sigma_k, label=f"rank_{comp.rank_k}"
        )
    if comp.eigenvalue_floor is not None:
        sigma_f = eigenvalue_floor_sigma(cov_sample.sigma, comp.eigenvalue_floor)
        cov_floor = covariance_from(
            cov_sample.tickers,
            cov_sample.mu,
            sigma_f,
            label=f"floor_{comp.eigenvalue_floor:.0e}",
        )

    frontier_sample = compute_frontier(cov_sample, cfg)
    frontier_lw = compute_frontier(cov_lw, cfg)
    frontier_rank_k = compute_frontier(cov_rank, cfg) if cov_rank else None
    frontier_floor = compute_frontier(cov_floor, cfg) if cov_floor else None

    sub = _subset_returns(returns, comp.kkt_tickers)
    _, r_sub = returns_to_numpy_sub(sub)
    mu_sub = r_sub.mean(axis=0)
    sigma_sub = np.cov(r_sub, rowvar=False, ddof=1)
    r_targets = [
        float(mu_sub.min()),
        float(np.median(mu_sub)),
        float(mu_sub.max() * 0.85),
    ]
    kkt_results = [
        verify_markowitz_kkt(
            mu_sub, sigma_sub, r, comp.kkt_tickers, solver=cfg.solver
        )
        for r in r_targets
    ]

    cumvar = np.cumsum(variance_explained(cov_sample.eigenvalues))
    eport = eigenportfolio_table(cov_sample, top_modes=3)

    return ComparisonBundle(
        sample=cov_sample,
        ledoit_wolf=cov_lw,
        rank_k=cov_rank,
        eigen_floor=cov_floor,
        lw_shrinkage=shrink,
        frontier_sample=frontier_sample,
        frontier_lw=frontier_lw,
        frontier_rank_k=frontier_rank_k,
        frontier_floor=frontier_floor,
        kkt_results=kkt_results,
        kkt_tickers=comp.kkt_tickers,
        cumulative_variance=cumvar,
        eigenportfolios=eport,
    )


def returns_to_numpy_sub(returns: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    tickers = [c for c in returns.columns if c != "date"]
    return tickers, returns.select(tickers).to_numpy()


def min_variance_comparison(bundle: ComparisonBundle, cfg: ProjectConfig) -> pl.DataFrame:
    rows: list[dict] = []
    for cov in (bundle.sample, bundle.ledoit_wolf, bundle.rank_k, bundle.eigen_floor):
        if cov is None:
            continue
        mvp = min_variance_portfolio(cov, cfg)
        rows.append(
            {
                "covariance": cov.label,
                "mu": mvp["mu"],
                "sigma": mvp["sigma"],
                "kappa": cov.condition_number,
                "min_eig": cov.min_eigenvalue,
            }
        )
    return pl.DataFrame(rows)
