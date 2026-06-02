"""Eigenstructure: explained variance and eigenportfolio loadings."""

from __future__ import annotations

import numpy as np
import polars as pl

from portfolio_linalg.covariance import CovarianceResult, variance_explained


def eigenportfolio_table(cov: CovarianceResult, *, top_modes: int = 3) -> pl.DataFrame:
    """Largest-variance eigenmodes with L1-normalized loadings (for interpretation)."""
    n = len(cov.tickers)
    top_modes = min(max(top_modes, 1), n)
    evals = cov.eigenvalues
    evecs = cov.eigenvectors
    shares = variance_explained(evals)
    rows: list[dict] = []
    for rank in range(1, top_modes + 1):
        idx = n - rank
        v = evecs[:, idx]
        denom = float(np.sum(np.abs(v)))
        w = v / denom if denom > 0 else v
        for t, wi in zip(cov.tickers, w, strict=True):
            rows.append(
                {
                    "mode_rank": rank,
                    "eigenvalue": float(evals[idx]),
                    "variance_share": float(shares[idx]),
                    "ticker": t,
                    "loading": float(wi),
                }
            )
    return pl.DataFrame(rows)
