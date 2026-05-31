"""Sample mean and covariance; eigenanalysis (NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class CovarianceResult:
    tickers: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    is_psd: bool
    min_eigenvalue: float
    condition_number: float


def returns_to_numpy(returns: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    tickers = [c for c in returns.columns if c != "date"]
    r = returns.select(tickers).to_numpy()
    return tickers, r


def estimate_mu_sigma(returns: pl.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sample mean vector and covariance matrix (columns = assets)."""
    tickers, r = returns_to_numpy(returns)
    mu = r.mean(axis=0)
    # rowvar=False -> columns are variables
    sigma = np.cov(r, rowvar=False, ddof=1)
    return tickers, mu, sigma


def analyze_sigma(sigma: np.ndarray, *, psd_tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, bool, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    min_eig = float(eigenvalues.min())
    is_psd = min_eig >= -psd_tol
    cond = float(eigenvalues.max() / max(eigenvalues.min(), psd_tol))
    return eigenvalues, eigenvectors, is_psd, min_eig, cond


def build_covariance(returns: pl.DataFrame) -> CovarianceResult:
    tickers, mu, sigma = estimate_mu_sigma(returns)
    eigenvalues, eigenvectors, is_psd, min_eig, cond = analyze_sigma(sigma)
    return CovarianceResult(
        tickers=tickers,
        mu=mu,
        sigma=sigma,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        is_psd=is_psd,
        min_eigenvalue=min_eig,
        condition_number=cond,
    )


def correlation_from_sigma(sigma: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(sigma))
    return sigma / np.outer(d, d)
