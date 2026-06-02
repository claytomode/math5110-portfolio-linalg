"""Sample mean and covariance; eigenanalysis (NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf


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
    label: str = "sample"


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


def covariance_from(
    tickers: list[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    label: str = "custom",
) -> CovarianceResult:
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
        label=label,
    )


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
        label="sample",
    )


def ledoit_wolf_sigma(returns: pl.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf shrinkage toward scaled identity (sklearn)."""
    _, r = returns_to_numpy(returns)
    lw = LedoitWolf().fit(r)
    return lw.covariance_, float(lw.shrinkage_)


def rank_k_sigma(sigma: np.ndarray, k: int) -> np.ndarray:
    """PSD reconstruction keeping the k largest eigenvalues."""
    n = sigma.shape[0]
    k = min(max(k, 1), n)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    lam = np.maximum(eigenvalues, 0.0)
    lam_trunc = np.zeros_like(lam)
    lam_trunc[-k:] = lam[-k:]
    return eigenvectors @ np.diag(lam_trunc) @ eigenvectors.T


def eigenvalue_floor_sigma(sigma: np.ndarray, floor: float) -> np.ndarray:
    """Lift all eigenvalues to at least floor (spectral perturbation)."""
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    lam = np.maximum(eigenvalues, floor)
    return eigenvectors @ np.diag(lam) @ eigenvectors.T


def variance_explained(eigenvalues: np.ndarray) -> np.ndarray:
    """Fraction of total variance (trace) explained by each eigenmode, ascending order."""
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return np.zeros_like(eigenvalues)
    return eigenvalues / total


def correlation_from_sigma(sigma: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(sigma))
    return sigma / np.outer(d, d)
