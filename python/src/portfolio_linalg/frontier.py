"""Mean-variance efficient frontier via CVXPY (convex QP)."""

from __future__ import annotations

import numpy as np
import polars as pl
import cvxpy as cp

from portfolio_linalg.config import ProjectConfig
from portfolio_linalg.covariance import CovarianceResult


def _solve_markowitz(
    mu: np.ndarray,
    sigma: np.ndarray,
    r_min: float,
    *,
    solver: str,
) -> tuple[np.ndarray, float, float]:
    n = len(mu)
    x = cp.Variable(n)
    risk = cp.quad_form(x, sigma)
    constraints = [
        cp.sum(x) == 1,
        mu @ x >= r_min,
        x >= 0,
    ]
    prob = cp.Problem(cp.Minimize(0.5 * risk), constraints)
    prob.solve(solver=solver, verbose=False)
    if x.value is None:
        raise RuntimeError(f"QP infeasible or failed at r_min={r_min:.6f}")
    weights = np.asarray(x.value).flatten()
    port_mu = float(mu @ weights)
    port_var = float(weights @ sigma @ weights)
    port_sigma = float(np.sqrt(max(port_var, 0.0)))
    return weights, port_mu, port_sigma


def r_min_grid(mu: np.ndarray, n_points: int) -> np.ndarray:
    """Target returns from min asset return to max asset return."""
    lo = float(mu.min())
    hi = float(mu.max())
    return np.linspace(lo, hi, n_points)


def compute_frontier(
    cov: CovarianceResult,
    cfg: ProjectConfig,
) -> pl.DataFrame:
    targets = r_min_grid(cov.mu, cfg.r_min_points)
    rows: list[dict] = []
    for r_min in targets:
        try:
            w, port_mu, port_sigma = _solve_markowitz(
                cov.mu, cov.sigma, float(r_min), solver=cfg.solver
            )
            rows.append(
                {
                    "r_min_target": r_min,
                    "mu": port_mu,
                    "sigma": port_sigma,
                    "weights": w.tolist(),
                }
            )
        except RuntimeError:
            continue
    if not rows:
        raise RuntimeError("No feasible frontier points; check data or constraints.")
    long = []
    for row in rows:
        for i, t in enumerate(cov.tickers):
            long.append(
                {
                    "r_min_target": row["r_min_target"],
                    "mu": row["mu"],
                    "sigma": row["sigma"],
                    "ticker": t,
                    "weight": row["weights"][i],
                }
            )
    return pl.DataFrame(long)


def min_variance_portfolio(
    cov: CovarianceResult,
    cfg: ProjectConfig | None = None,
    *,
    solver: str | None = None,
) -> dict:
    """Minimum variance on the grid (no return constraint beyond sum=1, x>=0)."""
    n = len(cov.mu)
    x = cp.Variable(n)
    risk = cp.quad_form(x, cov.sigma)
    prob = cp.Problem(
        cp.Minimize(0.5 * risk),
        [cp.sum(x) == 1, x >= 0],
    )
    sol = solver or (cfg.solver if cfg is not None else "OSQP")
    prob.solve(solver=sol, verbose=False)
    if x.value is None:
        raise RuntimeError("Min-variance QP failed")
    w = np.asarray(x.value).flatten()
    port_mu = float(cov.mu @ w)
    port_sigma = float(np.sqrt(max(float(w @ cov.sigma @ w), 0.0)))
    return {"weights": w, "mu": port_mu, "sigma": port_sigma}
