"""KKT verification: CVXPY vs SciPy on a small long-only Markowitz QP."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class KKTVerification:
    r_min: float
    weights_cvxpy: np.ndarray
    weights_scipy: np.ndarray
    max_weight_diff: float
    stationarity_residual: float
    comp_slack_max: float
    lambda_sum: float
    nu_return: float
    pi_long: np.ndarray
    stationarity: np.ndarray
    return_slack: float
    sum_slack: float
    return_constraint_active: bool
    zero_weight_tickers: list[str]
    solver_cvxpy: str
    scipy_success: bool


def _solve_cvxpy_with_duals(
    mu: np.ndarray,
    sigma: np.ndarray,
    r_min: float,
    *,
    solver: str,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    n = len(mu)
    x = cp.Variable(n)
    risk = cp.quad_form(x, sigma)
    c_sum = cp.sum(x) == 1
    c_return = mu @ x >= r_min
    c_long = x >= 0
    prob = cp.Problem(cp.Minimize(0.5 * risk), [c_sum, c_return, c_long])
    prob.solve(solver=solver, verbose=False)
    if x.value is None:
        raise RuntimeError(f"CVXPY failed at r_min={r_min:.6f}")
    w = np.asarray(x.value).flatten()
    lam = float(np.asarray(c_sum.dual_value).flatten()[0])
    nu = float(np.asarray(c_return.dual_value).flatten()[0])
    pi = np.asarray(c_long.dual_value).flatten()
    return w, lam, nu, pi, float(prob.value)


def _solve_scipy(
    mu: np.ndarray,
    sigma: np.ndarray,
    r_min: float,
) -> np.ndarray:
    n = len(mu)

    def objective(x: np.ndarray) -> float:
        return 0.5 * float(x @ sigma @ x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return sigma @ x

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0, "jac": lambda x: np.ones(n)},
        {"type": "ineq", "fun": lambda x: float(mu @ x - r_min), "jac": lambda x: mu},
    ]
    bounds = [(0.0, None)] * n
    x0 = np.ones(n) / n
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"SciPy SLSQP failed: {res.message}")
    return np.asarray(res.x).flatten()


def _stationarity(
    mu: np.ndarray,
    sigma: np.ndarray,
    x: np.ndarray,
    lam: float,
    nu: float,
    pi: np.ndarray,
) -> np.ndarray:
    """Grad L = Sigma x - lam*1 - nu*mu - pi (CVXPY dual convention)."""
    n = len(x)
    return sigma @ x - lam * np.ones(n) - nu * mu - pi


def _stationarity_residual(stationarity: np.ndarray) -> float:
    return float(np.linalg.norm(stationarity, ord=np.inf))


def verify_markowitz_kkt(
    mu: np.ndarray,
    sigma: np.ndarray,
    r_min: float,
    tickers: list[str] | None = None,
    *,
    solver: str = "OSQP",
    weight_tol: float = 1e-4,
    dual_tol: float = 1e-6,
) -> KKTVerification:
    w_cvx, lam, nu, pi, _ = _solve_cvxpy_with_duals(mu, sigma, r_min, solver=solver)
    w_sci = _solve_scipy(mu, sigma, r_min)
    stat = _stationarity(mu, sigma, w_cvx, lam, nu, pi)
    comp_slack = float(np.max(np.abs(pi * w_cvx)))
    ret_slack = float(r_min - mu @ w_cvx)
    sum_slack = float(abs(np.sum(w_cvx) - 1.0))
    labels = tickers or [str(i) for i in range(len(mu))]
    zero_w = [t for t, w in zip(labels, w_cvx, strict=True) if w < weight_tol]
    ret_active = nu > dual_tol and abs(ret_slack) < 1e-5
    return KKTVerification(
        r_min=r_min,
        weights_cvxpy=w_cvx,
        weights_scipy=w_sci,
        max_weight_diff=float(np.max(np.abs(w_cvx - w_sci))),
        stationarity_residual=_stationarity_residual(stat),
        comp_slack_max=comp_slack,
        lambda_sum=lam,
        nu_return=nu,
        pi_long=pi,
        stationarity=stat,
        return_slack=ret_slack,
        sum_slack=sum_slack,
        return_constraint_active=ret_active,
        zero_weight_tickers=zero_w,
        solver_cvxpy=solver,
        scipy_success=True,
    )


def kkt_latex_block() -> str:
    """Stationarity + feasibility + complementarity for the long-only Markowitz QP."""
    return r"""
\min_x \tfrac12 x^\top \Sigma x \quad
\text{s.t.}\quad \mathbf{1}^\top x = 1,\;\; \mu^\top x \ge r_{\min},\;\; x \ge 0.

L = \tfrac12 x^\top \Sigma x - \lambda(\mathbf{1}^\top x - 1) - \nu(r_{\min} - \mu^\top x) - \pi^\top x.

\text{Stationarity:}\quad \Sigma x - \lambda\mathbf{1} - \nu\mu - \pi = 0,\quad
\pi \ge 0,\; x \ge 0,\; \pi_i x_i = 0.
""".strip()


def format_kkt_report(result: KKTVerification, tickers: list[str]) -> str:
    lines = [
        f"Target return r_min = {result.r_min:.6f}",
        f"CVXPY vs SciPy max |dw| = {result.max_weight_diff:.2e}",
        f"KKT stationarity (inf-norm) = {result.stationarity_residual:.2e}",
        f"Complementary slackness max |pi_i * x_i| = {result.comp_slack_max:.2e}",
        f"Primal: |sum(x)-1| = {result.sum_slack:.2e}, return slack r_min - mu'x = {result.return_slack:.2e}",
        f"Duals: lambda (sum) = {result.lambda_sum:.6f}, nu (return) = {result.nu_return:.6f}",
        f"Return constraint active: {result.return_constraint_active}",
        f"Near-zero weights: {', '.join(result.zero_weight_tickers) or '(none)'}",
        "Weights (CVXPY):",
    ]
    for t, w in zip(tickers, result.weights_cvxpy, strict=True):
        lines.append(f"  {t}: {w:.4f}")
    lines.append("Stationarity residual per asset:")
    for t, r in zip(tickers, result.stationarity, strict=True):
        lines.append(f"  {t}: {r:.2e}")
    return "\n".join(lines)
