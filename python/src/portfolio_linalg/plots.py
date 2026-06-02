"""Matplotlib figures for report and presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import polars as pl

from portfolio_linalg.comparison import ComparisonBundle
from portfolio_linalg.covariance import CovarianceResult, correlation_from_sigma
from portfolio_linalg.spectral import eigenportfolio_table


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def figure_efficient_frontier(
    frontier: pl.DataFrame,
    *,
    annualized: bool = False,
    trading_days: int = 252,
) -> Figure:
    pts = frontier.select(["mu", "sigma"]).unique().sort("sigma")
    if annualized:
        scale = np.sqrt(trading_days)
        x = (pts["sigma"] * scale).to_list()
        y = (pts["mu"] * trading_days).to_list()
        xlab, ylab = "Annualized volatility", "Annualized expected return"
        title = f"Efficient frontier (annualized, {trading_days} trading days)"
    else:
        x = pts["sigma"].to_list()
        y = pts["mu"].to_list()
        xlab, ylab = "Portfolio volatility (sigma)", "Expected return (mu)"
        title = "Mean-variance efficient frontier (long-only)"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, "o-", linewidth=1.5, markersize=4)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figure_eigenvalues(cov: CovarianceResult) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(cov.eigenvalues)), cov.eigenvalues)
    ax.set_xlabel("Index (sorted)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Spectrum of Sigma (kappa ~ {cov.condition_number:.2e})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def figure_correlation_heatmap(cov: CovarianceResult) -> Figure:
    corr = correlation_from_sigma(cov.sigma)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cov.tickers)))
    ax.set_yticks(range(len(cov.tickers)))
    ax.set_xticklabels(cov.tickers, rotation=45, ha="right")
    ax.set_yticklabels(cov.tickers)
    ax.set_title("Return correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def figure_weights_vs_return(frontier: pl.DataFrame) -> Figure:
    wide = frontier.pivot(on="ticker", index="r_min_target", values="weight").sort(
        "r_min_target"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    tickers = [c for c in wide.columns if c != "r_min_target"]
    x = wide["r_min_target"].to_list()
    for t in tickers:
        ax.plot(x, wide[t].to_list(), label=t, linewidth=1)
    ax.set_xlabel("Target return r_min")
    ax.set_ylabel("Weight")
    ax.set_title("Frontier weights vs target return")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def all_figures(cov: CovarianceResult, frontier: pl.DataFrame) -> list[Figure]:
    """Build figures for inline display (caller should not close before showing)."""
    return [
        figure_efficient_frontier(frontier),
        figure_eigenvalues(cov),
        figure_correlation_heatmap(cov),
        figure_weights_vs_return(frontier),
    ]


def _save_figure(fig: Figure, path: Path) -> Path:
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_efficient_frontier(frontier: pl.DataFrame, out: Path) -> Path:
    return _save_figure(figure_efficient_frontier(frontier), out / "efficient_frontier.png")


def plot_eigenvalues(cov: CovarianceResult, out: Path) -> Path:
    return _save_figure(figure_eigenvalues(cov), out / "eigenvalues.png")


def plot_correlation_heatmap(cov: CovarianceResult, out: Path) -> Path:
    return _save_figure(figure_correlation_heatmap(cov), out / "correlation_heatmap.png")


def plot_weights_vs_return(frontier: pl.DataFrame, out: Path) -> Path:
    return _save_figure(figure_weights_vs_return(frontier), out / "weights_vs_return.png")


def generate_all(
    cov: CovarianceResult,
    frontier: pl.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_efficient_frontier(frontier, figures_dir),
        plot_eigenvalues(cov, figures_dir),
        plot_correlation_heatmap(cov, figures_dir),
        plot_weights_vs_return(frontier, figures_dir),
    ]


def figure_frontier_compare(bundle: ComparisonBundle) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    def _curve(frontier: pl.DataFrame, label: str, style: str) -> None:
        pts = frontier.select(["mu", "sigma"]).unique().sort("sigma")
        ax.plot(pts["sigma"].to_list(), pts["mu"].to_list(), style, label=label, linewidth=1.5)

    _curve(bundle.frontier_sample, "Sample Sigma", "o-")
    _curve(bundle.frontier_lw, f"Ledoit-Wolf (shrink={bundle.lw_shrinkage:.3f})", "s--")
    if bundle.frontier_rank_k is not None:
        _curve(bundle.frontier_rank_k, bundle.rank_k.label if bundle.rank_k else "rank-k", "^:")
    if bundle.frontier_floor is not None:
        _curve(bundle.frontier_floor, bundle.eigen_floor.label if bundle.eigen_floor else "floor", "x-.")

    ax.set_xlabel("Portfolio volatility (sigma)")
    ax.set_ylabel("Expected return (mu)")
    ax.set_title("Efficient frontier: covariance estimators")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figure_variance_explained(bundle: ComparisonBundle) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(bundle.cumulative_variance))
    ax.bar(x, bundle.sample.eigenvalues, alpha=0.5, label="Eigenvalue")
    ax.plot(x, bundle.cumulative_variance, "ro-", label="Cumulative share of trace")
    ax.set_xlabel("Mode index (ascending eigenvalue)")
    ax.set_ylabel("Magnitude / cumulative fraction")
    ax.set_title("Spectral structure of sample Sigma")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figure_eigenportfolios(cov: CovarianceResult, *, top_modes: int = 3) -> Figure:
    """Heatmap of L1-normalized eigenmode loadings (largest modes)."""
    tab = eigenportfolio_table(cov, top_modes=top_modes)
    modes = sorted(tab["mode_rank"].unique().to_list())
    tickers = cov.tickers
    mat = np.zeros((len(modes), len(tickers)))
    for i, m in enumerate(modes):
        sub = tab.filter(pl.col("mode_rank") == m).sort("ticker")
        for j, t in enumerate(tickers):
            row = sub.filter(pl.col("ticker") == t).row(0, named=True)
            mat[i, j] = row["loading"]
    fig, ax = plt.subplots(figsize=(9, 3 + 0.4 * len(modes)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([f"PC{len(tickers) - m + 1}" for m in modes])
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_title("Eigenportfolio loadings (L1-normalized, largest modes)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def figure_kappa_sensitivity(
    floor_values: list[float],
    min_var_sigmas: list[float],
    kappas: list[float],
) -> Figure:
    """Min-variance risk vs eigenvalue floor (conditioning experiment)."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.semilogx(floor_values, min_var_sigmas, "bo-", label="Min-var sigma")
    ax2.semilogx(floor_values, kappas, "rs--", label="kappa(Sigma)")
    ax1.set_xlabel("Eigenvalue floor")
    ax1.set_ylabel("Min-variance volatility")
    ax2.set_ylabel("Condition number")
    ax1.set_title("Sensitivity to spectral floor perturbation")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_comparison_figures(
    bundle: ComparisonBundle,
    figures_dir: Path,
    *,
    solver: str = "OSQP",
    eigen_floors: list[float] | None = None,
) -> list[Path]:
    from portfolio_linalg.covariance import covariance_from, eigenvalue_floor_sigma
    from portfolio_linalg.frontier import min_variance_portfolio

    figures_dir.mkdir(parents=True, exist_ok=True)
    floors = eigen_floors or [1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
    risks: list[float] = []
    kappas: list[float] = []
    for f in floors:
        s = eigenvalue_floor_sigma(bundle.sample.sigma, f)
        cov = covariance_from(bundle.sample.tickers, bundle.sample.mu, s)
        risks.append(min_variance_portfolio(cov, solver=solver)["sigma"])
        kappas.append(cov.condition_number)
    paths = [
        _save_figure(figure_frontier_compare(bundle), figures_dir / "frontier_compare.png"),
        _save_figure(figure_variance_explained(bundle), figures_dir / "variance_explained.png"),
        _save_figure(
            figure_kappa_sensitivity(floors, risks, kappas),
            figures_dir / "kappa_sensitivity.png",
        ),
        _save_figure(
            figure_eigenportfolios(bundle.sample),
            figures_dir / "eigenportfolios.png",
        ),
    ]
    return paths
