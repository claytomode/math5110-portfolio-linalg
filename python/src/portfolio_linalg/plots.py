"""Matplotlib figures for report and presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import polars as pl

from portfolio_linalg.covariance import CovarianceResult, correlation_from_sigma


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def figure_efficient_frontier(frontier: pl.DataFrame) -> Figure:
    pts = frontier.select(["mu", "sigma"]).unique().sort("sigma")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pts["sigma"].to_list(), pts["mu"].to_list(), "o-", linewidth=1.5, markersize=4)
    ax.set_xlabel("Portfolio volatility (sigma)")
    ax.set_ylabel("Expected return (mu)")
    ax.set_title("Mean-variance efficient frontier (long-only)")
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
