#!/usr/bin/env python3
"""Print interpretation stats from cached returns."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from portfolio_linalg.config import load_config
from portfolio_linalg.covariance import build_covariance
from portfolio_linalg.fetch_data import load_returns
from portfolio_linalg.frontier import compute_frontier
from portfolio_linalg.interpret import print_summary

cfg = load_config()
returns = load_returns(cfg)
cov = build_covariance(returns)
frontier = compute_frontier(cov, cfg)
print_summary(cov, cfg, frontier)
