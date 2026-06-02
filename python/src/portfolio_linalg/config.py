"""Load project YAML config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class ComparisonConfig:
    kkt_tickers: list[str]
    rank_k: int | None
    eigenvalue_floor: float | None


@dataclass(frozen=True)
class ProjectConfig:
    tickers: list[str]
    start_date: str
    end_date: str | None
    return_type: str
    r_min_points: int
    solver: str
    comparison: ComparisonConfig

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def figures_dir(self) -> Path:
        return ROOT / "figures"

    @property
    def figures_comparison_dir(self) -> Path:
        return ROOT / "figures" / "comparison"


def _load_comparison(raw: dict[str, Any]) -> ComparisonConfig:
    block = raw.get("comparison") or {}
    return ComparisonConfig(
        kkt_tickers=[str(t).upper() for t in block.get("kkt_tickers", ["SPY", "AGG", "GLD"])],
        rank_k=int(block["rank_k"]) if block.get("rank_k") is not None else None,
        eigenvalue_floor=float(block["eigenvalue_floor"])
        if block.get("eigenvalue_floor") is not None
        else None,
    )


def load_config(path: Path | None = None) -> ProjectConfig:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return ProjectConfig(
        tickers=[str(t).upper() for t in raw["tickers"]],
        start_date=str(raw["start_date"]),
        end_date=raw.get("end_date"),
        return_type=str(raw.get("return_type", "simple")),
        r_min_points=int(raw.get("r_min_points", 15)),
        solver=str(raw.get("solver", "OSQP")),
        comparison=_load_comparison(raw),
    )
