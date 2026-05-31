"""Download daily prices via httpx (Yahoo chart API) and build return series with Polars."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import polars as pl

from portfolio_linalg.config import ProjectConfig

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _fetch_yahoo(client: httpx.Client, ticker: str, *, period: str = "2y") -> pl.DataFrame:
    url = YAHOO_CHART.format(ticker=ticker)
    resp = client.get(url, params={"interval": "1d", "range": period}, follow_redirects=True)
    resp.raise_for_status()
    payload = resp.json()
    result = payload["chart"]["result"]
    if not result:
        raise ValueError(f"Yahoo chart returned no data for {ticker}")
    block = result[0]
    ts = block.get("timestamp") or []
    quote = (block.get("indicators") or {}).get("quote", [{}])[0]
    closes = quote.get("close") or []
    rows = [
        {
            "date": datetime.fromtimestamp(t, tz=timezone.utc).date(),
            "close": c,
            "ticker": ticker,
        }
        for t, c in zip(ts, closes, strict=False)
        if c is not None
    ]
    if not rows:
        raise ValueError(f"Yahoo chart had no valid closes for {ticker}")
    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )


def download_prices(
    cfg: ProjectConfig,
    *,
    timeout: float = 30.0,
    retries: int = 3,
) -> pl.DataFrame:
    """Fetch and align close prices for all tickers."""
    frames: list[pl.DataFrame] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-linalg/0.1; academic)"}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        for ticker in cfg.tickers:
            last_err: Exception | None = None
            for attempt in range(retries):
                try:
                    frames.append(_fetch_yahoo(client, ticker))
                    break
                except Exception as e:  # noqa: BLE001 — retry loop
                    last_err = e
                    if attempt == retries - 1:
                        raise
            else:
                if last_err:
                    raise last_err
    prices = pl.concat(frames)
    if cfg.start_date:
        prices = prices.filter(pl.col("date") >= pl.lit(cfg.start_date).str.to_date("%Y-%m-%d"))
    if cfg.end_date:
        prices = prices.filter(pl.col("date") <= pl.lit(cfg.end_date).str.to_date("%Y-%m-%d"))
    return prices.sort(["ticker", "date"])


def prices_to_returns(prices: pl.DataFrame, return_type: str = "simple") -> pl.DataFrame:
    """Wide return matrix: date x tickers."""
    rets = (
        prices.sort(["ticker", "date"])
        .with_columns(
            pl.col("close").pct_change().over("ticker").alias("ret"),
        )
        .drop_nulls("ret")
    )
    if return_type == "log":
        rets = rets.with_columns(pl.col("ret").log1p().alias("ret"))
    wide = rets.pivot(on="ticker", index="date", values="ret").sort("date")
    return wide


def save_returns(
    returns: pl.DataFrame,
    cfg: ProjectConfig,
    *,
    prices: pl.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Write parquet + metadata.json; optional CSV for graders."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cfg.data_dir / "returns.parquet"
    csv_path = cfg.data_dir / "returns.csv"
    meta_path = cfg.data_dir / "metadata.json"

    returns.write_parquet(parquet_path)
    returns.write_csv(csv_path)

    meta = {
        "source": "yahoo",
        "yahoo_url_template": YAHOO_CHART,
        "tickers": cfg.tickers,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "return_type": cfg.return_type,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_obs": returns.height,
        "n_assets": len(cfg.tickers),
    }
    if prices is not None:
        meta["date_min"] = str(prices["date"].min())
        meta["date_max"] = str(prices["date"].max())

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return parquet_path, meta_path


def load_returns(cfg: ProjectConfig) -> pl.DataFrame:
    path = cfg.data_dir / "returns.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No cached returns at {path}; run fetch first.")
    return pl.read_parquet(path)


def fetch_and_cache(cfg: ProjectConfig) -> pl.DataFrame:
    prices = download_prices(cfg)
    returns = prices_to_returns(prices, cfg.return_type)
    save_returns(returns, cfg, prices=prices)
    return returns
