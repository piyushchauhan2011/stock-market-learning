"""Price history fetching with a small local Parquet cache.

This is the data layer every other module builds on: one function to download
OHLCV bars, one function to pick the price series that indicators should use.

Educational use only — not financial advice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "AAPL"
"""Canonical example ticker used throughout the notebooks and report."""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Repository root — the folder that holds ``pyproject.toml``."""

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
"""Local Parquet cache, anchored to the repository root.

Deliberately absolute: JupyterLab starts a kernel inside ``notebooks/``, so a
working-directory-relative path would scatter a second cache there.
"""


def get_history(
    ticker: str = DEFAULT_TICKER,
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV bars for ``ticker``, using the local cache when possible.

    Parameters
    ----------
    ticker:
        Yahoo Finance symbol, e.g. ``"AAPL"``.
    period:
        Look-back window (``"1d"``, ``"5d"``, ``"1mo"``, ``"3mo"``, ``"6mo"``,
        ``"1y"``, ``"2y"``, ``"5y"``, ``"10y"``, ``"ytd"``, ``"max"``). Ignored
        when ``start`` is given.
    interval:
        Bar size: ``"1d"``, ``"1wk"``, ``"1mo"``, ``"1h"``, ``"15m"``, ``"5m"``, ...
    start, end:
        Explicit ``YYYY-MM-DD`` bounds; when given they take priority over
        ``period`` and are part of the cache key.

    Returns
    -------
    pandas.DataFrame
        Indexed by timestamp (usually a ``DatetimeIndex``) with at least the
        columns ``Open, High, Low, Close, Adj Close, Volume``. ``auto_adjust``
        is deliberately left ``False`` so both the raw ``Close`` and the
        dividend/split-adjusted ``Adj Close`` stay visible.

    Raises
    ------
    ValueError
        Yahoo returned no rows — usually a wrong symbol.
    RuntimeError
        Every Yahoo request failed (network down, rate limit) — the underlying
        message is included so notebooks show a clear cause.
    """
    cache = _cache_path(ticker, period, interval, start, end)
    if cache.exists():
        cached = _read_parquet(cache)
        if cached is not None and not cached.empty:
            return cached
        # Present but unreadable/empty: drop it and download again.
        cache.unlink(missing_ok=True)

    frame = _download(ticker, period, interval, start, end)
    if frame.empty:
        raise ValueError(f"no data returned for {ticker!r} — check the symbol")

    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache)
    return frame


def price_series(df: pd.DataFrame, adjusted: bool = True) -> pd.Series:
    """Return the price series indicators should be computed on.

    Uses ``Adj Close`` (dividends and splits folded in) when available, because
    raw ``Close`` jumps on split days and ignores dividends — both of which
    distort returns, moving averages and drawdowns. Falls back to ``Close``.
    """
    column = "Adj Close" if adjusted and "Adj Close" in df.columns else "Close"
    return df[column]


def _cache_path(
    ticker: str, period: str, interval: str, start: str | None, end: str | None
) -> Path:
    return CACHE_DIR / f"{ticker}_{period}_{interval}_{start}_{end}.parquet"


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _download(
    ticker: str, period: str, interval: str, start: str | None, end: str | None
) -> pd.DataFrame:
    """Ask Yahoo for bars, trying the single-ticker then the batch endpoint."""
    attempts = (
        (
            "Ticker.history",
            lambda: yf.Ticker(ticker).history(
                period=period, interval=interval, start=start, end=end, auto_adjust=False
            ),
        ),
        (
            "yf.download",
            lambda: _download_batch(ticker, period, interval, start, end),
        ),
    )

    failures: list[str] = []
    raised = True
    for label, attempt in attempts:
        try:
            frame = attempt()
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim below
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        raised = False
        if frame is not None and not frame.empty:
            return _flatten_columns(frame)
        failures.append(f"{label}: no rows returned")

    if raised:
        raise RuntimeError(
            f"could not fetch history for {ticker!r} — " + "; ".join(failures)
        )
    # Every call answered, none had data: an empty frame (caller raises ValueError).
    return pd.DataFrame()


def _download_batch(
    ticker: str, period: str, interval: str, start: str | None, end: str | None
) -> pd.DataFrame:
    if start or end:
        return yf.download(
            ticker, start=start, end=end, interval=interval,
            auto_adjust=False, progress=False,
        )
    return yf.download(
        ticker, period=period, interval=interval, auto_adjust=False, progress=False
    )


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the ``(Price, Ticker)`` MultiIndex some endpoints return."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    return frame
