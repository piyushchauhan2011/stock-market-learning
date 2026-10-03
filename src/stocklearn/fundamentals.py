"""Company fundamentals: the three financial statements plus a few ratios.

Yahoo renames statement line items between releases, so every lookup goes
through an ordered list of candidate names. Anything still missing yields
``None`` for that ratio instead of raising — the report renders "unavailable".

Educational use only — not financial advice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from stocklearn.data import CACHE_DIR

STATEMENTS = ("income_stmt", "balance_sheet", "cashflow")

# Ordered fallbacks: Yahoo has used each of these names over the years.
_KEYS = {
    "net_income": (
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares",
    ),
    "total_debt": ("Total Debt", "Long Term Debt", "Long-Term Debt"),
    "equity": ("Stockholders Equity", "Total Stockholder Equity"),
    "current_assets": ("Current Assets", "Total Current Assets"),
    "current_liabilities": ("Current Liabilities", "Total Current Liabilities"),
    "revenue": ("Total Revenue", "Revenue"),
    "gross_profit": ("Gross Profit",),
    "shares": ("Diluted Average Shares", "Basic Average Shares"),
}


def get_financials(ticker: str) -> dict:
    """Return ``{"income_stmt", "balance_sheet", "cashflow", "info"}`` for ``ticker``.

    Each statement is a DataFrame indexed by line item with one column per
    fiscal period (newest first); ``info`` is Yahoo's summary dict. All four are
    cached under ``data/cache/`` so repeated notebook runs stay offline.
    """
    t = yf.Ticker(ticker)
    out: dict = {}

    for name in STATEMENTS:
        cached = _read_parquet(_cache_path(ticker, name))
        if cached is not None and not cached.empty:
            out[name] = cached
            continue
        try:
            frame = getattr(t, name)
        except Exception:  # noqa: BLE001 - Yahoo throttles; keep going
            frame = pd.DataFrame()
        if frame is None:
            frame = pd.DataFrame()
        out[name] = frame
        if not frame.empty:
            _write_parquet(_cache_path(ticker, name), frame)

    info = _read_json(_cache_path(ticker, "info"))
    if info is None:
        try:
            info = dict(t.info or {})
        except Exception:  # noqa: BLE001 - throttled .info is common and non-fatal
            info = {}
        if info:
            _write_json(_cache_path(ticker, "info"), info)
    out["info"] = info
    return out


def compute_ratios(ticker: str) -> dict:
    """Return the six headline valuation/health ratios, ``None`` when unknown.

    Keys, always present: ``pe_ratio``, ``eps``, ``roe``, ``debt_to_equity``,
    ``gross_margin``, ``current_ratio``.

    Formulas (latest annual column of each statement):
    ``eps = net_income / shares_outstanding``,
    ``roe = net_income / equity``,
    ``debt_to_equity = total_debt / equity``,
    ``gross_margin = gross_profit / revenue``,
    ``current_ratio = current_assets / current_liabilities``.
    ``pe_ratio`` is Yahoo's trailing P/E.
    """
    fin = get_financials(ticker)
    income = fin["income_stmt"]
    balance = fin["balance_sheet"]
    info = fin["info"]

    net_income = _latest(income, _KEYS["net_income"])
    equity = _latest(balance, _KEYS["equity"])
    total_debt = _latest(balance, _KEYS["total_debt"])
    revenue = _latest(income, _KEYS["revenue"])
    gross_profit = _latest(income, _KEYS["gross_profit"])
    current_assets = _latest(balance, _KEYS["current_assets"])
    current_liabilities = _latest(balance, _KEYS["current_liabilities"])

    shares = _as_float(info.get("sharesOutstanding"))
    if not shares:
        shares = _latest(income, _KEYS["shares"])

    return {
        "pe_ratio": _as_float(info.get("trailingPE")),
        "eps": _divide(net_income, shares),
        "roe": _divide(net_income, equity),
        "debt_to_equity": _divide(total_debt, equity),
        "gross_margin": _divide(gross_profit, revenue),
        "current_ratio": _divide(current_assets, current_liabilities),
    }


def _latest(frame: pd.DataFrame, keys: tuple[str, ...]) -> float | None:
    """Most recent (leftmost) value of the first matching line item."""
    if frame is None or frame.empty:
        return None
    for key in keys:
        if key in frame.index:
            value = _as_float(frame.loc[key].iloc[0])
            if value is not None:
                return value
    return None


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _cache_path(ticker: str, name: str) -> Path:
    return CACHE_DIR / f"{ticker}_fin_{name}.parquet"


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass


def _read_json(path: Path) -> dict | None:
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(payload).to_json(path)
    except Exception:  # noqa: BLE001 - some info values are not JSON-serialisable
        pass
