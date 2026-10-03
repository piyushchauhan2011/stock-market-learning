"""Google Finance quote + recent daily bars, hand-rolled with the standard library.

A second data source next to :mod:`stocklearn.data` (Yahoo). Nothing here uses
``requests`` or a scraping library: one :func:`urllib.request.urlopen` call
fetches the public quote page, and the numbers are pulled out of the
``AF_initDataCallback`` JSON blobs Google embeds in the HTML. The parsing is
split into small pure functions, so the only I/O in the module is
:func:`_fetch_html` — which is what makes the parser testable without a network.

What the page actually gives you (verified against the live page, 2026):

- a **snapshot quote** — price, change, percent change, previous close, name,
  currency, exchange and the as-of timestamp;
- **~1 month of daily bars** (21 rows in practice), already split/dividend
  adjusted, so ``Adj Close == Close`` and there are no ``Dividends`` or
  ``Stock Splits`` columns at all;
- nothing else: no 52-week range, no market cap, no multi-year history. The
  ``?window=6M/1Y/5Y/MAX`` selector on that page is applied client-side from a
  private XHR we deliberately do not chase. For long history use
  :func:`stocklearn.data.get_history`.

Two accepted constraints: the **exchange is mandatory** in the URL
(``AAPL:NASDAQ``; a bare ``AAPL`` resolves to a page with no ticker data), and
there is **no cache** — one page fetch is cheap and always current.

Google's HTML markers are undocumented and can change without notice. When the
page stops matching, these functions raise ``ValueError``/``RuntimeError`` with
what went wrong; they never return guessed numbers.

Educational use only — not financial advice.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
"""A desktop browser UA string; Google serves the plain page to this."""

GOOGLE_QUOTE_URL = "https://www.google.com/finance/quote/"
"""Public quote page: the ticker and exchange are appended as ``TICKER:EXCHANGE``."""

REQUIRED_EXCHANGE = "NASDAQ"
"""Default exchange suffix — the URL is invalid without one."""

BLOCK_START = re.compile(r"AF_initDataCallback\(\{key: '([^']*)'")
"""Matches the start of one embedded data blob, capturing its ``ds:N`` key."""

QUOTE_KEYS = ("ds:2", "ds:8", "ds:13")
"""Blocks known to carry the quote row, in the order we try them."""

HISTORY_KEY = "ds:12"
"""Block carrying the daily OHLCV rows."""


@dataclass(frozen=True)
class Quote:
    """One snapshot quote, exactly as the Google Finance page states it."""

    ticker: str
    exchange: str
    name: str
    currency: str
    price: float
    change: float
    """Price-unit change against the previous close."""
    pct_change: float
    """Change as a fraction: ``+1.02%`` is ``0.0102``."""
    previous_close: float
    as_of: datetime
    """Timezone-aware UTC timestamp of the quote."""


def get_quote(ticker: str = "AAPL", exchange: str = REQUIRED_EXCHANGE) -> Quote:
    """Fetch the current quote for ``ticker`` from Google Finance.

    Parameters
    ----------
    ticker:
        Symbol as Google writes it, e.g. ``"AAPL"``, ``"KO"``.
    exchange:
        Exchange suffix — ``"NASDAQ"``, ``"NYSE"``, ``"NYSEARCA"``, ... It is
        required: a bare ticker has no page with quote data.

    Returns
    -------
    Quote
        Frozen dataclass with price, change, percent change, previous close,
        name, currency and the as-of timestamp.

    Raises
    ------
    RuntimeError
        The page could not be fetched (network down, HTTP error, timeout).
    ValueError
        The page was fetched but contained no quote row — normally a wrong
        symbol or exchange, but also what you get if Google changes its markup.
    """
    html = _fetch_html(f"{ticker}:{exchange}")
    row = _find_quote_row(_parse_blocks(html))
    if row is None:
        raise ValueError(
            f"no quote for {ticker!r} on {exchange!r} — check the symbol/exchange"
        )
    return _quote_from_row(row)


def get_history(ticker: str = "AAPL", exchange: str = REQUIRED_EXCHANGE) -> pd.DataFrame:
    """Fetch the daily bars Google Finance shows for ``ticker`` (~1 month).

    Columns match :func:`stocklearn.data.get_history` —
    ``Open, High, Low, Close, Adj Close, Volume`` — so the two are
    interchangeable everywhere downstream. ``Adj Close`` is a copy of
    ``Close``: Google's daily history is already adjusted for splits and
    dividends, and publishes no separate raw close.

    The index is a tz-naive ``DatetimeIndex`` of trading dates, oldest first,
    at most ~21 rows. There is no ``period`` parameter: the static page always
    embeds the same fixed window.

    Raises
    ------
    RuntimeError
        The page could not be fetched (network down, HTTP error, timeout).
    ValueError
        The page was fetched but contained no daily rows — usually a wrong
        symbol or exchange.
    """
    html = _fetch_html(f"{ticker}:{exchange}")
    rows = _find_daily_rows(_parse_blocks(html))
    if not rows:
        raise ValueError(
            f"no daily bars for {ticker!r} on {exchange!r} — check the symbol/exchange"
        )
    return _history_frame(rows)


def _fetch_html(symbol: str) -> str:
    """Download the quote page for ``symbol`` (``"AAPL:NASDAQ"``) as text."""
    request = urllib.request.Request(
        GOOGLE_QUOTE_URL + symbol, headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"could not fetch Google Finance for {symbol!r} — {type(exc).__name__}: {exc}"
        ) from exc


def _parse_blocks(html: str) -> dict[str, list]:
    """Extract every ``AF_initDataCallback`` blob from ``html`` as parsed JSON.

    The blob text is ``AF_initDataCallback({key: 'ds:N', hash: '…', data: <json>,
    sideChannel: {}});``. The value after ``data:`` is raw JSON on the current
    page; older pages wrapped it as ``function(){return …}``, which is unwrapped
    as a fallback. A blob that still will not parse is skipped rather than
    aborting the whole page — the set of keys shifts over time.
    """
    blocks: dict[str, list] = {}

    for match in BLOCK_START.finditer(html):
        try:
            start = html.index("data:", match.end()) + len("data:")
            end = html.index(" sideChannel", start)
        except ValueError:  # a truncated/reshaped block: skip it, keep the rest
            continue
        raw = html[start:end].rstrip(" ,")
        unwrapped = (
            re.sub(r"^function\(\)\s*\{\s*return\s*", "", raw.lstrip()).removesuffix("}")
        )

        for candidate in (raw, unwrapped):
            try:
                blocks[match.group(1)] = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            break

    return blocks


def _walk(node: object) -> Iterator[list]:
    """Yield ``node`` and every nested list inside it, depth-first."""
    if isinstance(node, list):
        yield node
        for child in node:
            yield from _walk(child)


def _find_quote_row(blocks: dict[str, list]) -> list | None:
    """Return the quote row from ``blocks``, or ``None`` when absent.

    The row looks like ``["/m/07zmbvf", ["AAPL","NASDAQ"], "Apple Inc", 0,
    "USD", [price, change, pct], null, previous_close, …]``. Blocks are tried in
    :data:`QUOTE_KEYS` order and the first structural match wins.
    """
    for key in QUOTE_KEYS:
        for candidate in _walk(blocks.get(key)):
            if _is_quote_row(candidate):
                return candidate
    return None


def _is_quote_row(row: list) -> bool:
    return (
        len(row) > 11
        and isinstance(row[0], str)
        and row[0].startswith("/m/")
        and isinstance(row[1], list)
        and len(row[1]) == 2
        and isinstance(row[5], list)
        and isinstance(row[5][0], (int, float))
    )


def _quote_from_row(row: list) -> Quote:
    """Map a quote row's positions onto a :class:`Quote`.

    Position map: ``[1]`` ``[ticker, exchange]``, ``[2]`` name, ``[4]``
    currency, ``[5]`` ``[price, change, pct_change]``, ``[7]`` previous close,
    ``[11]`` ``[unix seconds]``. Google states the percentage already multiplied
    by 100, so it is divided back into a fraction.
    """
    return Quote(
        ticker=str(row[1][0]),
        exchange=str(row[1][1]),
        name=str(row[2]),
        currency=str(row[4]),
        price=float(row[5][0]),
        change=float(row[5][1]),
        pct_change=float(row[5][2]) / 100.0,
        previous_close=float(row[7]),
        as_of=datetime.fromtimestamp(int(row[11][0]), tz=timezone.utc),
    )


def _find_daily_rows(blocks: dict[str, list]) -> list | None:
    """Return the list of daily bars from ``blocks``, or ``None`` when absent.

    Each bar is ``[open, close, high, low, "YYYY-MM-DDTHH:MM:SS±HH:MM", volume]``
    — note the order: open, close, high, low.
    """
    for candidate in _walk(blocks.get(HISTORY_KEY)):
        if candidate and all(_is_daily_bar(bar) for bar in candidate):
            return candidate
    return None


def _is_daily_bar(bar: list) -> bool:
    return (
        isinstance(bar, list)
        and len(bar) == 6
        and isinstance(bar[0], (int, float))
        and isinstance(bar[4], str)
    )


def _history_frame(rows: list) -> pd.DataFrame:
    """Build the OHLCV frame from Google's ``[open, close, high, low, ts, vol]`` rows."""
    index = pd.DatetimeIndex([pd.Timestamp(row[4]).date() for row in rows])
    closes = [float(row[1]) for row in rows]

    return pd.DataFrame(
        {
            "Open": [float(row[0]) for row in rows],
            "High": [float(row[2]) for row in rows],
            "Low": [float(row[3]) for row in rows],
            "Close": closes,
            # Google's daily history is already adjusted — one price, two columns,
            # so downstream code can keep using price_series() unchanged.
            "Adj Close": closes,
            "Volume": [int(row[5]) for row in rows],
        },
        index=index,
    )
