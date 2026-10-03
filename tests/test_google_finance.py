"""Offline parser tests for :mod:`stocklearn.google_finance`.

``_fetch_html`` is the only I/O in the module, so everything here runs without a
network: the page markup is pinned as a trimmed copy of the real HTML fragment
(the ``ds:2`` quote block and the ``ds:12`` daily-bar block), and the fetches are
monkeypatched. The fixture values are the ones the live page returned, which is
why the expected numbers are literal rather than recomputed.
"""

from __future__ import annotations

import datetime as dt
import urllib.error

import pandas as pd
import pytest

from stocklearn import google_finance
from stocklearn.google_finance import (
    Quote,
    _fetch_html,
    _find_daily_rows,
    _find_quote_row,
    _history_frame,
    _parse_blocks,
    _quote_from_row,
    get_history,
    get_quote,
)

# Verbatim copy of the quote row the live page returned (14 elements, including the
# as-of timestamp at [11]) — trimmed out of its surrounding nesting.
QUOTE_ROW = (
    "[\"/m/07zmbvf\",[\"AAPL\",\"NASDAQ\"],\"Apple Inc\",0,\"USD\","
    "[333.69,3.369995,1.0202214,2,2,2],null,330.32,\"#666666\",\"US\",\"/m/0k8z\","
    "[1790987400],\"America/New_York\",-14400]"
)

# Trimmed copy of the real page: the quote block plus the first two of the 21 daily bars.
# Nesting and delimiters (`data:`, ` sideChannel`) are exactly as Google sends them.
PAGE_FIXTURE = (
    "<html><body><script>"
    "AF_initDataCallback({key: 'ds:2', hash: '1', data: [[[["
    + QUOTE_ROW
    + "]]]], sideChannel: {}});</script>"
    "<script>"
    "AF_initDataCallback({key: 'ds:12', hash: '1', data: [[[\"AAPL\",\"NASDAQ\"],"
    "\"/m/07zmbvf\",\"USD\",[[[1,[2026,10,2,9,30,null,null,[-14400]],"
    "[2026,10,2,16,null,null,null,[-14400]]],null,[[324.87,328.21,330.81,324.11,"
    "\"2026-09-03T16:00:00-04:00\",37225839],[328.31,319.97,328.93,317.86,"
    "\"2026-09-04T16:00:00-04:00\",39606885]]]]]], sideChannel: {}});</script>"
    "</body></html>"
)

# Older pages wrapped the payload in `function(){return …}` — the parser unwraps it.
WRAPPED_FIXTURE = (
    "<script>AF_initDataCallback({key: 'ds:2', hash: '1', data: function(){return [[[["
    + QUOTE_ROW
    + "]]]]}, sideChannel: {}});</script>"
)

# A reshaped/truncated block sits next to a good one: the good one must survive.
BROKEN_FIXTURE = (
    "<script>AF_initDataCallback({key: 'ds:99', hash: '1', data: not json at all, "
    "sideChannel: {}});</script>" + PAGE_FIXTURE
)


def test_parse_blocks_extracts_every_data_block():
    blocks = _parse_blocks(PAGE_FIXTURE)

    assert sorted(blocks) == ["ds:12", "ds:2"]
    assert all(isinstance(blocks[key], list) for key in blocks)


def test_parse_blocks_unwraps_the_legacy_function_wrapper():
    blocks = _parse_blocks(WRAPPED_FIXTURE)

    assert _find_quote_row(blocks) is not None


def test_parse_blocks_skips_a_block_it_cannot_parse():
    blocks = _parse_blocks(BROKEN_FIXTURE)

    assert sorted(blocks) == ["ds:12", "ds:2"]


def test_quote_row_fields_map_to_the_quote_dataclass():
    row = _find_quote_row(_parse_blocks(PAGE_FIXTURE))

    assert row is not None
    quote = _quote_from_row(row)

    assert quote.ticker == "AAPL"
    assert quote.exchange == "NASDAQ"
    assert quote.name == "Apple Inc"
    assert quote.currency == "USD"
    assert quote.price == 333.69
    assert quote.change == 3.369995
    # The page states the percentage already multiplied by 100.
    assert quote.pct_change == pytest.approx(0.010202214)
    assert quote.previous_close == 330.32
    assert quote.as_of == dt.datetime(2026, 10, 3, 0, 30, tzinfo=dt.timezone.utc)


def test_quote_row_is_absent_from_an_empty_or_unrelated_page():
    assert _find_quote_row({}) is None
    assert _find_daily_rows({}) is None


def test_history_frame_records_open_close_high_low_in_yahoo_column_order():
    rows = _find_daily_rows(_parse_blocks(PAGE_FIXTURE))

    assert rows is not None and len(rows) == 2
    frame = _history_frame(rows)

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert frame.index.tolist() == [pd.Timestamp("2026-09-03"), pd.Timestamp("2026-09-04")]
    assert frame.index.tz is None
    # Google sends [open, close, high, low, …]: the close is the *second* element.
    assert frame["Open"].tolist() == [324.87, 328.31]
    assert frame["Close"].tolist() == [328.21, 319.97]
    assert frame["High"].tolist() == [330.81, 328.93]
    assert frame["Low"].tolist() == [324.11, 317.86]
    assert frame["Volume"].tolist() == [37225839, 39606885]
    # Already adjusted on the way out, so both price columns are the same series.
    assert (frame["Adj Close"] == frame["Close"]).all()


def test_get_quote_fetches_the_ticker_exchange_url(monkeypatch):
    seen = []
    monkeypatch.setattr(
        google_finance, "_fetch_html", lambda symbol: seen.append(symbol) or PAGE_FIXTURE
    )

    quote = get_quote("AAPL", exchange="NASDAQ")

    assert isinstance(quote, Quote)
    assert quote.price == 333.69
    assert seen == ["AAPL:NASDAQ"]


def test_get_history_returns_the_daily_frame(monkeypatch):
    monkeypatch.setattr(google_finance, "_fetch_html", lambda symbol: PAGE_FIXTURE)

    frame = get_history("AAPL", exchange="NASDAQ")

    assert frame.shape == (2, 6)
    assert frame["Close"].iloc[-1] == 319.97


def test_get_quote_raises_value_error_when_the_page_has_no_quote(monkeypatch):
    monkeypatch.setattr(google_finance, "_fetch_html", lambda symbol: "")

    with pytest.raises(ValueError, match="no quote for 'NOPE' on 'NASDAQ'"):
        get_quote("NOPE", exchange="NASDAQ")


def test_get_history_raises_value_error_when_the_page_has_no_bars(monkeypatch):
    monkeypatch.setattr(google_finance, "_fetch_html", lambda symbol: "")

    with pytest.raises(ValueError, match="no daily bars for 'NOPE' on 'NASDAQ'"):
        get_history("NOPE", exchange="NASDAQ")


def test_fetch_html_turns_a_network_failure_into_runtime_error(monkeypatch):
    def boom(request, timeout=None):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(google_finance.urllib.request, "urlopen", boom)

    with pytest.raises(RuntimeError, match="could not fetch Google Finance for 'AAPL:NASDAQ'"):
        _fetch_html("AAPL:NASDAQ")
