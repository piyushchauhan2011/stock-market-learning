"""Headlines from Yahoo Finance and public RSS feeds — no API key, no signup.

Two best-effort sources are combined:

1. Yahoo's own news list for the ticker (via :class:`yfinance.Search`, whose
   payload carries titles, publishers, links and publish times).
2. Public RSS: the Yahoo Finance headline feed and a Google News search feed.

Nothing here may raise on a network hiccup: news is enrichment, and an empty
list simply means neutral sentiment downstream.

Educational use only — not financial advice.
"""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import pandas as pd
import yfinance as yf

COLUMNS = ["title", "publisher", "link", "published", "source"]


def get_yfinance_news(ticker: str, limit: int = 10) -> list[dict]:
    """Headlines for ``ticker`` from Yahoo Finance's news list.

    Returns dicts with keys ``title, publisher, link, published, source``;
    ``published`` is a timezone-aware UTC timestamp or ``pd.NaT``.
    """
    entries: list[dict] = []
    for fetch in (
        lambda: yf.Search(ticker).news,
        lambda: yf.Ticker(ticker).news,
    ):
        try:
            payload = fetch() or []
        except Exception:  # noqa: BLE001 - Yahoo throttles and reshapes freely
            continue
        entries = [item for item in (_normalize(item) for item in payload) if item]
        if entries:
            break

    entries.sort(key=_published_sort_key, reverse=True)
    return entries[:limit]


def get_rss_news(ticker: str, limit: int = 10) -> list[dict]:
    """Headlines from Yahoo's headline RSS feed, then Google News as backup."""
    sources = (
        (f"https://finance.yahoo.com/rss/headline?s={quote(ticker)}", "yahoo-rss"),
        (
            "https://news.google.com/rss/search?q="
            f"{quote(ticker + ' stock')}&hl=en-US&gl=US&ceid=US:en",
            "google-news",
        ),
    )

    entries: list[dict] = []
    for url, source in sources:
        entries.extend(_parse_feed(url, source, limit))
    entries.sort(key=_published_sort_key, reverse=True)
    return entries[:limit]


def get_news(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Combine every source into one de-duplicated, newest-first table.

    Columns are exactly ``title, publisher, link, published, source``. Returns
    an empty frame with those columns when nothing is available.
    """
    rows = get_yfinance_news(ticker, limit=limit) + get_rss_news(ticker, limit=limit)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop_duplicates(subset="link").dropna(subset=["title"])
    frame = frame.sort_values("published", ascending=False, na_position="last")
    # The same story arrives from several feeds under different URLs (and Google
    # appends " - Publisher" to the title): collapse those, newest copy wins.
    keys = [_title_key(t, p or "") for t, p in zip(frame["title"], frame["publisher"])]
    frame = frame[~pd.Series(keys, index=frame.index).duplicated()]
    return frame.head(limit).reset_index(drop=True)


def _parse_feed(url: str, source: str, limit: int) -> list[dict]:
    try:
        parsed = feedparser.parse(url)
    except Exception:  # noqa: BLE001 - RSS is best-effort
        return []

    feed_title = ""
    if isinstance(parsed, dict):
        feed_title = (parsed.get("feed") or {}).get("title", "") or ""

    rows = []
    for entry in parsed.get("entries", [])[:limit] if isinstance(parsed, dict) else []:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        rows.append(
            {
                "title": title,
                "publisher": _feed_publisher(entry, feed_title),
                "link": link,
                "published": _feed_time(entry),
                "source": source,
            }
        )
    return rows


def _feed_publisher(entry: dict, feed_title: str) -> str:
    source = entry.get("source")
    if isinstance(source, dict) and source.get("title"):
        return source["title"]
    return entry.get("author") or feed_title


def _feed_time(entry: dict) -> pd.Timestamp:
    published = pd.to_datetime(entry.get("published"), utc=True, errors="coerce")
    if pd.isna(published):
        published = pd.to_datetime(entry.get("updated"), utc=True, errors="coerce")
    return published


def _normalize(item) -> dict | None:
    """Map one Yahoo news item to our flat shape.

    Handles both the legacy flat payload (``providerPublishTime``) and the
    newer nested shape (``content.pubDate``, ``content.provider``).
    """
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else {}

    title = (item.get("title") or content.get("title") or "").strip()
    link = _first_url(item.get("link"), item.get("clickThroughUrl"), content.get("canonicalUrl"), content.get("clickThroughUrl"))
    if not title or not link:
        return None

    return {
        "title": title,
        "publisher": _publisher(item, content),
        "link": link,
        "published": _timestamp(item.get("providerPublishTime"), content.get("pubDate")),
        "source": "yfinance",
    }


def _title_key(title: str, publisher: str) -> str:
    """Normalised headline used to spot the same story across feeds."""
    text = " ".join(str(title).split()).lower()
    suffix = " - " + " ".join(str(publisher).split()).lower()
    if len(suffix) > 3 and text.endswith(suffix):
        text = text[: -len(suffix)].strip()
    return text


def _publisher(item: dict, content: dict) -> str:
    if item.get("publisher"):
        return str(item["publisher"])
    provider = content.get("provider")
    if isinstance(provider, dict) and provider.get("displayName"):
        return str(provider["displayName"])
    if isinstance(provider, str):
        return provider
    return ""


def _first_url(*candidates) -> str:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
        if isinstance(candidate, dict) and candidate.get("url"):
            return str(candidate["url"])
    return ""


def _timestamp(epoch, iso: str | None) -> pd.Timestamp:
    if epoch is not None:
        try:
            return pd.Timestamp(datetime.fromtimestamp(float(epoch), tz=timezone.utc))
        except (TypeError, ValueError, OSError):
            pass
    return pd.to_datetime(iso, utc=True, errors="coerce")


def _published_sort_key(row: dict):
    published = row.get("published")
    if published is None or pd.isna(published):
        return datetime.min.replace(tzinfo=timezone.utc)
    return published.to_pydatetime()
