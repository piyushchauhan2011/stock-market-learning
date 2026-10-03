"""VADER sentiment scoring for financial headlines.

VADER is a rule/lexicon based scorer: fast, deterministic, no model download,
no API key. It understands punctuation and capitalisation, not context — the
notebook shows where it breaks (sarcasm, negation scope, headlines).

Educational use only — not financial advice.
"""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
"""VADER's own convention: compound >= 0.05 reads positive, <= -0.05 negative."""

_analyzer = SentimentIntensityAnalyzer()


def analyze_text(text: str) -> dict:
    """Return VADER's scores: keys exactly ``neg, neu, pos, compound``."""
    return _analyzer.polarity_scores(text)


def score_headlines(titles: list[str]) -> pd.DataFrame:
    """Score each headline; columns exactly ``title, compound, label``."""
    rows = [
        {
            "title": title,
            "compound": analyze_text(title)["compound"],
            "label": _label(analyze_text(title)["compound"]),
        }
        for title in titles
    ]
    return pd.DataFrame(rows, columns=["title", "compound", "label"])


def aggregate_sentiment(titles: list[str]) -> dict:
    """Summarise a headline list: mean score, label counts, and a verdict.

    Returns keys ``mean_compound, positive, negative, neutral, verdict`` where
    verdict is ``"bullish"`` (mean >= 0.05), ``"bearish"`` (<= -0.05) or
    ``"neutral"``. An empty list is neutral, not an error.
    """
    scored = score_headlines(titles)
    if scored.empty:
        return {
            "mean_compound": 0.0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "verdict": "neutral",
        }

    mean_compound = float(scored["compound"].mean())
    counts = scored["label"].value_counts()
    return {
        "mean_compound": mean_compound,
        "positive": int(counts.get("positive", 0)),
        "negative": int(counts.get("negative", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "verdict": _label(mean_compound),
    }


def _label(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= -POSITIVE_THRESHOLD:
        return "negative"
    return "neutral"
