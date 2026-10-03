"""The composite report: every signal the notebooks built, in one place.

This answers "will this stock go up?" the only honest way a textbook can: it
describes the current state (trend, momentum, volatility, valuation, news tone)
with fixed, explicit rules, tallies the bullish against the bearish readings,
and refuses to claim it knows the future. It is a checklist, not a crystal ball.

Educational use only — not financial advice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stocklearn.data import DEFAULT_TICKER, get_history, price_series
from stocklearn.fundamentals import compute_ratios
from stocklearn.indicators import (
    annualized_volatility,
    macd,
    max_drawdown,
    rsi,
    sma,
)
from stocklearn.news import get_news
from stocklearn.sentiment import aggregate_sentiment, score_headlines

DISCLAIMER = (
    "> Educational analysis only — not financial advice. "
    "Past performance does not predict future returns."
)
"""Exact closing line of every rendered report."""

CROSS_WINDOW = 5
"""A crossover only counts as "fresh" if it happened within this many bars."""

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
PE_CHEAP = 15.0
PE_EXPENSIVE = 40.0
SENTIMENT_THRESHOLD = 0.05

HEADLINES_SHOWN = 5


def summarize(ticker: str = DEFAULT_TICKER, period: str = "1y") -> dict:
    """Gather every descriptive signal for ``ticker`` into one nested dict.

    Never raises for missing fundamentals, missing news or a short history:
    unavailable inputs come back as ``None`` and the report says so.
    """
    history = get_history(ticker, period=period)
    close = price_series(history)

    trend = _trend(close)
    momentum = _momentum(close)
    ratios = compute_ratios(ticker)

    news = get_news(ticker, limit=10)
    titles = news["title"].tolist() if not news.empty else []
    sentiment = aggregate_sentiment(titles)
    headlines = _headlines(news)

    summary = {
        "ticker": ticker.upper(),
        "period": period,
        "as_of": pd.Timestamp(history.index[-1]).date().isoformat(),
        "last_close": float(close.iloc[-1]),
        "trend": trend,
        "momentum": momentum,
        "volatility": {
            "annualized_volatility": annualized_volatility(close),
            "max_drawdown": max_drawdown(close),
        },
        "valuation": {
            "pe_ratio": ratios["pe_ratio"],
            "roe": ratios["roe"],
            "debt_to_equity": ratios["debt_to_equity"],
        },
        "sentiment": sentiment,
        "headlines": headlines,
    }
    summary["signals"] = _signals(summary)
    summary["bull_bear_tally"] = _tally(summary)
    return summary


def format_report(summary: dict) -> str:
    """Render :func:`summarize` output as a markdown report."""
    trend = summary["trend"]
    momentum = summary["momentum"]
    volatility = summary["volatility"]
    valuation = summary["valuation"]
    sentiment = summary["sentiment"]
    tally = summary["bull_bear_tally"]

    lines = [
        f"# {summary['ticker']} — descriptive analysis",
        "",
        f"_As of {summary['as_of']} · {summary['period']} of daily bars · "
        f"last adjusted close ${summary['last_close']:,.2f}_",
        "",
        "## Trend",
        "",
        f"- **Regime: {trend['regime']}** "
        f"(close vs 50-day: {_value_or_na(trend['close_vs_sma50'])}, "
        f"50-day vs 200-day: {_value_or_na(trend['sma50_vs_sma200'])})",
        f"- 50-day SMA: {_money(trend['sma50'])} · 200-day SMA: {_money(trend['sma200'])}",
        f"- Latest crossover: {_cross_text(trend)}",
        "",
        "## Momentum",
        "",
        f"- RSI(14): {_number(momentum['rsi'], 1)} — {momentum['rsi_zone']}",
        f"- MACD(12/26/9): {_macd_text(momentum)}",
        "",
        "## Volatility",
        "",
        f"- Annualized volatility: {volatility['annualized_volatility']:.1%}",
        f"- Maximum drawdown: {volatility['max_drawdown']:.1%}",
        "",
        "## Valuation",
        "",
        "| Ratio | Value |",
        "| --- | --- |",
        f"| P/E (trailing) | {_number(valuation['pe_ratio'], 1)} |",
        f"| Return on equity | {_percent(valuation['roe'])} |",
        f"| Debt / equity | {_number(valuation['debt_to_equity'], 2)} |",
        "",
        "## News sentiment",
        "",
        f"- **Verdict: {sentiment['verdict']}** "
        f"(mean compound {sentiment['mean_compound']:+.2f} over "
        f"{sentiment['positive'] + sentiment['negative'] + sentiment['neutral']} headlines)",
        f"- Positive {sentiment['positive']} · Negative {sentiment['negative']} · "
        f"Neutral {sentiment['neutral']}",
        "",
    ]

    if summary["headlines"]:
        lines += [
            "| Headline | Source | Score | Tone |",
            "| --- | --- | --- | --- |",
        ]
        for item in summary["headlines"][:HEADLINES_SHOWN]:
            lines.append(
                f"| {item['title']} | {item['source']} | "
                f"{item['compound']:+.2f} | {item['label']} |"
            )
    else:
        lines.append("_No headlines available — sentiment treated as neutral._")

    lines += ["", "## Signals", ""]
    lines += [f"- {signal}" for signal in summary["signals"]]
    lines += [
        "",
        "## Bull / bear tally",
        "",
        f"Bullish {tally['bullish']} · Bearish {tally['bearish']} → "
        f"**{tally['verdict']}**",
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines)


def _trend(close: pd.Series) -> dict:
    last_close = float(close.iloc[-1])
    sma50 = _last(sma(close, 50))
    sma200 = _last(sma(close, 200))

    spread = (sma(close, 50) - sma(close, 200)).dropna()
    up_cross = (spread > 0) & (spread.shift(1) <= 0)
    down_cross = (spread < 0) & (spread.shift(1) >= 0)

    window = spread.iloc[-CROSS_WINDOW:]
    if window.empty:
        golden_cross = death_cross = False
    else:
        golden_cross = bool(up_cross.iloc[-CROSS_WINDOW:].any())
        death_cross = bool(down_cross.iloc[-CROSS_WINDOW:].any())

    last_cross, bars_ago = _last_cross(spread, up_cross, down_cross)

    return {
        "sma50": sma50,
        "sma200": sma200,
        "close_vs_sma50": _relation(last_close, sma50),
        "close_vs_sma200": _relation(last_close, sma200),
        "sma50_vs_sma200": _relation(sma50, sma200),
        "regime": _regime(last_close, sma50, sma200),
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "last_cross": last_cross,
        "last_cross_bars_ago": bars_ago,
    }


def _momentum(close: pd.Series) -> dict:
    rsi_value = _last(rsi(close))
    if rsi_value is None:
        zone = "unavailable"
    elif rsi_value > RSI_OVERBOUGHT:
        zone = "overbought"
    elif rsi_value < RSI_OVERSOLD:
        zone = "oversold"
    else:
        zone = "neutral"

    histogram = macd(close)["hist"]
    up = (histogram > 0) & (histogram.shift(1) <= 0)
    down = (histogram < 0) & (histogram.shift(1) >= 0)
    recent_up = up.iloc[-CROSS_WINDOW:]
    recent_down = down.iloc[-CROSS_WINDOW:]
    tail_up = bool(recent_up.any())
    tail_down = bool(recent_down.any())

    # Both directions can appear inside the window: the later one is the live cross.
    cross = None
    if tail_up or tail_down:
        last_up = recent_up.to_numpy().nonzero()[0].max() if tail_up else -1
        last_down = recent_down.to_numpy().nonzero()[0].max() if tail_down else -1
        cross = "bullish" if last_up > last_down else "bearish"

    return {
        "rsi": rsi_value,
        "rsi_zone": zone,
        "macd_histogram": _last(histogram),
        "macd_bullish_cross": tail_up,
        "macd_bearish_cross": tail_down,
        "macd_cross": cross,
    }


def _headlines(news: pd.DataFrame) -> list[dict]:
    if news.empty:
        return []
    scored = score_headlines(news["title"].tolist())
    return [
        {
            "title": row.title,
            "compound": float(row.compound),
            "label": row.label,
            "source": source,
        }
        for row, source in zip(scored.itertuples(), news["source"])
    ]


def _signals(summary: dict) -> list[str]:
    trend = summary["trend"]
    momentum = summary["momentum"]
    valuation = summary["valuation"]
    sentiment = summary["sentiment"]
    volatility = summary["volatility"]
    signals: list[str] = []

    if trend["close_vs_sma50"]:
        signals.append(f"price {trend['close_vs_sma50']} the 50-day SMA")
    if trend["close_vs_sma200"]:
        signals.append(f"price {trend['close_vs_sma200']} the 200-day SMA")
    if trend["sma50_vs_sma200"]:
        signals.append(f"50-day SMA {trend['sma50_vs_sma200']} the 200-day SMA")
    signals.append(f"trend regime: {trend['regime']}")
    if trend["golden_cross"]:
        signals.append(
            f"golden cross {trend['last_cross_bars_ago']} bars ago "
            "(50-day SMA crossed above the 200-day SMA)"
        )
    elif trend["death_cross"]:
        signals.append(
            f"death cross {trend['last_cross_bars_ago']} bars ago "
            "(50-day SMA crossed below the 200-day SMA)"
        )

    if momentum["rsi"] is not None:
        signals.append(f"RSI {momentum['rsi']:.1f} = {momentum['rsi_zone']}")
    if momentum["macd_cross"]:
        signals.append(
            f"MACD {momentum['macd_cross']} crossover within the last {CROSS_WINDOW} bars"
        )
    elif momentum["macd_histogram"] is not None:
        direction = "positive" if momentum["macd_histogram"] > 0 else "negative"
        signals.append(f"MACD histogram {direction} (no fresh crossover)")

    signals.append(
        f"annualized volatility {volatility['annualized_volatility']:.1%}, "
        f"max drawdown {volatility['max_drawdown']:.1%}"
    )

    pe = valuation["pe_ratio"]
    if pe is None:
        signals.append("P/E unavailable from Yahoo")
    else:
        signals.append(f"P/E {pe:.1f} = {_pe_word(pe)}")
    if valuation["roe"] is not None:
        signals.append(f"return on equity {valuation['roe']:.1%}")
    if valuation["debt_to_equity"] is not None:
        signals.append(f"debt/equity {valuation['debt_to_equity']:.2f}")

    signals.append(
        f"news sentiment {sentiment['verdict']} "
        f"(mean compound {sentiment['mean_compound']:+.2f})"
    )
    return signals


def _tally(summary: dict) -> dict:
    bullish = bearish = 0
    trend = summary["trend"]
    momentum = summary["momentum"]
    sentiment = summary["sentiment"]
    pe = summary["valuation"]["pe_ratio"]

    if trend["regime"] == "uptrend":
        bullish += 1
    elif trend["regime"] == "downtrend":
        bearish += 1

    if momentum["rsi_zone"] == "oversold":
        bullish += 1
    elif momentum["rsi_zone"] == "overbought":
        bearish += 1

    if momentum["macd_cross"] == "bullish":
        bullish += 1
    elif momentum["macd_cross"] == "bearish":
        bearish += 1

    if sentiment["verdict"] == "bullish":
        bullish += 1
    elif sentiment["verdict"] == "bearish":
        bearish += 1

    if pe is not None:
        if pe < PE_CHEAP:
            bullish += 1
        elif pe > PE_EXPENSIVE:
            bearish += 1

    if bullish > bearish:
        verdict = "bullish"
    elif bearish > bullish:
        verdict = "bearish"
    else:
        verdict = "neutral"
    return {"bullish": bullish, "bearish": bearish, "verdict": verdict}


def _pe_word(pe: float) -> str:
    if pe < PE_CHEAP:
        return "cheap"
    if pe > PE_EXPENSIVE:
        return "expensive"
    return "fair"


def _regime(last_close: float, sma50: float | None, sma200: float | None) -> str:
    if sma50 is None or sma200 is None:
        return "mixed"
    if last_close > sma50 and sma50 > sma200:
        return "uptrend"
    if last_close < sma50 and sma50 < sma200:
        return "downtrend"
    return "mixed"


def _last_cross(spread: pd.Series, up_cross: pd.Series, down_cross: pd.Series):
    """Most recent 50/200 SMA crossover anywhere in the series."""
    if spread.empty:
        return None, None
    crosses = pd.concat(
        [up_cross.rename("golden"), down_cross.rename("death")], axis=1
    ).dropna(how="all")
    hits = crosses[(crosses["golden"]) | (crosses["death"])]
    if hits.empty:
        return None, None
    position = spread.index.get_loc(hits.index[-1])
    direction = "golden" if bool(hits["golden"].iloc[-1]) else "death"
    return direction, int(len(spread) - 1 - position)


def _relation(value: float | None, reference: float | None) -> str | None:
    if value is None or reference is None:
        return None
    return "above" if value > reference else "below"


def _last(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _cross_text(trend: dict) -> str:
    if trend["golden_cross"]:
        return f"golden cross {trend['last_cross_bars_ago']} bars ago"
    if trend["death_cross"]:
        return f"death cross {trend['last_cross_bars_ago']} bars ago"
    if trend["last_cross"]:
        return (
            f"{trend['last_cross']} cross {trend['last_cross_bars_ago']} bars ago "
            f"(older than the last {CROSS_WINDOW} bars)"
        )
    return "none in the available history"


def _macd_text(momentum: dict) -> str:
    if momentum["macd_cross"]:
        return f"{momentum['macd_cross']} crossover within the last {CROSS_WINDOW} bars"
    if momentum["macd_histogram"] is None:
        return "unavailable"
    direction = "above" if momentum["macd_histogram"] > 0 else "below"
    return f"MACD line {direction} its signal line, no fresh crossover"


def _money(value: float | None) -> str:
    return "unavailable" if value is None else f"${value:,.2f}"


def _number(value: float | None, digits: int) -> str:
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1%}"


def _value_or_na(value: str | None) -> str:
    return value if value else "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Descriptive stock report (educational, not financial advice)."
    )
    parser.add_argument("ticker", nargs="?", default=DEFAULT_TICKER)
    parser.add_argument("--period", default="1y", help="history window, e.g. 1y, 6mo")
    parser.add_argument("--out", type=Path, help="also write the markdown to this path")
    args = parser.parse_args(argv)

    report = format_report(summarize(args.ticker, period=args.period))
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
