"""Technical indicators and risk statistics, hand-computed with pandas.

Every function here is a pure function of price series — no network, no files,
no hidden state — so the maths can be read, tested, and trusted. The formulas
are the textbook ones; each docstring states them explicitly.

Educational use only — not financial advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average: the mean of the last ``window`` closes."""
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average: weights decay with ``alpha = 2 / (span + 1)``.

    ``adjust=False`` keeps the recursive definition (``ema_t = a*x_t + (1-a)*ema_{t-1}``)
    so the value at each bar is the same in a chart and in a trading rule.
    """
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing), in the range 0-100.

    ``RSI = 100 - 100 / (1 + avg_gain / avg_loss)`` where gains and losses come
    from the bar-to-bar change and are smoothed with ``alpha = 1 / period``.
    A pure uptrend (``avg_loss == 0``) is pushed to ~100 instead of NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # Zero average loss => infinite relative strength => RSI 100.
    safe_loss = avg_loss.replace(0.0, np.finfo(float).eps)
    rs = avg_gain / safe_loss
    return 100 - 100 / (1 + rs)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Columns are exactly ``["macd", "signal", "hist"]``:
    ``macd = ema(fast) - ema(slow)``, ``signal = ema(macd, signal)``,
    ``hist = macd - signal``.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands.

    Columns are exactly ``["middle", "upper", "lower"]``:
    ``middle = sma(window)``, ``upper/lower = middle +/- num_std * rolling std``.
    """
    middle = sma(close, window)
    std = close.rolling(window).std()
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + num_std * std,
            "lower": middle - num_std * std,
        }
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing of the true range).

    ``TR = max(high - low, |high - prev_close|, |low - prev_close|)``.
    """
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple percentage change from one bar to the next."""
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    """Continuously compounded return: ``ln(price_t / price_{t-1})``."""
    return np.log(close / close.shift(1))


def annualized_volatility(close: pd.Series, periods_per_year: int = 252) -> float:
    """Standard deviation of returns scaled to a yearly figure."""
    return float(daily_returns(close).std() * np.sqrt(periods_per_year))


def max_drawdown(close: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction (e.g. ``-0.31``)."""
    return float((close / close.cummax() - 1).min())


def sharpe_ratio(close: pd.Series, periods_per_year: int = 252) -> float:
    """Return per unit of risk: ``mean(returns) / std(returns) * sqrt(periods)``.

    Simplification: the risk-free rate is taken as zero, so this is a
    risk-adjusted return measure rather than a precise Sharpe ratio.
    """
    returns = daily_returns(close)
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
