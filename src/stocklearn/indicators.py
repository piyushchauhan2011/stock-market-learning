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


FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
"""Retracement ratios of a swing: 0.0 sits at the high, 1.0 at the low."""

FIB_EXT_RATIOS = (1.0, 1.272, 1.618, 2.0, 2.618)
"""Extension ratios: 1.0 sits at the high, 1.618 at ``high + 0.618 * span``."""


def swing_highs_lows(close: pd.Series, window: int = 5) -> pd.DataFrame:
    """Local swing highs and lows of ``close``, one bar per confirmed extreme.

    A bar is a swing high when its close is the *strict* maximum of the
    ``window`` bars on either side of it (``close[t-window : t+window+1]``, so
    ``2*window + 1`` bars in total); a swing low is the strict minimum the same
    way. "Strict" matters: a value that ties for the extreme is not flagged, so
    a flat stretch produces no swing at all.

    The last ``window`` bars can never qualify — they are not yet surrounded by
    enough data to confirm a turning point — which is why a swing is a
    *lagging* label, not a live signal.

    Columns are exactly ``["high", "low"]``; every non-swing bar is ``NaN``.
    """
    values = close.to_numpy(dtype=float)
    highs = np.full(len(values), np.nan)
    lows = np.full(len(values), np.nan)

    for t in range(window, len(values) - window):
        segment = values[t - window : t + window + 1]
        if values[t] == segment.max() and (segment == values[t]).sum() == 1:
            highs[t] = values[t]
        if values[t] == segment.min() and (segment == values[t]).sum() == 1:
            lows[t] = values[t]

    return pd.DataFrame(
        {"high": highs, "low": lows}, index=close.index, dtype=float
    )


def recent_swing(close: pd.Series, window: int = 5) -> dict:
    """The most recent completed up-leg of ``close``: a low followed by a high.

    Returns ``{"low", "high", "low_bar", "high_bar"}`` where ``high_bar`` is the
    position of the *last* swing high and ``low_bar`` the position of the last
    swing low that came before it. Those two prices are the anchors Fibonacci
    retracement is measured between.

    ``low_bar``/``high_bar`` are 0-based positions into ``close`` (use them with
    ``.iloc``), not index labels. When no swing high exists, or none has a swing
    low before it, the dict is ``{"low": nan, "high": nan, "low_bar": -1,
    "high_bar": -1}`` — the caller decides what to do with a missing leg.
    """
    swings = swing_highs_lows(close, window)

    high_positions = np.flatnonzero(swings["high"].notna().to_numpy())
    low_positions = np.flatnonzero(swings["low"].notna().to_numpy())
    no_leg = {"low": np.nan, "high": np.nan, "low_bar": -1, "high_bar": -1}
    if high_positions.size == 0:
        return no_leg

    high_bar = int(high_positions[-1])
    earlier_lows = low_positions[low_positions < high_bar]
    if earlier_lows.size == 0:
        return no_leg

    low_bar = int(earlier_lows[-1])
    return {
        "low": float(close.iloc[low_bar]),
        "high": float(close.iloc[high_bar]),
        "low_bar": low_bar,
        "high_bar": high_bar,
    }


def fibonacci_levels(
    low: float, high: float, ratios: tuple[float, ...] = FIB_RATIOS
) -> dict[str, float]:
    """Retracement levels inside an up-leg from ``low`` to ``high``.

    ``level = high - (high - low) * ratio``, so ratio ``0.0`` is the high,
    ``0.5`` the midpoint and ``1.0`` the low. Keys are the ratios as strings,
    e.g. ``levels["0.618"]``. ``NaN`` inputs give ``NaN`` levels.
    """
    span = high - low
    return {str(ratio): high - span * ratio for ratio in ratios}


def fibonacci_extensions(
    low: float, high: float, ratios: tuple[float, ...] = FIB_EXT_RATIOS
) -> dict[str, float]:
    """Extension levels *beyond* the high of an up-leg from ``low`` to ``high``.

    ``level = high + (high - low) * (ratio - 1)``, so ratio ``1.0`` is the high
    itself and ``1.618`` is one golden-ratio span above it. Keys are the ratios
    as strings, e.g. ``extensions["1.618"]``. ``NaN`` inputs give ``NaN`` levels.
    """
    span = high - low
    return {str(ratio): high + span * (ratio - 1.0) for ratio in ratios}


def pivot_points(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    """Classic floor pivots, computed from the *previous* bar's H/L/C.

    ``P = (H + L + C) / 3`` then ``R1 = 2P - L``, ``S1 = 2P - H``,
    ``R2 = P + (H - L)``, ``S2 = P - (H - L)``, ``R3 = H + 2(P - L)`` and
    ``S3 = L - 2(H - P)``. (R3/S3 are mirror images of each other around P.)

    Columns are exactly ``["pivot", "r1", "r2", "r3", "s1", "s2", "s3"]``,
    indexed like ``close``. Row 0 is all ``NaN``: there is no previous bar to
    compute from, and today's levels must use what was known *before* today's
    open. For any later row, ``r1 > pivot > s1``.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    pivot = (prev_high + prev_low + prev_close) / 3
    spread = prev_high - prev_low

    return pd.DataFrame(
        {
            "pivot": pivot,
            "r1": 2 * pivot - prev_low,
            "r2": pivot + spread,
            "r3": prev_high + 2 * (pivot - prev_low),
            "s1": 2 * pivot - prev_high,
            "s2": pivot - spread,
            "s3": prev_low - 2 * (prev_high - pivot),
        }
    )
