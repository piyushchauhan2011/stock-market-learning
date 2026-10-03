"""The strategy interface: one function from market state to orders.

A strategy is deliberately tiny. It sees what a trader sees at a given moment —
the closes up to and including the current bar, the cash, the current position —
and it answers one question: *what orders do I want now?* It never sees the
future, and it never talks to a broker directly; the backtester sits in between.

Every strategy here is long-only, uses adjusted closes only for signals, and
returns ``[]`` while its indicators are still warming up (NaN), so a too-short
history produces no trades instead of nonsense.

Educational use only — not financial advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from stocklearn.broker import Bar
from stocklearn.indicators import bollinger, macd, rsi, sma
from stocklearn.orders import Order, market_buy, market_sell

ALL_IN_BUFFER = 0.98
"""Fraction of cash an all-in entry is allowed to spend.

The signal is computed on a bar's close, so the fill happens at the *next* bar's
open, which can gap away from the signal price. Keeping 2% back means a
just-affordable all-in entry still clears its cash check after a small gap-up
instead of being rejected.
"""


@dataclass
class Context:
    """Everything a strategy is allowed to know on one bar.

    Attributes
    ----------
    close:
        Adjusted closes through the **current** bar, inclusive. ``close.iloc[-1]``
        is the price that just finished printing — the only price a signal may use.
    cash:
        Spendable cash in the account.
    position_qty:
        Shares currently held; ``0`` when flat.
    timestamp:
        Timestamp of the current bar.
    bar:
        The current bar's OHLC. Present for context and teaching; a signal that
        READs ``bar.high`` or ``bar.low`` is using information it could not have
        had until the bar closed, so the built-ins below use ``close`` only.
    """

    close: pd.Series
    cash: float
    position_qty: int
    timestamp: pd.Timestamp
    bar: Bar


class Strategy(Protocol):
    """Anything with ``on_bar(ctx) -> list[Order]`` is a strategy."""

    def on_bar(self, ctx: Context) -> list[Order]:
        """Return the orders to place *now*; they fill on the next bar."""
        ...  # pragma: no cover - protocol definition


def entry_quantity(ctx: Context, qty: int | None) -> int:
    """Shares to buy: ``qty`` verbatim, or an all-in size when ``qty`` is ``None``.

    The all-in size spends at most :data:`ALL_IN_BUFFER` of the available cash and
    always asks for at least one share — if one share is unaffordable the broker
    rejects the order, which is the honest outcome, not a silent no-op.
    """
    if qty is not None:
        return int(qty)
    price = float(ctx.close.iloc[-1])
    if price <= 0:
        return 1
    return max(1, math.floor(ctx.cash * ALL_IN_BUFFER / price))


def _last(value: pd.Series) -> float | None:
    """Latest value of an indicator, or ``None`` while it is still NaN."""
    if len(value) == 0:
        return None
    latest = value.iloc[-1]
    if pd.isna(latest):
        return None
    return float(latest)


class BuyAndHold:
    """Buy once with everything you have, then never sell.

    The control case for every backtest: if a strategy cannot beat this after
    costs, it is not adding anything.
    """

    name = "Buy and hold"

    def __init__(self, qty: int | None = None) -> None:
        self.qty = qty

    def on_bar(self, ctx: Context) -> list[Order]:
        if ctx.position_qty > 0:
            return []
        return [market_buy(entry_quantity(ctx, self.qty))]


class SmaCross:
    """Classic trend following: buy when the fast SMA crosses above the slow one.

    Exits on the opposite cross. Both crossovers are *events*, not states: the
    strategy must see ``fast`` on the wrong side of ``slow`` on the previous bar
    and the right side on the current one. Staying above does not repeat the buy.
    """

    name = "SMA crossover"

    def __init__(self, fast: int = 50, slow: int = 200, qty: int | None = None) -> None:
        if fast >= slow:
            raise ValueError(f"fast window must be < slow window, got {fast} >= {slow}")
        self.fast = fast
        self.slow = slow
        self.qty = qty

    def on_bar(self, ctx: Context) -> list[Order]:
        fast = sma(ctx.close, self.fast)
        slow = sma(ctx.close, self.slow)
        if len(ctx.close) < 2:
            return []
        fast_now, fast_prev = _last(fast), _last(fast.iloc[:-1])
        slow_now, slow_prev = _last(slow), _last(slow.iloc[:-1])
        if None in (fast_now, fast_prev, slow_now, slow_prev):
            return []

        if fast_now > slow_now and fast_prev <= slow_prev and ctx.position_qty == 0:
            return [market_buy(entry_quantity(ctx, self.qty))]
        if fast_now < slow_now and fast_prev >= slow_prev and ctx.position_qty > 0:
            return [market_sell(ctx.position_qty)]
        return []


class RsiReversion:
    """Buy oversold, sell overbought.

    Betting against the move: RSI below ``buy_below`` means the recent selling
    looks overdone, above ``sell_above`` means the rally looks stretched. It
    tends to work in ranges and to get run over in a persistent trend.
    """

    name = "RSI reversion"

    def __init__(
        self,
        period: int = 14,
        buy_below: float = 30.0,
        sell_above: float = 70.0,
        qty: int | None = None,
    ) -> None:
        if buy_below >= sell_above:
            raise ValueError(
                f"buy_below must be < sell_above, got {buy_below} >= {sell_above}"
            )
        self.period = period
        self.buy_below = buy_below
        self.sell_above = sell_above
        self.qty = qty

    def on_bar(self, ctx: Context) -> list[Order]:
        value = _last(rsi(ctx.close, self.period))
        if value is None:
            return []
        if ctx.position_qty == 0 and value < self.buy_below:
            return [market_buy(entry_quantity(ctx, self.qty))]
        if ctx.position_qty > 0 and value > self.sell_above:
            return [market_sell(ctx.position_qty)]
        return []


class MacdCross:
    """Momentum crossover on the MACD histogram (MACD line minus its signal).

    The histogram crossing zero *is* the MACD line crossing its signal line, in
    one series instead of two.
    """

    name = "MACD crossover"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        qty: int | None = None,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.qty = qty

    def on_bar(self, ctx: Context) -> list[Order]:
        hist = macd(ctx.close, self.fast, self.slow, self.signal)["hist"]
        if len(hist) < 2:
            return []
        now = _last(hist)
        prev = _last(hist.iloc[:-1])
        if now is None or prev is None:
            return []

        if now > 0 and prev <= 0 and ctx.position_qty == 0:
            return [market_buy(entry_quantity(ctx, self.qty))]
        if now < 0 and prev >= 0 and ctx.position_qty > 0:
            return [market_sell(ctx.position_qty)]
        return []


class BollingerBreakout:
    """Buy a close above the upper band, exit on a close below the lower band.

    The opposite temperament to :class:`RsiReversion`: here strength is the
    signal. In a trend it rides; in a chop it buys high and sells low.
    """

    name = "Bollinger breakout"

    def __init__(
        self, window: int = 20, num_std: float = 2.0, qty: int | None = None
    ) -> None:
        self.window = window
        self.num_std = num_std
        self.qty = qty

    def on_bar(self, ctx: Context) -> list[Order]:
        bands = bollinger(ctx.close, self.window, self.num_std)
        close = _last(ctx.close)
        upper = _last(bands["upper"])
        lower = _last(bands["lower"])
        if None in (close, upper, lower):
            return []

        if ctx.position_qty == 0 and close > upper:
            return [market_buy(entry_quantity(ctx, self.qty))]
        if ctx.position_qty > 0 and close < lower:
            return [market_sell(ctx.position_qty)]
        return []
