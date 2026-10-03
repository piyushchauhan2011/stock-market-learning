"""A paper broker: bars in, positions/fills/P&L out. No network, no API keys.

The broker is the piece that turns *intent* (an :class:`~stocklearn.orders.Order`)
into *reality* (a fill price, a position, a realised trade). It is bar-driven: you
hand it one :class:`Bar` at a time with :meth:`PaperBroker.next_bar`, and it

1. fills whatever open orders that bar's price path allows,
2. closes positions whose take-profit / stop-loss bracket was hit,
3. marks the account to market on the bar's close.

That is exactly the sequence a real broker runs for you, just with the queue
made visible. Simplifications, all deliberate: long-only (a SELL reduces or
closes), whole shares, a flat per-fill ``commission`` (default ``0.0``), and no
slippage — fills happen at the bar's own prices, which is optimistic for
anything but a tiny size.

Educational use only — not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from stocklearn.orders import Order, OrderStatus, OrderType, Side


@dataclass(frozen=True)
class Bar:
    """One OHLC bar.

    Prices are expected on the *adjusted* scale (see :func:`bars_from_history`)
    so that signal prices and fill prices are directly comparable.
    """

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float


def bars_from_history(df: pd.DataFrame, adjusted: bool = True) -> list[Bar]:
    """Convert an OHLCV frame from :func:`stocklearn.data.get_history` to bars.

    When ``adjusted`` is true and both ``Adj Close`` and ``Close`` exist, every
    price in a row is multiplied by that row's ``Adj Close / Close`` factor. That
    puts the whole bar on the adjusted scale, so an indicator computed on
    ``price_series(df)`` (which is ``Adj Close``) and a fill computed from these
    bars are talking about the same prices. Missing factors fall back to 1.0 so a
    data gap cannot poison a whole history.

    Parameters
    ----------
    df:
        Frame indexed by timestamp with ``Open, High, Low, Close`` and,
        preferably, ``Adj Close``.
    adjusted:
        Set ``False`` to keep the raw, as-traded prices.

    Returns
    -------
    list[Bar]
        One bar per row, oldest first.
    """
    if df.empty:
        return []

    if adjusted and "Adj Close" in df.columns and "Close" in df.columns:
        factor = (df["Adj Close"] / df["Close"]).fillna(1.0)
    else:
        factor = pd.Series(1.0, index=df.index)

    ohlc = df[["Open", "High", "Low", "Close"]].astype(float)
    bars: list[Bar] = []
    for timestamp, open_, high, low, close, scale in zip(
        ohlc.index,
        ohlc["Open"].to_numpy(),
        ohlc["High"].to_numpy(),
        ohlc["Low"].to_numpy(),
        ohlc["Close"].to_numpy(),
        factor.to_numpy(dtype=float),
    ):
        bars.append(
            Bar(
                pd.Timestamp(timestamp),
                float(open_) * float(scale),
                float(high) * float(scale),
                float(low) * float(scale),
                float(close) * float(scale),
            )
        )
    return bars


@dataclass
class Position:
    """An open long position, plus the bracket that is watching it."""

    symbol: str
    quantity: int
    avg_price: float
    take_profit: float | None = None
    stop_loss: float | None = None
    opened_at: pd.Timestamp | None = None


@dataclass
class Trade:
    """A closed round trip: what was paid, what was received, what was left."""

    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float

    @property
    def return_pct(self) -> float:
        """P&L as a fraction of the entry notional (commissions included in P&L)."""
        notional = self.entry_price * self.quantity
        return self.pnl / notional if notional else 0.0


class PaperBroker:
    """An account that fills orders against bars you feed it.

    Parameters
    ----------
    initial_cash:
        Starting cash. Long entries are checked against it at fill time.
    commission:
        Flat fee charged per fill (each buy, each sell). Default ``0.0``.

    Attributes
    ----------
    cash:
        Spendable cash, updated by every fill.
    positions:
        ``{symbol: Position}`` — only symbols with a non-zero quantity appear.
    pending:
        Orders that are still ``OPEN`` and waiting for a bar.
    trades:
        Closed :class:`Trade` objects, oldest first.
    equity_curve:
        ``(timestamp, equity)`` appended on every :meth:`next_bar`.

    Notes
    -----
    A :class:`Bar` carries no symbol: one broker instance replays **one**
    instrument. Every position is therefore marked at the latest bar's close.
    """

    def __init__(self, initial_cash: float = 10_000.0, commission: float = 0.0) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.commission = float(commission)
        self.positions: dict[str, Position] = {}
        self.pending: list[Order] = []
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self._last_close: float | None = None

    # ------------------------------------------------------------------ orders

    def place_order(self, order: Order) -> Order:
        """Queue ``order`` for the next :meth:`next_bar`.

        Nothing is validated here beyond the structural checks the
        :class:`~stocklearn.orders.Order` already ran: whether the account can
        *afford* the trade depends on the price, and that is only known when the
        bar arrives. A too-expensive or over-sized order is rejected at fill
        time with a ``reject_reason``.
        """
        self.pending.append(order)
        return order

    def cancel(self, order_id: int) -> bool:
        """Cancel an open order by id.

        Returns ``True`` if it was still open and is now ``CANCELED``, ``False``
        if the id is unknown or the order already reached a terminal state.
        """
        for order in self.pending:
            if order.order_id == order_id and order.status is OrderStatus.OPEN:
                order.status = OrderStatus.CANCELED
                self.pending = [o for o in self.pending if o is not order]
                return True
        return False

    # ------------------------------------------------------------------ engine

    def next_bar(self, bar: Bar) -> None:
        """Advance the clock by one bar: fill, bracket-check, mark to market."""
        self._last_close = float(bar.close)

        for order in list(self.pending):
            if order.status is not OrderStatus.OPEN:
                continue
            fill_price = _fill_price(order, bar)
            if fill_price is None:
                continue
            self._fill(order, bar, fill_price)

        self.pending = [o for o in self.pending if o.status is OrderStatus.OPEN]

        for position in list(self.positions.values()):
            exit_price = _bracket_exit(position, bar)
            if exit_price is not None:
                self._close(position, bar, exit_price)

        self.equity_curve.append((bar.timestamp, self.equity))

    # ------------------------------------------------------------------ fills

    def _fill(self, order: Order, bar: Bar, fill_price: float) -> None:
        if order.side is Side.BUY:
            self._fill_buy(order, bar, fill_price)
        else:
            self._fill_sell(order, bar, fill_price)

    def _fill_buy(self, order: Order, bar: Bar, fill_price: float) -> None:
        cost = fill_price * order.quantity + self.commission
        if self.cash < cost:
            self._reject(order, "insufficient cash")
            return

        position = self.positions.get(order.symbol)
        if position is None:
            position = Position(symbol=order.symbol, quantity=0, avg_price=0.0)
            self.positions[order.symbol] = position

        if position.quantity == 0:
            position.opened_at = bar.timestamp

        total_cost = position.quantity * position.avg_price + order.quantity * fill_price
        position.quantity += order.quantity
        position.avg_price = total_cost / position.quantity

        # A bracket on the order replaces the position's bracket; omitting it
        # leaves whatever the position already had.
        if order.take_profit is not None:
            position.take_profit = order.take_profit
        if order.stop_loss is not None:
            position.stop_loss = order.stop_loss

        self.cash -= cost
        self._mark_filled(order, bar, fill_price)

    def _fill_sell(self, order: Order, bar: Bar, fill_price: float) -> None:
        position = self.positions.get(order.symbol)
        if position is None or position.quantity < order.quantity:
            self._reject(order, "insufficient position")
            return

        self._close(position, bar, fill_price, quantity=order.quantity)
        self._mark_filled(order, bar, fill_price)

    def _close(
        self,
        position: Position,
        bar: Bar,
        exit_price: float,
        quantity: int | None = None,
    ) -> Trade:
        """Book a sale out of ``position``; removes it when it hits zero."""
        quantity = position.quantity if quantity is None else quantity
        pnl = (exit_price - position.avg_price) * quantity - self.commission

        self.cash += exit_price * quantity - self.commission
        trade = Trade(
            symbol=position.symbol,
            entry_time=position.opened_at,
            exit_time=bar.timestamp,
            entry_price=position.avg_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
        )
        self.trades.append(trade)

        position.quantity -= quantity
        if position.quantity == 0:
            del self.positions[position.symbol]
        return trade

    @staticmethod
    def _mark_filled(order: Order, bar: Bar, fill_price: float) -> None:
        order.status = OrderStatus.FILLED
        order.fill_price = float(fill_price)
        order.filled_at = _utc_now()

    @staticmethod
    def _reject(order: Order, reason: str) -> None:
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason

    # ------------------------------------------------------------------ state

    @property
    def equity(self) -> float:
        """Cash plus every position marked at the latest bar's close."""
        close = self._last_close
        if close is None:
            return self.cash
        return self.cash + sum(p.quantity * close for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Sum of realised P&L over closed trades (open positions excluded)."""
        return sum(t.pnl for t in self.trades)


def _fill_price(order: Order, bar: Bar) -> float | None:
    """Price the order would fill at on ``bar``, or ``None`` if it would not.

    MARKET orders always fill, at the bar's open. LIMIT and STOP orders only
    fill if the bar traded through their trigger, and then get the *better* of
    the trigger and the open — which is how gaps help an entry and hurt an exit.
    """
    if order.order_type is OrderType.MARKET:
        return float(bar.open)

    if order.order_type is OrderType.LIMIT:
        limit = float(order.limit_price)
        if order.side is Side.BUY:
            if bar.low <= limit:
                return float(bar.open) if bar.open <= limit else limit
            return None
        if bar.high >= limit:
            return float(bar.open) if bar.open >= limit else limit
        return None

    stop = float(order.stop_price)
    if order.side is Side.BUY:
        if bar.high >= stop:
            return float(bar.open) if bar.open >= stop else stop
        return None
    if bar.low <= stop:
        return float(bar.open) if bar.open <= stop else stop
    return None


def _bracket_exit(position: Position, bar: Bar) -> float | None:
    """Exit price if a take-profit / stop-loss bracket fired on ``bar``.

    The stop-loss is checked first: when a single bar contains both triggers we
    cannot know which came first, so we assume the worse outcome. A gap below
    the stop fills at the open (worse than the stop); a gap above the
    take-profit fills at the open (better than the target).
    """
    if position.stop_loss is not None and bar.low <= position.stop_loss:
        return float(bar.open) if bar.open <= position.stop_loss else position.stop_loss
    if position.take_profit is not None and bar.high >= position.take_profit:
        return (
            float(bar.open) if bar.open >= position.take_profit else position.take_profit
        )
    return None


def _utc_now() -> datetime:
    """Wall-clock stamp for ``filled_at`` — when the broker acted, not the bar."""
    return datetime.now(timezone.utc)
