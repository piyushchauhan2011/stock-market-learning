"""Order model: what a trader wants to do, before any price is known.

An order is a *request*. It carries the intent (side, quantity, type, optional
take-profit / stop-loss bracket) and the lifecycle state that a broker updates
as bars arrive: ``open`` → ``filled`` (or ``canceled`` / ``rejected``).

The model here is deliberately long-only: a SELL order reduces or closes an
existing position, and brackets attach to entries only.

Educational use only — not financial advice.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from stocklearn.data import DEFAULT_TICKER

_ORDER_IDS = itertools.count(1)
"""Process-wide counter, so every order gets a stable, readable id."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Side(str, Enum):
    """Which way the order moves the position."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """How the order decides its fill price."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    """Where the order sits in its lifecycle."""

    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Order:
    """A single order request.

    Parameters
    ----------
    side:
        :class:`Side` — BUY opens/adds, SELL reduces/closes.
    quantity:
        Whole shares, must be a positive ``int``.
    symbol:
        Yahoo-style ticker. Defaults to :data:`stocklearn.data.DEFAULT_TICKER`
        (the factory helpers below pass it explicitly).
    order_type:
        :class:`OrderType` — MARKET, LIMIT or STOP.
    limit_price, stop_price:
        Required for the matching ``order_type``; must be positive.
    take_profit, stop_loss:
        Optional BUY-only bracket. Both are absolute prices: take-profit above
        the entry, stop-loss below it. They travel with the order and are
        attached to the resulting position when it fills.
    status, reject_reason, order_id, created_at, filled_at, fill_price:
        Lifecycle fields. A broker fills them in; construct-and-forget code only
        needs the first group.

    Raises
    ------
    ValueError
        If ``quantity`` is not a positive ``int``, if a LIMIT order lacks
        ``limit_price``, if a STOP order lacks ``stop_price``, if a SELL order
        tries to carry a bracket, or if a bracket has
        ``take_profit <= stop_loss``.
    """

    side: Side
    quantity: int
    symbol: str = DEFAULT_TICKER
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    status: OrderStatus = OrderStatus.OPEN
    reject_reason: str | None = None
    order_id: int = field(default_factory=lambda: next(_ORDER_IDS))
    created_at: datetime = field(default_factory=_utc_now)
    filled_at: datetime | None = None
    fill_price: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise ValueError(
                f"quantity must be a whole number of shares, got {self.quantity!r}"
            )
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")

        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required when order_type is LIMIT")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("stop_price is required when order_type is STOP")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError(f"limit_price must be > 0, got {self.limit_price}")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError(f"stop_price must be > 0, got {self.stop_price}")

        if self.side is Side.SELL and (
            self.take_profit is not None or self.stop_loss is not None
        ):
            raise ValueError(
                "take_profit / stop_loss only attach to BUY orders (entries)"
            )
        if (
            self.side is Side.BUY
            and self.take_profit is not None
            and self.stop_loss is not None
            and self.take_profit <= self.stop_loss
        ):
            raise ValueError(
                "take_profit must be above stop_loss, "
                f"got take_profit={self.take_profit} stop_loss={self.stop_loss}"
            )

    def __str__(self) -> str:  # pragma: no cover - display helper
        extras = ""
        if self.limit_price is not None:
            extras += f" @ limit {self.limit_price}"
        if self.stop_price is not None:
            extras += f" @ stop {self.stop_price}"
        if self.take_profit is not None or self.stop_loss is not None:
            extras += f" (TP {self.take_profit} / SL {self.stop_loss})"
        return (
            f"#{self.order_id} {self.side.value} {self.quantity} {self.symbol} "
            f"{self.order_type.value}{extras} [{self.status.value}]"
        )


def market_buy(
    quantity: int,
    symbol: str = DEFAULT_TICKER,
    *,
    take_profit: float | None = None,
    stop_loss: float | None = None,
) -> Order:
    """Buy ``quantity`` shares at whatever the next bar opens at."""
    return Order(
        side=Side.BUY,
        quantity=quantity,
        symbol=symbol,
        order_type=OrderType.MARKET,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )


def market_sell(quantity: int, symbol: str = DEFAULT_TICKER) -> Order:
    """Sell ``quantity`` shares at whatever the next bar opens at."""
    return Order(side=Side.SELL, quantity=quantity, symbol=symbol)


def limit_buy(
    quantity: int,
    price: float,
    symbol: str = DEFAULT_TICKER,
    *,
    take_profit: float | None = None,
    stop_loss: float | None = None,
) -> Order:
    """Buy ``quantity`` shares, but never above ``price``."""
    return Order(
        side=Side.BUY,
        quantity=quantity,
        symbol=symbol,
        order_type=OrderType.LIMIT,
        limit_price=price,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )


def limit_sell(quantity: int, price: float, symbol: str = DEFAULT_TICKER) -> Order:
    """Sell ``quantity`` shares, but never below ``price``."""
    return Order(
        side=Side.SELL,
        quantity=quantity,
        symbol=symbol,
        order_type=OrderType.LIMIT,
        limit_price=price,
    )


def stop_buy(
    quantity: int,
    price: float,
    symbol: str = DEFAULT_TICKER,
    *,
    take_profit: float | None = None,
    stop_loss: float | None = None,
) -> Order:
    """Buy ``quantity`` shares once the price trades up through ``price``."""
    return Order(
        side=Side.BUY,
        quantity=quantity,
        symbol=symbol,
        order_type=OrderType.STOP,
        stop_price=price,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )


def stop_sell(quantity: int, price: float, symbol: str = DEFAULT_TICKER) -> Order:
    """Sell ``quantity`` shares once the price trades down through ``price``."""
    return Order(
        side=Side.SELL,
        quantity=quantity,
        symbol=symbol,
        order_type=OrderType.STOP,
        stop_price=price,
    )
