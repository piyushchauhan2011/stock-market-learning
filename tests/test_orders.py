"""Order-model tests: construction, validation, and the factory helpers.

Every expectation here is read straight off the :class:`~stocklearn.orders.Order`
definition — these pin the contract the broker and the notebooks rely on.
"""

from __future__ import annotations

import pytest

from stocklearn.data import DEFAULT_TICKER
from stocklearn.orders import (
    Order,
    OrderStatus,
    OrderType,
    Side,
    limit_buy,
    market_buy,
    market_sell,
    stop_sell,
)


def test_market_buy_returns_an_open_buy_order():
    order = market_buy(10)

    assert order.side is Side.BUY
    assert order.order_type is OrderType.MARKET
    assert order.quantity == 10
    assert order.status is OrderStatus.OPEN
    assert order.symbol == DEFAULT_TICKER
    assert order.fill_price is None
    assert order.filled_at is None
    assert order.created_at.tzinfo is not None


def test_order_ids_are_unique_and_increasing():
    first = market_buy(1)
    second = market_sell(1)

    assert second.order_id > first.order_id


def test_zero_or_fractional_quantity_is_rejected():
    with pytest.raises(ValueError, match="quantity"):
        Order(side=Side.BUY, quantity=0)

    with pytest.raises(ValueError, match="quantity"):
        Order(side=Side.BUY, quantity=2.5)


def test_limit_order_requires_a_limit_price():
    with pytest.raises(ValueError, match="limit_price"):
        Order(side=Side.BUY, quantity=10, order_type=OrderType.LIMIT)


def test_stop_order_requires_a_stop_price():
    with pytest.raises(ValueError, match="stop_price"):
        Order(side=Side.BUY, quantity=10, order_type=OrderType.STOP)


def test_bracket_must_sit_the_right_way_round():
    order = market_buy(10, take_profit=110, stop_loss=95)
    assert (order.take_profit, order.stop_loss) == (110, 95)

    with pytest.raises(ValueError, match="take_profit must be above stop_loss"):
        market_buy(10, take_profit=90, stop_loss=95)


def test_brackets_cannot_attach_to_a_sell():
    # The sell factories take no bracket arguments at all; the guard itself
    # lives on Order, so a hand-built sell with a bracket is still refused.
    with pytest.raises(ValueError, match="BUY orders"):
        Order(side=Side.SELL, quantity=10, take_profit=110)

    with pytest.raises(ValueError, match="BUY orders"):
        Order(side=Side.SELL, quantity=10, stop_loss=90)


def test_factory_helpers_carry_their_trigger_prices():
    limit = limit_buy(5, 99.5)
    stop = stop_sell(5, 88.0)

    assert limit.order_type is OrderType.LIMIT
    assert limit.limit_price == 99.5
    assert limit.side is Side.BUY

    assert stop.order_type is OrderType.STOP
    assert stop.stop_price == 88.0
    assert stop.side is Side.SELL
