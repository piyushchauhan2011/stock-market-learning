"""Broker tests: fill rules, bracket priority, rejections, and P&L arithmetic.

Bars are built inline so every number below can be checked by hand; no
DataFrame, no network. ``AAA``/``DEFAULT_TICKER`` symbols are immaterial here —
the broker only needs the symbol on the position it opens.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stocklearn.broker import Bar, PaperBroker, bars_from_history
from stocklearn.orders import OrderStatus, limit_buy, market_buy, market_sell, stop_sell


def bar(open_: float, high: float, low: float, close: float, day: int = 1) -> Bar:
    return Bar(pd.Timestamp(f"2024-01-{day:02d}"), open_, high, low, close)


def test_market_buy_fills_at_the_next_open():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(market_buy(10))

    broker.next_bar(bar(100, 105, 99, 102))

    position = broker.positions["AAPL"]
    assert position.quantity == 10
    assert position.avg_price == pytest.approx(100.0)
    assert broker.cash == pytest.approx(9_000.0)
    assert position.opened_at == pd.Timestamp("2024-01-01")


def test_market_order_waits_for_the_next_bar():
    broker = PaperBroker(initial_cash=10_000)
    order = broker.place_order(market_buy(10))

    assert order.status is OrderStatus.OPEN
    assert broker.cash == pytest.approx(10_000.0)
    assert broker.positions == {}


def test_limit_buy_takes_the_better_price_when_the_bar_gaps_below():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(limit_buy(10, 100))

    # Open 98 is already below the limit, so the fill improves on it.
    broker.next_bar(bar(98, 99, 95, 97))

    assert broker.positions["AAPL"].avg_price == pytest.approx(98.0)
    assert broker.cash == pytest.approx(9_020.0)


def test_limit_buy_does_not_fill_when_the_bar_stays_above_the_limit():
    broker = PaperBroker(initial_cash=10_000)
    order = broker.place_order(limit_buy(10, 90))

    broker.next_bar(bar(95, 99, 91, 97))

    assert order.status is OrderStatus.OPEN
    assert broker.positions == {}
    assert broker.pending == [order]


def test_stop_sell_fills_through_a_gap_at_the_open():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(market_buy(10))
    broker.next_bar(bar(90, 95, 89, 92, day=1))

    broker.place_order(stop_sell(10, 100))
    broker.next_bar(bar(98, 99, 95, 96, day=2))

    # The stop was 100 but the bar opened at 98: the exit is worse, as it would be.
    trade = broker.trades[0]
    assert trade.exit_price == pytest.approx(98.0)
    assert trade.entry_price == pytest.approx(90.0)
    assert trade.pnl == pytest.approx(80.0)
    assert broker.positions == {}


def test_stop_loss_beats_take_profit_when_one_bar_hits_both():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(market_buy(10, take_profit=110, stop_loss=95))
    broker.next_bar(bar(100, 105, 99, 101, day=1))

    broker.next_bar(bar(101, 111, 94, 96, day=2))

    assert broker.positions == {}
    assert broker.total_pnl == pytest.approx(-50.0)
    assert broker.trades[0].exit_price == pytest.approx(95.0)


def test_take_profit_closes_the_position_alone():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(market_buy(10, take_profit=110))
    broker.next_bar(bar(100, 101, 99, 100, day=1))

    broker.next_bar(bar(101, 112, 100, 110, day=2))

    assert broker.positions == {}
    assert broker.trades[0].exit_price == pytest.approx(110.0)
    assert broker.total_pnl == pytest.approx(100.0)


def test_commission_is_charged_on_every_fill():
    broker = PaperBroker(initial_cash=10_000, commission=5)
    broker.place_order(market_buy(10))

    broker.next_bar(bar(100, 105, 99, 102))

    assert broker.cash == pytest.approx(10_000 - 1_000 - 5)


def test_commission_reduces_realised_pnl():
    broker = PaperBroker(initial_cash=10_000, commission=5)
    broker.place_order(market_buy(10))
    broker.next_bar(bar(100, 105, 99, 102, day=1))

    broker.place_order(market_sell(10))
    broker.next_bar(bar(110, 112, 108, 111, day=2))

    # (110 - 100) * 10 = 100 gross, minus the 5 charged on the exit fill.
    assert broker.total_pnl == pytest.approx(95.0)


def test_insufficient_cash_rejects_the_order_and_leaves_cash_alone():
    broker = PaperBroker(initial_cash=500)
    order = broker.place_order(market_buy(10))

    broker.next_bar(bar(100, 105, 99, 102))

    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason == "insufficient cash"
    assert broker.cash == pytest.approx(500.0)
    assert broker.positions == {}
    assert broker.pending == []


def test_selling_without_a_position_is_rejected():
    broker = PaperBroker(initial_cash=10_000)
    order = broker.place_order(market_sell(10))

    broker.next_bar(bar(100, 105, 99, 102))

    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason == "insufficient position"
    assert broker.cash == pytest.approx(10_000.0)


def test_selling_more_than_held_is_rejected():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(market_buy(5))
    broker.next_bar(bar(100, 105, 99, 102, day=1))

    order = broker.place_order(market_sell(6))
    broker.next_bar(bar(105, 106, 104, 105, day=2))

    assert order.status is OrderStatus.REJECTED
    assert broker.positions["AAPL"].quantity == 5


def test_cancel_only_works_on_a_still_open_order():
    broker = PaperBroker(initial_cash=10_000)
    order = broker.place_order(limit_buy(10, 90))

    assert broker.cancel(order.order_id) is True
    assert order.status is OrderStatus.CANCELED
    assert broker.pending == []
    # Already terminal: nothing left to cancel.
    assert broker.cancel(order.order_id) is False
    assert broker.cancel(9_999) is False


def test_equity_marks_the_position_at_the_bar_close():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(market_buy(10))

    broker.next_bar(bar(100, 105, 99, 105))

    assert broker.cash == pytest.approx(9_000.0)
    assert broker.equity == pytest.approx(9_000.0 + 10 * 105)
    assert len(broker.equity_curve) == 1
    stamp, value = broker.equity_curve[0]
    assert stamp == pd.Timestamp("2024-01-01")
    assert value == pytest.approx(10_050.0)


def test_adding_to_a_position_averages_the_entry_price():
    broker = PaperBroker(initial_cash=10_000)
    broker.place_order(market_buy(10))
    broker.next_bar(bar(100, 101, 99, 100, day=1))

    broker.place_order(market_buy(10))
    broker.next_bar(bar(110, 111, 109, 110, day=2))

    # (10 * 100 + 10 * 110) / 20 = 105, and the position's clock has not reset.
    position = broker.positions["AAPL"]
    assert position.avg_price == pytest.approx(105.0)
    assert position.opened_at == pd.Timestamp("2024-01-01")


def test_bars_from_history_scales_every_price_to_the_adjusted_basis():
    history = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [100.0],
            "Adj Close": [50.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
    )

    scaled = bars_from_history(history)
    raw = bars_from_history(history, adjusted=False)

    assert scaled[0].open == pytest.approx(50.0)
    assert scaled[0].high == pytest.approx(52.5)
    assert raw[0].high == pytest.approx(105.0)
