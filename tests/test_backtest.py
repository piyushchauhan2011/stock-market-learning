"""Backtest-engine tests: sizing, no-lookahead timing, metrics, and the footer.

All histories are synthetic so the expected equity is arithmetic, not a market
opinion. Indexes are daily and columns mirror :func:`stocklearn.data.get_history`
(``Adj Close`` present), so the engine takes its normal path.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stocklearn.backtest import run_backtest, summarize_backtest
from stocklearn.orders import OrderStatus, OrderType, Side, market_buy, market_sell
from stocklearn.report import DISCLAIMER
from stocklearn.strategy import BuyAndHold


def history_from_closes(closes: list[float]) -> pd.DataFrame:
    """Frame whose Open/Close/Adj Close are identical (no gaps, no dividends)."""
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    close = pd.Series([float(c) for c in closes], index=index)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close,
            "Volume": 0.0,
        }
    )


class BuyOnFirstRise:
    """Buy one share the moment the close ticks up from the previous bar."""

    name = "Buy on first rise"

    def on_bar(self, ctx):
        if ctx.position_qty:
            return []
        if len(ctx.close) < 2 or ctx.close.iloc[-1] <= ctx.close.iloc[-2]:
            return []
        return [market_buy(1)]


class RebuyEveryBar:
    """Signal that fires on every bar, for tests about repeated behaviour."""

    name = "Rebuy every bar"

    def on_bar(self, ctx):
        return [market_buy(1)]


class BuyRiseSellFall:
    """Buy one share after an up bar, sell it after a down bar."""

    name = "Buy rise sell fall"

    def on_bar(self, ctx):
        if len(ctx.close) < 2:
            return []
        if ctx.position_qty == 0 and ctx.close.iloc[-1] > ctx.close.iloc[-2]:
            return [market_buy(1)]
        if ctx.position_qty > 0 and ctx.close.iloc[-1] < ctx.close.iloc[-2]:
            return [market_sell(ctx.position_qty)]
        return []


def test_buy_and_hold_spends_almost_everything_and_keeps_the_rest_in_cash():
    result = run_backtest(BuyAndHold(), history_from_closes([100, 100, 101, 102, 103]))

    # 2% is held back so a gap-up cannot bankrupt the next-open fill:
    # floor(10_000 * 0.98 / 100) = 98 shares at bar 1's open of 100.
    assert result.metrics["final_equity"] == pytest.approx(200.0 + 98 * 103)
    assert result.metrics["total_return"] == pytest.approx(0.0294)
    assert result.metrics["initial_cash"] == pytest.approx(10_000.0)


def test_signals_execute_at_the_next_bar_not_their_own_bar():
    result = run_backtest(BuyOnFirstRise(), history_from_closes([10, 11, 12, 13]))

    assert len(result.orders) == 1
    order = result.orders[0]
    # Signal on bar 1 (close 11); the fill must be bar 2's open of 12.
    assert order.status is OrderStatus.FILLED
    assert order.fill_price == pytest.approx(12.0)

    assert len(result.equity_curve) == 4
    assert result.equity_curve.index[3] == pd.Timestamp("2024-01-04")


def test_buy_and_hold_result_has_no_closed_trades_and_a_bounded_drawdown():
    result = run_backtest(BuyAndHold(), history_from_closes([100, 100, 101, 102, 103]))

    assert result.metrics["n_trades"] == 0
    assert isinstance(result.metrics["n_trades"], int)
    assert result.metrics["max_drawdown"] <= 0
    assert result.metrics["win_rate"] == 0.0
    assert result.metrics["profit_factor"] == 0.0
    assert result.trades.empty
    assert list(result.trades.columns)[-1] == "return_pct"


def test_final_equity_includes_the_still_open_position():
    result = run_backtest(BuyOnFirstRise(), history_from_closes([10, 11, 12, 13]))

    # Bought 1 share at 12 on bar 2, still held at the last close of 13.
    assert result.metrics["n_trades"] == 0
    assert result.metrics["total_pnl"] == pytest.approx(0.0)
    assert result.metrics["final_equity"] == pytest.approx(10_000.0 - 12.0 + 13.0)


def test_cagr_needs_more_than_one_bar():
    result = run_backtest(BuyAndHold(), history_from_closes([100]))

    assert result.metrics["cagr"] is None
    assert result.metrics["n_trades"] == 0
    # One bar, no fills: equity is just the untouched cash.
    assert result.metrics["final_equity"] == pytest.approx(10_000.0)


def test_commission_is_deducted_from_cash_and_from_pnl():
    history = history_from_closes([100, 100, 100])
    result = run_backtest(BuyAndHold(qty=1), history, commission=1.0)

    # One fill: 100 spent plus 1 of commission.
    assert result.metrics["final_equity"] == pytest.approx(10_000.0 - 101.0 + 100.0)


def test_rebuy_every_bar_leaves_the_final_order_open():
    result = run_backtest(RebuyEveryBar(), history_from_closes([100, 100, 100, 100]))

    assert len(result.orders) == 4
    # Three bars to fill into (bars 1-3); the fourth signal has no bar left.
    assert [o.status for o in result.orders] == [
        OrderStatus.FILLED,
        OrderStatus.FILLED,
        OrderStatus.FILLED,
        OrderStatus.OPEN,
    ]
    assert all(order.side is Side.BUY for order in result.orders)
    assert all(order.order_type is OrderType.MARKET for order in result.orders)
    assert len(result.equity_curve) == 4


def test_summary_ends_with_the_shared_disclaimer():
    result = run_backtest(BuyAndHold(), history_from_closes([100, 100, 101, 102, 103]))
    summary = summarize_backtest(result)

    assert summary.startswith("# Backtest — Buy and hold")
    assert summary.endswith(DISCLAIMER)
    assert "Educational analysis only" in summary
    assert "| Metric | Value |" in summary


def test_summary_handles_a_run_with_no_closed_trades():
    result = run_backtest(BuyAndHold(), history_from_closes([100]))
    summary = summarize_backtest(result)

    assert "No closed trades" in summary
    assert "n/a (too short)" in summary
    assert summary.endswith(DISCLAIMER)


def test_a_run_with_only_winners_reports_an_infinite_profit_factor():
    result = run_backtest(BuyRiseSellFall(), history_from_closes([10, 11, 12, 13, 12, 14]))

    # Buy 1 share at bar 2's open of 12; sell at bar 5's open of 14.
    assert result.metrics["n_trades"] == 1
    assert result.metrics["total_pnl"] == pytest.approx(2.0)
    assert result.metrics["win_rate"] == pytest.approx(1.0)
    assert result.metrics["avg_win"] == pytest.approx(2.0)
    assert result.metrics["avg_loss"] == pytest.approx(0.0)
    assert result.metrics["profit_factor"] == float("inf")
    assert result.trades["return_pct"].iloc[0] == pytest.approx(2.0 / 12.0)

    summary = summarize_backtest(result)
    assert "∞ (no losing trades)" in summary
    assert "100.0%" in summary
