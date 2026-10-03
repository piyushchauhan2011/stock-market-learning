"""Bar-by-bar backtester: replay a strategy over history and measure it.

The engine is intentionally thin, because the interesting part is the *timing*
rule it enforces:

    a signal computed from bar ``i``'s close is executed at bar ``i + 1``'s open

Anything else quietly cheats. If the order filled at the close it was computed
from, the strategy would be trading on a price it already knew — the classic
look-ahead bias, and the single most common way a backtest lies to you.

What this engine does **not** model, on purpose: slippage, latency, partial
fills, borrow costs, taxes, and market impact. Commission is a flat per-fill fee
and defaults to ``0.0``. Results are an upper bound on a strategy's real
behaviour, not a forecast.

Educational use only — not financial advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from stocklearn.broker import Bar, PaperBroker, bars_from_history
from stocklearn.data import DEFAULT_TICKER, price_series
from stocklearn.indicators import annualized_volatility, max_drawdown, sharpe_ratio
from stocklearn.orders import Order
from stocklearn.report import DISCLAIMER
from stocklearn.strategy import Context

TRADE_COLUMNS = [
    "symbol",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl",
    "return_pct",
]
"""Columns of :attr:`BacktestResult.trades`, in order."""

TRADING_DAYS = 252
"""Bars per year assumed by the annualised metrics."""


@dataclass
class BacktestResult:
    """What a run produced.

    Attributes
    ----------
    equity_curve:
        Account equity after every bar, indexed by bar timestamp.
    trades:
        Closed round trips as a DataFrame with columns
        :data:`TRADE_COLUMNS`; empty when nothing ever closed.
    metrics:
        The headline numbers (see :func:`run_backtest`).
    orders:
        Every order the strategy asked for, including the ones that were
        canceled or rejected — the teaching aid for "what did it actually try
        to do?".
    strategy_name:
        Human label of the strategy that produced this result.
    commission:
        The flat per-fill fee the run was charged, echoed back for the summary.
    """

    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    orders: list[Order]
    strategy_name: str = "strategy"
    commission: float = 0.0


def run_backtest(
    strategy,
    history: pd.DataFrame,
    *,
    initial_cash: float = 10_000.0,
    commission: float = 0.0,
) -> BacktestResult:
    """Replay ``strategy`` bar by bar over ``history``.

    Parameters
    ----------
    strategy:
        Any object with ``on_bar(ctx) -> list[Order]`` (see
        :mod:`stocklearn.strategy`).
    history:
        OHLCV frame from :func:`stocklearn.data.get_history`. Signals are
        computed on ``Adj Close`` and fills use bars scaled to the same adjusted
        basis, so the two are directly comparable.
    initial_cash:
        Starting cash for the paper account.
    commission:
        Flat fee per fill.

    Returns
    -------
    BacktestResult
        With these ``metrics`` keys: ``initial_cash``, ``final_equity``,
        ``total_return``, ``cagr`` (``None`` for a one-bar history),
        ``annualized_volatility``, ``sharpe``, ``max_drawdown``, ``total_pnl``,
        ``n_trades``, ``win_rate``, ``profit_factor``, ``avg_win``, ``avg_loss``.
    """
    bars = bars_from_history(history, adjusted=True)
    close = price_series(history)
    broker = PaperBroker(initial_cash=initial_cash, commission=commission)
    placed: list[Order] = []

    for i, bar in enumerate(bars):
        # 1. Fill what the PREVIOUS bar decided; 2. then decide again.
        broker.next_bar(bar)
        position = broker.positions.get(DEFAULT_TICKER)
        ctx = Context(
            close=close.iloc[: i + 1],
            cash=broker.cash,
            position_qty=position.quantity if position else 0,
            timestamp=bar.timestamp,
            bar=bar,
        )
        for order in strategy.on_bar(ctx):
            broker.place_order(order)
            placed.append(order)

    equity_curve = _equity_series(broker.equity_curve)
    trades = _trades_frame(broker.trades)
    metrics = _metrics(equity_curve, trades, initial_cash)
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        orders=placed,
        strategy_name=getattr(strategy, "name", type(strategy).__name__),
        commission=float(commission),
    )


def summarize_backtest(result: BacktestResult) -> str:
    """Render a :class:`BacktestResult` as a markdown report.

    Metrics table, the last ten trades, the limitations of the replay, and the
    standard :data:`stocklearn.report.DISCLAIMER` footer.
    """
    m = result.metrics
    rows = [
        ("Initial cash", _money(m["initial_cash"])),
        ("Final equity", _money(m["final_equity"])),
        ("Total return", _pct(m["total_return"])),
        ("CAGR", "n/a (too short)" if m["cagr"] is None else _pct(m["cagr"])),
        ("Annualised volatility", _pct(m["annualized_volatility"])),
        ("Sharpe ratio", _num(m["sharpe"])),
        ("Max drawdown", _pct(m["max_drawdown"])),
        ("Realised P&L", _money(m["total_pnl"])),
        ("Trades", str(m["n_trades"])),
        ("Win rate", _pct(m["win_rate"])),
        ("Profit factor", _profit_factor(m["profit_factor"])),
        ("Average win", _money(m["avg_win"])),
        ("Average loss", _money(m["avg_loss"])),
    ]

    lines = [
        f"# Backtest — {result.strategy_name}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", "## Trades (last 10)", ""]
    if result.trades.empty:
        lines.append("_No closed trades — the strategy never completed a round trip._")
    else:
        tail = result.trades.tail(10)
        lines += [
            "| Entry | Exit | Qty | Entry px | Exit px | P&L | Return |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in tail.itertuples(index=False):
            lines.append(
                f"| {_timestamp(row.entry_time)} | {_timestamp(row.exit_time)} | "
                f"{row.quantity} | {_money(row.entry_price)} | {_money(row.exit_price)} | "
                f"{_money(row.pnl)} | {_pct(row.return_pct)} |"
            )

    lines += [
        "",
        (
            "> Historical replay: each signal is computed from a bar's close and filled at "
            "the next bar's open, with no slippage, latency or partial fills — only the "
            f"flat commission of {_money(result.commission)} per fill."
        ),
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- helpers


def _equity_series(curve: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    if not curve:
        return pd.Series(dtype=float)
    timestamps = [timestamp for timestamp, _ in curve]
    values = [value for _, value in curve]
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)


def _trades_frame(trades: list) -> pd.DataFrame:
    rows = [
        {
            "symbol": t.symbol,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "pnl": t.pnl,
            "return_pct": t.return_pct,
        }
        for t in trades
    ]
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _metrics(
    equity_curve: pd.Series, trades: pd.DataFrame, initial_cash: float
) -> dict:
    final_equity = (
        float(equity_curve.iloc[-1]) if len(equity_curve) else float(initial_cash)
    )
    n_bars = len(equity_curve)
    total_return = final_equity / initial_cash - 1 if initial_cash else 0.0
    cagr = None
    if n_bars > 1 and initial_cash > 0:
        years = TRADING_DAYS / max(n_bars - 1, 1)
        cagr = (final_equity / initial_cash) ** years - 1

    # A flat all-cash account has zero dispersion: volatility and Sharpe are 0,
    # not NaN, so the report and any downstream comparison stay well defined.
    if n_bars < 2 or float(equity_curve.std()) == 0.0:
        volatility, sharpe = 0.0, 0.0
    else:
        volatility = _finite(annualized_volatility(equity_curve))
        sharpe = _finite(sharpe_ratio(equity_curve))

    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(losses):
        profit_factor = float(wins.sum() / abs(losses.sum()))
    elif len(wins):
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(max_drawdown(equity_curve)) if n_bars else 0.0,
        "total_pnl": float(pnl.sum()),
        "n_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()) if len(trades) else 0.0,
        "profit_factor": profit_factor,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def _finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    return f"{value:.2f}"


def _profit_factor(value: float) -> str:
    if value == float("inf"):
        return "∞ (no losing trades)"
    return f"{value:.2f}"


def _timestamp(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
