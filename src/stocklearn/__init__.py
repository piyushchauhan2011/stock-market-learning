"""stocklearn — small, readable building blocks for stock-market analysis.

Modules
-------
- :mod:`stocklearn.data`         price history fetch + local Parquet cache
- :mod:`stocklearn.indicators`   hand-computed technical indicators and risk stats
- :mod:`stocklearn.fundamentals` company statements and valuation ratios
- :mod:`stocklearn.news`         headlines from yfinance and public RSS feeds
- :mod:`stocklearn.sentiment`    VADER sentiment scoring of headlines
- :mod:`stocklearn.docs`         PDF / Excel / Word readers for research documents
- :mod:`stocklearn.report`       composite descriptive report ("will it go up?")
- :mod:`stocklearn.orders`       order model (market/limit/stop, take-profit/stop-loss)
- :mod:`stocklearn.broker`       paper broker: positions, fills, P&L accounting
- :mod:`stocklearn.strategy`     strategy interface + example strategies
- :mod:`stocklearn.backtest`     bar-by-bar backtester + metrics
- :mod:`stocklearn.google_finance`  Google Finance quote + ~1 month of daily bars (hand-rolled)

Educational use only — not financial advice.
"""

__version__ = "0.1.0"
