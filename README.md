# Stock Market Learning

A hands-on, notebook-driven introduction to stock-market analysis in Python.
Fifteen ordered notebooks (00–14) walk through price data, charts,
hand-computed technical indicators, risk statistics, company fundamentals, news
sentiment, research documents, a composite "will this stock go up?" report,
advanced support/resistance levels, and finally the trading layer — orders, a
paper broker, strategies and backtests — all built on a small, readable package
(`stocklearn`) you can read end to end.

> **Educational analysis only — not financial advice.** Nothing here predicts
> prices: the indicators and the final report are descriptive textbook heuristics.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages the interpreter and the environment).
  With [mise](https://mise.jdx.dev/): `mise use -g uv`.
- No Python install needed: `.python-version` pins **Python 3.12**, and `uv`
  downloads it on first sync. (3.12 rather than 3.14 for rock-solid
  pandas/numpy/matplotlib wheels.)

## Setup

```bash
uv sync                       # creates .venv/ and installs everything
uv run python scripts/make_sample_docs.py   # (re)generate sample_docs/ fixtures
uv run jupyter lab            # opens the notebook UI in your browser
```

Everything after that runs through `uv run …` — no manual `pip install`, no
virtualenv activation.

## Notebooks — the curriculum

Open `notebooks/` in JupyterLab and work through them in order. Each notebook
sets the ticker in one place (`ticker = "AAPL"`), so swap it and re-run.

| Notebook | What you'll learn |
| --- | --- |
| `00_setup_and_first_prices.ipynb` | Tickers, OHLCV columns, Adj Close vs Close, timeframes/intervals, first pandas steps |
| `01_charts.ipynb` | Line/area charts, return histograms, candlesticks with volume, seaborn styling — and when each misleads |
| `02_returns_and_statistics.ipynb` | Simple vs log returns, volatility, drawdown, Sharpe ratio, growth of \$1, comparing two tickers |
| `03_moving_averages.ipynb` | SMA vs EMA, window length and smoothness, golden/death crosses and their lag |
| `04_oscillators_rsi_macd.ipynb` | RSI zones, MACD line/signal/histogram, crossovers, what momentum does and does not say |
| `05_bollinger_and_volatility.ipynb` | Bollinger bands, %B, bandwidth squeezes, ATR as absolute volatility |
| `06_fundamentals.ipynb` | Income statement, balance sheet, cash flow; P/E, EPS, ROE, debt/equity, gross margin, current ratio |
| `07_news_and_sentiment.ipynb` | Where headlines come from, VADER scoring, aggregate verdict, and its failure modes |
| `08_reading_documents.ipynb` | Extracting text/tables from PDF, Excel and Word research documents |
| `09_full_analysis.ipynb` | Assembling every signal into one descriptive report |
| `10_orders_and_execution.ipynb` | The order lifecycle: market vs limit vs stop, what fills where, and take-profit / stop-loss brackets |
| `11_backtesting_strategies.ipynb` | Defining a strategy, the no-lookahead rule, and reading return / CAGR / Sharpe / drawdown / win rate / profit factor |
| `12_paper_trading.ipynb` | Placing orders by hand, stepping bars, watching fills and P&L — and why paper trading flatters you |
| `13_google_finance.ipynb` | A second hand-rolled data source: the Google Finance quote, its ~1 month of daily bars, and what the public page does *not* give you |
| `14_fibonacci_and_levels.ipynb` | Swing highs/lows, Fibonacci retracement and extensions, and classic pivot points — and how traders use them to place entry, stop and target |

## The `stocklearn` package

`src/stocklearn/` is deliberately small: each module is one layer, and the
notebooks call it rather than duplicating logic.

| Module | Responsibility |
| --- | --- |
| `data.py` | `get_history()` (Yahoo OHLCV, Parquet-cached in `data/cache/`), `price_series()` |
| `google_finance.py` | `get_quote()` + `get_history()` — hand-rolled Google Finance (no library): snapshot quote + ~1 month of daily OHLCV, exchange required |
| `indicators.py` | Hand-computed `sma`, `ema`, `rsi`, `macd`, `bollinger`, `atr`, plus returns, volatility, drawdown, Sharpe, and the level tools `swing_highs_lows`/`recent_swing`, `fibonacci_levels`/`fibonacci_extensions`, `pivot_points` |
| `fundamentals.py` | `get_financials()` (income/balance/cash-flow + info), `compute_ratios()` |
| `news.py` | `get_news()` — Yahoo's news list plus Yahoo/Google RSS, de-duplicated |
| `sentiment.py` | VADER scoring: `analyze_text`, `score_headlines`, `aggregate_sentiment` |
| `docs.py` | `read_pdf`, `read_excel`, `read_docx` |
| `report.py` | `summarize()` + `format_report()` — the composite descriptive report |
| `orders.py` | `Order` + `market_buy`/`limit_sell`/`stop_buy`/… helpers — market, limit and stop orders, with optional take-profit / stop-loss brackets |
| `broker.py` | `PaperBroker` — fills orders against bars you feed it, tracks cash, positions, realised trades and equity |
| `strategy.py` | `Context` + `on_bar()` interface, and five example strategies (buy-and-hold, SMA cross, RSI reversion, MACD cross, Bollinger breakout) |
| `backtest.py` | `run_backtest()` — bar-by-bar replay with next-open fills, plus `summarize_backtest()` for the metrics table |

### Paper trading & backtesting

`orders.py` → `broker.py` → `strategy.py` → `backtest.py` add the layer that turns
analysis into *action*: place an order, watch it fill, watch profit and loss
unfold. The engine is hand-rolled (`for` loop over bars, no new dependency — the
same choice as the hand-computed indicators) because that is the point: an
event-driven loop shows the order mechanics that a vectorised backtester hides.

Deliberate limits, all visible in the code:

- **Long-only.** A sell reduces or closes a position; there is no short selling.
- **Historical replay, not live trading.** Bars come from `get_history()`; there
  is no streaming feed, so "paper trading" here means stepping through real past
  bars, one at a time.
- **No lookahead.** A signal computed from bar *i*'s close fills at bar *i+1*'s
  open, because that is the earliest price a trader could actually have got.
- **Flat commission, default `0.0`**, charged per fill; no slippage, no latency,
  no partial fills. Real trading is worse than this in every one of those ways.

### Run the report from the command line

```bash
uv run python -m stocklearn.report AAPL
uv run python -m stocklearn.report MSFT --period 6mo --out report.md
```

It prints a markdown report: trend regime, RSI/MACD state, volatility and
drawdown, valuation ratios, news sentiment with the top headlines, the raw
signal list, and the bull/bear tally. Unavailable inputs are labelled
"unavailable" rather than guessed.

## Tests

```bash
uv run pytest -q
```

`tests/test_indicators.py` pins the indicator mathematics against hand-computed
expectations (SMA/EMA definitions, RSI range and saturation, MACD histogram
identity, Bollinger ordering, ATR, drawdown, plus Fibonacci levels/extensions,
swing detection and pivot points). `tests/test_google_finance.py` pins the
Google Finance parser against a saved page fragment, so it never touches the
network. `tests/test_orders.py`, `tests/test_broker.py` and
`tests/test_backtest.py` do the same for the trading layer: order validation,
fill prices (including gap-through behaviour), bracket
priority, rejection paths, commission, position averaging, and the no-lookahead
timing rule.

## Layout

```
src/stocklearn/    the package (data, google_finance, indicators, fundamentals, news, sentiment,
                   docs, report, orders, broker, strategy, backtest)
notebooks/         the 00–14 curriculum
scripts/           make_sample_docs.py — regenerates sample_docs/ (idempotent)
sample_docs/       committed fixtures: earnings PDF, Excel model, Word research note
tests/             indicator, google-finance parser, order, broker and backtest unit tests
data/cache/        downloaded price/fundamental Parquet cache (git-ignored)
```

## Notes and limits

- Data comes from Yahoo Finance via `yfinance`; it can lag, throttle, or rename
  line items. Fundamentals lookups fall back through known alternative names
  and return `None` when a value is genuinely unavailable.
- News sentiment is lexicon-based (VADER): deterministic and offline, but blind
  to sarcasm and context. Empty news means a neutral verdict, not good news.
- Google Finance (via its public quote page) is snapshot-only — a quote plus
  about one month of daily bars, no multi-year history. Its HTML markers are
  undocumented and can change, so a parse failure raises `ValueError` /
  `RuntimeError` with the reason instead of returning wrong data.
- Everything is a simplification chosen for teaching. Backtests model whole
  shares, a flat commission and no slippage; taxes, survivorship bias, and
  walk-forward validation are all out of scope. A good-looking equity curve is
  as likely to be a curve-fit as a strategy.

> Educational analysis only — not financial advice. Past performance does not
> predict future returns.
