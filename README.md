# Stock Market Learning

A hands-on, notebook-driven introduction to stock-market analysis in Python.
Ten ordered notebooks (00–09) walk through price data, charts, hand-computed
technical indicators, risk statistics, company fundamentals, news sentiment,
research documents, and a composite "will this stock go up?" report — all built
on a small, readable package (`stocklearn`) you can read end to end.

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

## The `stocklearn` package

`src/stocklearn/` is deliberately small: each module is one layer, and the
notebooks call it rather than duplicating logic.

| Module | Responsibility |
| --- | --- |
| `data.py` | `get_history()` (Yahoo OHLCV, Parquet-cached in `data/cache/`), `price_series()` |
| `indicators.py` | Hand-computed `sma`, `ema`, `rsi`, `macd`, `bollinger`, `atr`, plus returns, volatility, drawdown, Sharpe |
| `fundamentals.py` | `get_financials()` (income/balance/cash-flow + info), `compute_ratios()` |
| `news.py` | `get_news()` — Yahoo's news list plus Yahoo/Google RSS, de-duplicated |
| `sentiment.py` | VADER scoring: `analyze_text`, `score_headlines`, `aggregate_sentiment` |
| `docs.py` | `read_pdf`, `read_excel`, `read_docx` |
| `report.py` | `summarize()` + `format_report()` — the composite descriptive report |

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
identity, Bollinger ordering, ATR, drawdown).

## Layout

```
src/stocklearn/    the package (data, indicators, fundamentals, news, sentiment, docs, report)
notebooks/         the 00–09 curriculum
scripts/           make_sample_docs.py — regenerates sample_docs/ (idempotent)
sample_docs/       committed fixtures: earnings PDF, Excel model, Word research note
tests/             indicator unit tests
data/cache/        downloaded price/fundamental Parquet cache (git-ignored)
```

## Notes and limits

- Data comes from Yahoo Finance via `yfinance`; it can lag, throttle, or rename
  line items. Fundamentals lookups fall back through known alternative names
  and return `None` when a value is genuinely unavailable.
- News sentiment is lexicon-based (VADER): deterministic and offline, but blind
  to sarcasm and context. Empty news means a neutral verdict, not good news.
- Everything is a simplification chosen for teaching. Position sizing, taxes,
  fees, survivorship bias, and walk-forward validation are all out of scope.

> Educational analysis only — not financial advice. Past performance does not
> predict future returns.
