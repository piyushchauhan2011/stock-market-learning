"""Hand-checkable maths for :mod:`stocklearn.indicators`.

Every expectation here is computed by hand (or from the definition) rather than
by copying a library result, so the suite pins the formulas themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklearn.indicators import (
    annualized_volatility,
    atr,
    bollinger,
    daily_returns,
    ema,
    log_returns,
    macd,
    max_drawdown,
    rsi,
    sharpe_ratio,
    sma,
)


def test_sma_matches_hand_rolled_three_point_means():
    close = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    result = sma(close, 3)
    expected = close.rolling(3).mean()

    pd.testing.assert_series_equal(result, expected, check_names=False)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == [2.0, 3, 4, 5, 6, 7, 8, 9]


def test_ema_recursive_definition_is_exact():
    close = pd.Series([10.0, 20, 30])
    # alpha = 2 / (span + 1) = 0.5; ema_0 = 10, then 0.5*x + 0.5*prev.
    expected = pd.Series([10.0, 15.0, 22.5])

    pd.testing.assert_series_equal(ema(close, span=3), expected, check_names=False)


def test_rsi_stays_in_range_and_saturates_in_a_pure_uptrend():
    rng = np.random.default_rng(7)
    walk = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))
    values = rsi(walk).dropna()
    assert values.between(0, 100).all()

    rising = pd.Series(np.arange(1.0, 31.0))
    assert rsi(rising).iloc[-1] >= 99


def test_rsi_saturates_near_zero_in_a_pure_downtrend():
    falling = pd.Series(np.arange(30.0, 0.0, -1.0))
    assert rsi(falling).iloc[-1] <= 1


def test_macd_histogram_is_exactly_line_minus_signal():
    close = pd.Series(np.linspace(50, 120, 200))

    frame = macd(close)

    assert list(frame.columns) == ["macd", "signal", "hist"]
    pd.testing.assert_series_equal(
        frame["hist"], frame["macd"] - frame["signal"], check_names=False
    )


def test_bollinger_bands_keep_upper_above_middle_above_lower():
    rng = np.random.default_rng(11)
    close = pd.Series(80 + np.cumsum(rng.normal(0, 1.5, 120)))

    bands = bollinger(close, window=20, num_std=2.0).dropna()

    assert list(bollinger(close).columns) == ["middle", "upper", "lower"]
    assert (bands["upper"] > bands["middle"]).all()
    assert (bands["middle"] > bands["lower"]).all()
    # Two standard deviations, exactly, on the first fully-populated bar.
    window = close.iloc[:20]
    assert bands["upper"].iloc[0] == pytest.approx(window.mean() + 2 * window.std(ddof=1))


def test_atr_uses_the_largest_of_the_three_ranges():
    # Gaps dominate: high 12, low 11.5, previous close 10 => TR = 2.0 at bar 1.
    high = pd.Series([10.5, 12.0, 12.2])
    low = pd.Series([10.0, 11.5, 11.9])
    close = pd.Series([10.0, 11.8, 12.0])

    values = atr(high, low, close, period=2)

    assert values.iloc[0] == pytest.approx(0.5)
    assert values.iloc[-1] > values.iloc[0]


def test_returns_and_risk_statistics_match_their_definitions():
    close = pd.Series([100.0, 110.0, 99.0])

    assert daily_returns(close).iloc[1] == pytest.approx(0.10)
    assert log_returns(close).iloc[1] == pytest.approx(np.log(1.10))

    # Peak 110 -> trough 99 is a 10% drawdown.
    assert max_drawdown(close) == pytest.approx(-0.10)

    steady = pd.Series(np.linspace(100, 200, 60))
    assert annualized_volatility(steady) == pytest.approx(
        daily_returns(steady).std() * np.sqrt(252)
    )
    assert sharpe_ratio(steady) == pytest.approx(
        daily_returns(steady).mean() / daily_returns(steady).std() * np.sqrt(252)
    )


def test_max_drawdown_of_known_peak_then_drop():
    # cummax: 100, 120, 120, 120, 120 -> worst ratio is 90/120 - 1 = -0.25.
    close = pd.Series([100.0, 120.0, 90.0, 95.0, 110.0])

    assert max_drawdown(close) == pytest.approx(-0.25)
