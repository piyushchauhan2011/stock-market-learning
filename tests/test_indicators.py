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
    fibonacci_extensions,
    fibonacci_levels,
    log_returns,
    macd,
    max_drawdown,
    pivot_points,
    recent_swing,
    rsi,
    sharpe_ratio,
    sma,
    swing_highs_lows,
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


def test_fibonacci_levels_run_from_the_high_down_to_the_low():
    # Leg 100 -> 200, span 100: 0.382 of the leg is 200 - 38.2 = 161.8.
    levels = fibonacci_levels(100.0, 200.0)

    assert levels["0.0"] == 200.0
    assert levels["0.5"] == pytest.approx(150.0)
    assert levels["0.382"] == pytest.approx(161.8)
    assert levels["0.618"] == pytest.approx(138.2)
    assert levels["1.0"] == 100.0
    # Ordered high -> low as the ratio grows.
    assert list(levels) == ["0.0", "0.236", "0.382", "0.5", "0.618", "0.786", "1.0"]
    assert list(levels.values()) == sorted(levels.values(), reverse=True)


def test_fibonacci_extensions_project_past_the_high():
    # 1.0 is the high itself; 1.618 is one golden-ratio span above it (200 + 61.8).
    extensions = fibonacci_extensions(100.0, 200.0)

    assert extensions["1.0"] == 200.0
    assert extensions["1.272"] == pytest.approx(227.2)
    assert extensions["1.618"] == pytest.approx(261.8)
    assert extensions["2.618"] == pytest.approx(361.8)


def test_levels_of_an_unknown_leg_are_nan():
    levels = fibonacci_levels(float("nan"), 200.0)
    extensions = fibonacci_extensions(100.0, float("nan"))

    assert all(np.isnan(value) for value in levels.values())
    assert all(np.isnan(value) for value in extensions.values())


def test_swing_highs_lows_flags_only_strict_local_extremes():
    # 1 2 [3] 2 1 2 [3] 2 1 with window 2: peaks at 2 and 6, trough at 4, edges never.
    close = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])

    swings = swing_highs_lows(close, window=2)

    assert list(swings.columns) == ["high", "low"]
    assert swings["high"].notna().tolist() == [False, False, True, False, False, False, True, False, False]
    assert swings["low"].notna().tolist() == [False, False, False, False, True, False, False, False, False]
    assert swings["high"].iloc[2] == 3.0 and swings["high"].iloc[6] == 3.0
    assert swings["low"].iloc[4] == 1.0


def test_swing_highs_lows_ignores_a_plateau_because_the_test_is_strict():
    # The flat 5s tie for the maximum, so neither bar is a confirmed swing high.
    flat_top = pd.Series([1.0, 5.0, 5.0, 1.0])

    swings = swing_highs_lows(flat_top, window=1)

    assert swings["high"].isna().all()


def test_recent_swing_returns_the_last_low_before_the_last_high():
    close = pd.Series([5.0, 4.0, 3.0, 4.0, 6.0, 4.0, 3.0, 4.0, 7.0, 6.0, 5.0])

    leg = recent_swing(close, window=2)

    # Last swing high is the 7.0 at bar 8; the last swing low before it is the 3.0 at bar 6.
    assert leg == {"low": 3.0, "high": 7.0, "low_bar": 6, "high_bar": 8}


def test_recent_swing_is_empty_when_no_leg_exists():
    # A monotone rise has no confirmed high, so there is no leg to measure.
    leg = recent_swing(pd.Series(np.arange(10.0)), window=2)

    assert leg["low_bar"] == -1 and leg["high_bar"] == -1
    assert np.isnan(leg["low"]) and np.isnan(leg["high"])


def test_pivot_points_come_from_the_previous_bars_high_low_close():
    # Bar 1: H 110, L 100, C 105 -> P 105, R1 110, S1 100, R2 115, S2 95, R3 120, S3 90.
    high = pd.Series([0.0, 110.0, 0.0])
    low = pd.Series([0.0, 100.0, 0.0])
    close = pd.Series([0.0, 105.0, 0.0])

    pivots = pivot_points(high, low, close)

    assert list(pivots.columns) == ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"]
    assert pivots.iloc[0].isna().all()
    assert pivots.iloc[2].tolist() == [105.0, 110.0, 115.0, 120.0, 100.0, 95.0, 90.0]
    # R1/S1 and R3/S3 straddle the pivot symmetrically.
    assert pivots.iloc[2]["r1"] - pivots.iloc[2]["pivot"] == pytest.approx(
        pivots.iloc[2]["pivot"] - pivots.iloc[2]["s1"]
    )
    assert pivots.iloc[2]["r3"] - pivots.iloc[2]["pivot"] == pytest.approx(
        pivots.iloc[2]["pivot"] - pivots.iloc[2]["s3"]
    )
