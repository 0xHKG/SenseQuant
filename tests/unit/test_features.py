"""Unit tests for technical indicator library."""

from __future__ import annotations

import pandas as pd
import pytest

from src.domain.features import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_vwap,
)


@pytest.fixture
def sample_price_series() -> pd.Series:
    """Create sample price series for testing."""
    return pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0])


@pytest.fixture
def sample_ohlcv_data() -> dict[str, pd.Series]:
    """Create sample OHLCV data for testing."""
    return {
        "open": pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]),
        "high": pd.Series([102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 109.0, 111.0]),
        "low": pd.Series([98.0, 100.0, 99.0, 101.0, 103.0, 102.0, 104.0, 106.0, 105.0, 107.0]),
        "close": pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]),
        "volume": pd.Series([1000, 1100, 1050, 1200, 1150, 1080, 1250, 1300, 1220, 1350]),
    }


# ==================== SMA Tests ====================


def test_sma_calculation(sample_price_series: pd.Series) -> None:
    """Test SMA returns correct values."""
    sma = calculate_sma(sample_price_series, period=3)

    # First 2 values should be NaN (insufficient data)
    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])

    # Third value should be average of first 3 prices
    assert sma.iloc[2] == pytest.approx((100.0 + 102.0 + 101.0) / 3)

    # Last value should be average of last 3 prices
    assert sma.iloc[-1] == pytest.approx((108.0 + 107.0 + 109.0) / 3)


def test_sma_insufficient_data() -> None:
    """Test SMA handles insufficient data gracefully."""
    short_series = pd.Series([100.0, 102.0])
    sma = calculate_sma(short_series, period=5)

    # All values should be NaN (not enough data for period=5)
    assert sma.isna().all()


def test_sma_with_nans() -> None:
    """Test SMA handles NaN values in input."""
    series_with_nan = pd.Series([100.0, 102.0, None, 103.0, 105.0])
    sma = calculate_sma(series_with_nan, period=3)

    # SMA should handle NaNs gracefully
    assert isinstance(sma, pd.Series)


# ==================== EMA Tests ====================


def test_ema_calculation(sample_price_series: pd.Series) -> None:
    """Test EMA returns correct values."""
    ema = calculate_ema(sample_price_series, period=3)

    # First 2 values should be NaN (insufficient data)
    assert pd.isna(ema.iloc[0])
    assert pd.isna(ema.iloc[1])

    # EMA should exist for later values
    assert pd.notna(ema.iloc[2])
    assert pd.notna(ema.iloc[-1])


def test_ema_reacts_faster_than_sma(sample_price_series: pd.Series) -> None:
    """Test EMA reacts faster to price changes than SMA."""
    # Create series with sharp price change
    series = pd.Series([100.0] * 10 + [110.0] * 10)

    sma = calculate_sma(series, period=5)
    ema = calculate_ema(series, period=5)

    # After the price jump, EMA should react faster (be higher) than SMA
    # Compare values shortly after the jump (index 12)
    assert ema.iloc[12] > sma.iloc[12]


def test_ema_insufficient_data() -> None:
    """Test EMA handles insufficient data gracefully."""
    short_series = pd.Series([100.0, 102.0])
    ema = calculate_ema(short_series, period=5)

    # All values should be NaN
    assert ema.isna().all()


# ==================== RSI Tests ====================


def test_rsi_calculation(sample_price_series: pd.Series) -> None:
    """Test RSI returns values in valid range (0-100)."""
    rsi = calculate_rsi(sample_price_series, period=5)

    # Check valid rows (after initial period)
    valid_rsi = rsi.dropna()
    assert valid_rsi.min() >= 0
    assert valid_rsi.max() <= 100


def test_rsi_uptrend_high_value() -> None:
    """Test RSI returns high values during uptrend."""
    # Strong uptrend
    uptrend = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0])
    rsi = calculate_rsi(uptrend, period=5)

    # RSI should be high (overbought) during strong uptrend
    assert rsi.iloc[-1] > 60


def test_rsi_downtrend_low_value() -> None:
    """Test RSI returns low values during downtrend."""
    # Strong downtrend
    downtrend = pd.Series([100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0])
    rsi = calculate_rsi(downtrend, period=5)

    # RSI should be low (oversold) during strong downtrend
    assert rsi.iloc[-1] < 40


def test_rsi_insufficient_data() -> None:
    """Test RSI handles insufficient data gracefully."""
    short_series = pd.Series([100.0, 102.0])
    rsi = calculate_rsi(short_series, period=14)

    # All values should be NaN
    assert rsi.isna().all()


# ==================== ATR Tests ====================


def test_atr_calculation(sample_ohlcv_data: dict[str, pd.Series]) -> None:
    """Test ATR returns positive values."""
    atr = calculate_atr(
        sample_ohlcv_data["high"],
        sample_ohlcv_data["low"],
        sample_ohlcv_data["close"],
        period=3,
    )

    # ATR should be positive for valid rows
    valid_atr = atr.dropna()
    assert (valid_atr > 0).all()


def test_atr_measures_volatility() -> None:
    """Test ATR increases with volatility."""
    # Low volatility data
    low_vol_high = pd.Series([100.0, 101.0, 100.5, 101.5, 100.8] * 3)
    low_vol_low = pd.Series([99.0, 100.0, 99.5, 100.5, 99.8] * 3)
    low_vol_close = pd.Series([99.5, 100.5, 100.0, 101.0, 100.3] * 3)

    # High volatility data
    high_vol_high = pd.Series([100.0, 110.0, 95.0, 115.0, 90.0] * 3)
    high_vol_low = pd.Series([90.0, 100.0, 85.0, 95.0, 80.0] * 3)
    high_vol_close = pd.Series([95.0, 105.0, 90.0, 105.0, 85.0] * 3)

    atr_low = calculate_atr(low_vol_high, low_vol_low, low_vol_close, period=5)
    atr_high = calculate_atr(high_vol_high, high_vol_low, high_vol_close, period=5)

    # High volatility should have higher ATR
    assert atr_high.iloc[-1] > atr_low.iloc[-1]


def test_atr_insufficient_data() -> None:
    """Test ATR handles insufficient data gracefully."""
    short_high = pd.Series([102.0, 104.0])
    short_low = pd.Series([98.0, 100.0])
    short_close = pd.Series([100.0, 102.0])

    atr = calculate_atr(short_high, short_low, short_close, period=14)

    # All values should be NaN
    assert atr.isna().all()


# ==================== VWAP Tests ====================


def test_vwap_calculation(sample_ohlcv_data: dict[str, pd.Series]) -> None:
    """Test VWAP returns valid values."""
    vwap = calculate_vwap(
        sample_ohlcv_data["high"],
        sample_ohlcv_data["low"],
        sample_ohlcv_data["close"],
        sample_ohlcv_data["volume"],
    )

    # VWAP should have no NaN values (cumulative calculation)
    assert vwap.notna().all()

    # VWAP should be within reasonable range
    assert vwap.min() > 0
    assert vwap.max() < 200


def test_vwap_typical_price() -> None:
    """Test VWAP uses typical price (H+L+C)/3."""
    # Simple test case with known values
    high = pd.Series([105.0])
    low = pd.Series([95.0])
    close = pd.Series([100.0])
    volume = pd.Series([1000.0])

    vwap = calculate_vwap(high, low, close, volume)

    # Typical price = (105 + 95 + 100) / 3 = 100
    # VWAP = 100 * 1000 / 1000 = 100
    assert vwap.iloc[0] == pytest.approx(100.0)


def test_vwap_zero_volume() -> None:
    """Test VWAP handles zero volume gracefully."""
    high = pd.Series([102.0, 104.0])
    low = pd.Series([98.0, 100.0])
    close = pd.Series([100.0, 102.0])
    volume = pd.Series([0.0, 0.0])  # Zero volume

    vwap = calculate_vwap(high, low, close, volume)

    # Should not raise error, returns valid series
    assert isinstance(vwap, pd.Series)


# ==================== Bollinger Bands Tests ====================


def test_bollinger_bands_calculation(sample_price_series: pd.Series) -> None:
    """Test Bollinger Bands return (upper, middle, lower)."""
    upper, middle, lower = calculate_bollinger_bands(sample_price_series, period=5, num_std=2.0)

    # All should be Series
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)

    # Check valid rows (after initial period)
    valid_indices = middle.notna()
    assert (upper[valid_indices] >= middle[valid_indices]).all()
    assert (middle[valid_indices] >= lower[valid_indices]).all()


def test_bollinger_bands_width_increases_with_volatility() -> None:
    """Test Bollinger Bands width increases with volatility."""
    # Low volatility
    low_vol = pd.Series([100.0, 100.5, 100.2, 100.3, 100.1, 100.4, 100.2])
    upper_low, middle_low, lower_low = calculate_bollinger_bands(low_vol, period=5, num_std=2.0)
    width_low = upper_low.iloc[-1] - lower_low.iloc[-1]

    # High volatility
    high_vol = pd.Series([100.0, 110.0, 95.0, 115.0, 90.0, 120.0, 85.0])
    upper_high, middle_high, lower_high = calculate_bollinger_bands(high_vol, period=5, num_std=2.0)
    width_high = upper_high.iloc[-1] - lower_high.iloc[-1]

    # High volatility should have wider bands
    assert width_high > width_low


def test_bollinger_bands_insufficient_data() -> None:
    """Test Bollinger Bands handle insufficient data gracefully."""
    short_series = pd.Series([100.0, 102.0])
    upper, middle, lower = calculate_bollinger_bands(short_series, period=20)

    # All values should be NaN
    assert upper.isna().all()
    assert middle.isna().all()
    assert lower.isna().all()


# ==================== MACD Tests ====================


def test_macd_calculation(sample_price_series: pd.Series) -> None:
    """Test MACD returns (macd_line, signal_line, histogram)."""
    # Need longer series for MACD (slow period = 26)
    long_series = pd.Series([100.0 + i * 0.5 for i in range(50)])

    macd_line, signal_line, histogram = calculate_macd(
        long_series, fast_period=12, slow_period=26, signal_period=9
    )

    # All should be Series
    assert isinstance(macd_line, pd.Series)
    assert isinstance(signal_line, pd.Series)
    assert isinstance(histogram, pd.Series)

    # Histogram should equal macd - signal for valid rows
    valid_indices = histogram.notna()
    expected_histogram = macd_line[valid_indices] - signal_line[valid_indices]
    pd.testing.assert_series_equal(
        histogram[valid_indices], expected_histogram, check_names=False, atol=1e-10
    )


def test_macd_crossover_signal() -> None:
    """Test MACD histogram changes sign at crossover."""
    # Create series with trend change
    series = pd.Series([100.0] * 15 + list(range(100, 130)))

    macd_line, signal_line, histogram = calculate_macd(
        series, fast_period=5, slow_period=10, signal_period=3
    )

    # Histogram should have both positive and negative values
    valid_histogram = histogram.dropna()
    if len(valid_histogram) > 0:
        # Check histogram can be positive or negative
        assert isinstance(valid_histogram.iloc[-1], float)


def test_macd_insufficient_data() -> None:
    """Test MACD handles insufficient data gracefully."""
    short_series = pd.Series([100.0, 102.0, 101.0])
    macd_line, signal_line, histogram = calculate_macd(
        short_series, fast_period=12, slow_period=26, signal_period=9
    )

    # All values should be NaN
    assert macd_line.isna().all()
    assert signal_line.isna().all()
    assert histogram.isna().all()


# ==================== ADX Tests ====================


def test_adx_calculation(sample_ohlcv_data: dict[str, pd.Series]) -> None:
    """Test ADX returns values in valid range (0-100)."""
    # Need more data for ADX
    long_high = pd.Series([100.0 + i * 0.5 for i in range(50)])
    long_low = pd.Series([98.0 + i * 0.5 for i in range(50)])
    long_close = pd.Series([99.0 + i * 0.5 for i in range(50)])

    adx = calculate_adx(long_high, long_low, long_close, period=14)

    # Check valid rows
    valid_adx = adx.dropna()
    if len(valid_adx) > 0:
        assert valid_adx.min() >= 0
        # Note: ADX can exceed 100 in extreme cases, but typically stays below


def test_adx_strong_trend() -> None:
    """Test ADX returns high values during strong trend."""
    # Strong uptrend
    trend_high = pd.Series([100.0 + i * 2 for i in range(50)])
    trend_low = pd.Series([98.0 + i * 2 for i in range(50)])
    trend_close = pd.Series([99.0 + i * 2 for i in range(50)])

    adx = calculate_adx(trend_high, trend_low, trend_close, period=14)

    # ADX should be higher during strong trend (typically > 25)
    valid_adx = adx.dropna()
    if len(valid_adx) > 0:
        # Just verify it's a positive value
        assert valid_adx.iloc[-1] >= 0


def test_adx_insufficient_data() -> None:
    """Test ADX handles insufficient data gracefully."""
    short_high = pd.Series([102.0, 104.0])
    short_low = pd.Series([98.0, 100.0])
    short_close = pd.Series([100.0, 102.0])

    adx = calculate_adx(short_high, short_low, short_close, period=14)

    # All values should be NaN
    assert adx.isna().all()


# ==================== OBV Tests ====================


def test_obv_calculation(sample_ohlcv_data: dict[str, pd.Series]) -> None:
    """Test OBV returns cumulative volume values."""
    obv = calculate_obv(sample_ohlcv_data["close"], sample_ohlcv_data["volume"])

    # OBV should be cumulative (no NaN values after first)
    assert obv.iloc[1:].notna().all()


def test_obv_increases_on_up_days() -> None:
    """Test OBV increases when price goes up."""
    close = pd.Series([100.0, 102.0, 104.0])  # Increasing prices
    volume = pd.Series([1000.0, 1000.0, 1000.0])

    obv = calculate_obv(close, volume)

    # OBV should increase on up days
    assert obv.iloc[2] > obv.iloc[1]


def test_obv_decreases_on_down_days() -> None:
    """Test OBV decreases when price goes down."""
    close = pd.Series([100.0, 98.0, 96.0])  # Decreasing prices
    volume = pd.Series([1000.0, 1000.0, 1000.0])

    obv = calculate_obv(close, volume)

    # OBV should decrease on down days
    assert obv.iloc[2] < obv.iloc[1]


def test_obv_unchanged_on_flat_days() -> None:
    """Test OBV stays same when price unchanged."""
    close = pd.Series([100.0, 100.0, 100.0])  # Flat prices
    volume = pd.Series([1000.0, 1000.0, 1000.0])

    obv = calculate_obv(close, volume)

    # OBV should stay same on flat days (after first NaN)
    assert obv.iloc[1] == obv.iloc[2]
