"""Unit tests for intraday strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from src.app.config import Settings
from src.domain.strategies.intraday import compute_features, signal


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        intraday_sma_period=20,
        intraday_ema_period=50,
        intraday_rsi_period=14,
        intraday_atr_period=14,
        intraday_long_rsi_min=55,
        intraday_short_rsi_max=45,
        sentiment_pos_limit=0.15,
        sentiment_neg_limit=-0.15,
    )


@pytest.fixture
def sample_ohlc_df() -> pd.DataFrame:
    """Create sample OHLC DataFrame with enough data for indicators."""
    # Generate 100 bars of synthetic data
    data = []
    base_price = 100.0
    for i in range(100):
        open_price = base_price + (i % 10) - 5
        high_price = open_price + 2
        low_price = open_price - 2
        close_price = open_price + (i % 5) - 2
        volume = 1000 + i * 10
        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )
    return pd.DataFrame(data)


@pytest.fixture
def insufficient_data_df() -> pd.DataFrame:
    """Create DataFrame with insufficient data."""
    return pd.DataFrame(
        [
            {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000},
            {"open": 101.0, "high": 103.0, "low": 99.0, "close": 102.0, "volume": 1100},
        ]
    )


def test_compute_features_success(sample_ohlc_df: pd.DataFrame, settings: Settings) -> None:
    """Test compute_features with sufficient data."""
    df_features = compute_features(sample_ohlc_df, settings)

    # Check all feature columns are present
    assert "sma20" in df_features.columns
    assert "ema50" in df_features.columns
    assert "rsi14" in df_features.columns
    assert "atr14" in df_features.columns
    assert "valid" in df_features.columns

    # Check that valid rows exist
    assert df_features["valid"].sum() > 0

    # Check that last row has valid features
    last_row = df_features[df_features["valid"]].iloc[-1]
    assert pd.notna(last_row["sma20"])
    assert pd.notna(last_row["ema50"])
    assert pd.notna(last_row["rsi14"])
    assert pd.notna(last_row["atr14"])


def test_compute_features_insufficient_data(
    insufficient_data_df: pd.DataFrame, settings: Settings
) -> None:
    """Test compute_features with insufficient data sets valid=False."""
    df_features = compute_features(insufficient_data_df, settings)

    # All rows should be marked invalid
    assert df_features["valid"].sum() == 0
    assert all(df_features["valid"] == False)  # noqa: E712


def test_compute_features_missing_columns(settings: Settings) -> None:
    """Test compute_features raises ValueError for missing columns."""
    df_missing = pd.DataFrame([{"open": 100.0, "close": 101.0}])  # Missing high, low

    with pytest.raises(ValueError, match="Missing required columns"):
        compute_features(df_missing, settings)


def test_signal_long(sample_ohlc_df: pd.DataFrame, settings: Settings) -> None:
    """Test LONG signal generation when close > sma20 and rsi > threshold."""
    df_features = compute_features(sample_ohlc_df, settings)

    # Manually set conditions for LONG signal
    df_features.loc[df_features.index[-1], "close"] = 110.0
    df_features.loc[df_features.index[-1], "sma20"] = 100.0
    df_features.loc[df_features.index[-1], "rsi14"] = 60.0  # > 55
    df_features.loc[df_features.index[-1], "valid"] = True

    sig = signal(df_features, settings, sentiment=0.0)

    assert sig.direction == "LONG"
    assert sig.strength > 0
    assert sig.meta is not None
    assert sig.meta["reason"] == "long_sma_rsi_sentiment_ok"


def test_signal_short(sample_ohlc_df: pd.DataFrame, settings: Settings) -> None:
    """Test SHORT signal generation when close < sma20 and rsi < threshold."""
    df_features = compute_features(sample_ohlc_df, settings)

    # Manually set conditions for SHORT signal
    df_features.loc[df_features.index[-1], "close"] = 90.0
    df_features.loc[df_features.index[-1], "sma20"] = 100.0
    df_features.loc[df_features.index[-1], "rsi14"] = 40.0  # < 45
    df_features.loc[df_features.index[-1], "valid"] = True

    sig = signal(df_features, settings, sentiment=0.0)

    assert sig.direction == "SHORT"
    assert sig.strength > 0
    assert sig.meta is not None
    assert sig.meta["reason"] == "short_sma_rsi_sentiment_ok"


def test_signal_flat(sample_ohlc_df: pd.DataFrame, settings: Settings) -> None:
    """Test FLAT signal when no conditions are met."""
    df_features = compute_features(sample_ohlc_df, settings)

    # Set conditions that don't match LONG or SHORT
    df_features.loc[df_features.index[-1], "close"] = 100.0
    df_features.loc[df_features.index[-1], "sma20"] = 100.0
    df_features.loc[df_features.index[-1], "rsi14"] = 50.0  # Between thresholds
    df_features.loc[df_features.index[-1], "valid"] = True

    sig = signal(df_features, settings, sentiment=0.0)

    assert sig.direction == "FLAT"
    assert sig.strength == 0.0


def test_signal_long_blocked_by_negative_sentiment(
    sample_ohlc_df: pd.DataFrame, settings: Settings
) -> None:
    """Test LONG signal blocked by negative sentiment."""
    df_features = compute_features(sample_ohlc_df, settings)

    # Set conditions for LONG signal
    df_features.loc[df_features.index[-1], "close"] = 110.0
    df_features.loc[df_features.index[-1], "sma20"] = 100.0
    df_features.loc[df_features.index[-1], "rsi14"] = 60.0
    df_features.loc[df_features.index[-1], "valid"] = True

    # Negative sentiment below threshold
    sig = signal(df_features, settings, sentiment=-0.3)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "long_blocked_by_negative_sentiment"


def test_signal_short_blocked_by_positive_sentiment(
    sample_ohlc_df: pd.DataFrame, settings: Settings
) -> None:
    """Test SHORT signal blocked by positive sentiment."""
    df_features = compute_features(sample_ohlc_df, settings)

    # Set conditions for SHORT signal
    df_features.loc[df_features.index[-1], "close"] = 90.0
    df_features.loc[df_features.index[-1], "sma20"] = 100.0
    df_features.loc[df_features.index[-1], "rsi14"] = 40.0
    df_features.loc[df_features.index[-1], "valid"] = True

    # Positive sentiment above threshold
    sig = signal(df_features, settings, sentiment=0.3)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "short_blocked_by_positive_sentiment"


def test_signal_no_valid_rows(insufficient_data_df: pd.DataFrame, settings: Settings) -> None:
    """Test signal returns FLAT when no valid feature rows exist."""
    df_features = compute_features(insufficient_data_df, settings)

    sig = signal(df_features, settings, sentiment=0.0)

    assert sig.direction == "FLAT"
    assert sig.strength == 0.0
    assert sig.meta is not None
    assert sig.meta["reason"] == "no_valid_data"


def test_signal_missing_feature_columns(sample_ohlc_df: pd.DataFrame, settings: Settings) -> None:
    """Test signal raises ValueError for missing feature columns."""
    # DataFrame without features
    df_no_features = sample_ohlc_df.copy()

    with pytest.raises(ValueError, match="Missing required feature columns"):
        signal(df_no_features, settings, sentiment=0.0)


def test_signal_sentiment_default_none(sample_ohlc_df: pd.DataFrame, settings: Settings) -> None:
    """Test signal with sentiment=None defaults to 0.0."""
    df_features = compute_features(sample_ohlc_df, settings)

    # Set conditions for LONG signal
    df_features.loc[df_features.index[-1], "close"] = 110.0
    df_features.loc[df_features.index[-1], "sma20"] = 100.0
    df_features.loc[df_features.index[-1], "rsi14"] = 60.0
    df_features.loc[df_features.index[-1], "valid"] = True

    sig = signal(df_features, settings, sentiment=None)

    assert sig.direction == "LONG"
    assert sig.meta is not None
    assert sig.meta["sentiment"] == 0.0
