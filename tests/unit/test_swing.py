"""Unit tests for swing strategy."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.app.config import Settings
from src.domain.strategies.swing import SwingPosition, compute_features, signal


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        swing_sma_fast=20,
        swing_sma_slow=50,
        swing_rsi_period=14,
        swing_sl_pct=0.03,
        swing_tp_pct=0.06,
        swing_max_hold_days=15,
    )


@pytest.fixture
def sample_daily_df() -> pd.DataFrame:
    """Create sample daily OHLC DataFrame."""
    data = []
    base_date = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(100):
        data.append(
            {
                "ts": base_date + timedelta(days=i),
                "open": 100.0 + i * 0.5,
                "high": 102.0 + i * 0.5,
                "low": 98.0 + i * 0.5,
                "close": 101.0 + i * 0.5,
                "volume": 10000,
            }
        )
    return pd.DataFrame(data)


def test_compute_features_success(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test compute_features with sufficient data."""
    df_features = compute_features(sample_daily_df, settings)

    assert "sma_fast" in df_features.columns
    assert "sma_slow" in df_features.columns
    assert "rsi" in df_features.columns
    assert "valid" in df_features.columns
    assert df_features["valid"].sum() > 0


def test_compute_features_insufficient_data(settings: Settings) -> None:
    """Test compute_features with insufficient data."""
    df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2025-01-01"),
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 1000,
            }
        ]
    )
    df_features = compute_features(df, settings)

    assert df_features["valid"].sum() == 0


def test_compute_features_missing_columns(settings: Settings) -> None:
    """Test compute_features raises ValueError for missing columns."""
    df = pd.DataFrame([{"close": 100.0}])

    with pytest.raises(ValueError, match="Missing required columns"):
        compute_features(df, settings)


def test_signal_bullish_crossover(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test LONG signal on bullish crossover."""
    df_features = compute_features(sample_daily_df, settings)

    # Force crossover: yesterday fast <= slow, today fast > slow
    df_features.loc[df_features.index[-2], "sma_fast"] = 100.0
    df_features.loc[df_features.index[-2], "sma_slow"] = 100.5
    df_features.loc[df_features.index[-1], "sma_fast"] = 101.0
    df_features.loc[df_features.index[-1], "sma_slow"] = 100.0

    sig = signal(df_features, settings)

    assert sig.direction == "LONG"
    assert sig.meta is not None
    assert sig.meta["reason"] == "bull_cross"


def test_signal_bearish_crossunder(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test SHORT signal on bearish crossunder."""
    df_features = compute_features(sample_daily_df, settings)

    # Force crossunder: yesterday fast >= slow, today fast < slow
    df_features.loc[df_features.index[-2], "sma_fast"] = 100.0
    df_features.loc[df_features.index[-2], "sma_slow"] = 99.5
    df_features.loc[df_features.index[-1], "sma_fast"] = 99.0
    df_features.loc[df_features.index[-1], "sma_slow"] = 100.0

    sig = signal(df_features, settings)

    assert sig.direction == "SHORT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "bear_cross"


def test_signal_no_cross_noop(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test FLAT signal when no crossover."""
    df_features = compute_features(sample_daily_df, settings)

    # No crossover
    df_features.loc[df_features.index[-2], "sma_fast"] = 100.0
    df_features.loc[df_features.index[-2], "sma_slow"] = 99.0
    df_features.loc[df_features.index[-1], "sma_fast"] = 101.0
    df_features.loc[df_features.index[-1], "sma_slow"] = 99.5

    sig = signal(df_features, settings)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "noop"


def test_signal_stop_loss_trigger_long(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test SL exit for LONG position."""
    df_features = compute_features(sample_daily_df, settings)

    position = SwingPosition(
        symbol="TEST",
        direction="LONG",
        entry_price=100.0,
        entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
        qty=10,
    )

    # Today's close triggers SL (96.5 = -3.5% loss)
    df_features.loc[df_features.index[-1], "open"] = 99.0  # No gap
    df_features.loc[df_features.index[-1], "close"] = 96.5

    sig = signal(df_features, settings, position=position)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "sl_hit"


def test_signal_take_profit_trigger_long(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test TP exit for LONG position."""
    df_features = compute_features(sample_daily_df, settings)

    position = SwingPosition(
        symbol="TEST",
        direction="LONG",
        entry_price=100.0,
        entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
        qty=10,
    )

    # Today's close triggers TP (107 = +7% gain)
    df_features.loc[df_features.index[-1], "open"] = 101.0  # No gap
    df_features.loc[df_features.index[-1], "close"] = 107.0

    sig = signal(df_features, settings, position=position)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "tp_hit"


def test_signal_max_hold_exit(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test max hold exit."""
    df_features = compute_features(sample_daily_df, settings)

    # Position held for 16 days (> 15 max)
    position = SwingPosition(
        symbol="TEST",
        direction="LONG",
        entry_price=100.0,
        entry_date=df_features.iloc[-1]["ts"] - timedelta(days=16),
        qty=10,
    )

    # Set open/close to avoid gap exit and SL/TP (within -3% to +6% range)
    df_features.loc[df_features.index[-1], "open"] = 101.0
    df_features.loc[df_features.index[-1], "close"] = 102.0  # +2% gain, below TP

    sig = signal(df_features, settings, position=position)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "max_hold"


def test_signal_gap_exit_tp(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test gap exit at open (TP)."""
    df_features = compute_features(sample_daily_df, settings)

    position = SwingPosition(
        symbol="TEST",
        direction="LONG",
        entry_price=100.0,
        entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
        qty=10,
    )

    # Gap up at open triggers TP (108 = +8% gain)
    df_features.loc[df_features.index[-1], "open"] = 108.0
    df_features.loc[df_features.index[-1], "close"] = 105.0

    sig = signal(df_features, settings, position=position)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "gap_exit_tp"
    assert sig.meta["exit_price"] == 108.0


def test_signal_gap_exit_sl(sample_daily_df: pd.DataFrame, settings: Settings) -> None:
    """Test gap exit at open (SL)."""
    df_features = compute_features(sample_daily_df, settings)

    position = SwingPosition(
        symbol="TEST",
        direction="LONG",
        entry_price=100.0,
        entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
        qty=10,
    )

    # Gap down at open triggers SL (92 = -8% loss)
    df_features.loc[df_features.index[-1], "open"] = 92.0
    df_features.loc[df_features.index[-1], "close"] = 95.0

    sig = signal(df_features, settings, position=position)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "gap_exit_sl"
    assert sig.meta["exit_price"] == 92.0


def test_signal_insufficient_data(settings: Settings) -> None:
    """Test signal with no valid data."""
    df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2025-01-01"),
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 1000,
                "sma_fast": None,
                "sma_slow": None,
                "valid": False,
            }
        ]
    )

    sig = signal(df, settings)

    assert sig.direction == "FLAT"
    assert sig.meta is not None
    assert sig.meta["reason"] == "insufficient"
