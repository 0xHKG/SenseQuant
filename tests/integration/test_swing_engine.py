"""Integration tests for swing engine."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.domain.strategies.swing import SwingPosition
from src.domain.types import Bar
from src.services.engine import Engine


@pytest.fixture
def mock_breeze_client() -> MagicMock:
    """Create mock Breeze client."""
    client = MagicMock()
    client.authenticate = MagicMock()
    client.historical_bars = MagicMock()
    client.place_order = MagicMock()
    return client


@pytest.fixture
def sample_daily_bars() -> list[Bar]:
    """Generate sample daily bars."""
    ist = pytz.timezone("Asia/Kolkata")
    # Create bars ending on 2025-04-10 (same as mock time in tests)
    # 100 bars ending on 2025-04-10 means starting from 2025-01-01
    base_date = pd.Timestamp("2025-01-01", tz=ist)
    bars = []
    for i in range(100):
        ts = base_date + pd.Timedelta(days=i)
        bars.append(
            Bar(
                ts=ts,
                open=100.0 + i * 0.5,
                high=102.0 + i * 0.5,
                low=98.0 + i * 0.5,
                close=101.0 + i * 0.5,
                volume=10000,
            )
        )
    return bars


def test_run_swing_daily_entry_on_crossover(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test swing entry on bullish crossover."""
    # Force crossover: create downtrend then sharp reversal
    # Set bars 80-98 to create a downtrend (fast SMA will fall below slow SMA)
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0  # Downtrend
    # Then sharp reversal on last bar
    sample_daily_bars[-1].close = 180.0  # Sharp up, fast crosses above slow

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine.start()
            engine.run_swing_daily("TEST")

            # Verify position created
            assert "TEST" in engine._swing_positions


def test_run_swing_daily_tp_exit(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test swing TP exit."""
    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine.start()

            # Create open position
            engine._swing_positions["TEST"] = SwingPosition(
                symbol="TEST",
                direction="LONG",
                entry_price=100.0,
                entry_date=sample_daily_bars[-10].ts,
                qty=10,
            )

            # Force TP condition
            sample_daily_bars[-1].close = 107.0

            engine.run_swing_daily("TEST")

            # Verify position closed
            assert "TEST" not in engine._swing_positions


def test_run_swing_daily_holiday_skip(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test swing skips on holiday (no new bar)."""
    # Last bar is yesterday
    ist = pytz.timezone("Asia/Kolkata")
    yesterday = datetime(2025, 4, 9, tzinfo=ist)
    sample_daily_bars[-1].ts = pd.Timestamp(yesterday)

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine.start()
            engine.run_swing_daily("TEST")

            # No position should be created
            assert "TEST" not in engine._swing_positions


def test_run_swing_daily_gap_exit(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test swing gap exit."""
    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine.start()

            # Create open position
            engine._swing_positions["TEST"] = SwingPosition(
                symbol="TEST",
                direction="LONG",
                entry_price=100.0,
                entry_date=sample_daily_bars[-5].ts,
                qty=10,
            )

            # Force gap down to SL at open
            sample_daily_bars[-1].open = 92.0
            sample_daily_bars[-1].close = 95.0

            engine.run_swing_daily("TEST")

            # Verify position closed
            assert "TEST" not in engine._swing_positions


def test_run_swing_daily_dryrun_no_orders(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Test dryrun mode doesn't place orders."""
    # Force crossover
    sample_daily_bars[-2].close = 100.0
    sample_daily_bars[-1].close = 110.0

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine.start()
            engine.run_swing_daily("TEST")

            # Verify place_order was NOT called (dryrun mode)
            assert not mock_breeze_client.place_order.called
