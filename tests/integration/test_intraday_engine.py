"""Integration tests for intraday engine."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

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
def mock_sentiment() -> MagicMock:
    """Create mock sentiment provider."""
    sentiment_mock = MagicMock(return_value=0.0)
    return sentiment_mock


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Generate sample bars for testing."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    bars = []
    for i in range(100):
        ts = pd.Timestamp(now - timedelta(minutes=100 - i), tz=ist)
        bars.append(
            Bar(
                ts=ts,
                open=100.0 + i * 0.1,
                high=102.0 + i * 0.1,
                low=98.0 + i * 0.1,
                close=101.0 + i * 0.1,
                volume=1000 + i * 10,
            )
        )
    return bars


def test_tick_intraday_within_window(
    mock_breeze_client: MagicMock,
    mock_sentiment: MagicMock,
    sample_bars: list[Bar],
) -> None:
    """Test tick_intraday processes signal within trading window."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        with patch("src.services.engine.get_sentiment", mock_sentiment):
            # Mock current time to be within window (10:00 IST)
            ist = pytz.timezone("Asia/Kolkata")
            mock_time = datetime(2025, 10, 11, 10, 0, 0, tzinfo=ist)
            with patch("src.services.engine.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time

                engine = Engine(symbols=["RELIANCE"])
                engine.start()
                engine.tick_intraday("RELIANCE")

                # Verify historical_bars was called
                assert mock_breeze_client.historical_bars.called
                # Verify sentiment was fetched
                assert mock_sentiment.called


def test_tick_intraday_outside_window_before(
    mock_breeze_client: MagicMock,
    mock_sentiment: MagicMock,
) -> None:
    """Test tick_intraday skips processing before market open (9:15 IST)."""
    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        with patch("src.services.engine.get_sentiment", mock_sentiment):
            # Mock current time to be before window (9:00 IST)
            ist = pytz.timezone("Asia/Kolkata")
            mock_time = datetime(2025, 10, 11, 9, 0, 0, tzinfo=ist)
            with patch("src.services.engine.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time

                engine = Engine(symbols=["RELIANCE"])
                engine.start()
                engine.tick_intraday("RELIANCE")

                # Verify historical_bars was NOT called (outside window)
                assert not mock_breeze_client.historical_bars.called


def test_tick_intraday_outside_window_after(
    mock_breeze_client: MagicMock,
    mock_sentiment: MagicMock,
) -> None:
    """Test tick_intraday skips processing after market close (15:29 IST)."""
    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        with patch("src.services.engine.get_sentiment", mock_sentiment):
            # Mock current time to be after window (15:35 IST)
            ist = pytz.timezone("Asia/Kolkata")
            mock_time = datetime(2025, 10, 11, 15, 35, 0, tzinfo=ist)
            with patch("src.services.engine.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time

                engine = Engine(symbols=["RELIANCE"])
                engine.start()
                engine.tick_intraday("RELIANCE")

                # Verify historical_bars was NOT called (outside window)
                assert not mock_breeze_client.historical_bars.called


def test_tick_intraday_at_start_edge(
    mock_breeze_client: MagicMock,
    mock_sentiment: MagicMock,
    sample_bars: list[Bar],
) -> None:
    """Test tick_intraday processes at exactly 9:15 IST (start edge)."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        with patch("src.services.engine.get_sentiment", mock_sentiment):
            # Mock current time to be exactly at start (9:15 IST)
            ist = pytz.timezone("Asia/Kolkata")
            mock_time = datetime(2025, 10, 11, 9, 15, 0, tzinfo=ist)
            with patch("src.services.engine.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time

                engine = Engine(symbols=["RELIANCE"])
                engine.start()
                engine.tick_intraday("RELIANCE")

                # Verify historical_bars WAS called (at start edge)
                assert mock_breeze_client.historical_bars.called


def test_tick_intraday_at_end_edge(
    mock_breeze_client: MagicMock,
    mock_sentiment: MagicMock,
    sample_bars: list[Bar],
) -> None:
    """Test tick_intraday processes at exactly 15:29 IST (end edge)."""
    mock_breeze_client.historical_bars.return_value = sample_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        with patch("src.services.engine.get_sentiment", mock_sentiment):
            # Mock current time to be exactly at end (15:29 IST)
            ist = pytz.timezone("Asia/Kolkata")
            mock_time = datetime(2025, 10, 11, 15, 29, 0, tzinfo=ist)
            with patch("src.services.engine.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time

                engine = Engine(symbols=["RELIANCE"])
                engine.start()
                engine.tick_intraday("RELIANCE")

                # Verify historical_bars WAS called (at end edge)
                assert mock_breeze_client.historical_bars.called


def test_square_off_intraday_with_long_position(
    mock_breeze_client: MagicMock,
) -> None:
    """Test square_off_intraday closes long position."""
    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["RELIANCE"])
        engine.start()

        # Set up open long position
        engine._open_positions["RELIANCE"] = 100  # 100 shares long

        engine.square_off_intraday("RELIANCE")

        # In dryrun mode, should not call place_order, but should log to journal
        assert engine._open_positions["RELIANCE"] == 0


def test_square_off_intraday_with_short_position(
    mock_breeze_client: MagicMock,
) -> None:
    """Test square_off_intraday closes short position."""
    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["RELIANCE"])
        engine.start()

        # Set up open short position
        engine._open_positions["RELIANCE"] = -50  # 50 shares short

        engine.square_off_intraday("RELIANCE")

        # Position should be cleared
        assert engine._open_positions["RELIANCE"] == 0


def test_square_off_intraday_no_position(
    mock_breeze_client: MagicMock,
) -> None:
    """Test square_off_intraday with no open position does nothing."""
    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        engine = Engine(symbols=["RELIANCE"])
        engine.start()

        # No position set
        engine.square_off_intraday("RELIANCE")

        # Should not crash, position remains 0
        assert engine._open_positions.get("RELIANCE", 0) == 0


def test_tick_intraday_handles_fetch_error(
    mock_breeze_client: MagicMock,
    mock_sentiment: MagicMock,
) -> None:
    """Test tick_intraday handles historical_bars fetch error gracefully."""
    # Simulate fetch error
    mock_breeze_client.historical_bars.side_effect = Exception("Network error")

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        with patch("src.services.engine.get_sentiment", mock_sentiment):
            ist = pytz.timezone("Asia/Kolkata")
            mock_time = datetime(2025, 10, 11, 10, 0, 0, tzinfo=ist)
            with patch("src.services.engine.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time

                engine = Engine(symbols=["RELIANCE"])
                engine.start()

                # Should not crash
                engine.tick_intraday("RELIANCE")

                # Verify error was logged but execution continued
                assert mock_breeze_client.historical_bars.called


def test_tick_intraday_handles_empty_bars(
    mock_breeze_client: MagicMock,
    mock_sentiment: MagicMock,
) -> None:
    """Test tick_intraday handles empty bars gracefully."""
    mock_breeze_client.historical_bars.return_value = []

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        with patch("src.services.engine.get_sentiment", mock_sentiment):
            ist = pytz.timezone("Asia/Kolkata")
            mock_time = datetime(2025, 10, 11, 10, 0, 0, tzinfo=ist)
            with patch("src.services.engine.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time

                engine = Engine(symbols=["RELIANCE"])
                engine.start()

                # Should not crash
                engine.tick_intraday("RELIANCE")

                # Verify bars were fetched but processing stopped gracefully
                assert mock_breeze_client.historical_bars.called
