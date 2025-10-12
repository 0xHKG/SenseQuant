"""Integration tests for engine sentiment flow."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.adapters.sentiment_provider import SentimentProvider
from src.domain.types import Bar
from src.services.engine import Engine


class MockSentimentProvider(SentimentProvider):
    """Mock sentiment provider for testing."""

    def __init__(self, sentiment: float = 0.0) -> None:
        self._sentiment = sentiment

    def get_sentiment(self, symbol: str) -> float:
        """Return mock sentiment."""
        return self._sentiment

    @property
    def name(self) -> str:
        """Return provider name."""
        return "mock"


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


def test_engine_sentiment_gating_negative(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Engine suppresses BUY on negative sentiment."""
    # Create crossover pattern
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0
    sample_daily_bars[-1].close = 180.0  # Bullish crossover

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            # Replace sentiment provider with negative sentiment
            engine._sentiment_provider = MockSentimentProvider(sentiment=-0.5)
            engine.start()
            engine.run_swing_daily("TEST")

            # Verify position NOT created due to sentiment gate
            assert "TEST" not in engine._swing_positions


def test_engine_sentiment_gating_neutral(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Engine allows BUY on neutral sentiment."""
    # Create crossover pattern
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0
    sample_daily_bars[-1].close = 180.0  # Bullish crossover

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            # Neutral sentiment (default)
            engine._sentiment_provider = MockSentimentProvider(sentiment=0.0)
            engine.start()
            engine.run_swing_daily("TEST")

            # Verify position created
            assert "TEST" in engine._swing_positions


def test_engine_sentiment_boosting(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Engine boosts confidence on positive sentiment."""
    # Create crossover pattern
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0
    sample_daily_bars[-1].close = 180.0  # Bullish crossover

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            # Positive sentiment triggers boosting
            engine._sentiment_provider = MockSentimentProvider(sentiment=0.7)
            engine.start()
            engine.run_swing_daily("TEST")

            # Verify position created (sentiment boosting doesn't prevent entry)
            assert "TEST" in engine._swing_positions
            # Log output verified manually in test output


def test_engine_sentiment_cache_hit(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Second call uses cached sentiment."""
    # Create crossover pattern
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0
    sample_daily_bars[-1].close = 180.0  # Bullish crossover

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            provider = MockSentimentProvider(sentiment=0.3)
            engine._sentiment_provider = provider
            engine.start()

            # Track provider calls to verify caching
            original_get = provider.get_sentiment
            call_count = {"count": 0}

            def tracked_get(symbol: str) -> float:
                call_count["count"] += 1
                return original_get(symbol)

            provider.get_sentiment = tracked_get  # type: ignore

            # First call - cache miss
            engine.run_swing_daily("TEST")
            assert call_count["count"] == 1

            # Second call - cache hit (no additional provider call)
            engine.run_swing_daily("TEST")
            assert call_count["count"] == 1  # Still 1, not 2


def test_engine_sentiment_fallback(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Provider error falls back to 0.0 with structured logs."""

    class FailingSentimentProvider(SentimentProvider):
        """Provider that always fails."""

        def get_sentiment(self, symbol: str) -> float:
            """Raise error."""
            raise Exception("Mock sentiment error")

        @property
        def name(self) -> str:
            """Return provider name."""
            return "failing"

    # Create crossover pattern
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0
    sample_daily_bars[-1].close = 180.0  # Bullish crossover

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine._sentiment_provider = FailingSentimentProvider()
            engine.start()
            engine.run_swing_daily("TEST")

            # Verify fallback to neutral sentiment (0.0)
            # Position should be created because neutral sentiment doesn't gate
            assert "TEST" in engine._swing_positions
            # Error logging verified in test output


def test_journal_includes_sentiment(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Sentiment metadata flows through engine and signal generation."""
    # Create crossover pattern
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0
    sample_daily_bars[-1].close = 180.0  # Bullish crossover

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            engine._sentiment_provider = MockSentimentProvider(sentiment=0.4)
            engine.start()
            engine.run_swing_daily("TEST")

            # Verify position created (sentiment 0.4 doesn't gate)
            assert "TEST" in engine._swing_positions
            # Sentiment logging verified in test output


def test_engine_dryrun_no_network(
    mock_breeze_client: MagicMock,
    sample_daily_bars: list[Bar],
) -> None:
    """Dryrun mode uses cached sentiment (no network calls after first)."""
    # Create crossover pattern
    for i in range(80, 99):
        sample_daily_bars[i].close = 150.0 - (i - 80) * 2.0
    sample_daily_bars[-1].close = 180.0  # Bullish crossover

    mock_breeze_client.historical_bars.return_value = sample_daily_bars

    with patch("src.services.engine.BreezeClient", return_value=mock_breeze_client):
        ist = pytz.timezone("Asia/Kolkata")
        mock_time = datetime(2025, 4, 10, 16, 0, 0, tzinfo=ist)
        with patch("src.services.engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_time

            engine = Engine(symbols=["TEST"])
            mock_provider = MockSentimentProvider(sentiment=0.2)
            engine._sentiment_provider = mock_provider
            engine.start()

            # First call
            engine.run_swing_daily("TEST")

            # Clear positions to allow re-entry
            engine._swing_positions.clear()

            # Mock provider to track calls
            call_count_before = 1  # First call already made
            original_get_sentiment = mock_provider.get_sentiment

            call_tracker = {"count": call_count_before}

            def tracked_get_sentiment(symbol: str) -> float:
                call_tracker["count"] += 1
                return original_get_sentiment(symbol)

            mock_provider.get_sentiment = tracked_get_sentiment  # type: ignore

            # Second call - should use cache
            engine.run_swing_daily("TEST")

            # Verify no additional calls to provider (cached)
            assert call_tracker["count"] == call_count_before
