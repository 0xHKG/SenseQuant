"""Unit tests for sentiment cache."""

from __future__ import annotations

from unittest.mock import patch

from src.adapters.sentiment_provider import SentimentProvider, SentimentProviderError
from src.services.sentiment import SentimentCache


class MockSentimentProvider(SentimentProvider):
    """Mock sentiment provider for testing."""

    def __init__(self, sentiment: float = 0.5) -> None:
        self._sentiment = sentiment
        self.call_count = 0

    def get_sentiment(self, symbol: str) -> float:
        """Return mock sentiment."""
        self.call_count += 1
        return self._sentiment

    @property
    def name(self) -> str:
        """Return provider name."""
        return "mock"


class FailingSentimentProvider(SentimentProvider):
    """Mock sentiment provider that always fails."""

    def get_sentiment(self, symbol: str) -> float:
        """Raise an error."""
        raise SentimentProviderError("Mock error")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "failing"


def test_cache_hit() -> None:
    """Cache returns cached value within TTL."""
    cache = SentimentCache(ttl_seconds=10, rate_limit_per_min=10)
    provider = MockSentimentProvider(sentiment=0.7)

    # First call - cache miss
    value1, meta1 = cache.get("RELIANCE", provider)
    assert value1 == 0.7
    assert meta1["cache_hit"] is False
    assert provider.call_count == 1

    # Second call - cache hit
    value2, meta2 = cache.get("RELIANCE", provider)
    assert value2 == 0.7
    assert meta2["cache_hit"] is True
    assert "ttl_remaining" in meta2
    assert provider.call_count == 1  # No additional call


def test_cache_miss() -> None:
    """Cache fetches from provider on miss."""
    cache = SentimentCache(ttl_seconds=10, rate_limit_per_min=10)
    provider = MockSentimentProvider(sentiment=0.5)

    value, meta = cache.get("TCS", provider)

    assert value == 0.5
    assert meta["cache_hit"] is False
    assert meta["provider"] == "mock"
    assert meta["rate_limited"] is False
    assert provider.call_count == 1


def test_cache_expiry() -> None:
    """Expired cache entry triggers re-fetch."""
    cache = SentimentCache(ttl_seconds=10, rate_limit_per_min=10)
    provider = MockSentimentProvider(sentiment=0.6)

    with patch("src.services.sentiment.cache.time.time") as mock_time:
        # Start at t=0
        mock_time.return_value = 0.0

        # First call at t=0
        value1, meta1 = cache.get("INFY", provider)
        assert value1 == 0.6
        assert meta1["cache_hit"] is False
        assert provider.call_count == 1

        # Advance time past TTL (t=11)
        mock_time.return_value = 11.0

        # Second call after expiry - should re-fetch
        value2, meta2 = cache.get("INFY", provider)
        assert value2 == 0.6
        assert meta2["cache_hit"] is False  # Cache expired, re-fetched
        assert provider.call_count == 2  # Second call made


def test_rate_limit() -> None:
    """Rate limit returns cached/fallback value."""
    cache = SentimentCache(ttl_seconds=5, rate_limit_per_min=2)  # Only 2 requests/min, 5s TTL
    provider = MockSentimentProvider(sentiment=0.8)

    with patch("src.services.sentiment.cache.time.time") as mock_time:
        # Start at t=0
        mock_time.return_value = 0.0

        # First call succeeds at t=0
        value1, meta1 = cache.get("HDFC", provider)
        assert meta1["cache_hit"] is False
        assert value1 == 0.8

        # Advance time past cache TTL but within rate limit window (t=6)
        mock_time.return_value = 6.0

        # Second call (after cache expiry) succeeds at t=6
        value2, meta2 = cache.get("HDFC", provider)
        assert meta2["cache_hit"] is False
        assert value2 == 0.8

        # Advance time past second cache TTL (t=12)
        mock_time.return_value = 12.0

        # Third call should be rate-limited (2 requests already made within 60s, cache expired)
        value3, meta3 = cache.get("HDFC", provider)
        assert meta3.get("rate_limited") is True
        assert value3 == 0.8  # Returns cached value when rate-limited


def test_provider_error_with_cache() -> None:
    """Provider error returns stale cache if using same provider."""
    cache = SentimentCache(ttl_seconds=10, rate_limit_per_min=10)

    # Create a provider that will fail on second call
    class FlakySentimentProvider(SentimentProvider):
        def __init__(self) -> None:
            self.call_count = 0

        def get_sentiment(self, symbol: str) -> float:
            self.call_count += 1
            if self.call_count == 1:
                return 0.7
            raise SentimentProviderError("Mock error on second call")

        @property
        def name(self) -> str:
            return "flaky"

    flaky_provider = FlakySentimentProvider()

    with patch("src.services.sentiment.cache.time.time") as mock_time:
        # Start at t=0
        mock_time.return_value = 0.0

        # First call succeeds at t=0
        value1, meta1 = cache.get("RELIANCE", flaky_provider)
        assert value1 == 0.7

        # Advance past TTL to force cache expiry (t=11)
        mock_time.return_value = 11.0

        # Second call fails but returns stale cache
        value2, meta2 = cache.get("RELIANCE", flaky_provider)
        assert value2 == 0.7  # Stale cached value
        assert meta2["cache_hit"] is True
        assert meta2.get("stale") is True
        assert "error" in meta2


def test_provider_error_without_cache() -> None:
    """Provider error returns fallback 0.0."""
    cache = SentimentCache(ttl_seconds=60, rate_limit_per_min=10)
    provider = FailingSentimentProvider()

    value, meta = cache.get("TCS", provider, fallback=0.0)

    assert value == 0.0  # Fallback
    assert meta["cache_hit"] is False
    assert meta.get("fallback") is True
    assert "error" in meta


def test_rate_limit_window() -> None:
    """Rate limit resets after 60 seconds."""
    cache = SentimentCache(ttl_seconds=5, rate_limit_per_min=2)
    provider = MockSentimentProvider(sentiment=0.9)

    with patch("src.services.sentiment.cache.time.time") as mock_time:
        # Start at t=0
        mock_time.return_value = 0.0

        # First request at t=0
        value1, _ = cache.get("INFY", provider)
        assert value1 == 0.9

        # Advance past cache TTL (t=6)
        mock_time.return_value = 6.0

        # Second request at t=6 (cache expired, within rate limit window)
        value2, _ = cache.get("INFY", provider)
        assert value2 == 0.9

        # Advance past second cache TTL (t=12)
        mock_time.return_value = 12.0

        # Third request should be rate-limited (both requests within 60 seconds, cache expired)
        value3, meta3 = cache.get("INFY", provider)
        assert meta3.get("rate_limited") is True
        assert provider.call_count == 2  # Only 2 successful calls

        # Advance past rate limit window (t=62)
        mock_time.return_value = 62.0

        # Fourth request should succeed (first request at t=0 is now outside 60s window)
        value4, meta4 = cache.get("INFY", provider)
        assert meta4["cache_hit"] is False
        assert provider.call_count == 3  # Third successful call


def test_multiple_symbols_independent() -> None:
    """Different symbols have independent cache entries."""
    cache = SentimentCache(ttl_seconds=60, rate_limit_per_min=10)
    provider = MockSentimentProvider(sentiment=0.5)

    # Fetch for two different symbols
    value1, meta1 = cache.get("RELIANCE", provider)
    value2, meta2 = cache.get("TCS", provider)

    assert value1 == 0.5
    assert value2 == 0.5
    assert meta1["cache_hit"] is False
    assert meta2["cache_hit"] is False
    assert provider.call_count == 2  # Both fetched


def test_multiple_providers_independent() -> None:
    """Different providers have independent cache entries."""
    cache = SentimentCache(ttl_seconds=60, rate_limit_per_min=10)
    provider1 = MockSentimentProvider(sentiment=0.6)

    # Create second provider with different name
    class MockSentimentProvider2(SentimentProvider):
        def __init__(self, sentiment: float = 0.8) -> None:
            self._sentiment = sentiment

        def get_sentiment(self, symbol: str) -> float:
            return self._sentiment

        @property
        def name(self) -> str:
            return "mock2"

    provider2 = MockSentimentProvider2(sentiment=0.8)

    # Fetch same symbol with different providers
    value1, meta1 = cache.get("RELIANCE", provider1)
    value2, meta2 = cache.get("RELIANCE", provider2)

    assert value1 == 0.6
    assert value2 == 0.8
    assert meta1["cache_hit"] is False
    assert meta2["cache_hit"] is False
    assert meta1["provider"] == "mock"
    assert meta2["provider"] == "mock2"


def test_cache_clear() -> None:
    """Clear method removes all cached entries."""
    cache = SentimentCache(ttl_seconds=60, rate_limit_per_min=10)
    provider = MockSentimentProvider(sentiment=0.7)

    # Populate cache
    cache.get("RELIANCE", provider)
    assert provider.call_count == 1

    # Clear cache
    cache.clear()

    # Next fetch should be cache miss
    value, meta = cache.get("RELIANCE", provider)
    assert meta["cache_hit"] is False
    assert provider.call_count == 2  # Re-fetched after clear
