"""Unit tests for sentiment providers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
import requests

from src.services.sentiment.base import ProviderError, RateLimitExceeded
from src.services.sentiment.enhanced_cache import EnhancedSentimentCache
from src.services.sentiment.providers.news_api import NewsAPIProvider
from src.services.sentiment.providers.stub import StubSentimentProvider
from src.services.sentiment.providers.twitter_api import TwitterAPIProvider
from src.services.sentiment.registry import SentimentProviderRegistry
from src.services.sentiment.types import SentimentScore


class TestStubSentimentProvider:
    """Tests for stub sentiment provider."""

    def test_stub_returns_sentiment_score(self):
        """Test stub provider returns valid sentiment score."""
        provider = StubSentimentProvider()
        score = provider.get_sentiment("TEST")

        assert score is not None
        assert isinstance(score, SentimentScore)
        assert -1.0 <= score.value <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert score.source == "stub"
        assert score.metadata["is_mock"] is True

    def test_stub_with_fixed_value(self):
        """Test stub provider with fixed value."""
        provider = StubSentimentProvider(fixed_value=0.75)
        score = provider.get_sentiment("TEST")

        assert score is not None
        assert score.value == 0.75
        assert score.confidence == 0.5
        assert score.source == "stub"

    def test_stub_is_always_healthy(self):
        """Test stub provider is always healthy."""
        provider = StubSentimentProvider()
        assert provider.is_healthy() is True

    def test_stub_metadata(self):
        """Test stub provider metadata."""
        provider = StubSentimentProvider()
        metadata = provider.get_metadata()

        assert metadata.name == "stub"
        assert metadata.version == "1.0.0"
        assert metadata.rate_limit_per_minute == 1000
        assert metadata.supports_async is False


class TestNewsAPIProvider:
    """Tests for NewsAPI sentiment provider."""

    def test_newsapi_requires_api_key(self):
        """Test NewsAPI provider requires API key."""
        with pytest.raises(ValueError, match="API key is required"):
            NewsAPIProvider(api_key="")

    def test_newsapi_successful_fetch(self):
        """Test successful sentiment fetch from NewsAPI."""
        provider = NewsAPIProvider(api_key="test_key", rate_limit_per_minute=1000)

        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Stock surge on bullish profit growth",
                    "description": "Strong gains in quarterly earnings",
                },
                {
                    "title": "Market rally continues",
                    "description": "Positive momentum in tech sector",
                },
            ],
        }

        with patch("requests.get", return_value=mock_response):
            score = provider.get_sentiment("RELIANCE")

        assert score is not None
        assert isinstance(score, SentimentScore)
        assert score.source == "newsapi"
        assert score.value > 0  # Positive sentiment from positive keywords
        assert score.confidence > 0
        assert score.metadata["article_count"] == 2

    def test_newsapi_no_articles(self):
        """Test NewsAPI with no articles found."""
        provider = NewsAPIProvider(api_key="test_key", rate_limit_per_minute=1000)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "articles": []}

        with patch("requests.get", return_value=mock_response):
            score = provider.get_sentiment("UNKNOWN")

        assert score is None

    def test_newsapi_rate_limit_handling(self):
        """Test NewsAPI rate limit handling."""
        provider = NewsAPIProvider(api_key="test_key", rate_limit_per_minute=1000)

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(RateLimitExceeded, match="rate limit exceeded"):
                provider.get_sentiment("TEST")

        # Provider should be marked unhealthy
        assert provider.is_healthy() is False

    def test_newsapi_timeout_handling(self):
        """Test NewsAPI timeout handling."""
        provider = NewsAPIProvider(api_key="test_key", timeout=1)

        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            with pytest.raises(ProviderError, match="timeout"):
                provider.get_sentiment("TEST")

        assert provider.is_healthy() is False

    def test_newsapi_api_error_handling(self):
        """Test NewsAPI error response handling."""
        provider = NewsAPIProvider(api_key="test_key")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "error", "message": "Invalid API key"}

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(ProviderError, match="Invalid API key"):
                provider.get_sentiment("TEST")

    def test_newsapi_metadata(self):
        """Test NewsAPI provider metadata."""
        provider = NewsAPIProvider(api_key="test_key", rate_limit_per_minute=100)
        metadata = provider.get_metadata()

        assert metadata.name == "newsapi"
        assert metadata.version == "1.0.0"
        assert metadata.rate_limit_per_minute == 100


class TestTwitterAPIProvider:
    """Tests for Twitter API sentiment provider."""

    def test_twitter_requires_bearer_token(self):
        """Test Twitter provider requires bearer token."""
        with pytest.raises(ValueError, match="bearer token is required"):
            TwitterAPIProvider(bearer_token="")

    def test_twitter_successful_fetch(self):
        """Test successful sentiment fetch from Twitter."""
        provider = TwitterAPIProvider(bearer_token="test_token", rate_limit_per_minute=1000)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "text": "Bullish on this stock! Going to the moon rocket",
                    "public_metrics": {"like_count": 100, "retweet_count": 50},
                },
                {
                    "text": "Strong buy signal with great momentum",
                    "public_metrics": {"like_count": 200, "retweet_count": 75},
                },
            ]
        }

        with patch("requests.get", return_value=mock_response):
            score = provider.get_sentiment("RELIANCE")

        assert score is not None
        assert isinstance(score, SentimentScore)
        assert score.source == "twitter"
        assert score.value > 0  # Positive sentiment from positive keywords
        assert score.confidence > 0
        assert score.metadata["tweet_count"] == 2

    def test_twitter_no_tweets(self):
        """Test Twitter with no tweets found."""
        provider = TwitterAPIProvider(bearer_token="test_token", rate_limit_per_minute=1000)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("requests.get", return_value=mock_response):
            score = provider.get_sentiment("UNKNOWN")

        assert score is None

    def test_twitter_rate_limit_handling(self):
        """Test Twitter rate limit handling."""
        provider = TwitterAPIProvider(bearer_token="test_token", rate_limit_per_minute=1000)

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"x-rate-limit-reset": "1234567890"}

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(RateLimitExceeded, match="rate limit exceeded"):
                provider.get_sentiment("TEST")

        assert provider.is_healthy() is False

    def test_twitter_authentication_error(self):
        """Test Twitter authentication error handling."""
        provider = TwitterAPIProvider(bearer_token="invalid_token")

        mock_response = Mock()
        mock_response.status_code = 401

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(ProviderError, match="authentication failed"):
                provider.get_sentiment("TEST")

        assert provider.is_healthy() is False

    def test_twitter_timeout_handling(self):
        """Test Twitter timeout handling."""
        provider = TwitterAPIProvider(bearer_token="test_token", timeout=1)

        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            with pytest.raises(ProviderError, match="timeout"):
                provider.get_sentiment("TEST")

    def test_twitter_metadata(self):
        """Test Twitter provider metadata."""
        provider = TwitterAPIProvider(bearer_token="test_token", rate_limit_per_minute=450)
        metadata = provider.get_metadata()

        assert metadata.name == "twitter"
        assert metadata.version == "1.0.0"
        assert metadata.rate_limit_per_minute == 450


class TestSentimentProviderRegistry:
    """Tests for sentiment provider registry."""

    def test_registry_register_provider(self):
        """Test registering a provider."""
        registry = SentimentProviderRegistry()
        provider = StubSentimentProvider(fixed_value=0.5)

        registry.register("stub", provider, weight=1.0, priority=0)

        assert "stub" in registry.providers
        assert registry.weights["stub"] == 1.0
        assert "stub" in registry.fallback_order

    def test_registry_unregister_provider(self):
        """Test unregistering a provider."""
        registry = SentimentProviderRegistry()
        provider = StubSentimentProvider()

        registry.register("stub", provider)
        registry.unregister("stub")

        assert "stub" not in registry.providers
        assert "stub" not in registry.weights
        assert "stub" not in registry.fallback_order

    def test_registry_single_provider(self):
        """Test registry with single provider."""
        registry = SentimentProviderRegistry()
        provider = StubSentimentProvider(fixed_value=0.75)
        registry.register("stub", provider, weight=1.0, priority=0)

        score = registry.get_sentiment("TEST")

        assert score is not None
        assert score.value == 0.75
        assert score.source == "stub"

    def test_registry_weighted_average(self):
        """Test registry with weighted average of multiple providers."""
        registry = SentimentProviderRegistry()

        provider1 = StubSentimentProvider(fixed_value=0.6)
        provider2 = StubSentimentProvider(fixed_value=0.4)

        registry.register("provider1", provider1, weight=0.6, priority=0)
        registry.register("provider2", provider2, weight=0.4, priority=1)

        score = registry.get_sentiment("TEST", use_weighted_average=True)

        assert score is not None
        # Weighted average: 0.6 * 0.6 + 0.4 * 0.4 = 0.52
        assert abs(score.value - 0.52) < 0.01
        assert "hybrid" in score.source
        assert "provider1" in score.metadata["providers"]
        assert "provider2" in score.metadata["providers"]

    def test_registry_fallback_on_provider_failure(self):
        """Test registry fallback when provider fails."""
        registry = SentimentProviderRegistry()

        # First provider that will fail
        failing_provider = Mock()
        failing_provider.get_sentiment_with_timing.side_effect = ProviderError("API down")

        # Second provider that succeeds
        working_provider = StubSentimentProvider(fixed_value=0.5)

        registry.register("failing", failing_provider, weight=1.0, priority=0)
        registry.register("working", working_provider, weight=1.0, priority=1)

        score = registry.get_sentiment("TEST", use_weighted_average=False)

        assert score is not None
        assert score.value == 0.5
        assert score.source == "stub"

    def test_registry_circuit_breaker(self):
        """Test circuit breaker opens after consecutive failures."""
        registry = SentimentProviderRegistry(circuit_breaker_threshold=3)

        failing_provider = Mock()
        failing_provider.get_sentiment_with_timing.side_effect = ProviderError("Always fails")

        registry.register("failing", failing_provider, weight=1.0, priority=0)

        # Trigger 3 failures
        for _ in range(3):
            try:
                registry.get_sentiment("TEST")
            except Exception:
                pass

        # Circuit breaker should be open
        stats = registry._circuit_breakers["failing"]
        assert stats.is_open is True

    def test_registry_get_provider_health(self):
        """Test getting provider health metrics."""
        registry = SentimentProviderRegistry()
        provider = StubSentimentProvider()
        registry.register("stub", provider, weight=1.0, priority=0)

        # Make a successful request
        registry.get_sentiment("TEST")

        health = registry.get_provider_health()

        assert "stub" in health
        assert health["stub"]["is_healthy"] is True
        assert health["stub"]["circuit_breaker_open"] is False
        assert health["stub"]["success_rate"] == 100.0

    def test_registry_provider_stats(self):
        """Test provider statistics tracking."""
        registry = SentimentProviderRegistry()
        provider = StubSentimentProvider()
        registry.register("stub", provider, weight=1.0, priority=0)

        # Make multiple requests
        for _ in range(5):
            registry.get_sentiment("TEST")

        stats = registry.get_provider_stats("stub")

        assert "stub" in stats
        assert stats["stub"].successful_requests == 5
        assert stats["stub"].total_requests == 5
        assert stats["stub"].success_rate == 100.0

    def test_registry_reset_stats(self):
        """Test resetting provider statistics."""
        registry = SentimentProviderRegistry()
        provider = StubSentimentProvider()
        registry.register("stub", provider, weight=1.0, priority=0)

        # Make requests
        registry.get_sentiment("TEST")

        # Reset stats
        registry.reset_stats("stub")

        stats = registry.get_provider_stats("stub")
        assert stats["stub"].total_requests == 0


class TestEnhancedSentimentCache:
    """Tests for enhanced sentiment cache."""

    def test_cache_set_and_get(self):
        """Test basic cache set and get."""
        cache = EnhancedSentimentCache(default_ttl_seconds=3600)
        score = SentimentScore(
            value=0.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
        )

        cache.set("TEST", score)
        cached_score = cache.get("TEST", "test")

        assert cached_score is not None
        assert cached_score.value == 0.5
        assert cached_score.source == "test"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = EnhancedSentimentCache()
        cached_score = cache.get("NONEXISTENT", "test")

        assert cached_score is None

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = EnhancedSentimentCache(default_ttl_seconds=1)
        score = SentimentScore(
            value=0.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
        )

        cache.set("TEST", score)

        # Wait for expiration
        import time

        time.sleep(1.1)

        cached_score = cache.get("TEST", "test")
        assert cached_score is None

    def test_cache_provider_stats(self):
        """Test cache provider statistics."""
        cache = EnhancedSentimentCache()
        score = SentimentScore(
            value=0.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
        )

        # Cache miss
        cache.set("TEST", score)

        # Cache hit
        cache.get("TEST", "test")

        stats = cache.get_provider_stats("test")
        assert "test" in stats
        assert stats["test"].hits == 1
        assert stats["test"].misses == 1

    def test_cache_record_error(self):
        """Test recording provider errors."""
        cache = EnhancedSentimentCache()
        cache.record_error("test", "API error")

        stats = cache.get_provider_stats("test")
        assert stats["test"].errors == 1
        assert stats["test"].last_error_msg == "API error"

    def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = EnhancedSentimentCache(default_ttl_seconds=1)
        score = SentimentScore(
            value=0.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
        )

        cache.set("TEST1", score)
        cache.set("TEST2", score)

        import time

        time.sleep(1.1)

        removed = cache.cleanup_expired()
        assert removed == 2

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = EnhancedSentimentCache()
        score = SentimentScore(
            value=0.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
        )

        cache.set("TEST1", score)
        cache.set("TEST2", score)

        cache.clear()

        assert cache.get("TEST1", "test") is None
        assert cache.get("TEST2", "test") is None

    def test_cache_max_size_eviction(self):
        """Test cache eviction when max size reached."""
        cache = EnhancedSentimentCache(max_cache_size=2)
        score = SentimentScore(
            value=0.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
        )

        cache.set("TEST1", score)
        cache.set("TEST2", score)
        cache.set("TEST3", score)  # Should evict oldest

        # TEST1 should be evicted
        assert cache.get("TEST1", "test") is None
        assert cache.get("TEST2", "test") is not None
        assert cache.get("TEST3", "test") is not None
