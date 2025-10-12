"""Integration tests for sentiment provider pipeline."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.app.config import Settings
from src.services.engine import Engine
from src.services.sentiment.factory import create_sentiment_registry
from src.services.sentiment.providers.stub import StubSentimentProvider
from src.services.sentiment.registry import SentimentProviderRegistry


class TestSentimentProviderIntegration:
    """Integration tests for sentiment provider pipeline."""

    def test_engine_with_stub_registry(self):
        """Test Engine with stub sentiment registry."""
        registry = SentimentProviderRegistry()
        registry.register("stub", StubSentimentProvider(fixed_value=0.5), weight=1.0, priority=0)

        engine = Engine(symbols=["TEST"], sentiment_registry=registry)

        # Engine should have registry
        assert engine._sentiment_registry is not None

        # Get health should work
        health = engine.get_sentiment_health()
        assert health is not None
        assert "stub" in health
        assert health["stub"]["is_healthy"] is True

    def test_engine_without_registry_backward_compat(self):
        """Test Engine without registry (backward compatibility)."""
        engine = Engine(symbols=["TEST"])

        # Engine should work without registry
        assert engine._sentiment_registry is None

        # Get health should return None
        health = engine.get_sentiment_health()
        assert health is None

    def test_multi_provider_sentiment_aggregation(self):
        """Test sentiment aggregation from multiple providers."""
        registry = SentimentProviderRegistry()

        # Register multiple providers with different values
        provider1 = StubSentimentProvider(fixed_value=0.8)
        provider2 = StubSentimentProvider(fixed_value=0.4)

        registry.register("provider1", provider1, weight=0.6, priority=0)
        registry.register("provider2", provider2, weight=0.4, priority=1)

        # Get weighted sentiment
        score = registry.get_sentiment("TEST", use_weighted_average=True)

        assert score is not None
        # Weighted average: 0.8 * 0.6 + 0.4 * 0.4 = 0.64
        assert abs(score.value - 0.64) < 0.01
        assert "hybrid" in score.source
        assert len(score.metadata["providers"]) == 2

    def test_provider_fallback_on_failure(self):
        """Test provider fallback when one fails."""
        registry = SentimentProviderRegistry()

        # First provider that fails
        failing_provider = Mock()
        failing_provider.get_sentiment_with_timing.return_value = (None, 100.0)

        # Second provider that works
        working_provider = StubSentimentProvider(fixed_value=0.7)

        registry.register("failing", failing_provider, weight=1.0, priority=0)
        registry.register("working", working_provider, weight=1.0, priority=1)

        # Get sentiment should fallback to working provider
        score = registry.get_sentiment("TEST", use_weighted_average=False)

        assert score is not None
        assert score.value == 0.7
        assert score.source == "stub"

    def test_circuit_breaker_integration(self):
        """Test circuit breaker prevents repeated failures."""
        registry = SentimentProviderRegistry(circuit_breaker_threshold=2)

        # Provider that always fails
        failing_provider = Mock()
        failing_provider.get_sentiment_with_timing.side_effect = Exception("Always fails")

        # Backup provider
        backup_provider = StubSentimentProvider(fixed_value=0.5)

        registry.register("failing", failing_provider, weight=1.0, priority=0)
        registry.register("backup", backup_provider, weight=1.0, priority=1)

        # First two requests should try failing provider
        for _ in range(2):
            registry.get_sentiment("TEST")

        # Circuit breaker should be open now
        health = registry.get_provider_health()
        assert health["failing"]["circuit_breaker_open"] is True

        # Next request should skip failing provider entirely
        score = registry.get_sentiment("TEST")
        assert score is not None
        assert score.source == "stub"  # Only backup provider used

    def test_factory_creates_registry_from_settings_no_providers(self):
        """Test factory creates stub registry when no providers enabled."""
        settings = Settings(
            sentiment_enable_newsapi=False,
            sentiment_enable_twitter=False,
        )

        registry = create_sentiment_registry(settings)

        # Should have stub provider as fallback
        assert "stub" in registry.providers
        assert len(registry.providers) == 1

    def test_provider_stats_tracking(self):
        """Test provider statistics are tracked correctly."""
        registry = SentimentProviderRegistry()
        provider = StubSentimentProvider(fixed_value=0.5)
        registry.register("stub", provider, weight=1.0, priority=0)

        # Make multiple requests
        for i in range(10):
            registry.get_sentiment(f"TEST{i}")

        # Check stats
        stats = registry.get_provider_stats("stub")
        assert stats["stub"].total_requests == 10
        assert stats["stub"].successful_requests == 10
        assert stats["stub"].success_rate == 100.0

    def test_health_check_reflects_provider_state(self):
        """Test health check accurately reflects provider state."""
        registry = SentimentProviderRegistry()

        # Healthy provider
        healthy_provider = StubSentimentProvider()

        # Unhealthy provider (mock)
        unhealthy_provider = Mock()
        unhealthy_provider.is_healthy.return_value = False
        unhealthy_provider.get_sentiment_with_timing.return_value = (None, 0.0)

        registry.register("healthy", healthy_provider, weight=1.0, priority=0)
        registry.register("unhealthy", unhealthy_provider, weight=1.0, priority=1)

        health = registry.get_provider_health()

        assert health["healthy"]["is_healthy"] is True
        assert health["unhealthy"]["is_healthy"] is False

    def test_sentiment_score_validation(self):
        """Test sentiment score validation."""
        from datetime import datetime, timezone

        from src.services.sentiment.types import SentimentScore

        # Valid score
        score = SentimentScore(
            value=0.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
        )
        assert score.value == 0.5

        # Invalid value (out of range)
        with pytest.raises(ValueError, match="Sentiment value must be between"):
            SentimentScore(
                value=1.5, confidence=0.8, source="test", timestamp=datetime.now(timezone.utc)
            )

        # Invalid confidence (out of range)
        with pytest.raises(ValueError, match="Confidence must be between"):
            SentimentScore(
                value=0.5, confidence=1.5, source="test", timestamp=datetime.now(timezone.utc)
            )
