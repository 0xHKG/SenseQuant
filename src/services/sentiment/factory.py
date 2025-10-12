"""Factory functions for creating sentiment provider registries."""

from __future__ import annotations

import logging

from src.app.config import Settings
from src.services.sentiment.providers.news_api import NewsAPIProvider
from src.services.sentiment.providers.stub import StubSentimentProvider
from src.services.sentiment.providers.twitter_api import TwitterAPIProvider
from src.services.sentiment.registry import SentimentProviderRegistry

logger = logging.getLogger(__name__)


def create_sentiment_registry(settings: Settings) -> SentimentProviderRegistry:
    """Create sentiment provider registry from settings.

    Automatically registers enabled providers based on configuration.
    Falls back to stub provider if no real providers are configured.

    Args:
        settings: Application settings

    Returns:
        Configured SentimentProviderRegistry
    """
    registry = SentimentProviderRegistry(
        circuit_breaker_threshold=settings.sentiment_circuit_breaker_threshold,
        circuit_breaker_cooldown_minutes=settings.sentiment_circuit_breaker_cooldown,
    )

    # Parse weights and fallback order
    try:
        weights = settings.get_sentiment_provider_weights()
        fallback_order = settings.get_sentiment_provider_fallback_order()
    except ValueError as e:
        logger.error(f"Invalid sentiment provider configuration: {e}")
        logger.warning("Falling back to stub provider")
        registry.register("stub", StubSentimentProvider(), weight=1.0, priority=0)
        return registry

    providers_registered = 0

    # Register NewsAPI provider
    if settings.sentiment_enable_newsapi:
        if not settings.sentiment_newsapi_api_key:
            logger.warning("NewsAPI enabled but no API key provided, skipping")
        else:
            try:
                provider = NewsAPIProvider(
                    api_key=settings.sentiment_newsapi_api_key,
                    endpoint=settings.sentiment_newsapi_endpoint,
                    rate_limit_per_minute=settings.sentiment_newsapi_rate_limit,
                    timeout=settings.sentiment_provider_timeout,
                )
                weight = weights.get("newsapi", 0.5)
                priority = fallback_order.index("newsapi") if "newsapi" in fallback_order else 0
                registry.register("newsapi", provider, weight=weight, priority=priority)
                providers_registered += 1
                logger.info(f"Registered NewsAPI provider (weight={weight}, priority={priority})")
            except Exception as e:
                logger.error(f"Failed to register NewsAPI provider: {e}")

    # Register Twitter provider
    if settings.sentiment_enable_twitter:
        if not settings.sentiment_twitter_bearer_token:
            logger.warning("Twitter enabled but no bearer token provided, skipping")
        else:
            try:
                twitter_provider = TwitterAPIProvider(
                    bearer_token=settings.sentiment_twitter_bearer_token,
                    endpoint=settings.sentiment_twitter_endpoint,
                    rate_limit_per_minute=settings.sentiment_twitter_rate_limit,
                    timeout=settings.sentiment_provider_timeout,
                )
                weight = weights.get("twitter", 0.5)
                priority = fallback_order.index("twitter") if "twitter" in fallback_order else 1
                registry.register("twitter", twitter_provider, weight=weight, priority=priority)
                providers_registered += 1
                logger.info(f"Registered Twitter provider (weight={weight}, priority={priority})")
            except Exception as e:
                logger.error(f"Failed to register Twitter provider: {e}")

    # Fallback to stub provider if no providers registered
    if providers_registered == 0:
        logger.warning("No sentiment providers configured, using stub provider")
        registry.register("stub", StubSentimentProvider(), weight=1.0, priority=0)

    return registry
