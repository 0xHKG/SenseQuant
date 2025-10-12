"""Sentiment analysis services with pluggable provider architecture."""

from src.services.sentiment.base import (
    ProviderError,
    RateLimiter,
    RateLimitExceeded,
    SentimentProvider,
)
from src.services.sentiment.cache import SentimentCache
from src.services.sentiment.enhanced_cache import EnhancedSentimentCache
from src.services.sentiment.factory import create_sentiment_registry
from src.services.sentiment.providers.news_api import NewsAPIProvider
from src.services.sentiment.providers.stub import StubSentimentProvider
from src.services.sentiment.providers.twitter_api import TwitterAPIProvider
from src.services.sentiment.registry import SentimentProviderRegistry
from src.services.sentiment.types import (
    CircuitBreakerState,
    ProviderMetadata,
    ProviderStats,
    SentimentScore,
)

__all__ = [
    "SentimentProvider",
    "SentimentProviderRegistry",
    "SentimentScore",
    "ProviderMetadata",
    "ProviderStats",
    "CircuitBreakerState",
    "RateLimiter",
    "RateLimitExceeded",
    "ProviderError",
    "SentimentCache",
    "EnhancedSentimentCache",
    "StubSentimentProvider",
    "NewsAPIProvider",
    "TwitterAPIProvider",
    "create_sentiment_registry",
]
