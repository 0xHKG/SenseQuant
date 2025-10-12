"""TTL cache with rate-limit guard for sentiment providers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.adapters.sentiment_provider import SentimentProvider


@dataclass
class CacheEntry:
    """Cached sentiment entry with TTL and metadata."""

    value: float
    timestamp: float
    provider: str


class SentimentCache:
    """TTL cache with rate-limit guard for sentiment providers."""

    def __init__(self, ttl_seconds: int = 3600, rate_limit_per_min: int = 10) -> None:
        """
        Initialize sentiment cache.

        Args:
            ttl_seconds: Time-to-live for cached entries (default: 1 hour)
            rate_limit_per_min: Max requests per minute per symbol (default: 10)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds
        self._rate_limit = rate_limit_per_min
        self._request_times: dict[str, list[float]] = {}  # symbol -> [timestamps]

    def get(
        self, symbol: str, provider: SentimentProvider, fallback: float = 0.0
    ) -> tuple[float, dict[str, bool | float | str]]:
        """
        Get sentiment with caching and rate-limit guard.

        Args:
            symbol: Stock symbol
            provider: Sentiment provider instance
            fallback: Default value on errors (default: 0.0 = neutral)

        Returns:
            (sentiment_score, metadata)
            metadata includes: cache_hit, ttl_remaining, rate_limited, etc.
        """
        cache_key = f"sentiment:{symbol}:{provider.name}"
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            age = now - entry.timestamp
            if age < self._ttl:
                # Cache hit
                ttl_remaining = self._ttl - age
                logger.debug(
                    f"Sentiment cache hit for {symbol}",
                    extra={
                        "component": "sentiment_cache",
                        "symbol": symbol,
                        "provider": provider.name,
                        "ttl_remaining": ttl_remaining,
                    },
                )
                return entry.value, {
                    "cache_hit": True,
                    "ttl_remaining": ttl_remaining,
                    "provider": provider.name,
                }

        # Check rate limit
        if self._is_rate_limited(symbol):
            logger.warning(
                f"Sentiment rate limit exceeded for {symbol}",
                extra={"component": "sentiment_cache", "symbol": symbol},
            )
            # Return cached value if available, else fallback
            if cache_key in self._cache:
                return self._cache[cache_key].value, {
                    "cache_hit": False,
                    "rate_limited": True,
                    "provider": provider.name,
                }
            return fallback, {
                "cache_hit": False,
                "rate_limited": True,
                "fallback": True,
            }

        # Fetch from provider
        try:
            value = provider.get_sentiment(symbol)
            self._cache[cache_key] = CacheEntry(value, now, provider.name)
            self._record_request(symbol)

            logger.debug(
                f"Sentiment cache miss for {symbol}",
                extra={
                    "component": "sentiment_cache",
                    "symbol": symbol,
                    "provider": provider.name,
                    "value": value,
                },
            )

            return value, {
                "cache_hit": False,
                "provider": provider.name,
                "rate_limited": False,
            }
        except Exception as e:
            logger.error(
                f"Sentiment provider failed for {symbol}: {e}",
                extra={
                    "component": "sentiment_cache",
                    "symbol": symbol,
                    "provider": provider.name,
                    "error": str(e),
                },
            )
            # Return cached value if available, else fallback
            if cache_key in self._cache:
                return self._cache[cache_key].value, {
                    "cache_hit": True,
                    "stale": True,
                    "error": str(e),
                }
            return fallback, {
                "cache_hit": False,
                "fallback": True,
                "error": str(e),
            }

    def _is_rate_limited(self, symbol: str) -> bool:
        """
        Check if symbol has exceeded rate limit.

        Args:
            symbol: Stock symbol

        Returns:
            True if rate limit exceeded, False otherwise
        """
        now = time.time()
        window_start = now - 60  # Last 60 seconds

        if symbol not in self._request_times:
            return False

        # Clean old requests
        self._request_times[symbol] = [t for t in self._request_times[symbol] if t > window_start]

        return len(self._request_times[symbol]) >= self._rate_limit

    def _record_request(self, symbol: str) -> None:
        """
        Record a request timestamp for rate limiting.

        Args:
            symbol: Stock symbol
        """
        if symbol not in self._request_times:
            self._request_times[symbol] = []
        self._request_times[symbol].append(time.time())

    def clear(self) -> None:
        """Clear all cached entries (useful for testing)."""
        self._cache.clear()
        self._request_times.clear()
