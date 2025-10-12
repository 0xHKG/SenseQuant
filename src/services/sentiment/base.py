"""Abstract base class for sentiment providers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, cast

from src.services.sentiment.types import ProviderMetadata, SentimentScore

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class SentimentProvider(ABC):
    """Abstract interface for sentiment data providers.

    All sentiment providers must implement this interface to ensure
    consistent behavior across different data sources (NewsAPI, Twitter, etc.).
    """

    @abstractmethod
    def get_sentiment(self, symbol: str, **kwargs: Any) -> SentimentScore | None:
        """Fetch sentiment for a given symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            **kwargs: Provider-specific parameters

        Returns:
            SentimentScore if successful, None if no data available

        Raises:
            RateLimitExceeded: If rate limit is exceeded
            ProviderError: If provider encounters an error
        """
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if provider is currently available and healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ProviderMetadata:
        """Get provider metadata.

        Returns:
            ProviderMetadata describing provider capabilities
        """
        pass

    def get_sentiment_with_timing(
        self, symbol: str, **kwargs: Any
    ) -> tuple[SentimentScore | None, float]:
        """Fetch sentiment and measure latency.

        Args:
            symbol: Stock symbol
            **kwargs: Provider-specific parameters

        Returns:
            Tuple of (SentimentScore or None, latency_ms)
        """
        start_time = time.time()
        try:
            score = self.get_sentiment(symbol, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            return score, latency_ms
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return None, latency_ms


class RateLimitExceededError(Exception):
    """Raised when provider rate limit is exceeded."""

    pass


class ProviderError(Exception):
    """Raised when provider encounters an error."""

    pass


# Backward compatibility alias
RateLimitExceeded = RateLimitExceededError


class RateLimiter:
    """Simple token bucket rate limiter.

    Attributes:
        capacity: Maximum number of tokens (requests per minute)
        tokens: Current number of available tokens
        last_refill: Timestamp of last token refill
    """

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.capacity = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.last_refill = time.time()
        self.refill_rate = requests_per_minute / 60.0  # tokens per second

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens acquired, False if rate limit exceeded
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now

    def wait_time(self) -> float:
        """Calculate time to wait until next token is available.

        Returns:
            Seconds to wait
        """
        self._refill()
        if self.tokens >= 1:
            return 0.0
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate


def exponential_backoff(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0
) -> Callable[[F], F]:
    """Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitExceeded:
                    # Don't retry on rate limit - let caller handle it
                    raise
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise

                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception if last_exception else Exception("Unknown error")

        return cast(F, wrapper)

    return decorator
