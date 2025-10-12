"""Type definitions for sentiment analysis services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SentimentScore:
    """Standardized sentiment score returned by all providers.

    Attributes:
        value: Sentiment value between -1.0 (bearish) and 1.0 (bullish)
        confidence: Confidence score between 0.0 (low) and 1.0 (high)
        source: Provider name (e.g., "newsapi", "twitter", "hybrid")
        timestamp: When the sentiment was computed (UTC)
        metadata: Optional additional metadata from provider
    """

    value: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate sentiment score values."""
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"Sentiment value must be between -1.0 and 1.0, got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class ProviderMetadata:
    """Metadata describing a sentiment provider.

    Attributes:
        name: Provider name (e.g., "newsapi", "twitter")
        version: Provider version (e.g., "1.0.0")
        rate_limit_per_minute: Max requests per minute
        supports_async: Whether provider supports async operations
    """

    name: str
    version: str
    rate_limit_per_minute: int
    supports_async: bool = False


@dataclass
class ProviderStats:
    """Statistics for monitoring sentiment provider health.

    Attributes:
        total_requests: Total number of requests made
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        avg_latency_ms: Average latency in milliseconds
        last_success_ts: Timestamp of last successful request
        last_error: Last error message (if any)
        consecutive_failures: Number of consecutive failures
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    last_success_ts: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        return 100.0 - self.success_rate


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a sentiment provider.

    Attributes:
        is_open: Whether circuit is open (provider disabled)
        failure_count: Number of consecutive failures
        last_failure_ts: Timestamp of last failure
        cooldown_until: Timestamp when circuit can transition to half-open
        failure_threshold: Number of failures before opening circuit
    """

    is_open: bool = False
    failure_count: int = 0
    last_failure_ts: datetime | None = None
    cooldown_until: datetime | None = None
    failure_threshold: int = 5
