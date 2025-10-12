"""Sentiment provider registry with weighted averaging and fallback logic."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.services.sentiment.base import ProviderError, RateLimitExceeded, SentimentProvider
from src.services.sentiment.types import CircuitBreakerState, ProviderStats, SentimentScore

logger = logging.getLogger(__name__)


class SentimentProviderRegistry:
    """Registry managing multiple sentiment providers with fallback and weighted averaging.

    Supports:
    - Multiple providers with configurable weights
    - Fallback ordering when providers fail
    - Circuit breaker pattern to disable unhealthy providers
    - Provider health monitoring and statistics
    - Weighted averaging of multiple provider scores
    """

    def __init__(
        self,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_minutes: int = 30,
    ):
        """Initialize provider registry.

        Args:
            circuit_breaker_threshold: Number of consecutive failures before opening circuit
            circuit_breaker_cooldown_minutes: Minutes to wait before attempting half-open state
        """
        self.providers: dict[str, SentimentProvider] = {}
        self.weights: dict[str, float] = {}
        self.fallback_order: list[str] = []
        self._stats: dict[str, ProviderStats] = {}
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = timedelta(minutes=circuit_breaker_cooldown_minutes)

    def register(
        self,
        name: str,
        provider: SentimentProvider,
        weight: float = 1.0,
        priority: int = 0,
    ) -> None:
        """Register a sentiment provider.

        Args:
            name: Unique provider name (e.g., "newsapi", "twitter")
            provider: SentimentProvider instance
            weight: Weight for weighted averaging (default: 1.0)
            priority: Priority for fallback ordering (lower = higher priority)
        """
        if name in self.providers:
            logger.warning(f"Provider {name} already registered, replacing")

        self.providers[name] = provider
        self.weights[name] = weight
        self._stats[name] = ProviderStats()
        self._circuit_breakers[name] = CircuitBreakerState(
            failure_threshold=self.circuit_breaker_threshold
        )

        # Update fallback order based on priority
        self.fallback_order.append(name)
        self.fallback_order.sort(key=lambda x: priority)

        logger.info(
            f"Registered sentiment provider: {name} (weight={weight:.2f}, priority={priority})"
        )

    def unregister(self, name: str) -> None:
        """Unregister a sentiment provider.

        Args:
            name: Provider name to remove
        """
        if name in self.providers:
            del self.providers[name]
            del self.weights[name]
            del self._stats[name]
            del self._circuit_breakers[name]
            self.fallback_order.remove(name)
            logger.info(f"Unregistered sentiment provider: {name}")

    def get_sentiment(
        self,
        symbol: str,
        use_weighted_average: bool = True,
        **kwargs: Any,
    ) -> SentimentScore | None:
        """Fetch sentiment using registered providers with fallback logic.

        Args:
            symbol: Stock symbol
            use_weighted_average: If True, combine multiple provider scores; if False, use first successful
            **kwargs: Provider-specific parameters

        Returns:
            SentimentScore or None if all providers fail
        """
        if not self.providers:
            logger.warning("No sentiment providers registered")
            return None

        scores: list[tuple[SentimentScore, float, str]] = []  # (score, weight, provider_name)

        for provider_name in self.fallback_order:
            if not self._is_provider_available(provider_name):
                logger.debug(f"Skipping {provider_name} (circuit breaker open)")
                continue

            provider = self.providers[provider_name]
            weight = self.weights[provider_name]

            try:
                score, latency_ms = provider.get_sentiment_with_timing(symbol, **kwargs)

                if score:
                    scores.append((score, weight, provider_name))
                    self._record_success(provider_name, latency_ms)
                    logger.info(
                        f"{provider_name} sentiment for {symbol}: {score.value:.3f} "
                        f"(confidence={score.confidence:.2f}, latency={latency_ms:.1f}ms)"
                    )

                    # If not using weighted average, return first successful score
                    if not use_weighted_average:
                        return score
                else:
                    self._record_failure(provider_name, "No data returned", latency_ms)
                    logger.warning(f"{provider_name} returned no sentiment data for {symbol}")

            except RateLimitExceeded as e:
                self._record_failure(provider_name, str(e), 0.0)
                logger.warning(f"{provider_name} rate limit exceeded for {symbol}")
            except ProviderError as e:
                self._record_failure(provider_name, str(e), 0.0)
                logger.error(f"{provider_name} error for {symbol}: {e}")
            except Exception as e:
                self._record_failure(provider_name, str(e), 0.0)
                logger.error(f"{provider_name} unexpected error for {symbol}: {e}", exc_info=True)

        # If no providers returned data, return None
        if not scores:
            logger.warning(f"All sentiment providers failed for {symbol}")
            return None

        # If only one provider, return its score
        if len(scores) == 1:
            return scores[0][0]

        # Weighted average of multiple provider scores
        return self._compute_weighted_average(scores)

    def _compute_weighted_average(
        self, scores: list[tuple[SentimentScore, float, str]]
    ) -> SentimentScore:
        """Compute weighted average of multiple sentiment scores.

        Args:
            scores: List of (SentimentScore, weight, provider_name) tuples

        Returns:
            Aggregated SentimentScore
        """
        total_weight = sum(weight for _, weight, _ in scores)
        if total_weight == 0:
            total_weight = 1.0  # Fallback to equal weighting

        # Weighted average of values
        weighted_value = sum(score.value * weight for score, weight, _ in scores) / total_weight

        # Weighted average of confidence
        weighted_confidence = (
            sum(score.confidence * weight for score, weight, _ in scores) / total_weight
        )

        # Combine provider names
        provider_names = [name for _, _, name in scores]
        combined_source = f"hybrid[{','.join(provider_names)}]"

        return SentimentScore(
            value=weighted_value,
            confidence=weighted_confidence,
            source=combined_source,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "providers": provider_names,
                "weights": {name: weight for _, weight, name in scores},
                "individual_scores": {name: score.value for score, _, name in scores},
            },
        )

    def _is_provider_available(self, provider_name: str) -> bool:
        """Check if provider is available (circuit breaker closed).

        Args:
            provider_name: Provider name

        Returns:
            True if provider is available, False if circuit breaker is open
        """
        breaker = self._circuit_breakers[provider_name]

        # If circuit is not open, provider is available
        if not breaker.is_open:
            return True

        # Check if cooldown period has passed
        if breaker.cooldown_until and datetime.now(timezone.utc) >= breaker.cooldown_until:
            logger.info(f"Circuit breaker for {provider_name} entering half-open state")
            breaker.is_open = False
            breaker.failure_count = 0
            return True

        return False

    def _record_success(self, provider_name: str, latency_ms: float) -> None:
        """Record successful provider request.

        Args:
            provider_name: Provider name
            latency_ms: Request latency in milliseconds
        """
        stats = self._stats[provider_name]
        stats.total_requests += 1
        stats.successful_requests += 1
        stats.last_success_ts = datetime.now(timezone.utc)
        stats.consecutive_failures = 0

        # Update rolling average latency
        if stats.avg_latency_ms == 0:
            stats.avg_latency_ms = latency_ms
        else:
            # Exponential moving average with alpha=0.2
            stats.avg_latency_ms = 0.8 * stats.avg_latency_ms + 0.2 * latency_ms

        # Close circuit breaker if it was open
        breaker = self._circuit_breakers[provider_name]
        if breaker.is_open:
            logger.info(f"Circuit breaker for {provider_name} closed after successful request")
            breaker.is_open = False
            breaker.failure_count = 0

    def _record_failure(self, provider_name: str, error_msg: str, latency_ms: float) -> None:
        """Record failed provider request.

        Args:
            provider_name: Provider name
            error_msg: Error message
            latency_ms: Request latency in milliseconds
        """
        stats = self._stats[provider_name]
        stats.total_requests += 1
        stats.failed_requests += 1
        stats.last_error = error_msg
        stats.consecutive_failures += 1

        # Update breaker
        breaker = self._circuit_breakers[provider_name]
        breaker.failure_count += 1
        breaker.last_failure_ts = datetime.now(timezone.utc)

        # Open circuit breaker if threshold exceeded
        if breaker.failure_count >= breaker.failure_threshold and not breaker.is_open:
            breaker.is_open = True
            breaker.cooldown_until = datetime.now(timezone.utc) + self.circuit_breaker_cooldown
            logger.error(
                f"Circuit breaker OPEN for {provider_name} after {breaker.failure_count} failures. "
                f"Cooldown until {breaker.cooldown_until.isoformat()}"
            )

    def get_provider_stats(self, provider_name: str | None = None) -> dict[str, ProviderStats]:
        """Get statistics for one or all providers.

        Args:
            provider_name: Provider name (None for all providers)

        Returns:
            Dictionary mapping provider names to ProviderStats
        """
        if provider_name:
            if provider_name not in self._stats:
                raise ValueError(f"Unknown provider: {provider_name}")
            return {provider_name: self._stats[provider_name]}

        return self._stats.copy()

    def get_provider_health(self) -> dict[str, dict[str, Any]]:
        """Get health status for all providers.

        Returns:
            Dictionary mapping provider names to health metrics
        """
        health = {}

        for name, provider in self.providers.items():
            stats = self._stats[name]
            breaker = self._circuit_breakers[name]

            health[name] = {
                "is_healthy": provider.is_healthy() and not breaker.is_open,
                "circuit_breaker_open": breaker.is_open,
                "success_rate": stats.success_rate,
                "error_rate": stats.error_rate,
                "avg_latency_ms": stats.avg_latency_ms,
                "consecutive_failures": stats.consecutive_failures,
                "last_success": stats.last_success_ts.isoformat()
                if stats.last_success_ts
                else None,
                "last_error": stats.last_error,
            }

        return health

    def reset_stats(self, provider_name: str | None = None) -> None:
        """Reset statistics for one or all providers.

        Args:
            provider_name: Provider name (None for all providers)
        """
        if provider_name:
            if provider_name in self._stats:
                self._stats[provider_name] = ProviderStats()
                self._circuit_breakers[provider_name] = CircuitBreakerState(
                    failure_threshold=self.circuit_breaker_threshold
                )
                logger.info(f"Reset stats for {provider_name}")
        else:
            for name in self._stats:
                self._stats[name] = ProviderStats()
                self._circuit_breakers[name] = CircuitBreakerState(
                    failure_threshold=self.circuit_breaker_threshold
                )
            logger.info("Reset stats for all providers")
