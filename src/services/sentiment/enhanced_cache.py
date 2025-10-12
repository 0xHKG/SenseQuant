"""Enhanced sentiment cache with provider-level statistics and audit trails."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.services.sentiment.types import SentimentScore

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCacheEntry:
    """Enhanced cache entry with provider metadata and statistics.

    Attributes:
        score: Cached sentiment score
        timestamp: Cache entry creation time
        ttl_seconds: Time-to-live for this entry
        provider_name: Provider that generated this score
        access_count: Number of times this entry was accessed
        last_access: Last access timestamp
    """

    score: SentimentScore
    timestamp: float
    ttl_seconds: int
    provider_name: str
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if cache entry has expired.

        Returns:
            True if expired, False otherwise
        """
        age = time.time() - self.timestamp
        return age >= self.ttl_seconds

    def ttl_remaining(self) -> float:
        """Calculate remaining TTL in seconds.

        Returns:
            Remaining TTL (0 if expired)
        """
        age = time.time() - self.timestamp
        remaining = self.ttl_seconds - age
        return max(0.0, remaining)


@dataclass
class ProviderCacheStats:
    """Statistics for a sentiment provider's cache performance.

    Attributes:
        provider_name: Provider name
        hits: Number of cache hits
        misses: Number of cache misses
        errors: Number of provider errors
        total_latency_ms: Cumulative latency for all requests
        request_count: Total number of requests
        last_success_ts: Last successful fetch timestamp
        last_error_ts: Last error timestamp
        last_error_msg: Last error message
    """

    provider_name: str
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    request_count: int = 0
    last_success_ts: datetime | None = None
    last_error_ts: datetime | None = None
    last_error_msg: str | None = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.hits / self.request_count) * 100.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.errors / self.request_count) * 100.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count


class EnhancedSentimentCache:
    """Enhanced sentiment cache with provider-level statistics and audit trails.

    Features:
    - Per-provider cache statistics (hits, misses, latency)
    - Audit trail persistence (last successful payloads)
    - Configurable TTL per entry
    - Automatic cleanup of expired entries
    - Export/import for persistence across restarts
    """

    def __init__(
        self,
        default_ttl_seconds: int = 3600,
        max_cache_size: int = 1000,
        audit_dir: Path | None = None,
    ):
        """Initialize enhanced sentiment cache.

        Args:
            default_ttl_seconds: Default TTL for cache entries (default: 1 hour)
            max_cache_size: Maximum number of cache entries (default: 1000)
            audit_dir: Directory for audit trail persistence (None = disabled)
        """
        self._cache: dict[str, EnhancedCacheEntry] = {}
        self._stats: dict[str, ProviderCacheStats] = {}
        self.default_ttl_seconds = default_ttl_seconds
        self.max_cache_size = max_cache_size
        self.audit_dir = Path(audit_dir) if audit_dir else None

        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        symbol: str,
        provider_name: str | None = None,
    ) -> SentimentScore | None:
        """Get cached sentiment score for symbol.

        Args:
            symbol: Stock symbol
            provider_name: Optional provider filter (None = any provider)

        Returns:
            Cached SentimentScore or None if not found/expired
        """
        # Build cache key
        cache_key: str
        if provider_name:
            cache_key = f"{symbol}:{provider_name}"
        else:
            # Look for any provider
            best_entry = self._find_best_entry(symbol)
            if not best_entry:
                return None
            cache_key = best_entry

        # Check if entry exists and is valid
        entry = self._cache.get(cache_key)
        if not entry:
            return None

        if entry.is_expired():
            logger.debug(f"Cache entry expired for {cache_key}")
            del self._cache[cache_key]
            return None

        # Update access stats
        entry.access_count += 1
        entry.last_access = time.time()

        # Record cache hit
        if entry.provider_name not in self._stats:
            self._stats[entry.provider_name] = ProviderCacheStats(entry.provider_name)
        self._stats[entry.provider_name].hits += 1
        self._stats[entry.provider_name].request_count += 1

        logger.debug(
            f"Cache hit for {symbol} (provider={entry.provider_name}, "
            f"ttl_remaining={entry.ttl_remaining():.1f}s, access_count={entry.access_count})"
        )

        return entry.score

    def set(
        self,
        symbol: str,
        score: SentimentScore,
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache sentiment score for symbol.

        Args:
            symbol: Stock symbol
            score: Sentiment score to cache
            ttl_seconds: TTL override (None = use default)
        """
        cache_key = f"{symbol}:{score.source}"
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        # Enforce max cache size
        if len(self._cache) >= self.max_cache_size:
            self._evict_oldest()

        # Create cache entry
        entry = EnhancedCacheEntry(
            score=score,
            timestamp=time.time(),
            ttl_seconds=ttl,
            provider_name=score.source,
        )

        self._cache[cache_key] = entry

        # Record cache miss (new entry)
        if score.source not in self._stats:
            self._stats[score.source] = ProviderCacheStats(score.source)
        self._stats[score.source].misses += 1
        self._stats[score.source].request_count += 1
        self._stats[score.source].last_success_ts = datetime.now(timezone.utc)

        # Persist audit trail
        if self.audit_dir:
            self._persist_audit_trail(symbol, score)

        logger.debug(f"Cached sentiment for {symbol} (provider={score.source}, ttl={ttl}s)")

    def record_error(self, provider_name: str, error_msg: str) -> None:
        """Record provider error for statistics.

        Args:
            provider_name: Provider name
            error_msg: Error message
        """
        if provider_name not in self._stats:
            self._stats[provider_name] = ProviderCacheStats(provider_name)

        stats = self._stats[provider_name]
        stats.errors += 1
        stats.request_count += 1
        stats.last_error_ts = datetime.now(timezone.utc)
        stats.last_error_msg = error_msg

    def record_latency(self, provider_name: str, latency_ms: float) -> None:
        """Record provider latency for statistics.

        Args:
            provider_name: Provider name
            latency_ms: Request latency in milliseconds
        """
        if provider_name not in self._stats:
            self._stats[provider_name] = ProviderCacheStats(provider_name)

        stats = self._stats[provider_name]
        stats.total_latency_ms += latency_ms

    def get_provider_stats(self, provider_name: str | None = None) -> dict[str, ProviderCacheStats]:
        """Get cache statistics for provider(s).

        Args:
            provider_name: Provider name (None = all providers)

        Returns:
            Dictionary mapping provider names to statistics
        """
        if provider_name:
            if provider_name not in self._stats:
                return {provider_name: ProviderCacheStats(provider_name)}
            return {provider_name: self._stats[provider_name]}

        return self._stats.copy()

    def get_cache_info(self) -> dict[str, Any]:
        """Get overall cache information.

        Returns:
            Dictionary with cache metadata
        """
        total_entries = len(self._cache)
        expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())

        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "max_size": self.max_cache_size,
            "utilization": (total_entries / self.max_cache_size) * 100.0,
            "providers": list(self._stats.keys()),
        }

    def clear(self, provider_name: str | None = None) -> None:
        """Clear cache entries.

        Args:
            provider_name: Provider name (None = clear all)
        """
        if provider_name:
            # Clear entries for specific provider
            keys_to_delete = [
                key for key, entry in self._cache.items() if entry.provider_name == provider_name
            ]
            for key in keys_to_delete:
                del self._cache[key]
            logger.info(f"Cleared {len(keys_to_delete)} cache entries for {provider_name}")
        else:
            # Clear all entries
            self._cache.clear()
            logger.info("Cleared all cache entries")

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        keys_to_delete = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in keys_to_delete:
            del self._cache[key]

        if keys_to_delete:
            logger.info(f"Cleaned up {len(keys_to_delete)} expired cache entries")

        return len(keys_to_delete)

    def _find_best_entry(self, symbol: str) -> str | None:
        """Find best (most recent, non-expired) cache entry for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Cache key or None if not found
        """
        matching_entries = [
            (key, entry) for key, entry in self._cache.items() if key.startswith(f"{symbol}:")
        ]

        if not matching_entries:
            return None

        # Filter expired entries
        valid_entries = [(key, entry) for key, entry in matching_entries if not entry.is_expired()]

        if not valid_entries:
            return None

        # Return most recent entry
        best_key: str
        best_key, _ = max(valid_entries, key=lambda x: x[1].timestamp)
        return best_key

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry when max size reached."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")

    def _persist_audit_trail(self, symbol: str, score: SentimentScore) -> None:
        """Persist sentiment score to audit trail.

        Args:
            symbol: Stock symbol
            score: Sentiment score
        """
        if not self.audit_dir:
            return

        try:
            audit_file = self.audit_dir / f"{symbol}_{score.source}_last.json"
            data = {
                "symbol": symbol,
                "score": asdict(score),
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(audit_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to persist audit trail for {symbol}: {e}")
