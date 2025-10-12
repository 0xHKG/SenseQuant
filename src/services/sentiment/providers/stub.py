"""Stub sentiment provider for testing and development."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from src.services.sentiment.base import SentimentProvider
from src.services.sentiment.types import ProviderMetadata, SentimentScore


class StubSentimentProvider(SentimentProvider):
    """Stub sentiment provider returning mock sentiment scores.

    Used for testing and as fallback when no real providers are configured.
    Returns random sentiment values between -0.3 and 0.3 (neutral range).
    """

    def __init__(self, fixed_value: float | None = None):
        """Initialize stub provider.

        Args:
            fixed_value: If provided, always return this value (for testing)
        """
        self.fixed_value = fixed_value

    def get_sentiment(self, symbol: str, **kwargs: Any) -> SentimentScore | None:
        """Return mock sentiment score.

        Args:
            symbol: Stock symbol (unused in stub)
            **kwargs: Unused

        Returns:
            Mock SentimentScore with neutral sentiment
        """
        if self.fixed_value is not None:
            value = self.fixed_value
        else:
            # Random sentiment in neutral range
            value = random.uniform(-0.3, 0.3)

        return SentimentScore(
            value=value,
            confidence=0.5,  # Low confidence for stub data
            source="stub",
            timestamp=datetime.now(timezone.utc),
            metadata={"is_mock": True},
        )

    def is_healthy(self) -> bool:
        """Stub provider is always healthy.

        Returns:
            True
        """
        return True

    def get_metadata(self) -> ProviderMetadata:
        """Return stub provider metadata.

        Returns:
            ProviderMetadata for stub provider
        """
        return ProviderMetadata(
            name="stub",
            version="1.0.0",
            rate_limit_per_minute=1000,  # No real rate limit
            supports_async=False,
        )
