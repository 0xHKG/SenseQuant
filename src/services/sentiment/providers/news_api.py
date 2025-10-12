"""NewsAPI sentiment provider implementation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from src.services.sentiment.base import (
    ProviderError,
    RateLimiter,
    RateLimitExceeded,
    SentimentProvider,
    exponential_backoff,
)
from src.services.sentiment.types import ProviderMetadata, SentimentScore

logger = logging.getLogger(__name__)


class NewsAPIProvider(SentimentProvider):
    """Sentiment provider using NewsAPI for news articles.

    Features:
    - Rate limiting with token bucket algorithm
    - Exponential backoff for transient errors
    - Keyword-based sentiment analysis
    - Configurable timeout and retry logic

    Attributes:
        api_key: NewsAPI API key
        endpoint: NewsAPI base URL (default: https://newsapi.org/v2)
        rate_limiter: Rate limiter instance
        timeout: Request timeout in seconds
    """

    # Sentiment keywords for simple analysis
    POSITIVE_KEYWORDS = [
        "surge",
        "profit",
        "growth",
        "bullish",
        "gain",
        "rally",
        "soar",
        "jump",
        "record",
        "strong",
        "beat",
        "upgrade",
        "positive",
        "success",
        "momentum",
    ]

    NEGATIVE_KEYWORDS = [
        "loss",
        "decline",
        "bearish",
        "plunge",
        "crash",
        "fall",
        "drop",
        "weak",
        "miss",
        "downgrade",
        "negative",
        "concern",
        "risk",
        "warning",
        "slump",
    ]

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://newsapi.org/v2",
        rate_limit_per_minute: int = 100,
        timeout: int = 5,
    ):
        """Initialize NewsAPI provider.

        Args:
            api_key: NewsAPI API key
            endpoint: NewsAPI base URL (default: https://newsapi.org/v2)
            rate_limit_per_minute: Max requests per minute (default: 100)
            timeout: Request timeout in seconds (default: 5)

        Raises:
            ValueError: If api_key is empty
        """
        if not api_key:
            raise ValueError("NewsAPI API key is required")

        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.timeout = timeout
        self._is_healthy = True

    @exponential_backoff(max_retries=3, base_delay=1.0, max_delay=10.0)
    def get_sentiment(self, symbol: str, **kwargs: Any) -> SentimentScore | None:
        """Fetch sentiment from NewsAPI.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            **kwargs: Optional parameters:
                - language: Language filter (default: "en")
                - page_size: Number of articles (default: 20)
                - sort_by: Sort order (default: "publishedAt")

        Returns:
            SentimentScore or None if no articles found

        Raises:
            RateLimitExceeded: If rate limit is exceeded
            ProviderError: If API request fails
        """
        # Check rate limit
        if not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            raise RateLimitExceeded(f"NewsAPI rate limit exceeded, wait {wait_time:.1f}s")

        # Build request parameters
        params = {
            "q": symbol,
            "apiKey": self.api_key,
            "language": kwargs.get("language", "en"),
            "pageSize": kwargs.get("page_size", 20),
            "sortBy": kwargs.get("sort_by", "publishedAt"),
        }

        try:
            # Make API request
            response = requests.get(
                f"{self.endpoint}/everything", params=params, timeout=self.timeout
            )

            # Handle rate limit responses
            if response.status_code == 429:
                self._is_healthy = False
                raise RateLimitExceeded("NewsAPI rate limit exceeded (429)")

            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if data.get("status") == "error":
                error_msg = data.get("message", "Unknown error")
                self._is_healthy = False
                raise ProviderError(f"NewsAPI error: {error_msg}")

            articles = data.get("articles", [])

            if not articles:
                logger.info(f"No NewsAPI articles found for {symbol}")
                return None

            # Analyze sentiment from articles
            sentiment_value = self._analyze_articles(articles)
            confidence = self._compute_confidence(len(articles))

            self._is_healthy = True

            return SentimentScore(
                value=sentiment_value,
                confidence=confidence,
                source="newsapi",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "article_count": len(articles),
                    "query": symbol,
                    "language": params["language"],
                },
            )

        except requests.exceptions.Timeout as e:
            self._is_healthy = False
            raise ProviderError(f"NewsAPI request timeout after {self.timeout}s") from e
        except requests.exceptions.RequestException as e:
            self._is_healthy = False
            raise ProviderError(f"NewsAPI request failed: {e}") from e

    def _analyze_articles(self, articles: list[dict[str, Any]]) -> float:
        """Analyze sentiment from article titles and descriptions.

        Uses simple keyword-based sentiment analysis. In production, this
        should be replaced with a proper NLP model (VADER, BERT, etc.).

        Args:
            articles: List of article dictionaries

        Returns:
            Sentiment value between -1.0 and 1.0
        """
        if not articles:
            return 0.0

        total_score = 0.0
        articles_analyzed = 0

        # Analyze up to 20 most recent articles
        for article in articles[:20]:
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            text = f"{title} {description}"

            if not text.strip():
                continue

            # Count positive and negative keywords
            positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text)
            negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text)

            # Calculate article score
            article_score = (positive_count - negative_count) / max(
                1, positive_count + negative_count
            )
            total_score += article_score
            articles_analyzed += 1

        if articles_analyzed == 0:
            return 0.0

        # Average sentiment across articles
        avg_sentiment = total_score / articles_analyzed

        # Clamp to [-1.0, 1.0] range
        return max(-1.0, min(1.0, avg_sentiment))

    def _compute_confidence(self, article_count: int) -> float:
        """Compute confidence based on number of articles.

        More articles = higher confidence.

        Args:
            article_count: Number of articles analyzed

        Returns:
            Confidence between 0.0 and 1.0
        """
        if article_count == 0:
            return 0.0

        # Logarithmic scaling: 1 article = 0.3, 10 articles = 0.7, 50+ articles = 0.9
        if article_count >= 50:
            return 0.9
        elif article_count >= 20:
            return 0.8
        elif article_count >= 10:
            return 0.7
        elif article_count >= 5:
            return 0.6
        else:
            return 0.3 + (article_count - 1) * 0.05

    def is_healthy(self) -> bool:
        """Check if provider is healthy.

        Returns:
            True if last request succeeded, False otherwise
        """
        return self._is_healthy

    def get_metadata(self) -> ProviderMetadata:
        """Get provider metadata.

        Returns:
            ProviderMetadata for NewsAPI provider
        """
        return ProviderMetadata(
            name="newsapi",
            version="1.0.0",
            rate_limit_per_minute=self.rate_limiter.capacity,
            supports_async=False,
        )
