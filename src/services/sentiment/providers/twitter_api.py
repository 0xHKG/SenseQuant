"""Twitter API sentiment provider implementation."""

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


class TwitterAPIProvider(SentimentProvider):
    """Sentiment provider using Twitter API v2 for social media sentiment.

    Features:
    - Rate limiting with token bucket algorithm
    - Exponential backoff for transient errors
    - Keyword-based sentiment analysis on tweets
    - Configurable timeout and retry logic

    Attributes:
        bearer_token: Twitter API bearer token
        endpoint: Twitter API base URL (default: https://api.twitter.com/2)
        rate_limiter: Rate limiter instance
        timeout: Request timeout in seconds
    """

    # Sentiment keywords for simple analysis
    POSITIVE_KEYWORDS = [
        "bullish",
        "moon",
        "rocket",
        "buy",
        "long",
        "calls",
        "gains",
        "profit",
        "win",
        "surge",
        "rally",
        "breakout",
        "strong",
        "upgrade",
        "beat",
    ]

    NEGATIVE_KEYWORDS = [
        "bearish",
        "crash",
        "dump",
        "sell",
        "short",
        "puts",
        "loss",
        "rekt",
        "plunge",
        "fall",
        "weak",
        "downgrade",
        "miss",
        "warning",
        "fear",
    ]

    def __init__(
        self,
        bearer_token: str,
        endpoint: str = "https://api.twitter.com/2",
        rate_limit_per_minute: int = 450,
        timeout: int = 5,
    ):
        """Initialize Twitter API provider.

        Args:
            bearer_token: Twitter API bearer token
            endpoint: Twitter API base URL (default: https://api.twitter.com/2)
            rate_limit_per_minute: Max requests per minute (default: 450)
            timeout: Request timeout in seconds (default: 5)

        Raises:
            ValueError: If bearer_token is empty
        """
        if not bearer_token:
            raise ValueError("Twitter bearer token is required")

        self.bearer_token = bearer_token
        self.endpoint = endpoint.rstrip("/")
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.timeout = timeout
        self._is_healthy = True

    @exponential_backoff(max_retries=3, base_delay=1.0, max_delay=10.0)
    def get_sentiment(self, symbol: str, **kwargs: Any) -> SentimentScore | None:
        """Fetch sentiment from Twitter API.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            **kwargs: Optional parameters:
                - max_results: Number of tweets (default: 100)
                - use_cashtag: Use $SYMBOL format (default: True for US stocks)

        Returns:
            SentimentScore or None if no tweets found

        Raises:
            RateLimitExceeded: If rate limit is exceeded
            ProviderError: If API request fails
        """
        # Check rate limit
        if not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            raise RateLimitExceeded(f"Twitter rate limit exceeded, wait {wait_time:.1f}s")

        # Build query (use cashtag for better results)
        use_cashtag = kwargs.get("use_cashtag", False)
        query = f"${symbol}" if use_cashtag else symbol
        query += " -is:retweet lang:en"  # Exclude retweets, English only

        # Build request parameters
        params = {
            "query": query,
            "max_results": min(kwargs.get("max_results", 100), 100),  # API limit is 100
            "tweet.fields": "created_at,public_metrics",
        }

        try:
            # Make API request
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            response = requests.get(
                f"{self.endpoint}/tweets/search/recent",
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

            # Handle rate limit responses
            if response.status_code == 429:
                self._is_healthy = False
                # Try to get rate limit reset time
                reset_time = response.headers.get("x-rate-limit-reset", "unknown")
                raise RateLimitExceeded(
                    f"Twitter rate limit exceeded (429), resets at {reset_time}"
                )

            # Handle authentication errors
            if response.status_code == 401:
                self._is_healthy = False
                raise ProviderError("Twitter authentication failed (invalid bearer token)")

            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if "errors" in data:
                error_msg = data["errors"][0].get("message", "Unknown error")
                self._is_healthy = False
                raise ProviderError(f"Twitter API error: {error_msg}")

            tweets = data.get("data", [])

            if not tweets:
                logger.info(f"No tweets found for {symbol}")
                return None

            # Analyze sentiment from tweets
            sentiment_value = self._analyze_tweets(tweets)
            confidence = self._compute_confidence(len(tweets))

            self._is_healthy = True

            return SentimentScore(
                value=sentiment_value,
                confidence=confidence,
                source="twitter",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "tweet_count": len(tweets),
                    "query": query,
                },
            )

        except requests.exceptions.Timeout as e:
            self._is_healthy = False
            raise ProviderError(f"Twitter API request timeout after {self.timeout}s") from e
        except requests.exceptions.RequestException as e:
            self._is_healthy = False
            raise ProviderError(f"Twitter API request failed: {e}") from e

    def _analyze_tweets(self, tweets: list[dict[str, Any]]) -> float:
        """Analyze sentiment from tweet text.

        Uses simple keyword-based sentiment analysis. In production, this
        should be replaced with a proper NLP model fine-tuned on financial tweets.

        Args:
            tweets: List of tweet dictionaries

        Returns:
            Sentiment value between -1.0 and 1.0
        """
        if not tweets:
            return 0.0

        total_score = 0.0
        tweets_analyzed = 0

        for tweet in tweets:
            text = tweet.get("text", "").lower()

            if not text.strip():
                continue

            # Count positive and negative keywords
            positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text)
            negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text)

            # Weight by engagement metrics if available
            metrics = tweet.get("public_metrics", {})
            likes = metrics.get("like_count", 1)
            retweets = metrics.get("retweet_count", 1)
            engagement_weight = 1 + (likes + retweets) / 100.0  # Boost popular tweets

            # Calculate tweet score
            if positive_count + negative_count > 0:
                tweet_score = (positive_count - negative_count) / (positive_count + negative_count)
                total_score += tweet_score * engagement_weight
                tweets_analyzed += engagement_weight

        if tweets_analyzed == 0:
            return 0.0

        # Average sentiment across tweets
        avg_sentiment = total_score / tweets_analyzed

        # Clamp to [-1.0, 1.0] range
        return max(-1.0, min(1.0, avg_sentiment))

    def _compute_confidence(self, tweet_count: int) -> float:
        """Compute confidence based on number of tweets.

        More tweets = higher confidence.

        Args:
            tweet_count: Number of tweets analyzed

        Returns:
            Confidence between 0.0 and 1.0
        """
        if tweet_count == 0:
            return 0.0

        # Logarithmic scaling: 1 tweet = 0.2, 10 tweets = 0.6, 50+ tweets = 0.9
        if tweet_count >= 100:
            return 0.95
        elif tweet_count >= 50:
            return 0.9
        elif tweet_count >= 20:
            return 0.8
        elif tweet_count >= 10:
            return 0.6
        else:
            return 0.2 + (tweet_count - 1) * 0.04

    def is_healthy(self) -> bool:
        """Check if provider is healthy.

        Returns:
            True if last request succeeded, False otherwise
        """
        return self._is_healthy

    def get_metadata(self) -> ProviderMetadata:
        """Get provider metadata.

        Returns:
            ProviderMetadata for Twitter provider
        """
        return ProviderMetadata(
            name="twitter",
            version="1.0.0",
            rate_limit_per_minute=self.rate_limiter.capacity,
            supports_async=False,
        )
