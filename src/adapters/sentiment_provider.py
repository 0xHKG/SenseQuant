"""Sentiment analysis provider with abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from loguru import logger

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None  # type: ignore
    logger.warning("textblob not available")


class SentimentProviderError(Exception):
    """Exception raised when sentiment provider fails."""

    pass


class SentimentProvider(ABC):
    """Abstract base for sentiment providers."""

    @abstractmethod
    def get_sentiment(self, symbol: str) -> float:
        """
        Fetch sentiment score for symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")

        Returns:
            float: Sentiment in range [-1.0, 1.0]
                   -1.0: Very negative
                    0.0: Neutral
                   +1.0: Very positive

        Raises:
            SentimentProviderError: On fetch failures
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/caching."""
        pass


class StubSentimentProvider(SentimentProvider):
    """Stub provider returning neutral sentiment (0.0) for all symbols."""

    def get_sentiment(self, symbol: str) -> float:
        """
        Always return neutral sentiment.

        Args:
            symbol: Stock symbol

        Returns:
            0.0 (neutral sentiment)
        """
        logger.debug(
            f"Stub sentiment provider returning neutral for {symbol}",
            extra={"component": "sentiment", "symbol": symbol, "provider": "stub"},
        )
        return 0.0

    @property
    def name(self) -> str:
        """Return provider name."""
        return "stub"


class TextBlobSentimentProvider(SentimentProvider):
    """
    Simple TextBlob-based sentiment provider.

    Uses TextBlob polarity analysis on headlines.
    This is a basic implementation for demonstration purposes.
    """

    def __init__(self, headlines_source: Callable[[str], list[str]] | None = None) -> None:
        """
        Initialize TextBlob provider.

        Args:
            headlines_source: Optional callable that returns list of headlines for symbol
                             If None, returns neutral sentiment (0.0)
        """
        self._headlines_source = headlines_source

    def get_sentiment(self, symbol: str) -> float:
        """
        Compute sentiment from headlines using TextBlob.

        Args:
            symbol: Stock symbol

        Returns:
            Sentiment score [-1.0, 1.0]

        Raises:
            SentimentProviderError: If TextBlob is not available or analysis fails
        """
        if TextBlob is None:
            logger.warning(
                "TextBlob not available, returning neutral sentiment",
                extra={"component": "sentiment", "symbol": symbol},
            )
            return 0.0

        if self._headlines_source is None:
            logger.debug(
                "No headlines source configured, returning neutral",
                extra={"component": "sentiment", "symbol": symbol},
            )
            return 0.0

        try:
            headlines = self._headlines_source(symbol)
            if not headlines:
                return 0.0

            scores = []
            for headline in headlines:
                try:
                    polarity = TextBlob(headline).sentiment.polarity
                    scores.append(polarity)
                except Exception as e:
                    logger.warning(
                        f"Failed to analyze headline: {e}",
                        extra={
                            "component": "sentiment",
                            "symbol": symbol,
                            "headline": headline[:50],
                        },
                    )

            if not scores:
                return 0.0

            avg_sentiment: float = sum(scores) / len(scores)
            logger.debug(
                f"Computed sentiment for {symbol}: {avg_sentiment:.2f}",
                extra={
                    "component": "sentiment",
                    "symbol": symbol,
                    "num_headlines": len(headlines),
                    "sentiment": avg_sentiment,
                },
            )
            return float(avg_sentiment)

        except Exception as e:
            logger.error(
                f"Sentiment analysis failed for {symbol}: {e}",
                extra={"component": "sentiment", "symbol": symbol, "error": str(e)},
            )
            raise SentimentProviderError(f"Failed to get sentiment for {symbol}: {e}") from e

    @property
    def name(self) -> str:
        """Return provider name."""
        return "textblob"


# Legacy function for backward compatibility
def get_sentiment(symbol: str) -> float:
    """
    Legacy function returning neutral sentiment.

    This function is kept for backward compatibility.
    New code should use SentimentProvider instances.

    Args:
        symbol: Stock symbol

    Returns:
        0.0 (neutral sentiment)
    """
    logger.debug(
        f"Legacy get_sentiment called for {symbol}",
        extra={"component": "sentiment", "symbol": symbol},
    )
    return 0.0


def score_headlines(headlines: list[str]) -> float:
    """
    Compute simple baseline sentiment score from headlines.

    Args:
        headlines: List of headline strings

    Returns:
        Sentiment score in range [-1.0, 1.0]
    """
    if not headlines:
        return 0.0
    if TextBlob is None:
        logger.warning("TextBlob not available, returning neutral sentiment")
        return 0.0

    vals: list[float] = []
    for h in headlines:
        try:
            vals.append(TextBlob(h).sentiment.polarity)
        except Exception as e:
            logger.warning("sentiment failed for '{}': {}", h, e)
    return sum(vals) / len(vals) if vals else 0.0
