"""Sentiment analysis provider."""

from __future__ import annotations

from loguru import logger

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None  # type: ignore
    logger.warning("textblob not available")


def get_sentiment(symbol: str) -> float:
    """
    Fetch sentiment score for a given stock symbol.

    This is a stub implementation that returns neutral sentiment.
    In production, this would integrate with news/social media APIs.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS")

    Returns:
        Sentiment score in range [-1.0, 1.0], where:
        - -1.0 = very negative
        -  0.0 = neutral
        - +1.0 = very positive
    """
    logger.debug(
        f"Fetching sentiment for {symbol}",
        extra={"component": "sentiment", "symbol": symbol},
    )
    # Stub: return neutral sentiment
    # In production: fetch from news API, social media, etc.
    sentiment_score = 0.0
    logger.info(
        f"Sentiment score for {symbol}: {sentiment_score}",
        extra={"component": "sentiment", "symbol": symbol, "score": sentiment_score},
    )
    return sentiment_score


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
