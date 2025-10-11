"""Swing trading strategy."""

from __future__ import annotations

import pandas as pd

from src.domain.types import Signal, SignalDirection


class SwingStrategy:
    """Simple baseline swing strategy with sentiment filter."""

    def signal(self, df_daily: pd.DataFrame, sentiment_score: float = 0.0) -> Signal | None:
        """
        Generate swing trading signal from daily bars.

        Simple logic: 10/30 SMA cross with sentiment filter.

        Args:
            df_daily: DataFrame with 'close' column
            sentiment_score: Sentiment score in [-1, 1]

        Returns:
            Signal or None if no signal
        """
        if df_daily is None or df_daily.empty or "close" not in df_daily.columns:
            return None

        s = df_daily["close"].tail(90)
        if len(s) < 30:
            return None

        sma10 = s.rolling(10).mean().iloc[-1]
        sma30 = s.rolling(30).mean().iloc[-1]

        direction: SignalDirection
        if sma10 > sma30 and sentiment_score >= -0.2:
            direction = "LONG"
            return Signal(
                symbol="", direction=direction, strength=0.6, meta={"sma10": sma10, "sma30": sma30}
            )
        if sma10 < sma30 and sentiment_score <= 0.2:
            direction = "SHORT"
            return Signal(
                symbol="", direction=direction, strength=0.6, meta={"sma10": sma10, "sma30": sma30}
            )

        return None
