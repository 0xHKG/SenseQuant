import pandas as pd
from typing import Optional, Dict, Any

class SwingStrategy:
    def signal(self, df_daily: pd.DataFrame, sentiment_score: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Baseline: 10/30 SMA cross with sentiment as weak filter.
        """
        if df_daily is None or df_daily.empty or "close" not in df_daily.columns:
            return None
        s = df_daily["close"].tail(90)
        sma10 = s.rolling(10).mean().iloc[-1]
        sma30 = s.rolling(30).mean().iloc[-1]
        if sma10 > sma30 and sentiment_score >= -0.2:
            return {"action": "BUY"}
        if sma10 < sma30 and sentiment_score <= 0.2:
            return {"action": "SELL"}
        return None

