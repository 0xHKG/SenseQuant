import pandas as pd
from typing import Optional, Dict, Any
from loguru import logger

class IntradayStrategy:
    def __init__(self, sentiment_gate: bool = True, neg_cutoff: float = -0.4):
        self.sentiment_gate = sentiment_gate
        self.neg_cutoff = neg_cutoff

    def signal(self, df_minute: pd.DataFrame, sentiment_score: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Very simple baseline:
        - If last close > rolling_mean(20) and sentiment >= cutoff => BUY
        - Else if last close < rolling_mean(20) and sentiment <= -abs(cutoff) => SELL
        """
        if df_minute is None or df_minute.empty:
            return None
        if not {"close"}.issubset(set(df_minute.columns)):
            logger.warning("intraday.signal: missing 'close' column")
            return None
        s = df_minute["close"].tail(50)
        ma20 = s.rolling(20).mean().iloc[-1]
        last = s.iloc[-1]
        if self.sentiment_gate and sentiment_score < self.neg_cutoff:
            return None
        if last > ma20:
            return {"action": "BUY"}
        elif last < ma20 and sentiment_score <= -abs(self.neg_cutoff):
            return {"action": "SELL"}
        return None

