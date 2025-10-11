from textblob import TextBlob
from typing import List
from loguru import logger

def score_headlines(headlines: List[str]) -> float:
    """Very simple baseline sentiment score [-1,1]. Replace with robust provider later."""
    if not headlines:
        return 0.0
    vals = []
    for h in headlines:
        try:
            vals.append(TextBlob(h).sentiment.polarity)
        except Exception as e:
            logger.warning("sentiment failed for '{}': {}", h, e)
    return sum(vals)/len(vals) if vals else 0.0

