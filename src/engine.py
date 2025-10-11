from datetime import datetime, time
from typing import List
from loguru import logger

from .config import settings
from .breeze_client import BreezeClient
from .strategies.intraday import IntradayStrategy
from .strategies.swing import SwingStrategy

def is_market_open(now=None) -> bool:
    now = now or datetime.now()
    return now.weekday() < 5 and time(9,15) <= now.time() <= time(15,29)

class Engine:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.client = BreezeClient(settings.breeze_api_key, settings.breeze_api_secret,
                                   settings.breeze_session_token, dry_run=(settings.mode!="live"))
        self.intraday = IntradayStrategy()
        self.swing = SwingStrategy()

    def start(self):
        logger.info("Engine start. Mode={}, Symbols={}", settings.mode, self.symbols)
        self.client.authenticate()

    def tick_intraday(self, symbol: str):
        # placeholder: pull last N minutes bars; here we just call historical with short window
        df = self.client.get_historical(symbol, interval="1minute",
                                        from_dt=datetime.now().strftime("%Y-%m-%d 09:15"),
                                        to_dt=datetime.now().strftime("%Y-%m-%d %H:%M"))
        sig = self.intraday.signal(df, sentiment_score=0.0)
        if sig:
            order = {"stock_code": symbol, "action": sig["action"], "quantity": 1, "order_type": "MARKET"}
            res = self.client.place_order(order)
            logger.info("Intraday {} -> {}", symbol, res)

    def daily_swing(self, symbol: str):
        df = self.client.get_historical(symbol, interval="1day",
                                        from_dt=(datetime.now().replace(day=1)).strftime("%Y-%m-%d"),
                                        to_dt=datetime.now().strftime("%Y-%m-%d"))
        sig = self.swing.signal(df, sentiment_score=0.0)
        if sig:
            order = {"stock_code": symbol, "action": sig["action"], "quantity": 1, "order_type": "MARKET"}
            res = self.client.place_order(order)
            logger.info("Swing {} -> {}", symbol, res)

