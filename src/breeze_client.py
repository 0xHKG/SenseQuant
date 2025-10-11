from typing import Optional, Dict, Any
from loguru import logger
import pandas as pd

try:
    from breeze_connect import BreezeConnect
except Exception as e:
    BreezeConnect = None  # allow import without package in tests
    logger.warning("breeze_connect not available: {}", e)

class BreezeClient:
    def __init__(self, api_key: str, api_secret: str, session_token: str, dry_run: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.dry_run = dry_run
        self._client: Optional[BreezeConnect] = None

    def authenticate(self) -> None:
        if self.dry_run:
            logger.info("DRYRUN: skipping Breeze auth")
            return
        if BreezeConnect is None:
            raise RuntimeError("breeze_connect not installed")
        self._client = BreezeConnect(api_key=self.api_key)
        self._client.generate_session(api_secret=self.api_secret, session_token=self.session_token)
        logger.info("Breeze session established")

    def get_ltp(self, symbol: str) -> Optional[float]:
        try:
            if self.dry_run:
                logger.info("DRYRUN get_ltp {}", symbol)
                return 0.0
            assert self._client is not None, "Authenticate first"
            resp = self._client.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
            # adapt to actual response shape if different
            price = None
            if isinstance(resp, dict):
                succ = resp.get("Success") or resp.get("success")
                if succ and isinstance(succ, list) and succ:
                    price = float(succ[0].get("ltp") or succ[0].get("Ltp") or 0.0)
            return price
        except Exception as e:
            logger.exception("get_ltp failed for {}: {}", symbol, e)
            return None

    def get_historical(self, symbol: str, interval: str, from_dt: str, to_dt: str) -> pd.DataFrame:
        """
        interval like: '1minute','5minute','1day' (match Breeze)
        dates in 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD'
        """
        try:
            if self.dry_run:
                logger.info("DRYRUN historical {} {} {} {}", symbol, interval, from_dt, to_dt)
                return pd.DataFrame(columns=["datetime","open","high","low","close","volume"]).set_index("datetime")
            assert self._client is not None, "Authenticate first"
            resp = self._client.get_historical_data(interval=interval, from_date=from_dt, to_date=to_dt,
                                                    stock_code=symbol, exchange_code="NSE", product_type="cash")
            df = pd.DataFrame(resp if isinstance(resp, list) else resp.get("Success", []))
            if not df.empty:
                # normalize column names
                cols = {c.lower(): c for c in df.columns}
                # ensure lower-case keys
                df.columns = [c.lower() for c in df.columns]
                # parse datetime
                dt_col = "datetime" if "datetime" in df.columns else "time" if "time" in df.columns else None
                if dt_col:
                    df[dt_col] = pd.to_datetime(df[dt_col])
                    df = df.set_index(dt_col)
            return df
        except Exception as e:
            logger.exception("get_historical failed for {}: {}", symbol, e)
            return pd.DataFrame()

    def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        params: {
          stock_code, action ('BUY'|'SELL'), quantity, order_type ('MARKET'|'LIMIT'),
          price (if LIMIT), exchange_code='NSE', product='cash'
        }
        """
        try:
            if self.dry_run:
                logger.info("DRYRUN place_order {}", params)
                return {"status": "DRYRUN", "params": params}
            assert self._client is not None, "Authenticate first"
            default = {"exchange_code": "NSE", "product": "cash"}
            payload = {**default, **params}
            res = self._client.place_order(**payload)
            logger.info("Order response: {}", res)
            return res if isinstance(res, dict) else {"raw": res}
        except Exception as e:
            logger.exception("place_order failed: {}", e)
            return {"error": str(e)}

