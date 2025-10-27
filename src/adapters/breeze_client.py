"""Breeze API client adapter with robust error handling and retries."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytz
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.types import Bar, OrderResponse, OrderSide, OrderType

try:
    from breeze_connect import BreezeConnect
except Exception as e:
    BreezeConnect = None  # type: ignore
    logger.warning("breeze_connect not available: {}", e, extra={"component": "breeze"})


# ============================================================================
# Exception Taxonomy
# ============================================================================


class BreezeError(Exception):
    """Base exception for all Breeze API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        raw_response: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.raw_response = raw_response

    def is_transient(self) -> bool:
        """Override in subclasses to indicate if error is retryable."""
        return False


class BreezeAuthError(BreezeError):
    """Authentication failure (401, invalid credentials)."""

    pass


class BreezeRateLimitError(BreezeError):
    """Rate limit exceeded (HTTP 429)."""

    def is_transient(self) -> bool:
        return True

    def get_retry_after(self) -> int | None:
        """Extract 'Retry-After' header value in seconds if present."""
        if self.raw_response and "retry_after" in self.raw_response:
            try:
                return int(self.raw_response["retry_after"])
            except (ValueError, TypeError):
                pass
        return None


class BreezeTransientError(BreezeError):
    """Transient errors: network issues, 5xx server errors, timeouts."""

    def is_transient(self) -> bool:
        return True


class BreezeOrderRejectedError(BreezeError):
    """Order rejected by exchange (insufficient funds, invalid symbol, etc.)."""

    pass


def is_transient(e: Exception) -> bool:
    """
    Helper to classify if an exception is retryable.

    Args:
        e: Exception instance

    Returns:
        True if error is transient and should be retried

    Examples:
        >>> is_transient(BreezeTransientError("timeout"))
        True
        >>> is_transient(BreezeAuthError("invalid key"))
        False
    """
    if isinstance(e, BreezeError):
        return e.is_transient()
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
    return False


# ============================================================================
# Symbol Mapping
# ============================================================================


def _load_symbol_mappings() -> dict[str, str]:
    """Load NSE→ISEC symbol mappings from symbol_mappings.json.

    Returns:
        Dict mapping NSE symbols to ISEC stock codes.
        Empty dict if file not found or error loading.

    Example:
        >>> mappings = _load_symbol_mappings()
        >>> mappings.get("RELIANCE", "RELIANCE")
        'RELIND'
        >>> mappings.get("TCS", "TCS")
        'TCS'
    """
    try:
        # Path relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        mappings_file = project_root / "data" / "historical" / "metadata" / "symbol_mappings.json"

        if not mappings_file.exists():
            logger.warning(
                f"Symbol mappings file not found: {mappings_file}. Using fallback mapping.",
                extra={"component": "breeze"},
            )
            # Fallback to hardcoded RELIANCE mapping
            return {"RELIANCE": "RELIND"}

        with open(mappings_file) as f:
            data = json.load(f)

        mappings = data.get("mappings", {})
        logger.debug(
            f"Loaded {len(mappings)} symbol mappings from {mappings_file}",
            extra={"component": "breeze"},
        )
        return mappings

    except Exception as e:
        logger.warning(
            f"Error loading symbol mappings: {e}. Using fallback mapping.",
            extra={"component": "breeze"},
        )
        # Fallback to hardcoded RELIANCE mapping
        return {"RELIANCE": "RELIND"}


# Module-level symbol mapping cache (loaded once on import)
SYMBOL_MAPPINGS = _load_symbol_mappings()


# ============================================================================
# BreezeClient
# ============================================================================


class BreezeClient:
    """Wrapper for Breeze API with robust error handling, retries, and dry-run support."""

    DEFAULT_TIMEOUT = 30  # seconds

    def __init__(
        self, api_key: str, api_secret: str, session_token: str, dry_run: bool = True
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.dry_run = dry_run
        self._client: Any = None

    def authenticate(self) -> None:
        """
        Establish session with Breeze API.

        In dry-run mode, skips authentication.
        In live mode, calls breeze_connect SDK to generate session.

        Raises:
            BreezeAuthError: If credentials are invalid (401)
            BreezeTransientError: If network/server error occurs

        Examples:
            >>> client = BreezeClient(api_key="key", api_secret="secret", session_token="token", dry_run=False)
            >>> client.authenticate()  # Session established
        """
        if self.dry_run:
            logger.info("DRYRUN: skipping Breeze authentication", extra={"component": "breeze"})
            return

        logger.info("Authenticating with Breeze API", extra={"component": "breeze"})
        try:
            if BreezeConnect is None:
                raise RuntimeError("breeze_connect not installed")

            self._client = BreezeConnect(api_key=self.api_key)
            self._call_with_retry(
                "generate_session", api_secret=self.api_secret, session_token=self.session_token
            )
            logger.info("Breeze session established", extra={"component": "breeze"})
        except BreezeAuthError:
            logger.error(
                "Authentication failed: invalid credentials", extra={"component": "breeze"}
            )
            raise

    def latest_price(self, symbol: str) -> float:
        """
        Get last traded price for a symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")

        Returns:
            Last traded price as float (0.0 in dry-run or on error)

        Raises:
            BreezeTransientError: On network/timeout/server errors (retried)
            BreezeError: On other API errors

        Examples:
            >>> client = BreezeClient(..., dry_run=False)
            >>> client.authenticate()
            >>> price = client.latest_price("RELIANCE")
            >>> assert isinstance(price, float) and price >= 0
        """
        if self.dry_run:
            logger.info("DRYRUN: latest_price", extra={"component": "breeze", "symbol": symbol})
            return 0.0

        try:
            response = self._call_with_retry(
                "get_quotes", stock_code=symbol, exchange_code="NSE", product_type="cash"
            )
            # Parse LTP from response
            success = response.get("Success", [])
            if success and isinstance(success, list) and len(success) > 0:
                ltp = success[0].get("ltp") or success[0].get("last")
                if ltp:
                    return float(ltp)

            logger.warning(
                "No LTP found in response", extra={"component": "breeze", "symbol": symbol}
            )
            return 0.0

        except BreezeError:
            logger.error(
                "Failed to get latest price", extra={"component": "breeze", "symbol": symbol}
            )
            return 0.0

    def historical_bars(
        self,
        symbol: str,
        interval: Literal["1minute", "5minute", "1day"],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[Bar]:
        """
        Fetch historical OHLCV bars.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            interval: Bar interval
            start: Start datetime (timezone-aware)
            end: End datetime (timezone-aware)

        Returns:
            List of Bar objects with IST timezone-aware timestamps

        Raises:
            BreezeTransientError: On network/timeout/server errors (retried)
            BreezeError: On other API errors

        Examples:
            >>> client = BreezeClient(..., dry_run=False)
            >>> client.authenticate()
            >>> start = pd.Timestamp("2025-01-01 09:15", tz="Asia/Kolkata")
            >>> end = pd.Timestamp("2025-01-01 15:30", tz="Asia/Kolkata")
            >>> bars = client.historical_bars("RELIANCE", "1minute", start, end)
            >>> assert all(isinstance(b, Bar) for b in bars)
        """
        if self.dry_run:
            logger.info(
                "DRYRUN: Loading historical_bars from cached data",
                extra={"component": "breeze", "symbol": symbol, "interval": interval},
            )
            # Load from data/historical/{symbol}/{interval}/*.csv
            from pathlib import Path
            import glob

            data_dir = Path("data/historical") / symbol / interval
            if not data_dir.exists():
                logger.warning(
                    f"No cached data found at {data_dir}",
                    extra={"component": "breeze", "symbol": symbol, "interval": interval},
                )
                return []

            # Load all CSV files in the directory
            csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
            if not csv_files:
                logger.warning(
                    f"No CSV files found in {data_dir}",
                    extra={"component": "breeze", "symbol": symbol, "interval": interval},
                )
                return []

            bars = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                # Convert to Bar objects
                for _, row in df.iterrows():
                    ts = pd.Timestamp(row['timestamp'], tz='Asia/Kolkata')
                    # Filter by date range
                    if start <= ts <= end:
                        bars.append(Bar(
                            ts=ts,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=int(row['volume']),
                            symbol=symbol,
                        ))

            logger.info(
                f"Loaded {len(bars)} bars from cached data",
                extra={"component": "breeze", "symbol": symbol, "interval": interval, "bars": len(bars)},
            )
            return bars

        # Map NSE exchange code to ISEC stock code
        # Load mappings from symbol_mappings.json (e.g., RELIANCE→RELIND, INFY→INFTEC)
        # Symbols not in mapping use their NSE code as-is (e.g., TCS→TCS)
        stock_code = SYMBOL_MAPPINGS.get(symbol, symbol)

        try:
            # Use v2 API (more reliable than v1, supports 1-second intervals)
            # Date format: ISO8601 with timezone (YYYY-MM-DDTHH:MM:SS.000Z)
            response = self._call_with_retry(
                "get_historical_data_v2",
                interval=interval,
                from_date=start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                to_date=end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                stock_code=stock_code,
                exchange_code="NSE",
                product_type="cash",
            )
            return self._normalize_bars(response, symbol, interval)

        except BreezeError:
            logger.error(
                "Failed to fetch historical bars",
                extra={"component": "breeze", "symbol": symbol, "interval": interval},
            )
            return []

    def get_historical(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: Literal["1minute", "5minute", "1day"] = "1day",
    ) -> pd.DataFrame:
        """Fetch historical data and return as DataFrame (wrapper for historical_bars).

        This method provides compatibility with scripts/fetch_historical_data.py
        which expects a DataFrame return format.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            from_date: Start datetime
            to_date: End datetime
            interval: Bar interval (default: "1day")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            BreezeTransientError: On network/timeout/server errors
            BreezeError: On other API errors
        """
        import pandas as pd

        bars = self.historical_bars(symbol, interval, from_date, to_date)

        if not bars:
            return pd.DataFrame()

        # Convert bars to DataFrame
        data = {
            "timestamp": [b.ts for b in bars],  # Bar uses 'ts' not 'timestamp'
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }

        return pd.DataFrame(data)

    def fetch_historical_chunk(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: Literal["1minute", "5minute", "1day"] = "1day",
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch historical data for a date range chunk using v2 API.

        This is the recommended method for fetching historical data in production.
        It uses get_historical_data_v2 (more reliable than v1), handles stock code
        mapping (RELIANCE → RELIND), and returns a DataFrame.

        Features:
        - Uses v2 API with ISO8601 timestamps
        - Automatic stock code mapping (NSE → ISEC codes)
        - Retry logic with exponential backoff
        - Empty DataFrame on no data (not an error)

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            start_date: Chunk start datetime
            end_date: Chunk end datetime
            interval: Bar interval ("1minute", "5minute", "1day", etc.)
            max_retries: Maximum retry attempts on transient errors

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Empty DataFrame if no data available for the range

        Raises:
            BreezeTransientError: On network/timeout errors after all retries
            BreezeError: On non-retryable API errors

        Example:
            >>> from datetime import datetime
            >>> client = BreezeClient(..., dry_run=False)
            >>> client.authenticate()
            >>> df = client.fetch_historical_chunk(
            ...     "RELIANCE",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 3, 31),
            ...     interval="1day"
            ... )
            >>> assert "timestamp" in df.columns
            >>> assert len(df) > 0  # Should have Q1 2024 data
        """
        import pandas as pd

        if self.dry_run:
            logger.info(
                "DRYRUN: fetch_historical_chunk",
                extra={
                    "component": "breeze",
                    "symbol": symbol,
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d"),
                    "interval": interval,
                },
            )
            return pd.DataFrame()

        # Use the underlying historical_bars method which already has:
        # - v2 API
        # - Stock code mapping
        # - Retry logic
        # - Error handling
        bars = self.historical_bars(symbol, interval, start_date, end_date)

        if not bars:
            logger.debug(
                f"No data returned for {symbol} {start_date.date()} to {end_date.date()}",
                extra={"component": "breeze", "symbol": symbol},
            )
            return pd.DataFrame()

        # Convert to DataFrame
        data = {
            "timestamp": [b.ts for b in bars],  # Bar uses 'ts' not 'timestamp'
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }

        df = pd.DataFrame(data)
        logger.debug(
            f"Fetched {len(df)} bars for {symbol} ({start_date.date()} to {end_date.date()})",
            extra={"component": "breeze", "symbol": symbol, "bars": len(df)},
        )

        return df

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType = "MARKET",
        price: float | None = None,
    ) -> OrderResponse:
        """
        Place an order via Breeze API.

        In dry-run mode, returns deterministic mock response without API call.
        In live mode, submits order and retries once on transient errors.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            side: "BUY" or "SELL"
            qty: Order quantity (positive integer)
            order_type: "MARKET" or "LIMIT"
            price: Limit price (required for LIMIT orders)

        Returns:
            OrderResponse with order_id, status, and raw API response

        Raises:
            BreezeOrderRejectedError: If order is rejected by exchange
            BreezeTransientError: On network/timeout errors (retried once)
            BreezeError: On other API errors

        Examples:
            >>> client = BreezeClient(..., dry_run=True)
            >>> response = client.place_order("RELIANCE", "BUY", 1)
            >>> assert response.status == "FILLED"
            >>> assert "DRYRUN" in response.order_id
        """
        if self.dry_run:
            return self._mock_order_response(symbol, side, qty, order_type, price)

        try:
            # First attempt
            response = self._place_order_impl(symbol, side, qty, order_type, price)
            return response

        except BreezeTransientError:
            # Single retry on transient errors
            logger.warning(
                "Transient error placing order, retrying once",
                extra={"component": "breeze", "symbol": symbol, "side": side, "qty": qty},
            )
            time.sleep(2)
            response = self._place_order_impl(symbol, side, qty, order_type, price)
            return response

        except BreezeOrderRejectedError:
            logger.error(
                "Order rejected by exchange",
                extra={"component": "breeze", "symbol": symbol, "side": side, "qty": qty},
            )
            raise

    # ========================================================================
    # Internal Methods
    # ========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(is_transient),  # type: ignore[arg-type]
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    def _call_with_retry(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """
        Internal wrapper for Breeze SDK calls with timeout and retry.

        Args:
            method: Breeze SDK method name (e.g., "get_quotes")
            **kwargs: Arguments to pass to the method

        Returns:
            Raw API response dict

        Raises:
            BreezeTransientError: On connection/timeout/5xx errors
            BreezeAuthError: On 401 authentication failure
            BreezeRateLimitError: On 429 rate limit
            BreezeError: On other API errors
        """
        if self.dry_run:
            raise RuntimeError("_call_with_retry should not be invoked in dry-run mode")

        try:
            sdk_method = getattr(self._client, method)
            response = sdk_method(**kwargs)

            # Parse SDK response (Breeze API returns dict with "Status" key)
            if isinstance(response, dict):
                status = response.get("Status", 200)
                if status == 401:
                    raise BreezeAuthError(
                        "Authentication failed", status_code=401, raw_response=response
                    )
                elif status == 429:
                    error = BreezeRateLimitError(
                        "Rate limit exceeded", status_code=429, raw_response=response
                    )
                    self._handle_rate_limit(error)
                    raise error
                elif status >= 500:
                    raise BreezeTransientError(
                        f"Server error {status}", status_code=status, raw_response=response
                    )
                elif status >= 400 and method == "place_order":
                    # Order rejections are handled specially
                    error_msg = response.get("Error", "Order rejected")
                    raise BreezeOrderRejectedError(
                        error_msg, status_code=status, raw_response=response
                    )
                elif status >= 400:
                    raise BreezeError(
                        f"Client error {status}", status_code=status, raw_response=response
                    )

            return response  # type: ignore

        except (ConnectionError, TimeoutError) as e:
            raise BreezeTransientError(f"Network error: {e}") from e
        except BreezeError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in Breeze SDK call", exc_info=True, extra={"component": "breeze"}
            )
            raise BreezeError(f"Unexpected error: {e}") from e

    def _place_order_impl(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType,
        price: float | None,
    ) -> OrderResponse:
        """Internal order placement implementation."""
        try:
            response = self._call_with_retry(
                "place_order",
                stock_code=symbol,
                exchange_code="NSE",
                product="cash",
                action=side,
                order_type=order_type,
                quantity=str(qty),
                price=str(price) if price else "0",
            )

            # Parse response
            success = response.get("Success", {})
            order_id = success.get("order_id", "UNKNOWN")

            logger.info(
                "Order placed successfully",
                extra={
                    "component": "breeze",
                    "order_id": str(order_id),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                },
            )

            return OrderResponse(order_id=str(order_id), status="PLACED", raw=response)

        except BreezeError:
            raise

    def _handle_rate_limit(self, error: BreezeRateLimitError) -> None:
        """
        Handle HTTP 429 rate limit with respect for Retry-After header.

        Args:
            error: BreezeRateLimitError with optional retry_after value
        """
        retry_after = error.get_retry_after()
        if retry_after:
            logger.warning(
                "Rate limit exceeded, respecting Retry-After",
                extra={"component": "breeze", "retry_after": retry_after},
            )
            time.sleep(retry_after)
        else:
            logger.warning(
                "Rate limit exceeded, using default backoff", extra={"component": "breeze"}
            )

    def _normalize_bars(
        self, payload: dict[str, Any] | list[Any], symbol: str, interval: str
    ) -> list[Bar]:
        """
        Normalize Breeze API response to list of Bar DTOs.

        Ensures:
        - Timestamps are IST timezone-aware (pytz.timezone("Asia/Kolkata"))
        - Numeric fields (open, high, low, close, volume) are correct types
        - Empty/malformed data returns empty list (no crash)

        Args:
            payload: Raw API response (dict with "Success" key or list)
            symbol: Stock symbol
            interval: Bar interval (for logging)

        Returns:
            List of Bar objects, sorted by timestamp ascending

        Examples:
            >>> raw = {"Success": [{"datetime": "2025-01-01 09:15:00", "open": 100, ...}]}
            >>> bars = client._normalize_bars(raw, "RELIANCE", "1minute")
            >>> assert bars[0].ts.tz.zone == "Asia/Kolkata"
        """
        ist = pytz.timezone("Asia/Kolkata")
        bars: list[Bar] = []

        # Extract raw data
        if isinstance(payload, dict):
            raw_data = payload.get("Success", [])
        elif isinstance(payload, list):
            raw_data = payload
        else:
            logger.warning(
                "Unexpected payload type for bars", extra={"component": "breeze", "symbol": symbol}
            )
            return []

        if not raw_data:
            return []

        for item in raw_data:
            try:
                # Parse datetime (assume Breeze returns "datetime" or "time" field)
                dt_str = item.get("datetime") or item.get("time")
                if not dt_str:
                    continue

                # Parse and localize to IST
                ts = pd.to_datetime(dt_str)
                if ts.tz is None:
                    ts = ts.tz_localize(ist)
                else:
                    ts = ts.tz_convert(ist)

                # Parse numeric fields with safe conversion
                bar = Bar(
                    ts=ts,
                    open=float(item.get("open", 0)),
                    high=float(item.get("high", 0)),
                    low=float(item.get("low", 0)),
                    close=float(item.get("close", 0)),
                    volume=int(item.get("volume", 0)),
                )
                bars.append(bar)

            except (ValueError, TypeError, KeyError) as e:
                logger.warning(
                    "Failed to parse bar",
                    extra={"component": "breeze", "symbol": symbol, "error": str(e)},
                )
                continue

        # Sort by timestamp ascending
        bars.sort(key=lambda b: b.ts)

        logger.debug(
            "Normalized bars",
            extra={"component": "breeze", "symbol": symbol, "count": len(bars)},
        )

        return bars

    def _mock_order_response(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType,
        price: float | None,
    ) -> OrderResponse:
        """
        Generate deterministic mock order response for dry-run mode.

        Order ID is stable hash: "DRYRUN-" + hash(symbol, side, qty, timestamp_floor_minute)

        Args:
            symbol: Stock symbol
            side: Order side
            qty: Order quantity
            order_type: Order type
            price: Limit price (optional)

        Returns:
            OrderResponse with deterministic order_id and FILLED status

        Examples:
            >>> # Same inputs at same minute yield same order_id
            >>> r1 = client._mock_order_response("RELIANCE", "BUY", 1, "MARKET", None)
            >>> r2 = client._mock_order_response("RELIANCE", "BUY", 1, "MARKET", None)
            >>> assert r1.order_id == r2.order_id
        """
        # Floor timestamp to minute for stability within same minute
        now = datetime.now()
        ts_floor = now.replace(second=0, microsecond=0)

        # Create stable hash
        hash_input = f"{symbol}|{side}|{qty}|{ts_floor.isoformat()}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        order_id = f"DRYRUN-{hash_digest.upper()}"

        logger.info(
            "DRYRUN order",
            extra={
                "component": "breeze",
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "price": price,
            },
        )

        return OrderResponse(
            order_id=order_id,
            status="FILLED",
            raw={"dryrun": True, "symbol": symbol, "side": side, "qty": qty},
        )
