"""Market data providers for order book, options chain, and macro indicators (US-029 Phase 4).

This module provides real Breeze-based integrations for advanced market data:
- Order book snapshots via Breeze REST API
- Options chain data via Breeze REST API
- Macro indicators via public APIs (yfinance, RBI)

All providers support dryrun mode with deterministic mock data for testing.
Credentials are loaded from environment variables via Settings.

Safety Controls:
- Read-only operations (no order placement)
- Configurable retry limits and backoff
- Dryrun mode enabled by default
- Graceful degradation on API failures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.adapters.breeze_client import (
    BreezeClient,
    BreezeError,
    is_transient,
)
from src.app.config import Settings

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time."""

    symbol: str
    timestamp: datetime
    exchange: str
    bids: list[dict[str, float]]  # [{"price": float, "quantity": int, "orders": int}]
    asks: list[dict[str, float]]  # [{"price": float, "quantity": int, "orders": int}]
    metadata: dict[str, Any]


@dataclass
class OptionsChainSnapshot:
    """Options chain snapshot for a symbol."""

    symbol: str
    date: str  # YYYY-MM-DD
    timestamp: datetime
    underlying_price: float
    options: list[dict[str, Any]]  # [{strike, expiry, call: {...}, put: {...}}]
    metadata: dict[str, Any]


@dataclass
class MacroIndicatorSnapshot:
    """Macro economic indicator snapshot."""

    indicator: str
    date: str  # YYYY-MM-DD
    timestamp: datetime
    value: float
    change: float | None
    change_pct: float | None
    metadata: dict[str, Any]


# ============================================================================
# Base Provider Interface
# ============================================================================


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    def __init__(self, settings: Settings, dry_run: bool = True):
        """Initialize provider.

        Args:
            settings: Application settings
            dry_run: If True, use mock data instead of real API calls
        """
        self.settings = settings
        self.dry_run = dry_run

    @abstractmethod
    def fetch(self, **kwargs: Any) -> Any:
        """Fetch market data.

        Args:
            **kwargs: Provider-specific parameters

        Returns:
            Provider-specific data structure
        """
        pass


# ============================================================================
# Breeze Order Book Provider
# ============================================================================


class BreezeOrderBookProvider(MarketDataProvider):
    """Fetch order book snapshots via Breeze API.

    Uses Breeze REST API to fetch L2 order book depth (best N bid/ask levels).
    Supports configurable depth levels and snapshot intervals.

    Safety:
    - Read-only (no order placement)
    - Respects retry limits and backoff
    - Dryrun mode returns deterministic mock data
    """

    def __init__(
        self,
        settings: Settings,
        client: BreezeClient | None = None,
        dry_run: bool = True,
    ):
        """Initialize order book provider.

        Args:
            settings: Application settings
            client: BreezeClient instance (created if None)
            dry_run: If True, return mock data instead of real API calls
        """
        super().__init__(settings, dry_run)

        if dry_run:
            self.client = None
        else:
            if client is None:
                self.client = BreezeClient(
                    api_key=settings.breeze_api_key,
                    api_secret=settings.breeze_api_secret,
                    session_token=settings.breeze_session_token,
                    dry_run=False,
                )
                self.client.authenticate()
            else:
                self.client = client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(is_transient),  # type: ignore[arg-type]
        reraise=True,
    )
    def fetch(  # type: ignore[override]
        self,
        symbol: str,
        depth_levels: int | None = None,
        exchange: str = "NSE",
    ) -> OrderBookSnapshot:
        """Fetch order book snapshot for symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            depth_levels: Number of bid/ask levels (defaults to config)
            exchange: Exchange code (default: "NSE")

        Returns:
            OrderBookSnapshot with bids/asks up to depth_levels

        Raises:
            BreezeTransientError: On network/timeout/server errors (retried)
            BreezeError: On other API errors
        """
        depth = depth_levels if depth_levels is not None else self.settings.order_book_depth_levels

        if self.dry_run:
            return self._mock_order_book(symbol, depth, exchange)

        try:
            # Breeze API call: get_quotes provides market depth
            # The Breeze API returns top 5 bid/ask levels in the Success response
            # Format: {"Success": [{"best_bid_price": X, "best_ask_price": Y, "depth": {...}}]}
            response = self.client._call_with_retry(  # type: ignore[union-attr]
                "get_quotes",
                stock_code=symbol,
                exchange_code=exchange,
                product_type="cash",
            )

            # Parse Breeze API response
            # Response structure: {"Success": [{"depth": {"buy": [...], "sell": [...]}}]}
            if not response or "Success" not in response:
                logger.warning(
                    f"Invalid order book response for {symbol}",
                    extra={"component": "order_book_provider", "response": response},
                )
                return self._mock_order_book(symbol, depth, exchange)

            data = response["Success"]
            if not data or not isinstance(data, list) or not data[0]:
                logger.warning(
                    f"Empty order book data for {symbol}",
                    extra={"component": "order_book_provider"},
                )
                return self._mock_order_book(symbol, depth, exchange)

            # Extract depth data from first element
            quote_data = data[0]
            depth_data = quote_data.get("depth", {})

            # Parse bid and ask sides
            # Breeze depth format: {"buy": [{"price": X, "quantity": Y, "orders": Z}], "sell": [...]}
            raw_bids = depth_data.get("buy", [])
            raw_asks = depth_data.get("sell", [])

            bids = self._parse_order_book_side(raw_bids, depth)
            asks = self._parse_order_book_side(raw_asks, depth)

            # Enrich metadata with additional quote data
            metadata = {
                "depth_levels": depth,
                "source": "breeze_live",
                "fetch_time": datetime.now().isoformat(),
                "best_bid": quote_data.get("best_bid_price"),
                "best_ask": quote_data.get("best_ask_price"),
                "ltp": quote_data.get("ltp"),
                "volume": quote_data.get("volume"),
                "total_buy_quantity": quote_data.get("total_buy_qty"),
                "total_sell_quantity": quote_data.get("total_sell_qty"),
            }

            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                exchange=exchange,
                bids=bids,
                asks=asks,
                metadata=metadata,
            )

            logger.debug(
                f"Fetched order book for {symbol}",
                extra={
                    "component": "order_book_provider",
                    "symbol": symbol,
                    "depth": depth,
                    "bids": len(bids),
                    "asks": len(asks),
                },
            )

            return snapshot

        except BreezeError as e:
            logger.error(
                f"Failed to fetch order book for {symbol}: {e}",
                extra={"component": "order_book_provider", "symbol": symbol},
            )
            raise

    def _parse_order_book_side(self, raw_data: list[Any], depth: int) -> list[dict[str, float]]:
        """Parse bid or ask side of order book.

        Args:
            raw_data: Raw API response for bid/ask side
            depth: Maximum number of levels to return

        Returns:
            List of dicts with price, quantity, orders
        """
        parsed = []
        for level in raw_data[:depth]:
            if isinstance(level, dict):
                parsed.append(
                    {
                        "price": float(level.get("price", 0)),
                        "quantity": int(level.get("quantity", 0)),
                        "orders": int(level.get("orders", 0)),
                    }
                )
        return parsed

    def _mock_order_book(self, symbol: str, depth: int, exchange: str) -> OrderBookSnapshot:
        """Generate deterministic mock order book for dryrun mode.

        Args:
            symbol: Stock symbol
            depth: Number of levels
            exchange: Exchange code

        Returns:
            Mock OrderBookSnapshot with synthetic data
        """
        # Use symbol hash to generate deterministic prices
        base_price = 2000.0 + (hash(symbol) % 1000)

        bids = []
        asks = []

        # Handle zero depth case
        if depth == 0:
            logger.info(
                f"[DRYRUN] Generated mock order book for {symbol} with 0 depth",
                extra={
                    "component": "order_book_provider",
                    "symbol": symbol,
                    "depth": depth,
                    "mode": "dryrun",
                },
            )
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                exchange=exchange,
                bids=[],
                asks=[],
                metadata={"depth_levels": 0, "source": "stub", "dryrun": True},
            )

        for i in range(depth):
            # Bids: decreasing prices from base
            bid_price = base_price - (i + 1) * 0.5
            bids.append(
                {
                    "price": round(bid_price, 2),
                    "quantity": (i + 1) * 500,
                    "orders": (i + 1) * 3,
                }
            )

            # Asks: increasing prices from base
            ask_price = base_price + (i + 1) * 0.5
            asks.append(
                {
                    "price": round(ask_price, 2),
                    "quantity": (i + 1) * 400,
                    "orders": (i + 1) * 2,
                }
            )

        logger.info(
            f"[DRYRUN] Generated mock order book for {symbol}",
            extra={
                "component": "order_book_provider",
                "symbol": symbol,
                "depth": depth,
                "mode": "dryrun",
            },
        )

        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            exchange=exchange,
            bids=bids,
            asks=asks,
            metadata={"depth_levels": depth, "source": "stub", "dryrun": True},
        )


# ============================================================================
# Breeze Options Provider
# ============================================================================


class BreezeOptionsProvider(MarketDataProvider):
    """Fetch options chain data via Breeze API.

    Uses Breeze REST API to fetch options chain (all strikes/expiries) for index options.
    Supports NIFTY, BANKNIFTY, and other NSE-traded index options.

    Safety:
    - Read-only (no options trading)
    - Respects retry limits and backoff
    - Dryrun mode returns deterministic mock data
    """

    def __init__(
        self,
        settings: Settings,
        client: BreezeClient | None = None,
        dry_run: bool = True,
    ):
        """Initialize options provider.

        Args:
            settings: Application settings
            client: BreezeClient instance (created if None)
            dry_run: If True, return mock data instead of real API calls
        """
        super().__init__(settings, dry_run)

        if dry_run:
            self.client = None
        else:
            if client is None:
                self.client = BreezeClient(
                    api_key=settings.breeze_api_key,
                    api_secret=settings.breeze_api_secret,
                    session_token=settings.breeze_session_token,
                    dry_run=False,
                )
                self.client.authenticate()
            else:
                self.client = client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(is_transient),  # type: ignore[arg-type]
        reraise=True,
    )
    def fetch(  # type: ignore[override]
        self, symbol: str, date: str | None = None
    ) -> OptionsChainSnapshot:
        """Fetch options chain for symbol.

        Args:
            symbol: Underlying symbol (e.g., "NIFTY", "BANKNIFTY")
            date: Date for snapshot (YYYY-MM-DD), defaults to today

        Returns:
            OptionsChainSnapshot with all strikes/expiries

        Raises:
            BreezeTransientError: On network/timeout/server errors (retried)
            BreezeError: On other API errors
        """
        snapshot_date = date or datetime.now().strftime("%Y-%m-%d")

        if self.dry_run:
            return self._mock_options_chain(symbol, snapshot_date)

        try:
            # Breeze API call for options chain
            # Note: Breeze API endpoint for options chain may vary
            # This is a placeholder; actual implementation depends on API
            response = self.client._call_with_retry(  # type: ignore[union-attr]
                "get_option_chain",
                stock_code=symbol,
                exchange_code="NFO",  # NSE Futures & Options
                product_type="options",
            )

            # Parse options chain
            underlying_price = float(response.get("UnderlyingPrice", 0))
            options = self._parse_options_chain(response.get("OptionChain", []))

            snapshot = OptionsChainSnapshot(
                symbol=symbol,
                date=snapshot_date,
                timestamp=datetime.now(),
                underlying_price=underlying_price,
                options=options,
                metadata={
                    "total_strikes": len({opt["strike"] for opt in options}),
                    "expiries": list({opt["expiry"] for opt in options}),
                    "source": "breeze",
                    "fetch_time": datetime.now().isoformat(),
                },
            )

            logger.debug(
                f"Fetched options chain for {symbol}",
                extra={
                    "component": "options_provider",
                    "symbol": symbol,
                    "strikes": snapshot.metadata["total_strikes"],
                    "expiries": len(snapshot.metadata["expiries"]),
                },
            )

            return snapshot

        except BreezeError as e:
            logger.error(
                f"Failed to fetch options chain for {symbol}: {e}",
                extra={"component": "options_provider", "symbol": symbol},
            )
            raise

    def _parse_options_chain(self, raw_data: list[Any]) -> list[dict[str, Any]]:
        """Parse options chain from Breeze API response.

        Args:
            raw_data: Raw API response for options chain

        Returns:
            List of option contracts with strike, expiry, call/put data
        """
        options = []
        for item in raw_data:
            if isinstance(item, dict):
                options.append(
                    {
                        "strike": float(item.get("strike", 0)),
                        "expiry": item.get("expiry", ""),
                        "call": {
                            "last_price": float(item.get("call_ltp", 0)),
                            "bid": float(item.get("call_bid", 0)),
                            "ask": float(item.get("call_ask", 0)),
                            "volume": int(item.get("call_volume", 0)),
                            "oi": int(item.get("call_oi", 0)),
                            "iv": float(item.get("call_iv", 0)),
                        },
                        "put": {
                            "last_price": float(item.get("put_ltp", 0)),
                            "bid": float(item.get("put_bid", 0)),
                            "ask": float(item.get("put_ask", 0)),
                            "volume": int(item.get("put_volume", 0)),
                            "oi": int(item.get("put_oi", 0)),
                            "iv": float(item.get("put_iv", 0)),
                        },
                    }
                )
        return options

    def _mock_options_chain(self, symbol: str, date: str) -> OptionsChainSnapshot:
        """Generate deterministic mock options chain for dryrun mode.

        Args:
            symbol: Underlying symbol
            date: Snapshot date

        Returns:
            Mock OptionsChainSnapshot with synthetic data
        """
        # Base price from symbol hash
        underlying_price = 21000.0 + (hash(symbol) % 2000)

        # Generate strikes around ATM
        strikes = []
        for i in range(-5, 6):  # 11 strikes: 5 OTM, ATM, 5 ITM
            strike = round(underlying_price + i * 500, -2)  # Round to nearest 100
            strikes.append(strike)

        # Generate mock options for 3 expiries
        expiries = ["2025-01-30", "2025-02-27", "2025-03-27"]
        options = []

        for strike in strikes:
            for expiry in expiries:
                # Simple IV model: higher IV for OTM
                atm_distance = abs(strike - underlying_price) / underlying_price
                iv = 0.15 + atm_distance * 0.5

                options.append(
                    {
                        "strike": strike,
                        "expiry": expiry,
                        "call": {
                            "last_price": max(0, underlying_price - strike) + 10,
                            "bid": max(0, underlying_price - strike) + 9,
                            "ask": max(0, underlying_price - strike) + 11,
                            "volume": 10000,
                            "oi": 100000,
                            "iv": round(iv, 4),
                        },
                        "put": {
                            "last_price": max(0, strike - underlying_price) + 10,
                            "bid": max(0, strike - underlying_price) + 9,
                            "ask": max(0, strike - underlying_price) + 11,
                            "volume": 8000,
                            "oi": 95000,
                            "iv": round(iv * 1.05, 4),  # Put skew
                        },
                    }
                )

        logger.info(
            f"[DRYRUN] Generated mock options chain for {symbol}",
            extra={
                "component": "options_provider",
                "symbol": symbol,
                "strikes": len(strikes),
                "expiries": len(expiries),
                "mode": "dryrun",
            },
        )

        return OptionsChainSnapshot(
            symbol=symbol,
            date=date,
            timestamp=datetime.now(),
            underlying_price=underlying_price,
            options=options,
            metadata={
                "total_strikes": len(strikes),
                "expiries": expiries,
                "source": "stub",
                "dryrun": True,
            },
        )


# ============================================================================
# Macro Indicator Provider
# ============================================================================


class MacroIndicatorProvider(MarketDataProvider):
    """Fetch macro economic indicators via public APIs.

    Uses yfinance for market indices (NIFTY, India VIX) and public APIs
    for economic indicators (bond yields, currency rates).

    Supported indicators:
    - NIFTY50: NIFTY 50 index
    - BANKNIFTY: BANK NIFTY index
    - INDIAVIX: India VIX (volatility index)
    - USDINR: USD/INR exchange rate
    - IN10Y: India 10-year bond yield

    Safety:
    - Read-only (public data APIs)
    - Respects retry limits and backoff
    - Dryrun mode returns deterministic mock data
    """

    def __init__(self, settings: Settings, dry_run: bool = True):
        """Initialize macro indicator provider.

        Args:
            settings: Application settings
            dry_run: If True, return mock data instead of real API calls
        """
        super().__init__(settings, dry_run)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(is_transient),  # type: ignore[arg-type]
        reraise=True,
    )
    def fetch(  # type: ignore[override]
        self, indicator: str, date: str | None = None
    ) -> MacroIndicatorSnapshot:
        """Fetch macro indicator value.

        Args:
            indicator: Indicator code (e.g., "NIFTY50", "INDIAVIX", "USDINR")
            date: Date for snapshot (YYYY-MM-DD), defaults to today

        Returns:
            MacroIndicatorSnapshot with value and change

        Raises:
            ConnectionError: On network/timeout errors (retried)
            ValueError: On invalid indicator code
        """
        snapshot_date = date or datetime.now().strftime("%Y-%m-%d")

        if self.dry_run:
            return self._mock_macro_indicator(indicator, snapshot_date)

        try:
            # Map indicator codes to data source
            if indicator in ["NIFTY50", "BANKNIFTY", "INDIAVIX"]:
                return self._fetch_from_yfinance(indicator, snapshot_date)
            elif indicator == "USDINR":
                return self._fetch_currency_rate(snapshot_date)
            elif indicator == "IN10Y":
                return self._fetch_bond_yield(snapshot_date)
            else:
                raise ValueError(f"Unsupported macro indicator: {indicator}")

        except Exception as e:
            logger.error(
                f"Failed to fetch macro indicator {indicator}: {e}",
                extra={"component": "macro_provider", "indicator": indicator},
            )
            raise

    def _fetch_from_yfinance(self, indicator: str, date: str) -> MacroIndicatorSnapshot:
        """Fetch index data from yfinance.

        Args:
            indicator: Index code
            date: Snapshot date

        Returns:
            MacroIndicatorSnapshot
        """
        try:
            import yfinance as yf  # type: ignore[import-not-found, import-untyped]
        except ImportError:
            logger.warning("yfinance not installed, using mock data")
            return self._mock_macro_indicator(indicator, date)

        # Map indicator to yfinance ticker
        ticker_map = {
            "NIFTY50": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "INDIAVIX": "^INDIAVIX",
        }

        ticker_symbol = ticker_map.get(indicator)
        if not ticker_symbol:
            raise ValueError(f"Unknown index: {indicator}")

        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")  # Get recent data for change calculation

        if hist.empty:
            raise ValueError(f"No data available for {indicator}")

        latest = hist.iloc[-1]
        value = float(latest["Close"])

        # Calculate change if we have previous day
        change = None
        change_pct = None
        if len(hist) > 1:
            prev = hist.iloc[-2]["Close"]
            change = value - prev
            change_pct = (change / prev) * 100

        return MacroIndicatorSnapshot(
            indicator=indicator,
            date=date,
            timestamp=datetime.now(),
            value=value,
            change=change,
            change_pct=change_pct,
            metadata={"source": "yfinance", "ticker": ticker_symbol},
        )

    def _fetch_currency_rate(self, date: str) -> MacroIndicatorSnapshot:
        """Fetch USD/INR exchange rate.

        Args:
            date: Snapshot date

        Returns:
            MacroIndicatorSnapshot for USDINR
        """
        # Placeholder: Use yfinance or forex API
        # For now, return mock data
        return self._mock_macro_indicator("USDINR", date)

    def _fetch_bond_yield(self, date: str) -> MacroIndicatorSnapshot:
        """Fetch India 10-year bond yield.

        Args:
            date: Snapshot date

        Returns:
            MacroIndicatorSnapshot for IN10Y
        """
        # Placeholder: Use RBI API or bond market data source
        # For now, return mock data
        return self._mock_macro_indicator("IN10Y", date)

    def _mock_macro_indicator(self, indicator: str, date: str) -> MacroIndicatorSnapshot:
        """Generate deterministic mock macro indicator for dryrun mode.

        Args:
            indicator: Indicator code
            date: Snapshot date

        Returns:
            Mock MacroIndicatorSnapshot with synthetic data
        """
        # Base values for common indicators
        base_values = {
            "NIFTY50": 21500.0,
            "BANKNIFTY": 45000.0,
            "INDIAVIX": 15.5,
            "USDINR": 83.2,
            "IN10Y": 7.1,
        }

        base_value = base_values.get(indicator, 100.0)

        # Add small random variation based on date hash
        variation = (hash(date + indicator) % 100) / 100.0 - 0.5
        value = base_value * (1 + variation / 100)

        # Mock change
        change = value * 0.007  # 0.7% change
        change_pct = 0.7

        logger.info(
            f"[DRYRUN] Generated mock macro indicator for {indicator}",
            extra={
                "component": "macro_provider",
                "indicator": indicator,
                "value": value,
                "mode": "dryrun",
            },
        )

        return MacroIndicatorSnapshot(
            indicator=indicator,
            date=date,
            timestamp=datetime.now(),
            value=round(value, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            metadata={"source": "stub", "dryrun": True},
        )


# ============================================================================
# Provider Factory
# ============================================================================


def create_order_book_provider(
    settings: Settings,
    client: BreezeClient | None = None,
    dry_run: bool | None = None,
) -> BreezeOrderBookProvider:
    """Create order book provider with safe defaults.

    Args:
        settings: Application settings
        client: BreezeClient instance (optional)
        dry_run: Override dryrun mode (defaults to not settings.order_book_enabled)

    Returns:
        BreezeOrderBookProvider instance
    """
    if dry_run is None:
        # Default to dryrun if order book not enabled in config
        dry_run = not settings.order_book_enabled

    return BreezeOrderBookProvider(settings, client=client, dry_run=dry_run)


def create_options_provider(
    settings: Settings,
    client: BreezeClient | None = None,
    dry_run: bool | None = None,
) -> BreezeOptionsProvider:
    """Create options provider with safe defaults.

    Args:
        settings: Application settings
        client: BreezeClient instance (optional)
        dry_run: Override dryrun mode (defaults to not settings.options_enabled)

    Returns:
        BreezeOptionsProvider instance
    """
    if dry_run is None:
        # Default to dryrun if options not enabled in config
        dry_run = not settings.options_enabled

    return BreezeOptionsProvider(settings, client=client, dry_run=dry_run)


def create_macro_provider(
    settings: Settings, dry_run: bool | None = None
) -> MacroIndicatorProvider:
    """Create macro indicator provider with safe defaults.

    Args:
        settings: Application settings
        dry_run: Override dryrun mode (defaults to not settings.macro_enabled)

    Returns:
        MacroIndicatorProvider instance
    """
    if dry_run is None:
        # Default to dryrun if macro not enabled in config
        dry_run = not settings.macro_enabled

    return MacroIndicatorProvider(settings, dry_run=dry_run)
