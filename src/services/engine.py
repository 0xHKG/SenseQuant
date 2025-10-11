"""Main trading engine."""

from __future__ import annotations

import signal
import sys
from datetime import datetime, time, timedelta
from typing import Any

import pandas as pd
import pytz
from loguru import logger

from src.adapters.breeze_client import BreezeClient
from src.adapters.sentiment_provider import get_sentiment
from src.app.config import settings
from src.domain.strategies import intraday as intraday_strategy
from src.domain.strategies.swing import SwingStrategy
from src.services.journal import TradeJournal


def is_market_open(now: datetime | None = None) -> bool:
    """Check if market is open (Mon-Fri, 9:15-15:29 IST)."""
    if now is None:
        now = datetime.now()
    return now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 29)


class Engine:
    """Trading engine orchestrating strategies and execution."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols
        self.client = BreezeClient(
            settings.breeze_api_key,
            settings.breeze_api_secret,
            settings.breeze_session_token,
            dry_run=(settings.mode != "live"),
        )
        self.swing = SwingStrategy()
        self.journal = TradeJournal()
        self._running = True
        self._open_positions: dict[str, int] = {}  # symbol -> qty (positive=long, negative=short)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _shutdown_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.warning("Received signal {}, shutting down gracefully...", signum)
        self._running = False
        self.journal.close()
        logger.info("Shutdown complete")
        sys.exit(0)

    def start(self) -> None:
        """Start the engine and authenticate with broker."""
        logger.info("Engine start. Mode={}, Symbols={}", settings.mode, self.symbols)
        self.client.authenticate()

    def square_off_intraday(self, symbol: str) -> None:
        """
        Square off all intraday positions for symbol at EOD.

        Called at 15:29 IST to close all open positions.
        In dry-run mode: logs only. In live mode: places market orders.

        Args:
            symbol: Stock symbol to square off
        """
        if symbol not in self._open_positions or self._open_positions[symbol] == 0:
            logger.info(
                f"No open position for {symbol} to square off",
                extra={"component": "engine", "symbol": symbol},
            )
            return

        qty = self._open_positions[symbol]
        side = "SELL" if qty > 0 else "BUY"
        abs_qty = abs(qty)

        logger.info(
            f"Squaring off {symbol}: {side} {abs_qty} shares",
            extra={"component": "engine", "symbol": symbol, "side": side, "qty": abs_qty},
        )

        if settings.mode == "dryrun":
            # Log only
            self.journal.log(
                symbol=symbol,
                action=side,
                qty=abs_qty,
                price=0.0,  # Unknown in dry-run without fetch
                pnl=0.0,
                reason="eod_square_off",
                mode=settings.mode,
                order_id="DRYRUN",
                status="FILLED",
            )
        else:
            # Place real market order
            try:
                response = self.client.place_order(
                    symbol=symbol,
                    side=side,  # type: ignore[arg-type]
                    qty=abs_qty,
                    order_type="MARKET",
                )
                self.journal.log(
                    symbol=symbol,
                    action=side,
                    qty=abs_qty,
                    price=0.0,  # Filled price unknown until confirmation
                    pnl=0.0,
                    reason="eod_square_off",
                    mode=settings.mode,
                    order_id=response.order_id,
                    status=response.status,
                )
            except Exception as e:
                logger.error(
                    f"Failed to square off {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )

        # Clear position
        self._open_positions[symbol] = 0

    def tick_intraday(self, symbol: str) -> None:
        """
        Process intraday tick for symbol.

        Fetches recent bars, computes features, generates signal with sentiment gating,
        and logs decision to journal. Respects 9:15-15:29 IST trading window.

        Args:
            symbol: Stock symbol to process
        """
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)

        # Check if within intraday window (9:15-15:29 IST)
        market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now_ist.replace(hour=15, minute=29, second=0, microsecond=0)

        if not (market_start <= now_ist <= market_end):
            logger.debug(
                f"Outside intraday window for {symbol}",
                extra={"component": "engine", "symbol": symbol, "time_ist": now_ist.isoformat()},
            )
            return

        # Fetch historical bars for feature computation
        lookback_minutes = settings.intraday_feature_lookback_minutes
        start_ts = pd.Timestamp(now_ist - timedelta(minutes=lookback_minutes), tz=ist)
        end_ts = pd.Timestamp(now_ist, tz=ist)

        try:
            bars = self.client.historical_bars(
                symbol=symbol,
                interval=settings.intraday_bar_interval,
                start=start_ts,
                end=end_ts,
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch bars for {symbol}: {e}",
                extra={"component": "engine", "symbol": symbol, "error": str(e)},
            )
            return

        if not bars:
            logger.warning(
                f"No bars returned for {symbol}",
                extra={"component": "engine", "symbol": symbol},
            )
            return

        # Convert bars to DataFrame
        df = pd.DataFrame(
            [
                {
                    "ts": bar.ts,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        )

        # Compute features
        try:
            df_features = intraday_strategy.compute_features(df, settings)
        except Exception as e:
            logger.error(
                f"Feature computation failed for {symbol}: {e}",
                extra={"component": "engine", "symbol": symbol, "error": str(e)},
            )
            return

        # Get sentiment
        sentiment = get_sentiment(symbol)

        # Generate signal
        try:
            sig = intraday_strategy.signal(df_features, settings, sentiment=sentiment)
        except Exception as e:
            logger.error(
                f"Signal generation failed for {symbol}: {e}",
                extra={"component": "engine", "symbol": symbol, "error": str(e)},
            )
            return

        # Log decision to journal
        self.journal.log(
            symbol=symbol,
            action=sig.direction,
            qty=0,  # No actual order placed in this tick (just signal evaluation)
            price=df["close"].iloc[-1] if not df.empty else 0.0,
            pnl=0.0,
            reason=sig.meta.get("reason", "signal_generated") if sig.meta else "signal_generated",
            mode=settings.mode,
            order_id="",
            status="SIGNAL",
            strategy="intraday",
            meta_json=str(sig.meta) if sig.meta else "",
        )

        logger.info(
            f"Intraday tick processed for {symbol}: {sig.direction}",
            extra={
                "component": "engine",
                "symbol": symbol,
                "direction": sig.direction,
                "strength": sig.strength,
            },
        )

    def daily_swing(self, symbol: str) -> None:
        """Process daily swing signal for symbol."""
        # Placeholder for swing strategy
        logger.info("daily_swing placeholder for {}", symbol)
        # TODO: Implement with historical_bars(symbol, "1day", start_ts, end_ts)

    def stop(self) -> None:
        """Stop the engine and clean up resources."""
        logger.info("Stopping engine...")
        self.journal.close()
        self._running = False
