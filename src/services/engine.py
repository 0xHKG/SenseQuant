"""Main trading engine."""

from __future__ import annotations

import signal
import sys
import time as time_module
from datetime import datetime, time, timedelta
from typing import Any, Literal

import pandas as pd
import pytz
from loguru import logger

from src.adapters.breeze_client import BreezeClient
from src.adapters.sentiment_provider import StubSentimentProvider
from src.app.config import settings
from src.domain.strategies import intraday as intraday_strategy
from src.domain.strategies.intraday import IntradayPosition
from src.domain.types import StudentConfig
from src.services.data_feed import DataFeed
from src.services.journal import TradeJournal
from src.services.monitoring import MonitoringService
from src.services.risk_manager import RiskManager
from src.services.sentiment import SentimentCache
from src.services.sentiment.registry import SentimentProviderRegistry
from src.services.teacher_student import StudentModel


def is_market_open(now: datetime | None = None) -> bool:
    """Check if market is open (Mon-Fri, 9:15-15:29 IST)."""
    if now is None:
        now = datetime.now()
    return now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 29)


class Engine:
    """Trading engine orchestrating strategies and execution."""

    def __init__(
        self,
        symbols: list[str],
        data_feed: DataFeed | None = None,
        sentiment_registry: SentimentProviderRegistry | None = None,
    ) -> None:
        """Initialize trading engine.

        Args:
            symbols: List of symbols to trade
            data_feed: Optional DataFeed for backtest/dryrun modes (uses live API if None)
            sentiment_registry: Optional SentimentProviderRegistry for pluggable sentiment providers
                               (uses stub provider if None for backward compatibility)
        """
        self.symbols = symbols
        self.data_feed = data_feed
        self.client = BreezeClient(
            settings.breeze_api_key,
            settings.breeze_api_secret,
            settings.breeze_session_token,
            dry_run=(settings.mode != "live"),
        )
        self.journal = TradeJournal()
        self._running = True
        self._intraday_positions: dict[str, IntradayPosition] = {}  # symbol -> IntradayPosition
        self._swing_positions: dict[str, Any] = {}  # symbol -> SwingPosition

        # Initialize sentiment provider and cache
        # Support both new registry-based approach and legacy stub provider
        self._sentiment_registry = sentiment_registry
        self._sentiment_provider = StubSentimentProvider()
        self._sentiment_cache = SentimentCache(
            ttl_seconds=settings.sentiment_cache_ttl,
            rate_limit_per_min=settings.sentiment_rate_limit_per_min,
        )

        # Initialize risk manager
        self._risk_manager = RiskManager(
            starting_capital=settings.starting_capital,
            mode=settings.position_sizing_mode,
            risk_per_trade_pct=settings.risk_per_trade_pct,
            atr_multiplier=settings.atr_multiplier,
            max_position_value_per_symbol=settings.max_position_value_per_symbol,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            trading_fee_bps=settings.trading_fee_bps,
            slippage_bps=settings.slippage_bps,
        )

        # Initialize Student model if enabled
        self._student_model: StudentModel | None = None
        if settings.enable_student_inference:
            if settings.student_model_path and settings.student_metadata_path:
                try:
                    # Create minimal config (paths only needed for loading)
                    config = StudentConfig(
                        teacher_metadata_path="",  # Not needed for load
                        teacher_labels_path="",  # Not needed for load
                    )
                    self._student_model = StudentModel(config)
                    self._student_model.load(
                        settings.student_model_path,
                        settings.student_metadata_path,
                    )
                    # US-021 Phase 2: Log model version on startup
                    model_version = (
                        self._student_model.metadata.get("model_version", "unknown")
                        if self._student_model.metadata
                        else "unknown"
                    )
                    logger.info(
                        f"Student model loaded successfully (version: {model_version})",
                        extra={
                            "component": "engine",
                            "model_path": settings.student_model_path,
                            "model_version": model_version,
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load Student model: {e}",
                        extra={
                            "component": "engine",
                            "error": str(e),
                        },
                    )
            else:
                logger.warning(
                    "Student inference enabled but paths not configured",
                    extra={"component": "engine"},
                )

        # Initialize monitoring service
        self._monitoring: MonitoringService | None = None
        if settings.enable_monitoring:
            try:
                self._monitoring = MonitoringService(settings)
                logger.info(
                    "Monitoring service initialized",
                    extra={"component": "engine"},
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize monitoring: {e}",
                    extra={"component": "engine", "error": str(e)},
                )

        # Initialize telemetry writer (US-018 Phase 4)
        import random as random_module
        from pathlib import Path as PathLib

        from src.services.accuracy_analyzer import TelemetryWriter

        self._telemetry_writer: TelemetryWriter | None = None
        self._last_telemetry_flush = datetime.now()
        self._random = random_module  # For sampling

        if settings.live_telemetry_enabled:
            try:
                telemetry_dir = PathLib(settings.telemetry_storage_path) / "live"
                telemetry_dir.mkdir(parents=True, exist_ok=True)

                self._telemetry_writer = TelemetryWriter(
                    output_dir=telemetry_dir,
                    format="csv",
                    compression=False,
                    buffer_size=50,  # Smaller buffer for live mode
                )

                logger.info(
                    "Live telemetry enabled",
                    extra={
                        "component": "engine",
                        "throttle_seconds": settings.live_telemetry_throttle_seconds,
                        "sample_rate": settings.live_telemetry_sample_rate,
                        "storage_path": str(telemetry_dir),
                    },
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize telemetry writer: {e}",
                    extra={"component": "engine", "error": str(e)},
                )
                self._telemetry_writer = None

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _shutdown_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.warning("Received signal {}, shutting down gracefully...", signum)
        self._running = False
        self.journal.close()

        # Close telemetry writer (US-018 Phase 4)
        if self._telemetry_writer is not None:
            try:
                self._telemetry_writer.close()
                logger.info("Telemetry writer closed")
            except Exception as e:
                logger.error(f"Error closing telemetry writer: {e}")

        logger.info("Shutdown complete")
        sys.exit(0)

    def _capture_telemetry_trace(
        self,
        symbol: str,
        position: IntradayPosition,
        exit_price: float,
        exit_fees: float,
        realized_pnl: float,
        gross_pnl: float,
        reason: str,
    ) -> None:
        """Capture telemetry trace for closed position.

        US-018 Phase 4: Records prediction trace with throttling and sampling.

        Args:
            symbol: Stock symbol
            position: Closed IntradayPosition
            exit_price: Exit price
            exit_fees: Exit fees
            realized_pnl: Realized PnL (net of fees)
            gross_pnl: Gross PnL (before fees)
            reason: Exit reason
        """
        # Check throttling
        if not self._should_emit_telemetry():
            return

        # Check sampling rate
        if self._random.random() > settings.live_telemetry_sample_rate:
            logger.debug(
                f"Telemetry sample rate check: skipping trace for {symbol}",
                extra={"component": "engine"},
            )
            return

        try:
            from src.services.accuracy_analyzer import PredictionTrace

            # Calculate holding period in minutes
            exit_time = datetime.now()
            holding_period_minutes = int((exit_time - position.entry_time).total_seconds() / 60)

            # Calculate realized return percentage
            position_value = position.entry_price * position.qty
            realized_return_pct = (
                (realized_pnl / position_value) * 100 if position_value > 0 else 0.0
            )

            # Determine actual direction based on realized return (0.3% threshold for intraday)
            if realized_return_pct > 0.3:
                actual_direction = "LONG"
            elif realized_return_pct < -0.3:
                actual_direction = "SHORT"
            else:
                actual_direction = "NOOP"

            # Calculate slippage estimate (difference between expected and actual)
            # In live trading, we don't have perfect signal price, but we can estimate
            slippage_bps = settings.slippage_bps
            slippage_pct = slippage_bps / 100.0  # Convert basis points to percentage

            # Build features dict (extract from position if available)
            features = {}
            if hasattr(position, "signal_features") and position.signal_features:
                features = position.signal_features
            else:
                # Minimal feature set from position
                features = {
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                }

            # Build metadata
            metadata = {
                "exit_reason": reason,
                "entry_fees": position.entry_fees,
                "exit_fees": exit_fees,
                "total_fees": position.entry_fees + exit_fees,
                "position_value": position_value,
                "gross_pnl": gross_pnl,
                "slippage_pct": slippage_pct,
                "mode": settings.mode,
            }

            # Create prediction trace
            trace = PredictionTrace(
                timestamp=position.entry_time.to_pydatetime()
                if hasattr(position.entry_time, "to_pydatetime")
                else position.entry_time,
                symbol=symbol,
                strategy="intraday",
                predicted_direction=position.direction,
                actual_direction=actual_direction,
                predicted_confidence=0.7,  # Default confidence for live trades
                entry_price=position.entry_price,
                exit_price=exit_price,
                holding_period_minutes=holding_period_minutes,
                realized_return_pct=realized_return_pct,
                features=features,
                metadata=metadata,
            )

            # Write trace (non-blocking, buffered)
            self._telemetry_writer.write_trace(trace)

            logger.info(
                f"Captured telemetry trace for {symbol}",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "predicted": position.direction,
                    "actual": actual_direction,
                    "return_pct": realized_return_pct,
                },
            )

        except Exception as e:
            # Non-blocking: log error but don't crash trading
            logger.error(
                f"Failed to capture telemetry trace for {symbol}: {e}",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "error": str(e),
                },
            )

    def _should_emit_telemetry(self) -> bool:
        """Check if enough time has elapsed to emit telemetry (throttling).

        US-018 Phase 4: Implements throttling to minimize overhead in live trading.

        Returns:
            True if telemetry should be emitted, False otherwise
        """
        if self._telemetry_writer is None:
            return False

        elapsed = (datetime.now() - self._last_telemetry_flush).total_seconds()
        throttle_seconds = settings.live_telemetry_throttle_seconds

        if elapsed >= throttle_seconds:
            self._last_telemetry_flush = datetime.now()
            logger.debug(
                f"Telemetry throttle check: emitting (elapsed={elapsed:.1f}s >= threshold={throttle_seconds}s)",
                extra={"component": "engine"},
            )
            return True

        logger.debug(
            f"Telemetry throttle check: skipping (elapsed={elapsed:.1f}s < threshold={throttle_seconds}s)",
            extra={"component": "engine"},
        )
        return False

    def start(self) -> None:
        """Start the engine and authenticate with broker."""
        logger.info("Engine start. Mode={}, Symbols={}", settings.mode, self.symbols)
        self.client.authenticate()

    def square_off_intraday(self, symbol: str) -> None:
        """
        Square off all intraday positions for symbol at EOD (3:20 PM IST).

        Calculates fees, realized PnL, records trade with risk manager,
        and checks circuit breaker.

        Args:
            symbol: Stock symbol to square off
        """
        if symbol not in self._intraday_positions:
            logger.info(
                f"No open intraday position for {symbol} to square off",
                extra={"component": "engine", "symbol": symbol},
            )
            return

        position = self._intraday_positions[symbol]

        # For EOD square-off, use current market price (simplified: use entry price for dry-run)
        exit_price = position.entry_price  # In production, fetch real-time price

        logger.info(
            f"EOD square-off for {symbol}: closing {position.direction} position of {position.qty} shares",
            extra={
                "component": "engine",
                "symbol": symbol,
                "direction": position.direction,
                "qty": position.qty,
            },
        )

        # Close position with risk management
        self._close_intraday_position(symbol, exit_price, reason="auto_square_off_eod")

        # Check circuit breaker after trade
        if self._risk_manager.is_circuit_breaker_active():
            logger.error(
                "CIRCUIT BREAKER ACTIVATED from EOD square-off! Squaring off all positions",
                extra={"component": "engine"},
            )
            self._square_off_all_positions()

    def _close_intraday_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """
        Close intraday position with fee tracking and risk recording.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            reason: Reason for closing (e.g., "signal_exit", "auto_square_off_eod")
        """
        if symbol not in self._intraday_positions:
            logger.warning(
                f"No intraday position to close for {symbol}",
                extra={"component": "engine", "symbol": symbol},
            )
            return

        position = self._intraday_positions[symbol]

        # Calculate exit fees
        exit_fees = self._risk_manager.calculate_fees(position.qty, exit_price)

        # Calculate gross PnL
        direction_multiplier = 1.0 if position.direction == "LONG" else -1.0
        gross_pnl = (exit_price - position.entry_price) * position.qty * direction_multiplier

        # Calculate realized PnL (net of fees)
        realized_pnl = gross_pnl - position.entry_fees - exit_fees

        # Journal with fees
        side = "SELL" if position.direction == "LONG" else "BUY"
        self.journal.log(
            symbol=symbol,
            action=side,
            qty=position.qty,
            price=exit_price,
            pnl=realized_pnl,
            reason=reason,
            mode=settings.mode,
            order_id="INTRADAY_EXIT",
            status="EXIT",
            strategy="intraday",
            meta_json=str(
                {
                    "entry_fees": position.entry_fees,
                    "exit_fees": exit_fees,
                    "gross_pnl": gross_pnl,
                    "realized_pnl": realized_pnl,
                    "entry_time": position.entry_time.isoformat(),
                }
            ),
        )

        logger.info(
            f"Closed intraday position for {symbol}: realized_pnl={realized_pnl:.2f}",
            extra={
                "component": "engine",
                "symbol": symbol,
                "realized_pnl": realized_pnl,
                "gross_pnl": gross_pnl,
                "total_fees": position.entry_fees + exit_fees,
            },
        )

        # Update risk manager and check circuit breaker
        self._risk_manager.update_position(
            symbol=symbol,
            qty=position.qty,
            price=exit_price,
            is_opening=False,
        )

        total_fees = position.entry_fees + exit_fees
        self._risk_manager.record_trade(
            symbol=symbol,
            realized_pnl=realized_pnl,
            fees=total_fees,
        )

        # Capture telemetry trace (US-018 Phase 4)
        self._capture_telemetry_trace(
            symbol=symbol,
            position=position,
            exit_price=exit_price,
            exit_fees=exit_fees,
            realized_pnl=realized_pnl,
            gross_pnl=gross_pnl,
            reason=reason,
        )

        # Remove position
        del self._intraday_positions[symbol]

    def tick_intraday(self, symbol: str) -> None:
        """
        Process intraday tick for symbol.

        Fetches recent bars, computes features, generates signal with sentiment gating,
        and logs decision to journal. Respects 9:15-15:29 IST trading window.

        Args:
            symbol: Stock symbol to process
        """
        # Start performance tracking
        tick_start_time = time_module.time()

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
        start_ts = pd.Timestamp(now_ist - timedelta(minutes=lookback_minutes))
        end_ts = pd.Timestamp(now_ist)

        # Use DataFeed if provided (for backtest/dryrun with CSV)
        if self.data_feed is not None and settings.mode in ("backtest", "dryrun"):
            try:
                df_bars = self.data_feed.get_historical_bars(
                    symbol=symbol,
                    from_date=start_ts.to_pydatetime(),
                    to_date=end_ts.to_pydatetime(),
                    interval=settings.intraday_bar_interval,
                )

                if df_bars.empty:
                    logger.warning(
                        f"No bars returned from DataFeed for {symbol}",
                        extra={"component": "engine", "symbol": symbol},
                    )
                    return

                # Convert DataFrame to Bar-like dict format
                bars = [
                    type(
                        "Bar",
                        (),
                        {
                            "ts": row["timestamp"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"],
                        },
                    )()
                    for _, row in df_bars.iterrows()
                ]

            except Exception as e:
                logger.error(
                    f"Failed to fetch bars from DataFeed for {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )
                return
        else:
            # Use live BreezeClient (default for live mode or when data_feed not provided)
            try:
                bars = self.client.historical_bars(
                    symbol=symbol,
                    interval=settings.intraday_bar_interval,
                    start=start_ts,
                    end=end_ts,
                )
            except Exception as e:
                logger.error(
                    f"Failed to fetch bars from BreezeClient for {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )
                return

            if not bars:
                logger.warning(
                    f"No bars returned from BreezeClient for {symbol}",
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

        # Fetch sentiment with caching and track API latency
        sentiment_start_time = time_module.time()

        # Use new registry-based approach if available, otherwise fall back to legacy
        if self._sentiment_registry is not None:
            try:
                sentiment_result = self._sentiment_registry.get_sentiment(symbol)
                if sentiment_result:
                    sentiment_score = sentiment_result.value
                    sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000
                    logger.info(
                        f"Sentiment for {symbol}: {sentiment_result.value:.3f} "
                        f"(source={sentiment_result.source}, confidence={sentiment_result.confidence:.2f})",
                        extra={
                            "component": "engine",
                            "symbol": symbol,
                            "sentiment": sentiment_result.value,
                            "source": sentiment_result.source,
                            "confidence": sentiment_result.confidence,
                        },
                    )
                else:
                    sentiment_score = 0.0
                    sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000
                    logger.debug(
                        f"No sentiment data available for {symbol}, using neutral (0.0)",
                        extra={"component": "engine", "symbol": symbol},
                    )
            except Exception as e:
                sentiment_score = 0.0
                sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000
                logger.warning(
                    f"Sentiment fetch failed for {symbol}: {e}, using neutral (0.0)",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )
        else:
            # Legacy path: use stub provider with cache
            sentiment_score, sentiment_meta = self._sentiment_cache.get(
                symbol, self._sentiment_provider, fallback=0.0
            )
            sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000

        # Record sentiment API latency
        if self._monitoring and self._monitoring.settings.monitoring_enable_performance_tracking:
            self._monitoring.record_performance_metric(
                "sentiment_api_latency",
                sentiment_latency_ms,
                {"symbol": symbol, "strategy": "intraday"},
            )

        # Generate signal
        try:
            sig = intraday_strategy.signal(df_features, settings, sentiment=sentiment_score)
        except Exception as e:
            logger.error(
                f"Signal generation failed for {symbol}: {e}",
                extra={"component": "engine", "symbol": symbol, "error": str(e)},
            )
            return

        # US-021 Phase 2: Generate Student prediction if enabled (intraday)
        student_meta: dict[str, Any] = {}
        student_adjusted_strength = sig.strength  # Default to original strength

        if self._student_model is not None and sig.direction in ("LONG", "SHORT"):
            try:
                # Extract features for Student prediction
                latest_row = df_features.iloc[-1]
                feature_dict = {
                    col: float(latest_row[col])
                    for col in self._student_model.feature_cols
                    if col in df_features.columns
                }

                # Make prediction
                prediction = self._student_model.predict_single(feature_dict, symbol=symbol)

                student_meta = {
                    "probability": prediction.probability,
                    "decision": prediction.decision,
                    "confidence": prediction.confidence,
                    "model_version": prediction.model_version,
                }

                # US-021 Phase 3: Record prediction for monitoring (outcome unknown at entry)
                if self._monitoring and self._monitoring.settings.student_monitoring_enabled:
                    # Convert decision (0/1) to direction string for monitoring
                    decision_str = "BUY" if prediction.decision == 1 else "SELL"
                    self._monitoring.record_student_prediction(
                        symbol=symbol,
                        prediction=decision_str,
                        probability=prediction.probability,
                        confidence=prediction.confidence,
                        actual_outcome=None,  # Will be updated on exit
                        model_version=prediction.model_version,
                    )

                # Adjust signal strength based on student confidence (if confidence threshold met)
                if prediction.confidence >= settings.student_model_confidence_threshold:
                    # Student agrees with signal: boost confidence (decision: 1 = BUY, 0 = SELL)
                    if (sig.direction == "LONG" and prediction.decision == 1) or (
                        sig.direction == "SHORT" and prediction.decision == 0
                    ):
                        student_adjusted_strength = min(1.0, sig.strength * 1.2)
                        logger.info(
                            f"Student prediction AGREES with {sig.direction} signal for {symbol}, "
                            f"boosting strength {sig.strength:.2f} -> {student_adjusted_strength:.2f}",
                            extra={"component": "student", "symbol": symbol, **student_meta},
                        )
                    # Student disagrees with signal: reduce confidence or suppress
                    else:
                        student_adjusted_strength = sig.strength * 0.5
                        logger.warning(
                            f"Student prediction DISAGREES with {sig.direction} signal for {symbol}, "
                            f"reducing strength {sig.strength:.2f} -> {student_adjusted_strength:.2f}",
                            extra={"component": "student", "symbol": symbol, **student_meta},
                        )
                else:
                    logger.info(
                        f"Student confidence {prediction.confidence:.2f} below threshold "
                        f"{settings.student_model_confidence_threshold:.2f}, not adjusting signal",
                        extra={"component": "student", "symbol": symbol},
                    )

            except Exception as e:
                logger.warning(
                    f"Student prediction failed for {symbol}: {e}",
                    extra={"component": "student", "symbol": symbol, "error": str(e)},
                )

        # Get current position
        current_position = self._intraday_positions.get(symbol)
        current_price = df_features.iloc[-1]["close"] if "close" in df_features.columns else 0.0

        # Extract ATR for position sizing
        atr = df_features.iloc[-1].get("atr14", 0.0) if "atr14" in df_features.columns else 0.0

        # ENTRY LOGIC: Open new position if LONG or SHORT signal and no current position
        if sig.direction in ("LONG", "SHORT") and current_position is None:
            entry_price = current_price
            signal_strength = (
                student_adjusted_strength  # Use student-adjusted strength (US-021 Phase 2)
            )
            reason = sig.meta.get("reason", "intraday_signal") if sig.meta else "intraday_signal"

            # Calculate position size with risk management
            pos_size = self._risk_manager.calculate_position_size(
                symbol=symbol,
                price=entry_price,
                atr=atr,
                signal_strength=signal_strength,
            )

            # Check risk limits
            risk_check = self._risk_manager.can_open_position(
                symbol=symbol,
                qty=pos_size.qty,
                price=entry_price,
            )

            if not risk_check.allowed:
                logger.warning(
                    f"Intraday position blocked for {symbol}: {risk_check.reason}",
                    extra={
                        "component": "engine",
                        "symbol": symbol,
                        "reason": risk_check.reason,
                        "breaker_active": risk_check.breaker_active,
                    },
                )
                return

            qty = risk_check.allowed_qty
            entry_fees = self._risk_manager.calculate_fees(qty, entry_price)

            # Create position with fees
            # Type assertion: we know sig.direction is LONG or SHORT here (checked in if condition)
            direction: Literal["LONG", "SHORT"] = sig.direction  # type: ignore[assignment]
            self._intraday_positions[symbol] = IntradayPosition(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=pd.Timestamp(df_features.iloc[-1]["ts"]),
                qty=qty,
                entry_fees=entry_fees,
            )

            # Update risk manager position tracking
            self._risk_manager.update_position(
                symbol=symbol,
                qty=qty,
                price=entry_price,
                is_opening=True,
            )

            # Journal with risk metadata (US-021 Phase 2: include student metadata)
            side = "BUY" if sig.direction == "LONG" else "SELL"
            self.journal.log(
                symbol=symbol,
                action=side,
                qty=qty,
                price=entry_price,
                pnl=0.0,
                reason=reason,
                mode=settings.mode,
                order_id="INTRADAY_ENTRY",
                status="ENTRY",
                strategy="intraday",
                meta_json=str(
                    {
                        **(sig.meta if sig.meta else {}),
                        "position_size_rationale": pos_size.rationale,
                        "risk_check": risk_check.reason,
                        "entry_fees": entry_fees,
                        "signal_strength": signal_strength,
                        "signal_strength_original": sig.strength,  # Original strength before student adjustment
                        "sentiment_score": sentiment_score,
                        "atr": atr,
                        "student_prediction": student_meta,  # Student model metadata (US-021 Phase 2)
                    }
                ),
            )

            logger.info(
                f"Opened intraday {sig.direction} position for {symbol}: qty={qty}, price={entry_price:.2f}",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "direction": sig.direction,
                    "qty": qty,
                    "entry_fees": entry_fees,
                },
            )

        # EXIT LOGIC: Close position if FLAT signal or opposite direction
        elif current_position is not None:
            should_exit = False
            exit_reason = ""

            if sig.direction == "FLAT":
                should_exit = True
                exit_reason = sig.meta.get("reason", "flat_signal") if sig.meta else "flat_signal"
            elif sig.direction != current_position.direction:
                should_exit = True
                exit_reason = "opposite_signal"

            if should_exit:
                exit_price = current_price
                self._close_intraday_position(symbol, exit_price, reason=exit_reason)

                # Check circuit breaker after trade
                if self._risk_manager.is_circuit_breaker_active():
                    logger.error(
                        "CIRCUIT BREAKER ACTIVATED from intraday trade! Squaring off all positions",
                        extra={"component": "engine"},
                    )
                    self._square_off_all_positions()

        # Log signal evaluation even if no action taken
        else:
            logger.debug(
                f"Intraday signal for {symbol}: {sig.direction} (no action)",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "direction": sig.direction,
                    "strength": sig.strength,
                    "has_position": current_position is not None,
                },
            )

        # Record tick latency performance metric
        if self._monitoring and self._monitoring.settings.monitoring_enable_performance_tracking:
            tick_latency_ms = (time_module.time() - tick_start_time) * 1000
            self._monitoring.record_performance_metric(
                "intraday_tick_latency",
                tick_latency_ms,
                {"symbol": symbol},
            )

        # Record monitoring metrics
        self._record_monitoring_metrics()

    def run_swing_daily(self, symbol: str) -> None:
        """
        Process daily swing evaluation for symbol.

        Runs post-market close (typically 16:00 IST). Fetches last N daily bars,
        computes features, generates entry/exit signals, and journals decisions.

        Holiday-safe: skips if no new bar available today.

        Args:
            symbol: Stock symbol to process
        """
        # Start performance tracking
        swing_start_time = time_module.time()

        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)

        # Fetch historical bars for feature computation
        lookback_days = settings.swing_feature_lookback_days
        start_ts = pd.Timestamp(now_ist - timedelta(days=lookback_days))
        end_ts = pd.Timestamp(now_ist)

        # Use DataFeed if provided (for backtest/dryrun with CSV)
        if self.data_feed is not None and settings.mode in ("backtest", "dryrun"):
            try:
                df_bars = self.data_feed.get_historical_bars(
                    symbol=symbol,
                    from_date=start_ts.to_pydatetime(),
                    to_date=end_ts.to_pydatetime(),
                    interval=settings.swing_bar_interval,
                )

                if df_bars.empty:
                    logger.warning(
                        f"No daily bars returned from DataFeed for {symbol}",
                        extra={"component": "engine", "symbol": symbol},
                    )
                    return

                # Convert DataFrame to Bar-like dict format
                bars = [
                    type(
                        "Bar",
                        (),
                        {
                            "ts": row["timestamp"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"],
                        },
                    )()
                    for _, row in df_bars.iterrows()
                ]

            except Exception as e:
                logger.error(
                    f"Failed to fetch daily bars from DataFeed for {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )
                return
        else:
            # Use live BreezeClient (default for live mode or when data_feed not provided)
            try:
                bars = self.client.historical_bars(
                    symbol=symbol,
                    interval=settings.swing_bar_interval,
                    start=start_ts,
                    end=end_ts,
                )
            except Exception as e:
                logger.error(
                    f"Failed to fetch daily bars from BreezeClient for {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )
                return

            if not bars:
                logger.warning(
                    f"No daily bars returned from BreezeClient for {symbol}",
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

        # Holiday check: ensure we have today's bar
        today_date = now_ist.date()
        last_bar_date = pd.Timestamp(df.iloc[-1]["ts"]).date()
        if last_bar_date < today_date:
            logger.info(
                f"No new bar for {symbol} today (holiday or weekend)",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "last_bar": str(last_bar_date),
                },
            )
            return

        # Compute features
        try:
            from src.domain.strategies.swing import (
                SwingPosition,
                compute_features,
            )
            from src.domain.strategies.swing import (
                signal as swing_signal,
            )

            df_features = compute_features(df, settings)
        except Exception as e:
            logger.error(
                f"Feature computation failed for {symbol}: {e}",
                extra={"component": "engine", "symbol": symbol, "error": str(e)},
            )
            return

        # Get current position (if any)
        current_position = self._swing_positions.get(symbol)

        # Fetch sentiment with caching and track API latency
        sentiment_start_time = time_module.time()
        sentiment_meta: dict[str, Any] = {}

        # Use new registry-based approach if available, otherwise fall back to legacy
        if self._sentiment_registry is not None:
            try:
                sentiment_result = self._sentiment_registry.get_sentiment(symbol)
                if sentiment_result:
                    sentiment_score = sentiment_result.value
                    sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000
                    sentiment_meta = sentiment_result.metadata
                    logger.info(
                        f"Sentiment for {symbol}: {sentiment_result.value:.3f} "
                        f"(source={sentiment_result.source}, confidence={sentiment_result.confidence:.2f})",
                        extra={
                            "component": "engine",
                            "symbol": symbol,
                            "sentiment": sentiment_result.value,
                            "source": sentiment_result.source,
                            "confidence": sentiment_result.confidence,
                        },
                    )
                else:
                    sentiment_score = 0.0
                    sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000
                    logger.debug(
                        f"No sentiment data available for {symbol}, using neutral (0.0)",
                        extra={"component": "engine", "symbol": symbol},
                    )
            except Exception as e:
                sentiment_score = 0.0
                sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000
                logger.warning(
                    f"Sentiment fetch failed for {symbol}: {e}, using neutral (0.0)",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )
        else:
            # Legacy path: use stub provider with cache
            sentiment_score, sentiment_meta = self._sentiment_cache.get(
                symbol, self._sentiment_provider, fallback=0.0
            )
            sentiment_latency_ms = (time_module.time() - sentiment_start_time) * 1000

        # Record sentiment API latency
        if self._monitoring and self._monitoring.settings.monitoring_enable_performance_tracking:
            self._monitoring.record_performance_metric(
                "sentiment_api_latency",
                sentiment_latency_ms,
                {"symbol": symbol, "strategy": "swing"},
            )

        # Generate signal with sentiment (before student prediction)
        try:
            sig = swing_signal(
                df_features,
                settings,
                position=current_position,
                sentiment_score=sentiment_score,
                sentiment_meta=sentiment_meta,
            )
        except Exception as e:
            logger.error(
                f"Swing signal generation failed for {symbol}: {e}",
                extra={"component": "engine", "symbol": symbol, "error": str(e)},
            )
            return

        # US-021 Phase 2: Generate Student prediction if enabled (swing)
        student_meta: dict[str, Any] = {}
        student_adjusted_strength = getattr(sig, "strength", 1.0)  # Default to original strength

        if self._student_model is not None and sig.direction in ("LONG", "SHORT"):
            try:
                # Extract features for Student prediction
                latest_row = df_features.iloc[-1]
                feature_dict = {
                    col: float(latest_row[col])
                    for col in self._student_model.feature_cols
                    if col in df_features.columns
                }

                # Make prediction
                prediction = self._student_model.predict_single(feature_dict, symbol=symbol)

                student_meta = {
                    "probability": prediction.probability,
                    "decision": prediction.decision,
                    "confidence": prediction.confidence,
                    "model_version": prediction.model_version,
                }

                # US-021 Phase 3: Record prediction for monitoring (outcome unknown at entry)
                if self._monitoring and self._monitoring.settings.student_monitoring_enabled:
                    # Convert decision (0/1) to direction string for monitoring
                    decision_str = "BUY" if prediction.decision == 1 else "SELL"
                    self._monitoring.record_student_prediction(
                        symbol=symbol,
                        prediction=decision_str,
                        probability=prediction.probability,
                        confidence=prediction.confidence,
                        actual_outcome=None,  # Will be updated on exit
                        model_version=prediction.model_version,
                    )

                # Adjust signal strength based on student confidence (if confidence threshold met)
                if prediction.confidence >= settings.student_model_confidence_threshold:
                    # Student agrees with signal: boost confidence (decision: 1 = BUY, 0 = SELL)
                    if (sig.direction == "LONG" and prediction.decision == 1) or (
                        sig.direction == "SHORT" and prediction.decision == 0
                    ):
                        original_strength = student_adjusted_strength
                        student_adjusted_strength = min(1.0, student_adjusted_strength * 1.2)
                        logger.info(
                            f"Student prediction AGREES with {sig.direction} signal for {symbol}, "
                            f"boosting strength {original_strength:.2f} -> {student_adjusted_strength:.2f}",
                            extra={"component": "student", "symbol": symbol, **student_meta},
                        )
                    # Student disagrees with signal: reduce confidence or suppress
                    else:
                        original_strength = student_adjusted_strength
                        student_adjusted_strength = student_adjusted_strength * 0.5
                        logger.warning(
                            f"Student prediction DISAGREES with {sig.direction} signal for {symbol}, "
                            f"reducing strength {original_strength:.2f} -> {student_adjusted_strength:.2f}",
                            extra={"component": "student", "symbol": symbol, **student_meta},
                        )
                else:
                    logger.info(
                        f"Student confidence {prediction.confidence:.2f} below threshold "
                        f"{settings.student_model_confidence_threshold:.2f}, not adjusting signal",
                        extra={"component": "student", "symbol": symbol},
                    )

            except Exception as e:
                logger.warning(
                    f"Student prediction failed for {symbol}: {e}",
                    extra={
                        "component": "student",
                        "symbol": symbol,
                        "error": str(e),
                    },
                )

        # Process signal
        if sig.direction == "FLAT" and sig.meta and sig.meta.get("reason") == "hold":
            # Continue holding, no action needed
            logger.debug(
                f"Holding swing position for {symbol}",
                extra={"component": "engine", "symbol": symbol},
            )
            return

        # Exit signal
        if sig.direction == "FLAT" and current_position is not None:
            exit_price = (
                sig.meta.get("exit_price", df_features.iloc[-1]["close"])
                if sig.meta
                else df_features.iloc[-1]["close"]
            )
            reason = sig.meta.get("reason", "exit") if sig.meta else "exit"

            # Calculate exit fees
            exit_fees = self._risk_manager.calculate_fees(current_position.qty, exit_price)

            # Calculate gross PnL
            direction_multiplier = 1.0 if current_position.direction == "LONG" else -1.0
            gross_pnl = (
                (exit_price - current_position.entry_price)
                * current_position.qty
                * direction_multiplier
            )

            # Calculate realized PnL (net of fees)
            realized_pnl = gross_pnl - current_position.entry_fees - exit_fees

            # Update position with exit fees and realized PnL
            current_position.exit_fees = exit_fees
            current_position.realized_pnl = realized_pnl

            logger.info(
                f"Exit calculation for {symbol}: "
                f"gross_pnl={gross_pnl:.2f}, "
                f"entry_fees={current_position.entry_fees:.2f}, "
                f"exit_fees={exit_fees:.2f}, "
                f"realized_pnl={realized_pnl:.2f}",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "gross_pnl": gross_pnl,
                    "realized_pnl": realized_pnl,
                },
            )

            # Place exit order (or journal in dryrun)
            side = "SELL" if current_position.direction == "LONG" else "BUY"

            if settings.mode == "dryrun":
                self.journal.log(
                    symbol=symbol,
                    action=side,
                    qty=current_position.qty,
                    price=exit_price,
                    pnl=realized_pnl,
                    reason=reason,
                    mode=settings.mode,
                    order_id="DRYRUN",
                    status="EXIT",
                    strategy="swing",
                    meta_json=str(
                        {
                            **(sig.meta if sig.meta else {}),
                            "entry_fees": current_position.entry_fees,
                            "exit_fees": exit_fees,
                            "gross_pnl": gross_pnl,
                            "realized_pnl": realized_pnl,
                        }
                    ),
                )
            else:
                # Live mode: place real exit order
                try:
                    response = self.client.place_order(
                        symbol=symbol,
                        side=side,  # type: ignore[arg-type]
                        qty=current_position.qty,
                        order_type="MARKET",
                    )
                    self.journal.log(
                        symbol=symbol,
                        action=side,
                        qty=current_position.qty,
                        price=exit_price,
                        pnl=realized_pnl,
                        reason=reason,
                        mode=settings.mode,
                        order_id=response.order_id,
                        status=response.status,
                        strategy="swing",
                        meta_json=str(
                            {
                                **(sig.meta if sig.meta else {}),
                                "entry_fees": current_position.entry_fees,
                                "exit_fees": exit_fees,
                                "gross_pnl": gross_pnl,
                                "realized_pnl": realized_pnl,
                            }
                        ),
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to place exit order for {symbol}: {e}",
                        extra={
                            "component": "engine",
                            "symbol": symbol,
                            "error": str(e),
                        },
                    )

            # Update risk manager with trade result
            self._risk_manager.update_position(
                symbol=symbol,
                qty=current_position.qty,
                price=exit_price,
                is_opening=False,
            )

            # Record trade for circuit breaker check
            total_fees = current_position.entry_fees + exit_fees
            self._risk_manager.record_trade(
                symbol=symbol,
                realized_pnl=realized_pnl,
                fees=total_fees,
            )

            # Check circuit breaker
            if self._risk_manager.is_circuit_breaker_active():
                logger.error(
                    "CIRCUIT BREAKER ACTIVATED! Squaring off all open positions",
                    extra={"component": "engine"},
                )
                # Square off all open swing positions
                self._square_off_all_swing_positions()

            # Clear position
            del self._swing_positions[symbol]
            logger.info(
                f"Swing exit for {symbol}: {reason}",
                extra={"component": "engine", "symbol": symbol, "realized_pnl": realized_pnl},
            )
            return

        # Entry signal
        if sig.direction in ("LONG", "SHORT") and current_position is None:
            entry_price = df_features.iloc[-1]["close"]

            # Calculate position size with risk management (US-021 Phase 2: use student-adjusted strength)
            atr = df_features.iloc[-1].get("atr", 0.0) if "atr" in df_features.columns else 0.0
            signal_strength = student_adjusted_strength  # Use student-adjusted strength

            pos_size = self._risk_manager.calculate_position_size(
                symbol=symbol,
                price=entry_price,
                atr=atr,
                signal_strength=signal_strength,
            )

            # Log position sizing
            logger.info(
                f"Position sizing for {symbol}: qty={pos_size.qty}, "
                f"rationale={pos_size.rationale}, warnings={pos_size.warnings}",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "qty": pos_size.qty,
                    "rationale": pos_size.rationale,
                },
            )

            # Check risk limits
            risk_check = self._risk_manager.can_open_position(
                symbol=symbol,
                qty=pos_size.qty,
                price=entry_price,
            )

            if not risk_check.allowed:
                logger.warning(
                    f"Position blocked for {symbol}: {risk_check.reason}",
                    extra={
                        "component": "engine",
                        "symbol": symbol,
                        "reason": risk_check.reason,
                        "breaker_active": risk_check.breaker_active,
                    },
                )
                return

            # Use allowed qty (may be reduced due to caps)
            qty = risk_check.allowed_qty

            # Calculate entry fees
            entry_fees = self._risk_manager.calculate_fees(qty, entry_price)

            logger.info(
                f"Entry fees for {symbol}: {entry_fees:.2f} INR",
                extra={"component": "engine", "symbol": symbol, "entry_fees": entry_fees},
            )

            # US-021 Phase 2: Include student metadata in journal
            meta_with_student = {
                **(sig.meta if sig.meta else {}),
                "signal_strength": signal_strength,
                "signal_strength_original": getattr(sig, "strength", 1.0),
                "student_prediction": student_meta,
                "position_size_rationale": pos_size.rationale,
                "entry_fees": entry_fees,
                "atr": atr,
            }

            if settings.mode == "dryrun":
                self.journal.log(
                    symbol=symbol,
                    action=sig.direction,
                    qty=qty,
                    price=entry_price,
                    pnl=0.0,
                    reason=(sig.meta.get("reason", "entry") if sig.meta else "entry"),
                    mode=settings.mode,
                    order_id="DRYRUN",
                    status="ENTRY",
                    strategy="swing",
                    meta_json=str(meta_with_student),
                )
            else:
                # Live mode: place real entry order
                side = "BUY" if sig.direction == "LONG" else "SELL"
                try:
                    response = self.client.place_order(
                        symbol=symbol,
                        side=side,  # type: ignore[arg-type]
                        qty=qty,
                        order_type="MARKET",
                    )
                    self.journal.log(
                        symbol=symbol,
                        action=sig.direction,
                        qty=qty,
                        price=entry_price,
                        pnl=0.0,
                        reason=(sig.meta.get("reason", "entry") if sig.meta else "entry"),
                        mode=settings.mode,
                        order_id=response.order_id,
                        status=response.status,
                        strategy="swing",
                        meta_json=str(meta_with_student),
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to place entry order for {symbol}: {e}",
                        extra={
                            "component": "engine",
                            "symbol": symbol,
                            "error": str(e),
                        },
                    )
                    return

            # Create position tracking with fees
            self._swing_positions[symbol] = SwingPosition(
                symbol=symbol,
                direction=sig.direction,  # type: ignore[arg-type]
                entry_price=entry_price,
                entry_date=pd.Timestamp(df_features.iloc[-1]["ts"]),
                qty=qty,
                entry_fees=entry_fees,
            )

            # Update risk manager position tracking
            self._risk_manager.update_position(
                symbol=symbol,
                qty=qty,
                price=entry_price,
                is_opening=True,
            )

            logger.info(
                f"Swing entry for {symbol}: {sig.direction}",
                extra={
                    "component": "engine",
                    "symbol": symbol,
                    "price": entry_price,
                },
            )

        # Record swing daily latency performance metric
        if self._monitoring and self._monitoring.settings.monitoring_enable_performance_tracking:
            swing_latency_ms = (time_module.time() - swing_start_time) * 1000
            self._monitoring.record_performance_metric(
                "swing_daily_latency",
                swing_latency_ms,
                {"symbol": symbol},
            )

        # Record monitoring metrics
        self._record_monitoring_metrics()

    def _square_off_all_positions(self) -> None:
        """
        Square off all open positions (intraday + swing) on circuit breaker.

        Called when circuit breaker is triggered to forcefully close all positions
        and halt trading.
        """
        logger.error(
            "Circuit breaker activated! Forcing square-off of all positions",
            extra={
                "component": "engine",
                "intraday_positions": len(self._intraday_positions),
                "swing_positions": len(self._swing_positions),
            },
        )

        # Square off all intraday positions
        intraday_symbols = list(self._intraday_positions.keys())
        for symbol in intraday_symbols:
            try:
                position = self._intraday_positions[symbol]
                exit_price = position.entry_price  # Simplified for circuit breaker
                exit_fees = self._risk_manager.calculate_fees(position.qty, exit_price)

                direction_multiplier = 1.0 if position.direction == "LONG" else -1.0
                gross_pnl = (
                    (exit_price - position.entry_price) * position.qty * direction_multiplier
                )
                realized_pnl = gross_pnl - position.entry_fees - exit_fees

                side = "SELL" if position.direction == "LONG" else "BUY"
                self.journal.log(
                    symbol=symbol,
                    action=side,
                    qty=position.qty,
                    price=exit_price,
                    pnl=realized_pnl,
                    reason="circuit_breaker_forced_exit",
                    mode=settings.mode,
                    order_id="CIRCUIT_BREAKER",
                    status="FORCED_EXIT",
                    strategy="intraday",
                    meta_json=str(
                        {
                            "entry_fees": position.entry_fees,
                            "exit_fees": exit_fees,
                            "realized_pnl": realized_pnl,
                            "circuit_breaker": True,
                        }
                    ),
                )

                self._risk_manager.update_position(
                    symbol=symbol,
                    qty=position.qty,
                    price=exit_price,
                    is_opening=False,
                )

                del self._intraday_positions[symbol]

                logger.info(
                    f"Forced exit intraday position for {symbol} due to circuit breaker",
                    extra={"component": "engine", "symbol": symbol, "strategy": "intraday"},
                )
            except Exception as e:
                logger.error(
                    f"Failed to force close intraday position {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )

        # Square off all swing positions
        self._square_off_all_swing_positions()

    def _square_off_all_swing_positions(self) -> None:
        """Square off all open swing positions (circuit breaker)."""
        symbols_to_close = list(self._swing_positions.keys())
        logger.warning(
            f"Force closing {len(symbols_to_close)} swing positions due to circuit breaker",
            extra={"component": "engine", "symbols": symbols_to_close},
        )

        for symbol in symbols_to_close:
            try:
                position = self._swing_positions[symbol]
                # For circuit breaker, we force close at current price (use last known)
                # In a real system, we'd fetch current price or use market order
                exit_price = position.entry_price  # Simplified for v1

                # Calculate exit fees
                exit_fees = self._risk_manager.calculate_fees(position.qty, exit_price)

                # Calculate realized PnL
                direction_multiplier = 1.0 if position.direction == "LONG" else -1.0
                gross_pnl = (
                    (exit_price - position.entry_price) * position.qty * direction_multiplier
                )
                realized_pnl = gross_pnl - position.entry_fees - exit_fees

                # Journal the forced exit
                side = "SELL" if position.direction == "LONG" else "BUY"
                self.journal.log(
                    symbol=symbol,
                    action=side,
                    qty=position.qty,
                    price=exit_price,
                    pnl=realized_pnl,
                    reason="circuit_breaker_forced_exit",
                    mode=settings.mode,
                    order_id="CIRCUIT_BREAKER",
                    status="FORCED_EXIT",
                    strategy="swing",
                    meta_json=str(
                        {
                            "entry_fees": position.entry_fees,
                            "exit_fees": exit_fees,
                            "realized_pnl": realized_pnl,
                            "circuit_breaker": True,
                        }
                    ),
                )

                # Update risk manager
                self._risk_manager.update_position(
                    symbol=symbol,
                    qty=position.qty,
                    price=exit_price,
                    is_opening=False,
                )

                # Remove position
                del self._swing_positions[symbol]

                logger.info(
                    f"Forced exit for {symbol} due to circuit breaker",
                    extra={"component": "engine", "symbol": symbol, "realized_pnl": realized_pnl},
                )
            except Exception as e:
                logger.error(
                    f"Failed to force close {symbol}: {e}",
                    extra={"component": "engine", "symbol": symbol, "error": str(e)},
                )

    def _record_monitoring_metrics(self) -> None:
        """Collect and record metrics for monitoring."""
        if not self._monitoring:
            return

        try:
            # Collect current state
            metrics = {
                "heartbeat": {
                    "last_tick": datetime.now().isoformat(),
                },
                "positions": {
                    "count": len(self._swing_positions) + len(self._intraday_positions),
                    "swing_count": len(self._swing_positions),
                    "intraday_count": len(self._intraday_positions),
                    "symbols": list(
                        set(
                            list(self._swing_positions.keys())
                            + list(self._intraday_positions.keys())
                        )
                    ),
                },
                "pnl": {
                    "daily": self._risk_manager.get_daily_stats().get("pnl", 0.0),
                    "daily_loss_pct": self._risk_manager.get_daily_stats().get(
                        "daily_loss_pct", 0.0
                    ),
                },
                "risk": {
                    "circuit_breaker_active": self._risk_manager.is_circuit_breaker_active(),
                    "max_position_value": settings.max_position_value_per_symbol,
                },
                "connectivity": {
                    "breeze_authenticated": True,  # If we got here, we're authenticated
                    "last_api_call": datetime.now().isoformat(),
                },
            }

            # Record to monitoring service
            self._monitoring.record_tick(metrics)

        except Exception as e:
            # Don't let monitoring failures crash the engine
            logger.error(
                f"Failed to record monitoring metrics: {e}",
                extra={"component": "engine", "error": str(e)},
            )

    def get_sentiment_health(self) -> dict[str, Any] | None:
        """Get health status of sentiment providers.

        Returns:
            Dictionary mapping provider names to health metrics if registry is available,
            None if using legacy stub provider.
        """
        if self._sentiment_registry is not None:
            return self._sentiment_registry.get_provider_health()
        return None

    def stop(self) -> None:
        """Stop the engine and clean up resources."""
        logger.info("Stopping engine...")
        self.journal.close()
        self._running = False
