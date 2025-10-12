"""Backtesting engine for strategy validation on historical data."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.adapters.breeze_client import BreezeClient
from src.app.config import Settings
from src.domain.strategies import swing as swing_strategy
from src.domain.strategies.intraday import IntradayPosition
from src.domain.strategies.swing import SwingPosition
from src.domain.types import BacktestConfig, BacktestResult, Bar
from src.services.accuracy_analyzer import PredictionTrace, TelemetryWriter
from src.services.data_feed import DataFeed, IntervalType
from src.services.risk_manager import RiskManager


class Backtester:
    """Backtest engine for validating trading strategies on historical data.

    Simulates trading with identical logic to live engine, tracking equity,
    positions, and trades. Calculates comprehensive performance metrics.
    """

    def __init__(
        self,
        config: BacktestConfig,
        client: BreezeClient | None = None,
        settings: Settings | None = None,
        data_feed: DataFeed | None = None,
        enable_telemetry: bool = False,
        telemetry_dir: Path | None = None,
    ) -> None:
        """Initialize backtester with configuration.

        Args:
            config: Backtest configuration
            client: BreezeClient for data fetching (deprecated, use data_feed instead)
            settings: Application settings (uses global settings if None)
            data_feed: DataFeed for historical data (preferred over client)
            enable_telemetry: Enable prediction telemetry capture
            telemetry_dir: Directory for telemetry files (uses settings default if None)
        """
        from src.app.config import settings as global_settings

        self.config = config
        self.client = client
        self.data_feed = data_feed
        self.settings = settings or global_settings

        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # Initialize tracking
        self.equity: float = config.initial_capital
        self.equity_curve: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.positions: dict[str, IntradayPosition | SwingPosition] = {}

        # Initialize risk manager
        self._risk_manager = RiskManager(
            starting_capital=config.initial_capital,
            mode=self.settings.position_sizing_mode,
            risk_per_trade_pct=self.settings.risk_per_trade_pct,
            atr_multiplier=self.settings.atr_multiplier,
            max_position_value_per_symbol=self.settings.max_position_value_per_symbol,
            max_daily_loss_pct=self.settings.max_daily_loss_pct,
            trading_fee_bps=self.settings.trading_fee_bps,
            slippage_bps=self.settings.slippage_bps,
        )

        # Initialize telemetry writer if enabled
        self.enable_telemetry = enable_telemetry
        self.telemetry_writer: TelemetryWriter | None = None

        if self.enable_telemetry:
            # Use provided directory or fall back to settings
            output_dir = telemetry_dir or Path(self.settings.telemetry_storage_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.telemetry_writer = TelemetryWriter(
                output_dir=output_dir,
                format="csv",
                compression=self.settings.telemetry_compression,
                buffer_size=self.settings.telemetry_buffer_size,
                max_file_size_mb=self.settings.telemetry_max_file_size_mb,
            )

            logger.info(
                "Telemetry capture enabled",
                extra={
                    "component": "backtest",
                    "output_dir": str(output_dir),
                    "sample_rate": self.settings.telemetry_sample_rate,
                },
            )
        else:
            logger.info(
                "Telemetry capture disabled",
                extra={"component": "backtest"},
            )

        logger.info(
            "Initialized Backtester",
            extra={
                "component": "backtest",
                "symbols": config.symbols,
                "date_range": [config.start_date, config.end_date],
                "strategy": config.strategy,
                "initial_capital": config.initial_capital,
            },
        )

    def run(self) -> BacktestResult:
        """Execute complete backtest and return results.

        Returns:
            BacktestResult with metrics, equity curve, trades, and artifact paths
        """
        logger.info(
            "Starting backtest",
            extra={
                "component": "backtest",
                "symbols": self.config.symbols,
                "strategy": self.config.strategy,
            },
        )

        # Run backtest for each symbol
        for symbol in self.config.symbols:
            logger.info(
                f"Backtesting {symbol}",
                extra={"component": "backtest", "symbol": symbol},
            )

            # Load historical data
            bars = self._load_data(symbol)

            if not bars:
                logger.warning(
                    f"No data for {symbol}, skipping",
                    extra={"component": "backtest", "symbol": symbol},
                )
                continue

            # Run strategy-specific simulation
            if self.config.strategy in ("intraday", "both"):
                self._simulate_intraday(symbol, bars)

            if self.config.strategy in ("swing", "both"):
                self._simulate_swing(symbol, bars)

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Create DataFrames
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        # Save artifacts
        summary_path, equity_path, trades_path = self._save_artifacts(metrics, equity_df, trades_df)

        # Flush and close telemetry writer if enabled
        if self.enable_telemetry and self.telemetry_writer is not None:
            try:
                self.telemetry_writer.flush()
                self.telemetry_writer.close()
                logger.info(
                    "Telemetry flushed and closed",
                    extra={"component": "backtest"},
                )
            except Exception as e:
                logger.error(
                    f"Failed to close telemetry writer: {e}",
                    extra={"component": "backtest", "error": str(e)},
                )

        logger.info(
            "Backtest complete",
            extra={
                "component": "backtest",
                "total_trades": len(self.trades),
                "final_equity": self.equity,
                "total_return_pct": metrics.get("total_return_pct", 0.0),
            },
        )

        # US-019 Phase 2: Compute accuracy metrics from telemetry if available
        accuracy_metrics = None
        telemetry_dir_path = None

        if self.settings.telemetry_storage_path:
            telemetry_dir_path = Path(self.settings.telemetry_storage_path)

            # Check if telemetry was captured
            if telemetry_dir_path.exists():
                try:
                    from src.services.accuracy_analyzer import AccuracyAnalyzer

                    analyzer = AccuracyAnalyzer()
                    traces = analyzer.load_traces(telemetry_dir_path)

                    if traces:
                        accuracy_metrics = analyzer.compute_metrics(traces)
                        logger.info(
                            f"Computed accuracy metrics: precision={accuracy_metrics.precision.get('LONG', 0.0):.2%}, "
                            f"hit_ratio={accuracy_metrics.hit_ratio:.2%}",
                            extra={"component": "backtest", "total_traces": len(traces)},
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to compute accuracy metrics: {e}",
                        extra={"component": "backtest", "error": str(e)},
                    )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades_df,
            metadata={
                "run_date": datetime.now().isoformat(),
                "random_seed": self.config.random_seed,
                "settings_snapshot": {
                    "position_sizing_mode": self.settings.position_sizing_mode,
                    "risk_per_trade_pct": self.settings.risk_per_trade_pct,
                    "trading_fee_bps": self.settings.trading_fee_bps,
                    "slippage_bps": self.settings.slippage_bps,
                },
            },
            summary_path=summary_path,
            equity_path=equity_path,
            trades_path=trades_path,
            accuracy_metrics=accuracy_metrics,  # US-019
            telemetry_dir=telemetry_dir_path,  # US-019
        )

    def _load_data(self, symbol: str) -> list[Any]:
        """Load historical bars for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of Bar objects
        """
        start_ts = pd.Timestamp(self.config.start_date)
        end_ts = pd.Timestamp(self.config.end_date)

        # Use DataFeed if provided (preferred method)
        if self.data_feed is not None:
            try:
                # Use resolution from config (US-018: supports minute bars)
                resolution: IntervalType = getattr(self.config, "resolution", "1day")
                df = self.data_feed.get_historical_bars(
                    symbol=symbol,
                    from_date=start_ts.to_pydatetime(),
                    to_date=end_ts.to_pydatetime(),
                    interval=resolution,
                )

                if df.empty:
                    logger.warning(
                        f"No data returned from DataFeed for {symbol}",
                        extra={"component": "backtest", "symbol": symbol},
                    )
                    return []

                # Convert DataFrame to list of Bar objects
                bars = [
                    Bar(
                        ts=row["timestamp"],
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                    for _, row in df.iterrows()
                ]

                logger.info(
                    f"Loaded {len(bars)} bars from DataFeed for {symbol}",
                    extra={
                        "component": "backtest",
                        "symbol": symbol,
                        "bars": len(bars),
                    },
                )
                return bars

            except Exception as e:
                logger.error(
                    f"Failed to load data from DataFeed for {symbol}: {e}",
                    extra={"component": "backtest", "symbol": symbol, "error": str(e)},
                )
                return []

        # Fallback to legacy BreezeClient method (backward compatibility)
        if self.config.data_source == "breeze":
            if self.client is None:
                raise ValueError(
                    "BreezeClient required for data_source='breeze' when data_feed not provided"
                )

            try:
                # Use resolution from config (US-018: supports minute bars)
                interval_resolution: IntervalType = getattr(self.config, "resolution", "1day")
                bars = self.client.historical_bars(
                    symbol=symbol,
                    interval=interval_resolution,
                    start=start_ts,
                    end=end_ts,
                )
                logger.info(
                    f"Loaded {len(bars)} bars from BreezeClient for {symbol}",
                    extra={
                        "component": "backtest",
                        "symbol": symbol,
                        "bars": len(bars),
                    },
                )
                return bars
            except Exception as e:
                logger.error(
                    f"Failed to load data from BreezeClient for {symbol}: {e}",
                    extra={"component": "backtest", "symbol": symbol, "error": str(e)},
                )
                return []

        elif self.config.data_source == "csv":
            raise ValueError(
                "CSV data source requires data_feed parameter. "
                "Use: Backtester(config, data_feed=CSVDataFeed(csv_directory))"
            )

        elif self.config.data_source == "teacher":
            raise NotImplementedError("Teacher data source not yet implemented")

        else:
            raise ValueError(f"Unknown data source: {self.config.data_source}")

    def _simulate_swing(self, symbol: str, bars: list[Any]) -> None:
        """Simulate swing strategy on historical bars.

        Args:
            symbol: Stock symbol
            bars: List of daily Bar objects
        """
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
            df_features = swing_strategy.compute_features(df, self.settings)
        except Exception as e:
            logger.error(
                f"Feature computation failed for {symbol}: {e}",
                extra={"component": "backtest", "symbol": symbol, "error": str(e)},
            )
            return

        # Filter to valid rows
        valid_df = df_features[df_features["valid"]]

        if valid_df.empty:
            logger.warning(
                f"No valid features for {symbol}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        # Simulate each day
        for idx in range(len(valid_df)):
            current_df = valid_df.iloc[: idx + 1]

            if len(current_df) < 2:
                continue  # Need at least 2 rows for crossover

            current_position = self.positions.get(symbol)

            # Generate signal
            try:
                # Type-check: ensure we're passing SwingPosition or None
                swing_position = (
                    current_position
                    if isinstance(current_position, SwingPosition) or current_position is None
                    else None
                )
                sig = swing_strategy.signal(
                    current_df,
                    self.settings,
                    position=swing_position,
                    sentiment_score=0.0,  # No sentiment in backtest
                )
            except Exception as e:
                logger.warning(
                    f"Signal generation failed: {e}",
                    extra={"component": "backtest", "error": str(e)},
                )
                continue

            current_row = current_df.iloc[-1]
            current_date = pd.Timestamp(current_row["ts"])
            current_price = float(current_row["close"])

            # Update equity curve
            self._update_equity_curve(current_date, symbol)

            # Process signal
            if sig.direction == "FLAT" and current_position is not None:
                # Exit signal
                if sig.meta and sig.meta.get("reason") == "hold":
                    continue  # Continue holding

                exit_price = (
                    sig.meta.get("exit_price", current_price) if sig.meta else current_price
                )
                self._close_swing_position(symbol, exit_price, current_date, sig.meta or {})

            elif sig.direction in ("LONG", "SHORT") and current_position is None:
                # Entry signal
                self._open_swing_position(
                    symbol, sig.direction, current_price, current_date, current_row
                )

    def _simulate_intraday(self, symbol: str, bars: list[Any]) -> None:
        """Simulate intraday strategy on historical bars.

        US-018 Phase 3: Now uses minute-level data when available (resolution='1minute').
        Falls back to daily bar proxy for backward compatibility.

        Args:
            symbol: Stock symbol
            bars: List of Bar objects (minute or daily resolution)
        """
        from src.domain.strategies.intraday import compute_features
        from src.domain.strategies.intraday import signal as intraday_signal

        if not bars:
            return

        # Check resolution - if daily bars, log warning and skip
        resolution = getattr(self.config, "resolution", "1day")
        if resolution == "1day":
            logger.warning(
                "Intraday backtest using daily bars (simplified) - skipping actual trades",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        # Convert bars to DataFrame for feature computation
        bar_data = []
        for bar in bars:
            bar_data.append(
                {
                    "timestamp": bar.ts,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

        df = pd.DataFrame(bar_data)

        # Compute features (requires sufficient data)
        try:
            df = compute_features(df, self.settings)
        except Exception as e:
            logger.error(
                f"Failed to compute features for {symbol}: {e}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        # Skip if insufficient valid data
        if not df["valid"].any():
            logger.warning(
                f"No valid feature rows for {symbol}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        # Simulate minute-by-minute
        logger.info(
            f"Starting intraday simulation for {symbol} with {len(df)} minute bars",
            extra={"component": "backtest", "symbol": symbol, "bars": len(df)},
        )

        # Track position state
        current_position = None
        entry_signal_meta = None

        # Iterate through each minute bar
        for idx in range(len(df)):
            row = df.iloc[idx]

            # Skip invalid rows
            if not row["valid"]:
                continue

            # Get current bar data
            current_time = row["timestamp"]
            current_close = row["close"]

            # Check if position exists
            if symbol in self.positions:
                current_position = self.positions[symbol]

            # Generate signal on current data up to this point
            history_df = df.iloc[: idx + 1]
            sig = intraday_signal(history_df, self.settings, sentiment=None)

            # Position management logic
            if current_position is None:
                # No position - check for entry signal
                if sig.direction in ["LONG", "SHORT"]:
                    self._open_intraday_position(
                        symbol=symbol,
                        direction=sig.direction,
                        entry_price=current_close,
                        entry_time=current_time,
                        row=row,
                    )
                    # Store signal metadata for telemetry
                    entry_signal_meta = sig.meta
                    if symbol in self.positions:
                        current_position = self.positions[symbol]

            else:
                # Have position - check for exit conditions
                should_exit = False
                exit_reason = ""

                # Exit if signal direction changes
                if sig.direction != current_position.direction and sig.direction != "FLAT":
                    should_exit = True
                    exit_reason = "signal_reversal"

                # Exit if signal goes flat
                elif sig.direction == "FLAT":
                    should_exit = True
                    exit_reason = "signal_flat"

                # Exit on last bar of the day (15:30 close)
                time_of_day = current_time.time()
                if time_of_day.hour == 15 and time_of_day.minute >= 29:
                    should_exit = True
                    exit_reason = "eod_close"

                if should_exit:
                    self._close_intraday_position(
                        symbol=symbol,
                        exit_price=current_close,
                        exit_time=current_time,
                        reason=exit_reason,
                        signal_meta=entry_signal_meta or {},
                    )
                    current_position = None
                    entry_signal_meta = None

        # Close any remaining positions at end
        if symbol in self.positions:
            last_row = df.iloc[-1]
            self._close_intraday_position(
                symbol=symbol,
                exit_price=last_row["close"],
                exit_time=last_row["timestamp"],
                reason="backtest_end",
                signal_meta=entry_signal_meta or {},
            )

        logger.info(
            f"Completed intraday simulation for {symbol}",
            extra={"component": "backtest", "symbol": symbol},
        )

    def _open_intraday_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_time: pd.Timestamp,
        row: pd.Series,
    ) -> None:
        """Open new intraday position.

        Args:
            symbol: Stock symbol
            direction: LONG or SHORT
            entry_price: Entry price
            entry_time: Entry timestamp
            row: Current bar data
        """
        # Check risk limits
        allowed, reason = self._risk_manager.can_open_position(symbol, entry_price)
        if not allowed:
            logger.debug(
                f"Risk check failed for {symbol}: {reason}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        # Calculate position size
        qty = self._risk_manager.calculate_position_size(symbol, entry_price)
        if qty <= 0:
            logger.debug(
                f"Position size 0 for {symbol}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        # Calculate entry fees
        entry_fees = self._risk_manager.calculate_fees(qty, entry_price)
        position_value = qty * entry_price

        # Check if we have enough equity
        required_equity = position_value + entry_fees
        if required_equity > self.equity:
            logger.debug(
                f"Insufficient equity for {symbol}: required={required_equity:.2f}, available={self.equity:.2f}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        # Deduct fees from equity
        self.equity -= entry_fees

        # Create intraday position
        position = IntradayPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            qty=qty,
            entry_fees=entry_fees,
        )

        self.positions[symbol] = position

        # Update risk manager
        self._risk_manager.update_position(
            symbol=symbol,
            qty=qty,
            price=entry_price,
            is_opening=True,
        )

        # Record trade
        self.trades.append(
            {
                "timestamp": entry_time,
                "symbol": symbol,
                "action": "ENTRY",
                "direction": direction,
                "qty": qty,
                "price": entry_price,
                "fees": entry_fees,
                "pnl": 0.0,
                "equity": self.equity,
                "days_held": 0,
                "exit_reason": "",
            }
        )

        logger.debug(
            f"Opened {direction} intraday position: {symbol} @ {entry_price}",
            extra={"component": "backtest", "symbol": symbol, "qty": qty},
        )

    def _close_intraday_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str = "unknown",
        signal_meta: dict[str, Any] | None = None,
    ) -> None:
        """Close existing intraday position with telemetry capture.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Exit reason (signal_reversal, signal_flat, eod_close, etc.)
            signal_meta: Signal metadata from entry (RSI, SMA, sentiment, etc.)
        """
        # Combine into meta dict for backward compatibility
        meta = {"reason": reason}
        if signal_meta:
            meta.update(signal_meta)
        position = self.positions.get(symbol)
        if position is None:
            return

        # This function only handles intraday positions
        if not isinstance(position, IntradayPosition):
            logger.warning(
                f"Expected IntradayPosition but got {type(position).__name__}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        exit_fees = self._risk_manager.calculate_fees(position.qty, exit_price)

        # Calculate PnL
        direction_multiplier = 1.0 if position.direction == "LONG" else -1.0
        gross_pnl = (exit_price - position.entry_price) * position.qty * direction_multiplier
        realized_pnl = gross_pnl - position.entry_fees - exit_fees

        # Calculate realized return percentage
        position_value = position.entry_price * position.qty
        realized_return_pct = (realized_pnl / position_value) * 100 if position_value > 0 else 0.0

        # Update equity
        self.equity += realized_pnl

        # Update risk manager
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

        # Calculate holding period in minutes
        holding_period_minutes = int((exit_time - position.entry_time).total_seconds() / 60)

        # Record trade
        self.trades.append(
            {
                "timestamp": exit_time,
                "symbol": symbol,
                "action": "EXIT",
                "direction": position.direction,
                "qty": position.qty,
                "price": exit_price,
                "fees": exit_fees,
                "pnl": realized_pnl,
                "equity": self.equity,
                "days_held": (exit_time - position.entry_time).days,
                "exit_reason": meta.get("reason", "unknown"),
            }
        )

        # Capture telemetry trace if enabled
        if self.enable_telemetry and self.telemetry_writer is not None:
            try:
                # Respect sample rate
                should_capture = self.settings.telemetry_sample_rate >= 1.0 or (
                    self.settings.telemetry_sample_rate > 0.0
                    and random.random() < self.settings.telemetry_sample_rate
                )

                if should_capture:
                    # Determine actual direction based on realized return
                    # Use tighter threshold for intraday (0.3% vs 0.5% for swing)
                    if realized_return_pct > 0.3:
                        actual_direction = "LONG"
                    elif realized_return_pct < -0.3:
                        actual_direction = "SHORT"
                    else:
                        actual_direction = "NOOP"

                    # Prepare metadata
                    trace_metadata = {
                        "exit_reason": meta.get("reason", "unknown"),
                        "sl_hit": meta.get("sl_hit", False),
                        "tp_hit": meta.get("tp_hit", False),
                        "entry_fees": position.entry_fees,
                        "exit_fees": exit_fees,
                        "total_fees": total_fees,
                        "position_value": position_value,
                        "gross_pnl": gross_pnl,
                    }

                    # Extract features from signal metadata (US-018 Phase 3)
                    features = {}
                    if signal_meta:
                        # Include technical indicators from signal
                        for key in ["close", "sma20", "rsi14", "ema50", "vwap", "sentiment"]:
                            if key in signal_meta:
                                features[key] = signal_meta[key]

                    # Create prediction trace with explicit "intraday" strategy
                    trace = PredictionTrace(
                        timestamp=position.entry_time.to_pydatetime(),
                        symbol=symbol,
                        strategy="intraday",  # Explicit strategy name for telemetry
                        predicted_direction=position.direction,
                        actual_direction=actual_direction,
                        predicted_confidence=0.5,  # Default confidence for backtests
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        holding_period_minutes=holding_period_minutes,
                        realized_return_pct=realized_return_pct,
                        features=features,  # US-018 Phase 3: Include signal features
                        metadata=trace_metadata,
                    )

                    # Write trace
                    self.telemetry_writer.write_trace(trace)

                    logger.debug(
                        f"Captured intraday telemetry trace for {symbol}",
                        extra={
                            "component": "backtest",
                            "symbol": symbol,
                            "predicted": position.direction,
                            "actual": actual_direction,
                        },
                    )

            except Exception as e:
                logger.error(
                    f"Failed to capture intraday telemetry trace: {e}",
                    extra={
                        "component": "backtest",
                        "symbol": symbol,
                        "error": str(e),
                    },
                )

        logger.debug(
            f"Closed {position.direction} intraday position: {symbol} @ {exit_price}, PnL: {realized_pnl:.2f}",
            extra={"component": "backtest", "symbol": symbol, "pnl": realized_pnl},
        )

        # Remove position
        del self.positions[symbol]

    def _open_swing_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_date: pd.Timestamp,
        features_row: Any,
    ) -> None:
        """Open new swing position.

        Args:
            symbol: Stock symbol
            direction: "LONG" or "SHORT"
            entry_price: Entry price
            entry_date: Entry timestamp
            features_row: Feature row with ATR
        """
        # Calculate position size
        atr = features_row.get("atr14", 0.0) if "atr14" in features_row else 0.0

        pos_size = self._risk_manager.calculate_position_size(
            symbol=symbol,
            price=entry_price,
            atr=atr,
            signal_strength=0.8,  # Default strength
        )

        # Check risk limits
        risk_check = self._risk_manager.can_open_position(
            symbol=symbol,
            qty=pos_size.qty,
            price=entry_price,
        )

        if not risk_check.allowed:
            logger.debug(
                f"Position blocked: {risk_check.reason}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        qty = risk_check.allowed_qty
        entry_fees = self._risk_manager.calculate_fees(qty, entry_price)

        # Create position
        self.positions[symbol] = SwingPosition(
            symbol=symbol,
            direction=direction,  # type: ignore[arg-type]
            entry_price=entry_price,
            entry_date=entry_date,
            qty=qty,
            entry_fees=entry_fees,
        )

        # Update risk manager
        self._risk_manager.update_position(
            symbol=symbol,
            qty=qty,
            price=entry_price,
            is_opening=True,
        )

        # Record trade
        self.trades.append(
            {
                "timestamp": entry_date,
                "symbol": symbol,
                "action": "ENTRY",
                "direction": direction,
                "qty": qty,
                "price": entry_price,
                "fees": entry_fees,
                "pnl": 0.0,
                "equity": self.equity,
            }
        )

        logger.debug(
            f"Opened {direction} position: {symbol} @ {entry_price}",
            extra={"component": "backtest", "symbol": symbol, "qty": qty},
        )

    def _close_swing_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: pd.Timestamp,
        meta: dict[str, Any],
    ) -> None:
        """Close existing swing position.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_date: Exit timestamp
            meta: Signal metadata
        """
        position = self.positions.get(symbol)
        if position is None:
            return

        # This function only handles swing positions
        if not isinstance(position, SwingPosition):
            logger.warning(
                f"Expected SwingPosition but got {type(position).__name__}",
                extra={"component": "backtest", "symbol": symbol},
            )
            return

        exit_fees = self._risk_manager.calculate_fees(position.qty, exit_price)

        # Calculate PnL
        direction_multiplier = 1.0 if position.direction == "LONG" else -1.0
        gross_pnl = (exit_price - position.entry_price) * position.qty * direction_multiplier
        realized_pnl = gross_pnl - position.entry_fees - exit_fees

        # Calculate realized return percentage (per-share basis)
        position_value = position.entry_price * position.qty
        realized_return_pct = (realized_pnl / position_value) * 100 if position_value > 0 else 0.0

        # Update equity
        self.equity += realized_pnl

        # Update risk manager
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

        # Calculate holding period in minutes
        holding_period_minutes = int((exit_date - position.entry_date).total_seconds() / 60)

        # Record trade
        self.trades.append(
            {
                "timestamp": exit_date,
                "symbol": symbol,
                "action": "EXIT",
                "direction": position.direction,
                "qty": position.qty,
                "price": exit_price,
                "fees": exit_fees,
                "pnl": realized_pnl,
                "equity": self.equity,
                "days_held": (exit_date - position.entry_date).days,
                "exit_reason": meta.get("reason", "unknown"),
            }
        )

        # Capture telemetry trace if enabled
        if self.enable_telemetry and self.telemetry_writer is not None:
            try:
                # Respect sample rate (skip if rate is 0.0)
                should_capture = self.settings.telemetry_sample_rate >= 1.0 or (
                    self.settings.telemetry_sample_rate > 0.0
                    and random.random() < self.settings.telemetry_sample_rate
                )

                if should_capture:
                    # Determine actual direction based on realized return
                    if realized_return_pct > 0.5:
                        actual_direction = "LONG"
                    elif realized_return_pct < -0.5:
                        actual_direction = "SHORT"
                    else:
                        actual_direction = "NOOP"

                    # Prepare metadata
                    trace_metadata = {
                        "exit_reason": meta.get("reason", "unknown"),
                        "sl_hit": meta.get("sl_hit", False),
                        "tp_hit": meta.get("tp_hit", False),
                        "max_hold_hit": meta.get("max_hold_hit", False),
                        "entry_fees": position.entry_fees,
                        "exit_fees": exit_fees,
                        "total_fees": total_fees,
                        "position_value": position_value,
                        "gross_pnl": gross_pnl,
                    }

                    # Create prediction trace
                    trace = PredictionTrace(
                        timestamp=position.entry_date.to_pydatetime(),
                        symbol=symbol,
                        strategy="swing",  # Explicit strategy name for telemetry
                        predicted_direction=position.direction,
                        actual_direction=actual_direction,
                        predicted_confidence=0.5,  # Default confidence for backtests
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        holding_period_minutes=holding_period_minutes,
                        realized_return_pct=realized_return_pct,
                        features={},  # Empty for now, can be enhanced
                        metadata=trace_metadata,
                    )

                    # Write trace
                    self.telemetry_writer.write_trace(trace)

                    logger.debug(
                        f"Captured telemetry trace for {symbol}",
                        extra={
                            "component": "backtest",
                            "symbol": symbol,
                            "predicted": position.direction,
                            "actual": actual_direction,
                        },
                    )

            except Exception as e:
                logger.error(
                    f"Failed to capture telemetry trace: {e}",
                    extra={
                        "component": "backtest",
                        "symbol": symbol,
                        "error": str(e),
                    },
                )

        logger.debug(
            f"Closed {position.direction} position: {symbol} @ {exit_price}, PnL={realized_pnl:.2f}",
            extra={"component": "backtest", "symbol": symbol, "pnl": realized_pnl},
        )

        # Remove position
        del self.positions[symbol]

    def _update_equity_curve(self, timestamp: pd.Timestamp, symbol: str | None = None) -> None:
        """Update equity curve with current equity and open position value.

        Args:
            timestamp: Current timestamp
            symbol: Current symbol (optional)
        """
        # Calculate unrealized PnL from open positions
        # Simplified: not tracking intraday price changes
        position_count = len(self.positions)

        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": self.equity,
                "open_positions": position_count,
                "symbol": symbol or "",
            }
        )

    def _calculate_metrics(self) -> dict[str, float]:
        """Calculate comprehensive backtest metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.equity_curve:
            logger.warning("No equity curve data", extra={"component": "backtest"})
            return {}

        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        initial_capital = self.config.initial_capital
        final_equity = self.equity
        total_return = (final_equity - initial_capital) / initial_capital

        # CAGR
        start_date = pd.Timestamp(self.config.start_date)
        end_date = pd.Timestamp(self.config.end_date)
        years = (end_date - start_date).days / 365.25
        cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

        # Max Drawdown
        running_max = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - running_max) / running_max
        max_drawdown = float(drawdown.min())

        # Sharpe Ratio
        returns = equity_df["equity"].pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        # Trade metrics
        if not trades_df.empty:
            exit_trades = trades_df[trades_df["action"] == "EXIT"]

            if not exit_trades.empty:
                total_trades = len(exit_trades)
                winning_trades = len(exit_trades[exit_trades["pnl"] > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

                avg_win = (
                    exit_trades[exit_trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0.0
                )
                losing_trades = exit_trades[exit_trades["pnl"] <= 0]
                avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0.0

                total_fees = trades_df["fees"].sum()
            else:
                total_trades = 0
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0
                total_fees = 0.0
        else:
            total_trades = 0
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            total_fees = 0.0

        # Exposure
        days_with_positions = equity_df[equity_df["open_positions"] > 0].shape[0]
        total_days = len(equity_df)
        exposure = days_with_positions / total_days if total_days > 0 else 0.0

        metrics = {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": float(sharpe),
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "total_fees": float(total_fees),
            "exposure_pct": exposure * 100,
        }

        logger.info(
            "Metrics calculated",
            extra={"component": "backtest", "metrics": metrics},
        )

        return metrics

    def _save_artifacts(
        self,
        metrics: dict[str, float],
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> tuple[str, str, str]:
        """Save backtest artifacts to disk.

        Args:
            metrics: Performance metrics
            equity_df: Equity curve DataFrame
            trades_df: Trades log DataFrame

        Returns:
            Tuple of (summary_path, equity_path, trades_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/backtests")
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / f"backtest_{timestamp}_summary.json"
        equity_path = output_dir / f"backtest_{timestamp}_equity.csv"
        trades_path = output_dir / f"backtest_{timestamp}_trades.csv"

        # Save summary JSON
        summary = {
            "config": {
                "symbols": self.config.symbols,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "strategy": self.config.strategy,
                "initial_capital": self.config.initial_capital,
                "data_source": self.config.data_source,
                "random_seed": self.config.random_seed,
            },
            "metrics": metrics,
            "metadata": {
                "run_date": datetime.now().isoformat(),
                "settings": {
                    "position_sizing_mode": self.settings.position_sizing_mode,
                    "risk_per_trade_pct": self.settings.risk_per_trade_pct,
                    "trading_fee_bps": self.settings.trading_fee_bps,
                    "slippage_bps": self.settings.slippage_bps,
                },
            },
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save equity curve CSV
        equity_df.to_csv(equity_path, index=False)

        # Save trades log CSV
        if not trades_df.empty:
            trades_df.to_csv(trades_path, index=False)
        else:
            # Create empty file with headers
            pd.DataFrame(columns=["timestamp", "symbol", "action", "qty", "price", "pnl"]).to_csv(
                trades_path, index=False
            )

        logger.info(
            "Artifacts saved",
            extra={
                "component": "backtest",
                "summary": str(summary_path),
                "equity": str(equity_path),
                "trades": str(trades_path),
            },
        )

        return str(summary_path), str(equity_path), str(trades_path)
