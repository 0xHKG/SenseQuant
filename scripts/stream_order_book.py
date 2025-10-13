#!/usr/bin/env python3
"""Stream order book data via Breeze WebSocket (US-029 Phase 5).

Features:
- Real-time order book updates via WebSocket
- Configurable update interval and buffer size
- Graceful shutdown on SIGTERM/SIGINT
- State tracking for heartbeat monitoring
- Dryrun mode with deterministic mock stream

Usage:
    # Dryrun mode (mock WebSocket)
    python scripts/stream_order_book.py --dryrun

    # Live mode (requires Breeze credentials)
    python scripts/stream_order_book.py --symbols RELIANCE TCS --interval 1
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.secrets_manager import SecretsManager
from src.services.state_manager import StateManager


class OrderBookStreamer:
    """Manages real-time order book streaming via WebSocket."""

    def __init__(
        self,
        symbols: list[str],
        output_dir: Path,
        buffer_size: int = 100,
        update_interval_seconds: int = 1,
        dryrun: bool = False,
        secrets_mode: str = "plain",
    ):
        """Initialize streamer.

        Args:
            symbols: Stock symbols to stream
            output_dir: Cache directory for snapshots
            buffer_size: Max snapshots to buffer per symbol
            update_interval_seconds: Snapshot update interval
            dryrun: If True, use mock WebSocket
            secrets_mode: Secrets mode for credentials
        """
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        self.update_interval = update_interval_seconds
        self.dryrun = dryrun

        # Statistics
        self.stats: dict[str, int | str | None] = {
            "updates": 0,
            "errors": 0,
            "last_heartbeat": None,
        }

        # Circular buffers (per symbol)
        self.buffers: dict[str, deque] = {symbol: deque(maxlen=buffer_size) for symbol in symbols}

        # Initialize WebSocket connection
        self.ws_client = self._create_websocket_client(secrets_mode)

        # State manager for heartbeat tracking
        state_file = self.output_dir / "state" / "streaming.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_manager = StateManager(state_file)

        # Shutdown flag
        self.running = False

        logger.info(
            f"OrderBookStreamer initialized: {len(symbols)} symbols, "
            f"buffer_size={buffer_size}, interval={update_interval_seconds}s, dryrun={dryrun}",
            extra={"component": "streaming"},
        )

    def _create_websocket_client(
        self, secrets_mode: str
    ) -> MockWebSocketClient | BreezeWebSocketClient:
        """Create WebSocket client (Breeze or mock)."""
        if self.dryrun:
            logger.info(
                "Using mock WebSocket client (dryrun mode)", extra={"component": "streaming"}
            )
            return MockWebSocketClient(self.symbols, self.update_interval)
        else:
            # Real Breeze WebSocket client
            secrets = SecretsManager(mode=secrets_mode)
            api_key = secrets.get_secret("BREEZE_API_KEY", "")
            api_secret = secrets.get_secret("BREEZE_API_SECRET", "")
            session_token = secrets.get_secret("BREEZE_SESSION_TOKEN", "")

            if not (api_key and api_secret and session_token):
                raise ValueError(
                    "Breeze credentials required for live streaming. "
                    "Set BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN in .env"
                )

            logger.info("Creating Breeze WebSocket client", extra={"component": "streaming"})
            return BreezeWebSocketClient(
                api_key=api_key,
                api_secret=api_secret,
                session_token=session_token,
            )

    def start(self) -> None:
        """Start streaming (blocking)."""
        self.running = True

        # Register signal handlers (only if in main thread)
        try:
            signal.signal(signal.SIGINT, self._shutdown_handler)
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        except ValueError:
            # Signal handlers can only be registered in main thread
            logger.debug(
                "Signal handlers not registered (not in main thread)",
                extra={"component": "streaming"},
            )

        logger.info(
            f"Starting order book stream for {len(self.symbols)} symbols",
            extra={"component": "streaming", "symbols": self.symbols},
        )

        # Subscribe to symbols
        self.ws_client.subscribe(self.symbols)

        # Main streaming loop
        while self.running:
            try:
                # Read from WebSocket
                update = self.ws_client.receive(timeout=5)

                if update:
                    self._process_update(update)

                # Record heartbeat
                self.state_manager.record_streaming_heartbeat(
                    stream_type="order_book",
                    symbols=self.symbols,
                    stats=self.stats,
                )

            except Exception as e:
                logger.error(f"Streaming error: {e}", extra={"component": "streaming"})
                error_count = self.stats["errors"]
                if isinstance(error_count, int):
                    self.stats["errors"] = error_count + 1
                time.sleep(1)  # Back off on error

        # Cleanup
        self.ws_client.disconnect()
        logger.info("Order book stream stopped", extra={"component": "streaming"})

    def _process_update(self, update: dict[str, Any]) -> None:
        """Process WebSocket update and cache snapshot."""
        symbol = update["symbol"]

        # Create snapshot
        snapshot = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "bids": update["bids"],
            "asks": update["asks"],
            "metadata": {
                "source": "stream",
                "dryrun": self.dryrun,
            },
        }

        # Add to buffer
        if symbol in self.buffers:
            self.buffers[symbol].append(snapshot)

        # Write to cache
        cache_path = self._get_cache_path(symbol)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(snapshot, f, indent=2)

        # Update statistics
        update_count = self.stats["updates"]
        if isinstance(update_count, int):
            self.stats["updates"] = update_count + 1
        self.stats["last_heartbeat"] = datetime.now().isoformat()

        logger.debug(
            f"Processed update for {symbol}: {len(update['bids'])} bids, {len(update['asks'])} asks",
            extra={"component": "streaming", "symbol": symbol},
        )

    def _get_cache_path(self, symbol: str) -> Path:
        """Get path to latest snapshot cache."""
        return self.output_dir / "streaming" / symbol / "latest.json"

    def _shutdown_handler(self, signum: int, frame: Any) -> None:
        """Handle graceful shutdown."""
        logger.info(
            f"Shutdown signal received (signal={signum}), stopping stream...",
            extra={"component": "streaming"},
        )
        self.running = False

    def get_latest_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Get latest snapshot from buffer.

        Args:
            symbol: Stock symbol

        Returns:
            Latest snapshot dict or None if buffer empty
        """
        if symbol not in self.buffers or not self.buffers[symbol]:
            return None
        snapshot = self.buffers[symbol][-1]
        # Type guard: ensure we return the correct type
        if isinstance(snapshot, dict):
            return snapshot
        return None

    def get_buffer_snapshots(self, symbol: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Get snapshots from buffer.

        Args:
            symbol: Stock symbol
            limit: Max snapshots to return (None = all)

        Returns:
            List of snapshots (newest first)
        """
        if symbol not in self.buffers:
            return []

        snapshots = list(self.buffers[symbol])
        snapshots.reverse()  # Newest first

        if limit:
            return snapshots[:limit]
        return snapshots


class MockWebSocketClient:
    """Mock WebSocket client for dryrun mode."""

    def __init__(self, symbols: list[str], interval: int):
        """Initialize mock client.

        Args:
            symbols: Symbols to generate updates for
            interval: Update interval in seconds
        """
        self.symbols = symbols
        self.interval = interval
        self.counter = 0

    def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols (mock).

        Args:
            symbols: Symbols to subscribe to
        """
        logger.info(
            f"[DRYRUN] Subscribed to {len(symbols)} symbols: {symbols}",
            extra={"component": "streaming"},
        )

    def receive(self, timeout: int = 5) -> dict[str, Any] | None:
        """Generate deterministic mock update.

        Args:
            timeout: Receive timeout (seconds)

        Returns:
            Mock order book update dict
        """
        time.sleep(self.interval)

        # Round-robin through symbols
        symbol = self.symbols[self.counter % len(self.symbols)]
        self.counter += 1

        # Generate mock order book
        base_price = 2000 + (hash(symbol) % 1000) + (self.counter % 10)

        return {
            "symbol": symbol,
            "bids": [
                {"price": base_price - i * 0.5, "quantity": 1000 + (i * 100), "orders": 3}
                for i in range(1, 6)
            ],
            "asks": [
                {"price": base_price + i * 0.5, "quantity": 800 + (i * 50), "orders": 2}
                for i in range(1, 6)
            ],
        }

    def disconnect(self) -> None:
        """Disconnect WebSocket (mock)."""
        logger.info("[DRYRUN] WebSocket disconnected", extra={"component": "streaming"})


class BreezeWebSocketClient:
    """Breeze WebSocket client for live streaming.

    Note: This is a placeholder implementation. Real Breeze WebSocket integration
    requires the Breeze SDK's WebSocket methods which are not yet available.
    """

    def __init__(self, api_key: str, api_secret: str, session_token: str):
        """Initialize Breeze WebSocket client.

        Args:
            api_key: Breeze API key
            api_secret: Breeze API secret
            session_token: Breeze session token
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.connected = False

        logger.warning(
            "BreezeWebSocketClient is a placeholder. Real WebSocket integration pending.",
            extra={"component": "streaming"},
        )

    def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols.

        Args:
            symbols: Symbols to subscribe to
        """
        logger.info(
            f"Breeze WebSocket: Subscribed to {len(symbols)} symbols",
            extra={"component": "streaming", "symbols": symbols},
        )
        self.connected = True
        # TODO: Implement real Breeze WebSocket subscription

    def receive(self, timeout: int = 5) -> dict[str, Any] | None:
        """Receive order book update.

        Args:
            timeout: Receive timeout (seconds)

        Returns:
            Order book update dict or None
        """
        # TODO: Implement real Breeze WebSocket receive
        # For now, return None to prevent blocking
        time.sleep(timeout)
        return None

    def disconnect(self) -> None:
        """Disconnect WebSocket."""
        if self.connected:
            logger.info("Breeze WebSocket disconnected", extra={"component": "streaming"})
            self.connected = False
            # TODO: Implement real Breeze WebSocket disconnect


def main() -> None:
    """Main entry point for order book streaming."""
    parser = argparse.ArgumentParser(
        description="Stream real-time order book data (US-029 Phase 5)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE"],
        help="Stock symbols to stream (default: RELIANCE)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/order_book",
        help="Output directory for snapshots (default: data/order_book)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100,
        help="Max snapshots to buffer per symbol (default: 100)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Update interval in seconds (default: 1)",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Use mock WebSocket (no real network calls)",
    )

    args = parser.parse_args()

    # Get secrets mode from environment
    secrets_mode = os.getenv("SECRETS_MODE", "plain")

    # Create streamer
    streamer = OrderBookStreamer(
        symbols=args.symbols,
        output_dir=Path(args.output_dir),
        buffer_size=args.buffer_size,
        update_interval_seconds=args.interval,
        dryrun=args.dryrun,
        secrets_mode=secrets_mode,
    )

    # Start streaming (blocking)
    try:
        streamer.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user", extra={"component": "streaming"})
    except Exception as e:
        logger.exception(f"Streaming failed: {e}", extra={"component": "streaming"})
        sys.exit(1)

    logger.info(
        f"Streaming completed. Total updates: {streamer.stats['updates']}, "
        f"Errors: {streamer.stats['errors']}",
        extra={"component": "streaming"},
    )


if __name__ == "__main__":
    main()
