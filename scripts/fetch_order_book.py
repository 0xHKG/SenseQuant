#!/usr/bin/env python3
"""Fetch order book depth snapshots for configured symbols (US-029 Phase 1).

Downloads L2 order book snapshots (best N bid/ask price levels) for configured
symbols and stores them as JSON under data/order_book/<symbol>/<YYYY-MM-DD>/<HH-MM-SS>.json.

Features:
- Configurable depth levels (1-20 price levels)
- Incremental mode (fetch only new snapshots since last run)
- Dryrun mode (generate mock data without network calls)
- Retry logic with exponential backoff
- State tracking for fetch metadata

Usage:
    # Fetch order book snapshots in dryrun mode
    python scripts/fetch_order_book.py --dryrun

    # Fetch for specific symbols and time range
    python scripts/fetch_order_book.py \\
        --symbols RELIANCE TCS INFY \\
        --start-time 09:15:00 \\
        --end-time 15:30:00 \\
        --interval-seconds 60

    # Incremental mode (fetch only new snapshots)
    python scripts/fetch_order_book.py --incremental

    # Force re-fetch (ignore cache)
    python scripts/fetch_order_book.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.breeze_client import BreezeClient
from src.adapters.market_data_providers import create_order_book_provider
from src.app.config import settings
from src.services.secrets_manager import SecretsManager
from src.services.state_manager import StateManager


class OrderBookFetcher:
    """Manages order book snapshot downloads."""

    def __init__(
        self,
        output_dir: Path,
        depth_levels: int = 5,
        retry_limit: int = 3,
        retry_backoff_seconds: int = 2,
        dryrun: bool = False,
        force: bool = False,
        secrets_mode: str = "plain",
    ):
        """Initialize order book fetcher.

        Args:
            output_dir: Base directory for order book snapshots
            depth_levels: Number of price levels to capture (1-20)
            retry_limit: Maximum retry attempts
            retry_backoff_seconds: Base backoff delay in seconds
            dryrun: If True, skip network calls and generate mock data
            force: If True, ignore cache and re-fetch all snapshots
            secrets_mode: Secrets mode ("plain" or "encrypted")
        """
        self.output_dir = output_dir
        self.depth_levels = depth_levels
        self.retry_limit = retry_limit
        self.retry_backoff_seconds = retry_backoff_seconds
        self.dryrun = dryrun
        self.force = force

        # Statistics
        self.stats = {
            "fetched": 0,
            "cached": 0,
            "failed": 0,
            "total_requests": 0,
            "retries": 0,
        }

        # Initialize provider (US-029 Phase 4)
        self.provider = self._create_provider(secrets_mode)

    def _create_provider(self, secrets_mode: str):
        """Create order book provider with credentials from SecretsManager.

        Args:
            secrets_mode: Secrets mode ("plain" or "encrypted")

        Returns:
            BreezeOrderBookProvider instance
        """
        # Load secrets
        secrets = SecretsManager(mode=secrets_mode)

        # Create Breeze client if not dryrun
        client = None
        if not self.dryrun and settings.order_book_enabled:
            api_key = secrets.get_secret("BREEZE_API_KEY", "")
            api_secret = secrets.get_secret("BREEZE_API_SECRET", "")
            session_token = secrets.get_secret("BREEZE_SESSION_TOKEN", "")

            if api_key and api_secret and session_token:
                client = BreezeClient(
                    api_key=api_key,
                    api_secret=api_secret,
                    session_token=session_token,
                    dry_run=False,
                )
                logger.info("Initialized Breeze client for order book provider")
            else:
                logger.warning(
                    "Missing Breeze credentials, provider will use dryrun mode. "
                    "Set BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN in .env"
                )

        # Create provider
        provider = create_order_book_provider(settings, client=client, dry_run=self.dryrun)
        return provider

    def get_snapshot_path(self, symbol: str, timestamp: datetime) -> Path:
        """Get path to order book snapshot file.

        Args:
            symbol: Stock symbol
            timestamp: Snapshot timestamp

        Returns:
            Path to JSON file
        """
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H-%M-%S")
        return self.output_dir / symbol / date_str / f"{time_str}.json"

    def is_cached(self, symbol: str, timestamp: datetime) -> bool:
        """Check if order book snapshot already exists.

        Args:
            symbol: Stock symbol
            timestamp: Snapshot timestamp

        Returns:
            True if snapshot file exists and is non-empty
        """
        snapshot_path = self.get_snapshot_path(symbol, timestamp)
        return snapshot_path.exists() and snapshot_path.stat().st_size > 0

    def fetch_order_book_from_provider(self, symbol: str, timestamp: datetime) -> dict[str, Any]:
        """Fetch order book snapshot from provider (US-029 Phase 4).

        Args:
            symbol: Stock symbol
            timestamp: Snapshot timestamp

        Returns:
            Order book snapshot dict

        Raises:
            Exception: On fetch failure after retries
        """
        try:
            # Fetch snapshot from provider (handles retry internally)
            snapshot = self.provider.fetch(symbol=symbol, depth_levels=self.depth_levels)

            # Convert to dict format
            return {
                "symbol": snapshot.symbol,
                "timestamp": timestamp.isoformat(),
                "exchange": snapshot.exchange,
                "bids": snapshot.bids,
                "asks": snapshot.asks,
                "metadata": snapshot.metadata,
            }

        except Exception as e:
            logger.error(f"Provider fetch failed for {symbol}: {e}")
            raise

    def fetch_snapshot(self, symbol: str, timestamp: datetime) -> bool:
        """Fetch and save order book snapshot.

        Args:
            symbol: Stock symbol
            timestamp: Snapshot timestamp

        Returns:
            True if successful, False otherwise
        """
        snapshot_path = self.get_snapshot_path(symbol, timestamp)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Check cache
        if not self.force and self.is_cached(symbol, timestamp):
            logger.debug(
                f"Cached order book snapshot exists: {symbol} {timestamp_str}",
                extra={"symbol": symbol, "timestamp": timestamp_str, "status": "cached"},
            )
            self.stats["cached"] += 1
            return True

        # Fetch from provider (US-029 Phase 4)
        try:
            logger.info(
                f"Fetching order book snapshot: {symbol} {timestamp_str}",
                extra={
                    "symbol": symbol,
                    "timestamp": timestamp_str,
                    "dryrun": self.dryrun,
                },
            )

            snapshot = self.fetch_order_book_from_provider(symbol, timestamp)

            # Save snapshot
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(snapshot_path, "w") as f:
                json.dump(snapshot, f, indent=2)

            self.stats["fetched"] += 1
            logger.debug(
                f"Saved order book snapshot: {snapshot_path}",
                extra={
                    "symbol": symbol,
                    "timestamp": timestamp_str,
                    "depth": len(snapshot["bids"]),
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to fetch order book snapshot for {symbol}: {e}",
                extra={"symbol": symbol, "timestamp": timestamp_str, "error": str(e)},
            )
            self.stats["failed"] += 1
            return False

    def fetch_all(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int,
    ) -> dict[str, Any]:
        """Fetch order book snapshots for all symbols/times.

        Args:
            symbols: List of stock symbols
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            interval_seconds: Snapshot interval in seconds

        Returns:
            Summary statistics
        """
        logger.info(
            f"Starting order book snapshot fetch: {len(symbols)} symbols, "
            f"{start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}, "
            f"interval={interval_seconds}s",
            extra={
                "symbols": symbols,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "interval_seconds": interval_seconds,
                "dryrun": self.dryrun,
            },
        )

        # Generate time range
        current_time = start_time
        timestamps = []
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += timedelta(seconds=interval_seconds)

        total_tasks = len(symbols) * len(timestamps)
        completed = 0

        # Fetch for each symbol/timestamp
        for symbol in symbols:
            for timestamp in timestamps:
                self.stats["total_requests"] += 1
                self.fetch_snapshot(symbol, timestamp)
                completed += 1

                if completed % 10 == 0 or completed == total_tasks:
                    logger.info(
                        f"Progress: {completed}/{total_tasks} ({100 * completed / total_tasks:.1f}%)"
                    )

        # Generate summary
        summary = {
            "symbols": symbols,
            "time_range": {
                "start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "snapshots": len(timestamps),
            },
            "stats": self.stats.copy(),
            "output_dir": str(self.output_dir),
            "dryrun": self.dryrun,
        }

        logger.info(
            f"Order book snapshot fetch complete: "
            f"{self.stats['fetched']} fetched, "
            f"{self.stats['cached']} cached, "
            f"{self.stats['failed']} failed",
            extra=summary,
        )

        return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch order book depth snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help=f"Stock symbols to fetch (default: {settings.symbols})",
    )

    parser.add_argument(
        "--start-time",
        type=str,
        help="Start time HH:MM:SS (default: 09:15:00)",
    )

    parser.add_argument(
        "--end-time",
        type=str,
        help="End time HH:MM:SS (default: 15:30:00)",
    )

    parser.add_argument(
        "--interval-seconds",
        type=int,
        help=f"Snapshot interval in seconds (default: {settings.order_book_snapshot_interval_seconds})",
    )

    parser.add_argument(
        "--depth-levels",
        type=int,
        help=f"Number of price levels to capture (default: {settings.order_book_depth_levels})",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help=f"Output directory (default: {settings.order_book_output_dir})",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dryrun mode - generate mock data, no network calls",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch (ignore cache)",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode (fetch only new snapshots since last run)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check if order book ingestion is enabled
    if not settings.order_book_enabled and not args.dryrun:
        logger.warning(
            "Order book ingestion is disabled. "
            "Set ORDER_BOOK_ENABLED=true in .env to enable, or use --dryrun mode."
        )
        return 0

    # Parse configuration
    symbols = args.symbols or settings.symbols
    depth_levels = args.depth_levels or settings.order_book_depth_levels
    interval_seconds = args.interval_seconds or settings.order_book_snapshot_interval_seconds
    output_dir = Path(args.output_dir or settings.order_book_output_dir)

    # Parse time range
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if args.start_time:
        start_hour, start_minute, start_second = map(int, args.start_time.split(":"))
        start_time = today.replace(hour=start_hour, minute=start_minute, second=start_second)
    else:
        start_time = today.replace(hour=9, minute=15, second=0)

    if args.end_time:
        end_hour, end_minute, end_second = map(int, args.end_time.split(":"))
        end_time = today.replace(hour=end_hour, minute=end_minute, second=end_second)
    else:
        end_time = today.replace(hour=15, minute=30, second=0)

    # Incremental mode: fetch from last run
    state_manager = None
    if args.incremental:
        state_file = Path("data/state/order_book_fetch.json")
        state_manager = StateManager(state_file)

        # Get last fetch timestamp
        last_fetch = state_manager.get_last_market_data_fetch("order_book", symbols[0])
        if last_fetch:
            last_timestamp = datetime.fromisoformat(last_fetch["timestamp"])
            start_time = last_timestamp + timedelta(seconds=interval_seconds)
            logger.info(
                f"Incremental mode: fetching from {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            logger.info("Incremental mode: no previous fetch, using default start time")

    # Get secrets mode from environment (default: plain)
    secrets_mode = os.getenv("SECRETS_MODE", "plain")

    # Create fetcher
    fetcher = OrderBookFetcher(
        output_dir=output_dir,
        depth_levels=depth_levels,
        retry_limit=settings.order_book_retry_limit,
        retry_backoff_seconds=settings.order_book_retry_backoff_seconds,
        dryrun=args.dryrun,
        force=args.force,
        secrets_mode=secrets_mode,
    )

    # Fetch all snapshots
    try:
        summary = fetcher.fetch_all(symbols, start_time, end_time, interval_seconds)

        # Print summary
        print("\n" + "=" * 70)
        print("ORDER BOOK SNAPSHOT FETCH SUMMARY")
        print("=" * 70)
        print(f"Symbols: {', '.join(summary['symbols'])}")
        print(
            f"Time Range: {summary['time_range']['start']} to {summary['time_range']['end']} "
            f"({summary['time_range']['snapshots']} snapshots)"
        )
        print(f"Output Directory: {summary['output_dir']}")
        print(f"Depth Levels: {depth_levels}")
        print(f"Interval: {interval_seconds}s")
        print(f"Dryrun Mode: {summary['dryrun']}")
        print("-" * 70)
        print(f"Fetched: {summary['stats']['fetched']}")
        print(f"Cached: {summary['stats']['cached']}")
        print(f"Failed: {summary['stats']['failed']}")
        print(f"Total Requests: {summary['stats']['total_requests']}")
        print("=" * 70)

        # Update state if incremental mode
        if args.incremental and state_manager:
            for symbol in symbols:
                state_manager.record_market_data_fetch(
                    data_type="order_book",
                    symbol=symbol,
                    timestamp=end_time.isoformat(),
                    stats={
                        "fetched": summary["stats"]["fetched"],
                        "cached": summary["stats"]["cached"],
                        "failed": summary["stats"]["failed"],
                    },
                )

            logger.info(f"Updated state file: last fetch timestamp = {end_time.isoformat()}")

        # Return exit code
        if summary["stats"]["failed"] > 0:
            logger.warning(f"{summary['stats']['failed']} order book snapshots failed to fetch")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Order book snapshot fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
