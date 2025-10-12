#!/usr/bin/env python3
"""
Fetch sentiment snapshots for historical data batch training (US-024 Phase 3).

Downloads daily sentiment snapshots for configured symbols/date ranges using
sentiment provider registry (NewsAPI/Twitter/stub). Stores results as JSON Lines
under data/sentiment/<symbol>/<YYYY-MM-DD>.jsonl.

Features:
- Provider registry integration (multi-provider support)
- Caching (skip already-fetched dates)
- Retry/backoff for transient failures
- Dryrun mode (no network calls, use cached/mocked data)
- Progress logging and summary reports

Usage:
    # Fetch sentiment for default symbols/dates
    python scripts/fetch_sentiment_snapshots.py

    # Fetch specific symbols and date range
    python scripts/fetch_sentiment_snapshots.py \\
        --symbols RELIANCE TCS INFY \\
        --start-date 2024-01-01 \\
        --end-date 2024-03-31

    # Dryrun mode (no network calls)
    python scripts/fetch_sentiment_snapshots.py --dryrun

    # Force re-fetch (ignore cache)
    python scripts/fetch_sentiment_snapshots.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import tenacity
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.config import settings
from src.services.sentiment.factory import create_sentiment_registry
from src.services.sentiment.registry import SentimentProviderRegistry
from src.services.state_manager import StateManager


class SentimentSnapshotFetcher:
    """Manages sentiment snapshot downloads for batch training."""

    def __init__(
        self,
        output_dir: Path,
        registry: SentimentProviderRegistry,
        retry_limit: int = 3,
        retry_backoff_seconds: int = 2,
        dryrun: bool = False,
        force: bool = False,
    ):
        """Initialize sentiment snapshot fetcher.

        Args:
            output_dir: Base directory for sentiment snapshots
            registry: Sentiment provider registry
            retry_limit: Maximum retry attempts
            retry_backoff_seconds: Base backoff delay in seconds
            dryrun: If True, skip network calls and use cached/mocked data
            force: If True, ignore cache and re-fetch all snapshots
        """
        self.output_dir = output_dir
        self.registry = registry
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
        }

    def get_snapshot_path(self, symbol: str, date: datetime) -> Path:
        """Get path to sentiment snapshot file.

        Args:
            symbol: Stock symbol
            date: Date for snapshot

        Returns:
            Path to JSONL file
        """
        date_str = date.strftime("%Y-%m-%d")
        return self.output_dir / symbol / f"{date_str}.jsonl"

    def is_cached(self, symbol: str, date: datetime) -> bool:
        """Check if sentiment snapshot already exists.

        Args:
            symbol: Stock symbol
            date: Date for snapshot

        Returns:
            True if snapshot file exists and is non-empty
        """
        snapshot_path = self.get_snapshot_path(symbol, date)
        return snapshot_path.exists() and snapshot_path.stat().st_size > 0

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=30),
        retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number}/3 for sentiment fetch"
        ),
    )
    def fetch_sentiment_with_retry(self, symbol: str, date: datetime) -> dict[str, Any]:
        """Fetch sentiment for symbol/date with retry logic.

        Args:
            symbol: Stock symbol
            date: Date for sentiment

        Returns:
            Sentiment snapshot dict

        Raises:
            Exception: On fetch failure after retries
        """
        try:
            # Fetch sentiment from registry (uses fallback and weighted averaging)
            result = self.registry.get_sentiment(symbol)

            snapshot = {
                "symbol": symbol,
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "score": result.score,
                "confidence": result.confidence,
                "providers": [p.name for p in result.providers_used],
                "metadata": result.metadata or {},
            }

            return snapshot

        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {symbol} on {date}: {e}")
            raise

    def fetch_snapshot(self, symbol: str, date: datetime) -> bool:
        """Fetch and save sentiment snapshot for symbol/date.

        Args:
            symbol: Stock symbol
            date: Date for snapshot

        Returns:
            True if successful, False otherwise
        """
        snapshot_path = self.get_snapshot_path(symbol, date)
        date_str = date.strftime("%Y-%m-%d")

        # Check cache
        if not self.force and self.is_cached(symbol, date):
            logger.debug(
                f"Cached sentiment snapshot exists: {symbol} {date_str}",
                extra={"symbol": symbol, "date": date_str, "status": "cached"},
            )
            self.stats["cached"] += 1
            return True

        # Dryrun mode - create mock data
        if self.dryrun:
            logger.info(
                f"[DRYRUN] Would fetch sentiment: {symbol} {date_str}",
                extra={"symbol": symbol, "date": date_str, "mode": "dryrun"},
            )

            # Create mock snapshot
            mock_snapshot = {
                "symbol": symbol,
                "date": date_str,
                "timestamp": datetime.now().isoformat(),
                "score": 0.0,
                "confidence": 1.0,
                "providers": ["stub"],
                "metadata": {"dryrun": True},
            }

            # Save mock data
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(snapshot_path, "w") as f:
                f.write(json.dumps(mock_snapshot) + "\n")

            self.stats["fetched"] += 1
            return True

        # Fetch sentiment
        try:
            self.stats["total_requests"] += 1
            snapshot = self.fetch_sentiment_with_retry(symbol, date)

            # Save to JSONL
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(snapshot_path, "w") as f:
                f.write(json.dumps(snapshot) + "\n")

            logger.info(
                f"Fetched sentiment snapshot: {symbol} {date_str} "
                f"(score={snapshot['score']:.3f}, confidence={snapshot['confidence']:.3f})",
                extra={
                    "symbol": symbol,
                    "date": date_str,
                    "score": snapshot["score"],
                    "confidence": snapshot["confidence"],
                    "status": "success",
                },
            )

            self.stats["fetched"] += 1
            return True

        except Exception as e:
            logger.error(
                f"Failed to fetch sentiment snapshot: {symbol} {date_str}: {e}",
                extra={"symbol": symbol, "date": date_str, "error": str(e), "status": "failed"},
            )
            self.stats["failed"] += 1
            return False

    def fetch_all(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Fetch sentiment snapshots for all symbols/dates.

        Args:
            symbols: List of stock symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Summary statistics
        """
        logger.info(
            f"Starting sentiment snapshot fetch: {len(symbols)} symbols, "
            f"{start_date.date()} to {end_date.date()}",
            extra={
                "symbols": symbols,
                "start_date": str(start_date.date()),
                "end_date": str(end_date.date()),
                "dryrun": self.dryrun,
            },
        )

        # Generate date range
        current_date = start_date
        dates = []
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        total_tasks = len(symbols) * len(dates)
        completed = 0

        # Fetch for each symbol/date
        for symbol in symbols:
            for date in dates:
                self.fetch_snapshot(symbol, date)
                completed += 1

                if completed % 10 == 0 or completed == total_tasks:
                    logger.info(
                        f"Progress: {completed}/{total_tasks} "
                        f"({100 * completed / total_tasks:.1f}%)"
                    )

        # Generate summary
        summary = {
            "symbols": symbols,
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "days": len(dates),
            },
            "stats": self.stats.copy(),
            "output_dir": str(self.output_dir),
            "dryrun": self.dryrun,
        }

        logger.info(
            f"Sentiment snapshot fetch complete: "
            f"{self.stats['fetched']} fetched, "
            f"{self.stats['cached']} cached, "
            f"{self.stats['failed']} failed",
            extra=summary,
        )

        return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch sentiment snapshots for batch training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help=f"Stock symbols to fetch (default: {settings.historical_data_symbols})",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help=f"Start date YYYY-MM-DD (default: {settings.historical_data_start_date})",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help=f"End date YYYY-MM-DD (default: {settings.historical_data_end_date})",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help=f"Output directory (default: {settings.sentiment_snapshot_output_dir})",
    )

    parser.add_argument(
        "--providers",
        nargs="+",
        help=f"Sentiment providers to use (default: {settings.sentiment_snapshot_providers})",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dryrun mode - no network calls, use cached/mocked data",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch (ignore cache)",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode (fetch only new data since last run)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Lookback days for incremental mode (default: from settings)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check if sentiment snapshots are enabled
    if not settings.sentiment_snapshot_enabled and not args.dryrun:
        logger.warning(
            "Sentiment snapshot ingestion is disabled. "
            "Set SENTIMENT_SNAPSHOT_ENABLED=true in .env to enable, or use --dryrun mode."
        )
        return 0

    # Parse configuration
    symbols = args.symbols or settings.historical_data_symbols

    # US-024 Phase 4: Incremental mode support
    state_manager = None
    if args.incremental:
        state_file = Path("data/state/sentiment_fetch.json")
        state_manager = StateManager(state_file)

        # Get lookback days
        lookback_days = (
            args.lookback_days if args.lookback_days else settings.incremental_lookback_days
        )

        # Calculate date range
        end_date = datetime.now()

        # Find earliest last fetch
        earliest_last_fetch = None
        for symbol in symbols:
            last_fetch = state_manager.get_last_fetch_date(symbol)
            if last_fetch:
                if earliest_last_fetch is None or last_fetch < earliest_last_fetch:
                    earliest_last_fetch = last_fetch

        if earliest_last_fetch:
            start_date = earliest_last_fetch + timedelta(days=1)
            logger.info(f"Incremental mode: fetching from {start_date.strftime('%Y-%m-%d')}")
        else:
            start_date = datetime.now() - timedelta(days=lookback_days)
            logger.info(f"Incremental mode: no previous fetch, using {lookback_days}-day lookback")
    else:
        # Full mode
        start_date = datetime.strptime(
            args.start_date or settings.historical_data_start_date, "%Y-%m-%d"
        )
        end_date = datetime.strptime(args.end_date or settings.historical_data_end_date, "%Y-%m-%d")
    output_dir = Path(args.output_dir or settings.sentiment_snapshot_output_dir)
    providers = args.providers or settings.sentiment_snapshot_providers

    # Create provider registry
    if args.dryrun:
        # Dryrun mode - use stub provider
        from src.services.sentiment.providers.stub import StubSentimentProvider
        from src.services.sentiment.registry import SentimentProviderRegistry

        registry = SentimentProviderRegistry()
        registry.register("stub", StubSentimentProvider(), weight=1.0, priority=0)
        logger.info("Dryrun mode: using stub sentiment provider")
    else:
        # Use configured providers
        registry = create_sentiment_registry(
            providers=providers,
            api_keys={},  # API keys loaded from environment
        )

    # Create fetcher
    fetcher = SentimentSnapshotFetcher(
        output_dir=output_dir,
        registry=registry,
        retry_limit=settings.sentiment_snapshot_retry_limit,
        retry_backoff_seconds=settings.sentiment_snapshot_retry_backoff_seconds,
        dryrun=args.dryrun,
        force=args.force,
    )

    # Fetch all snapshots
    try:
        summary = fetcher.fetch_all(symbols, start_date, end_date)

        # Print summary
        print("\n" + "=" * 70)
        print("SENTIMENT SNAPSHOT FETCH SUMMARY")
        print("=" * 70)
        print(f"Symbols: {', '.join(summary['symbols'])}")
        print(
            f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']} "
            f"({summary['date_range']['days']} days)"
        )
        print(f"Output Directory: {summary['output_dir']}")
        print(f"Dryrun Mode: {summary['dryrun']}")
        print("-" * 70)
        print(f"Fetched: {summary['stats']['fetched']}")
        print(f"Cached: {summary['stats']['cached']}")
        print(f"Failed: {summary['stats']['failed']}")
        print(f"Total Requests: {summary['stats']['total_requests']}")
        print("=" * 70)

        # Return exit code
        if summary["stats"]["failed"] > 0:
            logger.warning(f"{summary['stats']['failed']} sentiment snapshots failed to fetch")
            return 1

        # US-024 Phase 4: Update state after successful fetch
        if args.incremental and state_manager:
            for symbol in symbols:
                state_manager.set_last_fetch_date(symbol, end_date)

            state_manager.set_last_run_info(
                run_type="incremental",
                success=summary["stats"]["failed"] == 0,
                symbols_processed=symbols,
                files_created=summary["stats"]["fetched"],
                errors=summary["stats"]["failed"],
            )

            logger.info(f"Updated state file: last fetch date = {end_date.strftime('%Y-%m-%d')}")

        return 0

    except Exception as e:
        logger.error(f"Sentiment snapshot fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
