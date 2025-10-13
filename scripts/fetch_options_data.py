#!/usr/bin/env python3
"""Fetch options chain data for configured symbols (US-029 Phase 1).

Downloads complete options chain (all strikes/expiries) for configured symbols
and stores them as JSON under data/options/<symbol>/<YYYY-MM-DD>.json.

Features:
- Complete chain fetch (calls + puts for all strikes/expiries)
- Incremental mode (fetch only if chain updated since last run)
- Dryrun mode (generate mock options chain data)
- Retry logic with exponential backoff
- State tracking for fetch metadata

Usage:
    # Fetch options chain in dryrun mode
    python scripts/fetch_options_data.py --dryrun

    # Fetch for specific symbols and date
    python scripts/fetch_options_data.py \\
        --symbols NIFTY BANKNIFTY \\
        --date 2025-01-15

    # Incremental mode (fetch only if updated)
    python scripts/fetch_options_data.py --incremental

    # Force re-fetch (ignore cache)
    python scripts/fetch_options_data.py --force
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
from src.adapters.market_data_providers import create_options_provider
from src.app.config import settings
from src.services.secrets_manager import SecretsManager
from src.services.state_manager import StateManager


class OptionsDataFetcher:
    """Manages options chain data downloads."""

    def __init__(
        self,
        output_dir: Path,
        retry_limit: int = 3,
        retry_backoff_seconds: int = 2,
        dryrun: bool = False,
        force: bool = False,
        secrets_mode: str = "plain",
    ):
        """Initialize options data fetcher.

        Args:
            output_dir: Base directory for options chain data
            retry_limit: Maximum retry attempts
            retry_backoff_seconds: Base backoff delay in seconds
            dryrun: If True, skip network calls and generate mock data
            force: If True, ignore cache and re-fetch all chains
            secrets_mode: Secrets mode ("plain" or "encrypted")
        """
        self.output_dir = output_dir
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
            "total_strikes": 0,
            "total_expiries": 0,
            "retries": 0,
        }

        # Initialize provider (US-029 Phase 4)
        self.provider = self._create_provider(secrets_mode)

    def _create_provider(self, secrets_mode: str):
        """Create options provider with credentials from SecretsManager.

        Args:
            secrets_mode: Secrets mode ("plain" or "encrypted")

        Returns:
            BreezeOptionsProvider instance
        """
        # Load secrets
        secrets = SecretsManager(mode=secrets_mode)

        # Create Breeze client if not dryrun
        client = None
        if not self.dryrun and settings.options_enabled:
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
                logger.info("Initialized Breeze client for options provider")
            else:
                logger.warning(
                    "Missing Breeze credentials, provider will use dryrun mode. "
                    "Set BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN in .env"
                )

        # Create provider
        provider = create_options_provider(settings, client=client, dry_run=self.dryrun)
        return provider

    def get_chain_path(self, symbol: str, date: datetime) -> Path:
        """Get path to options chain file.

        Args:
            symbol: Stock symbol
            date: Date for chain snapshot

        Returns:
            Path to JSON file
        """
        date_str = date.strftime("%Y-%m-%d")
        return self.output_dir / symbol / f"{date_str}.json"

    def is_cached(self, symbol: str, date: datetime) -> bool:
        """Check if options chain already exists.

        Args:
            symbol: Stock symbol
            date: Date for chain snapshot

        Returns:
            True if chain file exists and is non-empty
        """
        chain_path = self.get_chain_path(symbol, date)
        return chain_path.exists() and chain_path.stat().st_size > 0

    def fetch_options_chain_from_provider(self, symbol: str, date: datetime) -> dict[str, Any]:
        """Fetch options chain from provider (US-029 Phase 4).

        Args:
            symbol: Stock symbol
            date: Date for chain snapshot

        Returns:
            Options chain dict

        Raises:
            Exception: On fetch failure after retries
        """
        try:
            # Fetch chain from provider (handles retry internally)
            date_str = date.strftime("%Y-%m-%d")
            snapshot = self.provider.fetch(symbol=symbol, date=date_str)

            # Convert to dict format
            return {
                "symbol": snapshot.symbol,
                "date": snapshot.date,
                "timestamp": snapshot.timestamp.isoformat(),
                "underlying_price": snapshot.underlying_price,
                "options": snapshot.options,
                "metadata": snapshot.metadata,
            }

        except Exception as e:
            logger.error(f"Provider fetch failed for {symbol}: {e}")
            raise

    def fetch_chain(self, symbol: str, date: datetime) -> bool:
        """Fetch and save options chain.

        Args:
            symbol: Stock symbol
            date: Date for chain snapshot

        Returns:
            True if successful, False otherwise
        """
        chain_path = self.get_chain_path(symbol, date)
        date_str = date.strftime("%Y-%m-%d")

        # Check cache
        if not self.force and self.is_cached(symbol, date):
            logger.debug(
                f"Cached options chain exists: {symbol} {date_str}",
                extra={"symbol": symbol, "date": date_str, "status": "cached"},
            )
            self.stats["cached"] += 1
            return True

        # Fetch from provider (US-029 Phase 4)
        try:
            logger.info(
                f"Fetching options chain: {symbol} {date_str}",
                extra={"symbol": symbol, "date": date_str, "dryrun": self.dryrun},
            )

            chain = self.fetch_options_chain_from_provider(symbol, date)

            # Save chain
            chain_path.parent.mkdir(parents=True, exist_ok=True)
            with open(chain_path, "w") as f:
                json.dump(chain, f, indent=2)

            self.stats["fetched"] += 1
            self.stats["total_strikes"] += chain["metadata"]["total_strikes"]
            self.stats["total_expiries"] += len(chain["metadata"]["expiries"])

            logger.debug(
                f"Saved options chain: {chain_path}",
                extra={
                    "symbol": symbol,
                    "date": date_str,
                    "strikes": chain["metadata"]["total_strikes"],
                    "expiries": len(chain["metadata"]["expiries"]),
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to fetch options chain for {symbol}: {e}",
                extra={"symbol": symbol, "date": date_str, "error": str(e)},
            )
            self.stats["failed"] += 1
            return False

    def fetch_all(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Fetch options chains for all symbols/dates.

        Args:
            symbols: List of stock symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Summary statistics
        """
        logger.info(
            f"Starting options chain fetch: {len(symbols)} symbols, "
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
                self.stats["total_requests"] += 1
                self.fetch_chain(symbol, date)
                completed += 1

                if completed % 10 == 0 or completed == total_tasks:
                    logger.info(
                        f"Progress: {completed}/{total_tasks} ({100 * completed / total_tasks:.1f}%)"
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
            f"Options chain fetch complete: "
            f"{self.stats['fetched']} fetched, "
            f"{self.stats['cached']} cached, "
            f"{self.stats['failed']} failed",
            extra=summary,
        )

        return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch options chain data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Stock symbols to fetch (default: NIFTY, BANKNIFTY)",
    )

    parser.add_argument(
        "--date",
        type=str,
        help="Single date to fetch (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help=f"Output directory (default: {settings.options_output_dir})",
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
        help="Incremental mode (fetch only if updated since last run)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check if options ingestion is enabled
    if not settings.options_enabled and not args.dryrun:
        logger.warning(
            "Options chain ingestion is disabled. "
            "Set OPTIONS_ENABLED=true in .env to enable, or use --dryrun mode."
        )
        return 0

    # Parse configuration
    symbols = args.symbols or ["NIFTY", "BANKNIFTY"]
    output_dir = Path(args.output_dir or settings.options_output_dir)

    # Parse date range
    if args.date:
        start_date = datetime.strptime(args.date, "%Y-%m-%d")
        end_date = start_date
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        # Default: today
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date

    # Incremental mode
    state_manager = None
    if args.incremental:
        state_file = Path("data/state/options_fetch.json")
        state_manager = StateManager(state_file)

        # Get last fetch date
        last_fetch = state_manager.get_last_market_data_fetch("options", symbols[0])
        if last_fetch:
            last_date = datetime.fromisoformat(last_fetch["timestamp"])
            start_date = last_date + timedelta(days=1)
            logger.info(f"Incremental mode: fetching from {start_date.strftime('%Y-%m-%d')}")
        else:
            logger.info("Incremental mode: no previous fetch, using default date")

    # Get secrets mode from environment (default: plain)
    secrets_mode = os.getenv("SECRETS_MODE", "plain")

    # Create fetcher
    fetcher = OptionsDataFetcher(
        output_dir=output_dir,
        retry_limit=settings.options_retry_limit,
        retry_backoff_seconds=settings.options_retry_backoff_seconds,
        dryrun=args.dryrun,
        force=args.force,
        secrets_mode=secrets_mode,
    )

    # Fetch all chains
    try:
        summary = fetcher.fetch_all(symbols, start_date, end_date)

        # Print summary
        print("\n" + "=" * 70)
        print("OPTIONS CHAIN FETCH SUMMARY")
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
        print(f"Total Strikes: {summary['stats']['total_strikes']}")
        print(f"Total Expiries: {summary['stats']['total_expiries']}")
        print("=" * 70)

        # Update state if incremental mode
        if args.incremental and state_manager:
            for symbol in symbols:
                state_manager.record_market_data_fetch(
                    data_type="options",
                    symbol=symbol,
                    timestamp=end_date.isoformat(),
                    stats={
                        "fetched": summary["stats"]["fetched"],
                        "cached": summary["stats"]["cached"],
                        "failed": summary["stats"]["failed"],
                        "strikes": summary["stats"]["total_strikes"],
                        "expiries": summary["stats"]["total_expiries"],
                    },
                )

            logger.info(f"Updated state file: last fetch date = {end_date.strftime('%Y-%m-%d')}")

        # Return exit code
        if summary["stats"]["failed"] > 0:
            logger.warning(f"{summary['stats']['failed']} options chains failed to fetch")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Options chain fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
