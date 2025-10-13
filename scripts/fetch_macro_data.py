#!/usr/bin/env python3
"""Fetch macro economic indicator data (US-029 Phase 1).

Downloads configured macro economic indicators (e.g., NIFTY 50, India VIX, USD/INR, bond yields)
and stores them as JSON under data/macro/<indicator>/<YYYY-MM-DD>.json.

Features:
- Configurable indicator list
- Incremental mode (fetch only new data since last run)
- Dryrun mode (generate mock macro data)
- Retry logic with exponential backoff
- State tracking for fetch metadata

Usage:
    # Fetch macro data in dryrun mode
    python scripts/fetch_macro_data.py --dryrun

    # Fetch specific indicators and date range
    python scripts/fetch_macro_data.py \\
        --indicators NIFTY50 INDIAVIX USDINR IN10Y \\
        --start-date 2025-01-01 \\
        --end-date 2025-01-31

    # Incremental mode (fetch only new data)
    python scripts/fetch_macro_data.py --incremental

    # Force re-fetch (ignore cache)
    python scripts/fetch_macro_data.py --force
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

from src.adapters.market_data_providers import create_macro_provider
from src.app.config import settings
from src.services.state_manager import StateManager


class MacroDataFetcher:
    """Manages macro economic indicator downloads."""

    def __init__(
        self,
        output_dir: Path,
        retry_limit: int = 3,
        retry_backoff_seconds: int = 2,
        dryrun: bool = False,
        force: bool = False,
        secrets_mode: str = "plain",
    ):
        """Initialize macro data fetcher.

        Args:
            output_dir: Base directory for macro indicator data
            retry_limit: Maximum retry attempts
            retry_backoff_seconds: Base backoff delay in seconds
            dryrun: If True, skip network calls and generate mock data
            force: If True, ignore cache and re-fetch all data
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
            "total_data_points": 0,
            "retries": 0,
        }

        # Initialize provider (US-029 Phase 4)
        self.provider = self._create_provider(secrets_mode)

    def _create_provider(self, secrets_mode: str):
        """Create macro indicator provider with credentials from SecretsManager.

        Args:
            secrets_mode: Secrets mode ("plain" or "encrypted")

        Returns:
            MacroIndicatorProvider instance
        """
        # Note: MacroIndicatorProvider doesn't use Breeze client
        # It uses public APIs (yfinance, etc.) which may have their own API keys
        # SecretsManager would be used here if we need API keys for macro sources
        # For now, we pass dryrun mode directly

        # Create provider
        provider = create_macro_provider(settings, dry_run=self.dryrun)
        return provider

    def get_data_path(self, indicator: str, date: datetime) -> Path:
        """Get path to macro data file.

        Args:
            indicator: Macro indicator name
            date: Date for data point

        Returns:
            Path to JSON file
        """
        date_str = date.strftime("%Y-%m-%d")
        return self.output_dir / indicator / f"{date_str}.json"

    def is_cached(self, indicator: str, date: datetime) -> bool:
        """Check if macro data already exists.

        Args:
            indicator: Macro indicator name
            date: Date for data point

        Returns:
            True if data file exists and is non-empty
        """
        data_path = self.get_data_path(indicator, date)
        return data_path.exists() and data_path.stat().st_size > 0

    def fetch_macro_data_from_provider(self, indicator: str, date: datetime) -> dict[str, Any]:
        """Fetch macro indicator data from provider (US-029 Phase 4).

        Args:
            indicator: Macro indicator name
            date: Date for data point

        Returns:
            Macro data dict

        Raises:
            Exception: On fetch failure after retries
        """
        try:
            # Fetch data from provider (handles retry internally)
            date_str = date.strftime("%Y-%m-%d")
            snapshot = self.provider.fetch(indicator=indicator, date=date_str)

            # Convert to dict format
            return {
                "indicator": snapshot.indicator,
                "date": snapshot.date,
                "timestamp": snapshot.timestamp.isoformat(),
                "value": snapshot.value,
                "change": snapshot.change,
                "change_pct": snapshot.change_pct,
                "metadata": snapshot.metadata,
            }

        except Exception as e:
            logger.error(f"Provider fetch failed for {indicator}: {e}")
            raise

    def fetch_data(self, indicator: str, date: datetime) -> bool:
        """Fetch and save macro data.

        Args:
            indicator: Macro indicator name
            date: Date for data point

        Returns:
            True if successful, False otherwise
        """
        data_path = self.get_data_path(indicator, date)
        date_str = date.strftime("%Y-%m-%d")

        # Check cache
        if not self.force and self.is_cached(indicator, date):
            logger.debug(
                f"Cached macro data exists: {indicator} {date_str}",
                extra={"indicator": indicator, "date": date_str, "status": "cached"},
            )
            self.stats["cached"] += 1
            return True

        # Fetch from provider (US-029 Phase 4)
        try:
            logger.info(
                f"Fetching macro data: {indicator} {date_str}",
                extra={"indicator": indicator, "date": date_str, "dryrun": self.dryrun},
            )

            data = self.fetch_macro_data_from_provider(indicator, date)

            # Save data
            data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)

            self.stats["fetched"] += 1
            self.stats["total_data_points"] += 1

            logger.debug(
                f"Saved macro data: {data_path}",
                extra={
                    "indicator": indicator,
                    "date": date_str,
                    "value": data["value"],
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to fetch macro data for {indicator}: {e}",
                extra={"indicator": indicator, "date": date_str, "error": str(e)},
            )
            self.stats["failed"] += 1
            return False

    def fetch_all(
        self,
        indicators: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Fetch macro data for all indicators/dates.

        Args:
            indicators: List of macro indicators
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Summary statistics
        """
        logger.info(
            f"Starting macro data fetch: {len(indicators)} indicators, "
            f"{start_date.date()} to {end_date.date()}",
            extra={
                "indicators": indicators,
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

        total_tasks = len(indicators) * len(dates)
        completed = 0

        # Fetch for each indicator/date
        for indicator in indicators:
            for date in dates:
                self.stats["total_requests"] += 1
                self.fetch_data(indicator, date)
                completed += 1

                if completed % 10 == 0 or completed == total_tasks:
                    logger.info(
                        f"Progress: {completed}/{total_tasks} ({100 * completed / total_tasks:.1f}%)"
                    )

        # Generate summary
        summary = {
            "indicators": indicators,
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
            f"Macro data fetch complete: "
            f"{self.stats['fetched']} fetched, "
            f"{self.stats['cached']} cached, "
            f"{self.stats['failed']} failed",
            extra=summary,
        )

        return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch macro economic indicator data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--indicators",
        nargs="+",
        help=f"Macro indicators to fetch (default: {settings.macro_indicators})",
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
        help=f"Output directory (default: {settings.macro_output_dir})",
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
        help="Incremental mode (fetch only new data since last run)",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Lookback days for incremental mode (default: 7)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check if macro data ingestion is enabled
    if not settings.macro_enabled and not args.dryrun:
        logger.warning(
            "Macro data ingestion is disabled. "
            "Set MACRO_ENABLED=true in .env to enable, or use --dryrun mode."
        )
        return 0

    # Parse configuration
    indicators = args.indicators or settings.macro_indicators
    output_dir = Path(args.output_dir or settings.macro_output_dir)

    # Parse date range
    state_manager = None
    if args.incremental:
        state_file = Path("data/state/macro_fetch.json")
        state_manager = StateManager(state_file)

        # Get lookback days
        lookback_days = args.lookback_days or 7

        # Calculate date range
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Find earliest last fetch
        earliest_last_fetch = None
        for indicator in indicators:
            last_fetch = state_manager.get_last_market_data_fetch("macro", indicator)
            if last_fetch:
                last_date = datetime.fromisoformat(last_fetch["timestamp"])
                if earliest_last_fetch is None or last_date < earliest_last_fetch:
                    earliest_last_fetch = last_date

        if earliest_last_fetch:
            start_date = earliest_last_fetch + timedelta(days=1)
            logger.info(f"Incremental mode: fetching from {start_date.strftime('%Y-%m-%d')}")
        else:
            start_date = end_date - timedelta(days=lookback_days)
            logger.info(f"Incremental mode: no previous fetch, using {lookback_days}-day lookback")
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        # Default: last 7 days
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=7)

    # Get secrets mode from environment (default: plain)
    secrets_mode = os.getenv("SECRETS_MODE", "plain")

    # Create fetcher
    fetcher = MacroDataFetcher(
        output_dir=output_dir,
        retry_limit=settings.macro_retry_limit,
        retry_backoff_seconds=settings.macro_retry_backoff_seconds,
        dryrun=args.dryrun,
        force=args.force,
        secrets_mode=secrets_mode,
    )

    # Fetch all data
    try:
        summary = fetcher.fetch_all(indicators, start_date, end_date)

        # Print summary
        print("\n" + "=" * 70)
        print("MACRO DATA FETCH SUMMARY")
        print("=" * 70)
        print(f"Indicators: {', '.join(summary['indicators'])}")
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
        print(f"Total Data Points: {summary['stats']['total_data_points']}")
        print("=" * 70)

        # Update state if incremental mode
        if args.incremental and state_manager:
            for indicator in indicators:
                state_manager.record_market_data_fetch(
                    data_type="macro",
                    symbol=indicator,
                    timestamp=end_date.isoformat(),
                    stats={
                        "fetched": summary["stats"]["fetched"],
                        "cached": summary["stats"]["cached"],
                        "failed": summary["stats"]["failed"],
                        "data_points": summary["stats"]["total_data_points"],
                    },
                )

            logger.info(f"Updated state file: last fetch date = {end_date.strftime('%Y-%m-%d')}")

        # Return exit code
        if summary["stats"]["failed"] > 0:
            logger.warning(f"{summary['stats']['failed']} macro data fetches failed")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Macro data fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
