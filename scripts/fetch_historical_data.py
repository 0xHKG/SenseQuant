"""Fetch historical OHLCV data for configured symbols and date ranges (US-024).

This script downloads historical market data from Breeze API and stores it as CSV files
under data/historical/<symbol>/<interval>/YYYY-MM-DD.csv. It includes:

- Caching to avoid re-downloading existing data
- Exponential backoff retry logic for API failures
- Data validation (OHLC relationships, volume, timestamps)
- Dryrun mode for testing without network calls
- Progress logging and summary reports

Usage:
    # Download data for default symbols
    python scripts/fetch_historical_data.py

    # Download specific symbols and date range
    python scripts/fetch_historical_data.py \\
        --symbols RELIANCE TCS INFY \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31 \\
        --intervals 1minute 5minute 1day

    # Dryrun mode (no network calls)
    python scripts/fetch_historical_data.py --dryrun

    # Force re-download (ignore cache)
    python scripts/fetch_historical_data.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import tenacity
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.breeze_client import BreezeClient
from src.app.config import Settings
from src.services.state_manager import StateManager


class HistoricalDataFetcher:
    """Fetches and caches historical OHLCV data."""

    def __init__(
        self,
        settings: Settings,
        breeze_client: BreezeClient | None = None,
        dryrun: bool = False,
    ):
        """Initialize fetcher.

        Args:
            settings: Application settings
            breeze_client: Optional BreezeClient instance (will create if None)
            dryrun: If True, skip actual network calls
        """
        self.settings = settings
        self.breeze_client = breeze_client
        self.dryrun = dryrun
        self.output_dir = Path(settings.historical_data_output_dir)

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cached_hits": 0,
            "downloads": 0,
            "failures": 0,
            "total_rows": 0,
            "chunks_fetched": 0,
            "chunks_failed": 0,
        }

    def validate_date_range(self, start_date: str, end_date: str) -> tuple[datetime, datetime]:
        """Validate and parse date range.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            Tuple of (start_datetime, end_datetime)

        Raises:
            ValueError: If dates are invalid or start >= end
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {e}") from e

        if start_dt >= end_dt:
            raise ValueError(f"Start date {start_date} must be before end date {end_date}")

        # Reasonable bounds check (not more than 5 years)
        if (end_dt - start_dt).days > 1825:
            logger.warning("Date range exceeds 5 years - this may take a long time")

        return start_dt, end_dt

    def generate_date_list(self, start_dt: datetime, end_dt: datetime) -> list[str]:
        """Generate list of date strings between start and end.

        Args:
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of date strings in YYYY-MM-DD format
        """
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            dates.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += timedelta(days=1)
        return dates

    def split_date_range_into_chunks(
        self, start_dt: datetime, end_dt: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Split date range into chunks based on settings.historical_chunk_days.

        Args:
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        chunk_size = self.settings.historical_chunk_days
        chunks = []

        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(current_start + timedelta(days=chunk_size - 1), end_dt)
            chunks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)

        return chunks

    def get_cache_path(self, symbol: str, interval: str, date: str) -> Path:
        """Get cache file path for a symbol/interval/date combination.

        Args:
            symbol: Stock symbol
            interval: Time interval (1minute, 5minute, 1day, etc.)
            date: Date string (YYYY-MM-DD)

        Returns:
            Path to cache file
        """
        return self.output_dir / symbol / interval / f"{date}.csv"

    def is_cached(self, symbol: str, interval: str, date: str) -> bool:
        """Check if data is already cached.

        Args:
            symbol: Stock symbol
            interval: Time interval
            date: Date string

        Returns:
            True if valid cache file exists
        """
        cache_path = self.get_cache_path(symbol, interval, date)
        if not cache_path.exists():
            return False

        # Validate cache file has content
        try:
            df = pd.read_csv(cache_path)
            return len(df) > 0
        except Exception as e:
            logger.warning(f"Invalid cache file {cache_path}: {e}")
            return False

    def validate_ohlcv_data(
        self, df: pd.DataFrame, symbol: str, date: str
    ) -> tuple[bool, str | None]:
        """Validate OHLCV DataFrame.

        Args:
            df: DataFrame to validate
            symbol: Stock symbol (for logging)
            date: Date string (for logging)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check non-empty
        if len(df) == 0:
            return False, "Empty DataFrame"

        # Check for null values
        if df[required_cols].isnull().any().any():
            return False, "Contains null values"

        # Validate OHLC relationships
        invalid_ohlc = (
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        if invalid_ohlc.any():
            return False, f"Invalid OHLC relationships in {invalid_ohlc.sum()} rows"

        # Validate volume non-negative
        if (df["volume"] < 0).any():
            return False, "Negative volume values"

        # Validate timestamp ordering
        try:
            timestamps = pd.to_datetime(df["timestamp"])
            if not timestamps.is_monotonic_increasing:
                return False, "Timestamps not in ascending order"
        except Exception as e:
            return False, f"Invalid timestamps: {e}"

        return True, None

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=30),
        retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number}/3 after error: {retry_state.outcome.exception()}"
        ),
    )
    def fetch_with_retry(self, symbol: str, date: str, interval: str) -> pd.DataFrame | None:
        """Fetch historical data with retry logic.

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)
            interval: Time interval

        Returns:
            DataFrame with OHLCV data or None if failed

        Raises:
            ConnectionError: On network failures (will trigger retry)
            TimeoutError: On timeout (will trigger retry)
        """
        if self.dryrun:
            logger.info(f"[DRYRUN] Would fetch {symbol} {date} {interval}")
            # Return mock data in dryrun mode
            return pd.DataFrame(
                {
                    "timestamp": [f"{date}T09:15:00+05:30"],
                    "open": [2450.0],
                    "high": [2455.0],
                    "low": [2448.0],
                    "close": [2453.0],
                    "volume": [100000],
                }
            )

        if self.breeze_client is None:
            raise ValueError("BreezeClient not initialized (required for non-dryrun mode)")

        try:
            # Parse date for API call
            date_obj = datetime.strptime(date, "%Y-%m-%d")

            # Fetch from API
            df = self.breeze_client.get_historical(
                symbol=symbol,
                from_date=date_obj,
                to_date=date_obj,
                interval=interval,
            )

            return df

        except ConnectionError as e:
            logger.error(f"Connection error for {symbol} {date}: {e}")
            raise
        except TimeoutError as e:
            logger.error(f"Timeout for {symbol} {date}: {e}")
            raise
        except Exception as e:
            # Non-retryable errors
            logger.error(f"Failed to fetch {symbol} {date}: {e}")
            return None

    def save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str, date: str) -> None:
        """Save DataFrame to cache file.

        Args:
            df: DataFrame to save
            symbol: Stock symbol
            interval: Time interval
            date: Date string
        """
        cache_path = self.get_cache_path(symbol, interval, date)

        # Create directory
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # If file exists and is read-only, make it writable temporarily
        if cache_path.exists():
            try:
                cache_path.chmod(0o644)
            except Exception:
                pass  # Ignore permission errors

        # Save CSV
        df.to_csv(cache_path, index=False)

        # Make read-only to prevent accidental edits
        try:
            cache_path.chmod(0o444)
        except Exception:
            pass  # Ignore permission errors (e.g., in test environments)

        logger.info(f"Saved {len(df)} rows to {cache_path}")
        self.stats["total_rows"] += len(df)

    def fetch_symbol_date(
        self,
        symbol: str,
        date: str,
        interval: str,
        force: bool = False,
    ) -> bool:
        """Fetch data for a single symbol/date/interval combination.

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)
            interval: Time interval
            force: If True, ignore cache and re-download

        Returns:
            True if successful, False if failed
        """
        self.stats["total_requests"] += 1

        # Check cache
        if not force and self.is_cached(symbol, interval, date):
            logger.debug(f"Cache hit: {symbol} {date} {interval}")
            self.stats["cached_hits"] += 1
            return True

        # Fetch from API
        try:
            df = self.fetch_with_retry(symbol, date, interval)

            if df is None or len(df) == 0:
                logger.warning(f"No data returned for {symbol} {date} {interval}")
                self.stats["failures"] += 1
                return False

            # Validate
            is_valid, error_msg = self.validate_ohlcv_data(df, symbol, date)
            if not is_valid:
                logger.error(f"Validation failed for {symbol} {date}: {error_msg}")
                self.stats["failures"] += 1
                return False

            # Save to cache
            self.save_to_cache(df, symbol, interval, date)
            self.stats["downloads"] += 1
            return True

        except tenacity.RetryError as e:
            logger.error(f"All retries exhausted for {symbol} {date}: {e}")
            self.stats["failures"] += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error for {symbol} {date}: {e}")
            self.stats["failures"] += 1
            return False

    def fetch_symbol_date_range_chunked(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        interval: str,
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch data for a symbol/date-range using chunked ingestion.

        This method replaces day-by-day fetching with chunked API calls to improve
        performance and respect API rate limits. It uses BreezeClient.fetch_historical_chunk()
        which leverages the v2 API.

        Args:
            symbol: Stock symbol
            start_dt: Start datetime
            end_dt: End datetime
            interval: Time interval
            force: If True, ignore cache and re-download all chunks

        Returns:
            Combined DataFrame with all data from the date range

        Raises:
            RuntimeError: If any chunk fails to fetch when live data should exist
        """
        # Split into chunks
        chunks = self.split_date_range_into_chunks(start_dt, end_dt)

        logger.info(
            f"Fetching {symbol} {interval} from {start_dt.date()} to {end_dt.date()} "
            f"in {len(chunks)} chunk(s) (chunk_size={self.settings.historical_chunk_days} days)"
        )

        all_data = []
        failed_chunks = []

        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            # Check if we need to respect rate limits (not first chunk)
            if i > 1:
                delay = self.settings.breeze_rate_limit_delay_seconds
                logger.debug(f"Rate limiting: sleeping {delay}s before next chunk")
                time.sleep(delay)

            # Check cache for this chunk (simplified: check if any day in chunk is cached)
            chunk_cached = False
            if not force:
                # For simplicity, check if the middle date of the chunk is cached
                # (A more sophisticated approach would check all dates, but that defeats
                # the purpose of chunking - we want to reduce API calls)
                mid_date = chunk_start + (chunk_end - chunk_start) / 2
                mid_date_str = mid_date.strftime("%Y-%m-%d")
                if self.is_cached(symbol, interval, mid_date_str):
                    logger.debug(
                        f"Chunk {i}/{len(chunks)} partially cached: {chunk_start.date()} to {chunk_end.date()}"
                    )
                    chunk_cached = True
                    self.stats["cached_hits"] += 1

            if chunk_cached and not force:
                # Load from cache (load all days in chunk)
                logger.debug(f"Loading chunk {i}/{len(chunks)} from cache")
                chunk_dates = self.generate_date_list(chunk_start, chunk_end)
                for date_str in chunk_dates:
                    cache_path = self.get_cache_path(symbol, interval, date_str)
                    if cache_path.exists():
                        try:
                            df = pd.read_csv(cache_path)
                            if len(df) > 0:
                                all_data.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to load cache {cache_path}: {e}")
                continue

            # Fetch chunk from API
            try:
                logger.info(
                    f"Fetching chunk {i}/{len(chunks)}: {symbol} {chunk_start.date()} to {chunk_end.date()}"
                )

                if self.dryrun:
                    # Generate mock data for dryrun
                    mock_dates = pd.date_range(chunk_start, chunk_end, freq="D")
                    df_chunk = pd.DataFrame(
                        {
                            "timestamp": mock_dates,
                            "open": [2450.0] * len(mock_dates),
                            "high": [2455.0] * len(mock_dates),
                            "low": [2448.0] * len(mock_dates),
                            "close": [2453.0] * len(mock_dates),
                            "volume": [100000] * len(mock_dates),
                        }
                    )
                else:
                    # Use BreezeClient's chunked fetch method
                    if self.breeze_client is None:
                        raise ValueError(
                            "BreezeClient not initialized (required for non-dryrun mode)"
                        )

                    # Convert to timezone-aware timestamps for Breeze API
                    chunk_start_tz = pd.Timestamp(chunk_start, tz="UTC")
                    chunk_end_tz = pd.Timestamp(chunk_end, tz="UTC")

                    df_chunk = self.breeze_client.fetch_historical_chunk(
                        symbol=symbol,
                        start_date=chunk_start_tz,
                        end_date=chunk_end_tz,
                        interval=interval,
                    )

                if df_chunk is None or len(df_chunk) == 0:
                    logger.warning(
                        f"No data returned for chunk {i}/{len(chunks)}: "
                        f"{symbol} {chunk_start.date()} to {chunk_end.date()}"
                    )
                    failed_chunks.append((chunk_start, chunk_end))
                    self.stats["chunks_failed"] += 1
                    continue

                # Validate chunk
                is_valid, error_msg = self.validate_ohlcv_data(
                    df_chunk, symbol, str(chunk_start.date())
                )
                if not is_valid:
                    logger.error(f"Validation failed for chunk {i}/{len(chunks)}: {error_msg}")
                    failed_chunks.append((chunk_start, chunk_end))
                    self.stats["chunks_failed"] += 1
                    continue

                # Save chunk to cache (split by day for consistency with existing cache structure)
                df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"])
                for date_str in df_chunk["timestamp"].dt.strftime("%Y-%m-%d").unique():
                    df_day = df_chunk[df_chunk["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
                    self.save_to_cache(df_day, symbol, interval, date_str)

                all_data.append(df_chunk)
                self.stats["chunks_fetched"] += 1
                self.stats["downloads"] += 1

                logger.info(
                    f"✓ Chunk {i}/{len(chunks)} fetched: {len(df_chunk)} rows "
                    f"({chunk_start.date()} to {chunk_end.date()})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to fetch chunk {i}/{len(chunks)} "
                    f"({chunk_start.date()} to {chunk_end.date()}): {e}"
                )
                failed_chunks.append((chunk_start, chunk_end))
                self.stats["chunks_failed"] += 1
                self.stats["failures"] += 1

        # Raise error if any chunks failed (when live data should exist)
        if failed_chunks and not self.dryrun:
            error_msg = (
                f"Failed to fetch {len(failed_chunks)} chunk(s) for {symbol} {interval}: "
                f"{[(c[0].date(), c[1].date()) for c in failed_chunks]}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Combine all chunks
        if not all_data:
            logger.warning(f"No data fetched for {symbol} {interval} in date range")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates (in case of overlapping chunks)
        combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="first")

        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"✓ Combined {len(all_data)} chunk(s) into {len(combined_df)} rows for {symbol} {interval}"
        )

        return combined_df

    def fetch_all(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        intervals: list[str],
        force: bool = False,
    ) -> dict[str, Any]:
        """Fetch data for all symbols/dates/intervals using chunked ingestion.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            intervals: List of time intervals
            force: If True, ignore cache and re-download

        Returns:
            Summary statistics dictionary
        """
        # Validate date range
        start_dt, end_dt = self.validate_date_range(start_date, end_date)

        # Calculate number of chunks
        chunks = self.split_date_range_into_chunks(start_dt, end_dt)
        num_chunks = len(chunks)

        logger.info(
            f"Fetching data for {len(symbols)} symbols, "
            f"{(end_dt - start_dt).days + 1} days, {len(intervals)} intervals"
        )
        logger.info(
            f"Using chunked ingestion: {num_chunks} chunk(s) per symbol/interval "
            f"(chunk_size={self.settings.historical_chunk_days} days)"
        )
        logger.info(
            f"Total API requests: ~{len(symbols) * len(intervals) * num_chunks} "
            f"(vs {len(symbols) * len(intervals) * (end_dt - start_dt).days} without chunking)"
        )

        # Fetch data using chunked ingestion
        for symbol in symbols:
            for interval in intervals:
                try:
                    df = self.fetch_symbol_date_range_chunked(
                        symbol, start_dt, end_dt, interval, force=force
                    )

                    if df is None or len(df) == 0:
                        logger.warning(f"No data fetched for {symbol} {interval}")
                        self.stats["failures"] += 1

                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} {interval}: {e}")
                    self.stats["failures"] += 1

        # Generate summary
        summary = {
            "total_requests": self.stats["total_requests"],
            "cached_hits": self.stats["cached_hits"],
            "downloads": self.stats["downloads"],
            "failures": self.stats["failures"],
            "total_rows": self.stats["total_rows"],
            "chunks_fetched": self.stats["chunks_fetched"],
            "chunks_failed": self.stats["chunks_failed"],
            "cache_hit_rate": (
                self.stats["cached_hits"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0
                else 0.0
            ),
        }

        return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV data for configured symbols"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Stock symbols to download (default: from settings)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD, default: from settings)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD, default: from settings)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        help="Time intervals to download (default: from settings)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download (ignore cache)",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dryrun mode (no network calls, use mocks)",
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

    # Load settings
    settings = Settings()  # type: ignore[call-arg]

    # Use command-line args or settings defaults
    symbols = args.symbols if args.symbols else settings.historical_data_symbols
    intervals = args.intervals if args.intervals else settings.historical_data_intervals

    # US-024 Phase 4: Incremental mode support
    state_manager = None
    if args.incremental:
        state_file = Path("data/state/historical_fetch.json")
        state_manager = StateManager(state_file)

        # Get lookback days
        lookback_days = (
            args.lookback_days if args.lookback_days else settings.incremental_lookback_days
        )

        # Calculate date range based on last fetch or lookback
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Find earliest last fetch date across all symbols
        earliest_last_fetch = None
        for symbol in symbols:
            last_fetch = state_manager.get_last_fetch_date(symbol)
            if last_fetch:
                if earliest_last_fetch is None or last_fetch < earliest_last_fetch:
                    earliest_last_fetch = last_fetch

        if earliest_last_fetch:
            # Start from day after last fetch
            start_date = (earliest_last_fetch + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(
                f"Incremental mode: fetching from {start_date} (last fetch: {earliest_last_fetch.strftime('%Y-%m-%d')})"
            )
        else:
            # No previous fetch, use lookback
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            logger.info(f"Incremental mode: no previous fetch, using {lookback_days}-day lookback")
    else:
        # Full mode: use explicit dates or settings defaults
        start_date = args.start_date if args.start_date else settings.historical_data_start_date
        end_date = args.end_date if args.end_date else settings.historical_data_end_date

    logger.info("=" * 70)
    logger.info("HISTORICAL DATA FETCH")
    logger.info("=" * 70)
    logger.info(f"Mode: {'INCREMENTAL' if args.incremental else 'FULL'}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Output dir: {settings.historical_data_output_dir}")
    logger.info(f"Dryrun: {args.dryrun}")
    logger.info(f"Force re-download: {args.force}")
    logger.info("=" * 70)

    # Initialize BreezeClient
    # Determine if we should use dry_run mode
    use_dry_run = args.dryrun or settings.mode != "live"

    breeze_client = None
    if not args.dryrun:
        # Defensive checks: ensure credentials are present for live mode
        if not use_dry_run:
            if (
                not settings.breeze_api_key
                or not settings.breeze_api_secret
                or not settings.breeze_session_token
            ):
                logger.error(
                    "Missing required Breeze API credentials. Required: BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN"
                )
                raise ValueError(
                    "Missing required Breeze API credentials. "
                    "Please set BREEZE_API_KEY, BREEZE_API_SECRET, and BREEZE_SESSION_TOKEN in .env file."
                )

        try:
            breeze_client = BreezeClient(
                api_key=settings.breeze_api_key,
                api_secret=settings.breeze_api_secret,
                session_token=settings.breeze_session_token,
                dry_run=use_dry_run,
            )
            logger.info(f"BreezeClient initialized (dry_run={use_dry_run})")

            # Authenticate with Breeze API
            if not use_dry_run:
                breeze_client.authenticate()
                logger.info("Breeze API session established")
        except Exception as e:
            logger.error(f"Failed to initialize BreezeClient: {e}")
            return 1

    # Create fetcher
    fetcher = HistoricalDataFetcher(settings, breeze_client, dryrun=args.dryrun)

    # Fetch data
    try:
        summary = fetcher.fetch_all(symbols, start_date, end_date, intervals, force=args.force)
    except Exception as e:
        logger.error(f"Fatal error during fetch: {e}")
        return 1

    # Print summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total requests: {summary['total_requests']}")
    logger.info(f"Cache hits: {summary['cached_hits']} ({summary['cache_hit_rate']:.1%})")
    logger.info(f"New downloads: {summary['downloads']}")
    logger.info(f"Chunks fetched: {summary['chunks_fetched']}")
    logger.info(f"Chunks failed: {summary['chunks_failed']}")
    logger.info(f"Failures: {summary['failures']}")
    logger.info(f"Total rows: {summary['total_rows']}")
    logger.info("=" * 70)

    # US-024 Phase 4: Update state after successful fetch
    if args.incremental and state_manager:
        # Update last fetch date for each symbol
        fetch_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        for symbol in symbols:
            state_manager.set_last_fetch_date(symbol, fetch_end_date)

        # Save run info
        state_manager.set_last_run_info(
            run_type="incremental",
            success=summary["failures"] == 0,
            symbols_processed=symbols,
            files_created=summary["downloads"],
            errors=summary["failures"],
        )

        logger.info(f"Updated state file: last fetch date = {end_date}")

    # Return exit code based on failures
    if summary["failures"] > 0:
        logger.warning(f"{summary['failures']} requests failed")
        return 1

    logger.info("All requests completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
