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
from tqdm import tqdm

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
            "warnings": 0,  # US-028 Phase 6j: Track data quality warnings
            "duplicates_removed": 0,  # US-028 Phase 7 Initiative 1: Deduplication tracking
            "gaps_detected": 0,  # US-028 Phase 7 Initiative 1: Gap detection tracking
        }

        # US-028 Phase 7 Initiative 1: Rate limiting tracking
        self.request_times: list[float] = []  # Track request timestamps for rate limiting
        self.fetch_log_path = Path("data/historical/metadata/fetch_log.jsonl")
        self.fetch_log_path.parent.mkdir(parents=True, exist_ok=True)

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

    def detect_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[tuple[str, str]]:
        """Detect gaps in fetched data (US-028 Phase 7 Initiative 1).

        Compares expected trading days against actual data to identify missing dates.

        Args:
            df: DataFrame with fetched data
            symbol: Stock symbol
            start_dt: Expected start date
            end_dt: Expected end date

        Returns:
            List of gap tuples (start_date, end_date) as ISO format strings
        """
        if df is None or len(df) == 0:
            logger.warning(f"No data to check for gaps: {symbol}")
            return []

        # Generate expected trading days (all days, assuming no filtering)
        # In production, this would use a trading calendar (NSE holidays, etc.)
        expected_dates = pd.date_range(start_dt, end_dt, freq="D")

        # Get actual dates from data
        if "timestamp" in df.columns:
            actual_dates = pd.to_datetime(df["timestamp"]).dt.date
        elif "datetime" in df.columns:
            actual_dates = pd.to_datetime(df["datetime"]).dt.date
        else:
            logger.warning(f"No timestamp column found in data for {symbol}")
            return []

        actual_dates_set = set(actual_dates.unique())
        expected_dates_set = set(expected_dates.date)

        # Find missing dates
        missing_dates = sorted(expected_dates_set - actual_dates_set)

        if not missing_dates:
            return []

        # Group consecutive missing dates into gap ranges
        gaps = []
        gap_start = missing_dates[0]
        gap_end = missing_dates[0]

        for i in range(1, len(missing_dates)):
            current = missing_dates[i]
            prev = missing_dates[i - 1]

            # Check if consecutive (allowing for 1-day gap which might be weekend)
            if (current - prev).days <= 3:
                gap_end = current
            else:
                # Save current gap and start new one
                gaps.append((gap_start.isoformat(), gap_end.isoformat()))
                gap_start = current
                gap_end = current

        # Add final gap
        gaps.append((gap_start.isoformat(), gap_end.isoformat()))

        if gaps:
            logger.warning(
                f"Detected {len(gaps)} gap(s) in {symbol} data: {gaps}",
                extra={"component": "gap_detection", "gaps": len(gaps)},
            )
            self.stats["gaps_detected"] += len(gaps)

        return gaps

    def enforce_rate_limit(self) -> None:
        """Enforce rate limiting based on requests per minute (US-028 Phase 7 Initiative 1).

        Tracks request timestamps and sleeps if rate limit would be exceeded.
        """
        import time

        now = time.time()
        rate_limit = self.settings.breeze_rate_limit_requests_per_minute
        window_seconds = 60.0

        # Remove timestamps older than 1 minute
        self.request_times = [t for t in self.request_times if (now - t) < window_seconds]

        # Check if we've hit the rate limit
        if len(self.request_times) >= rate_limit:
            # Calculate how long to sleep
            oldest_request = self.request_times[0]
            sleep_time = window_seconds - (now - oldest_request) + 0.1  # Add small buffer
            if sleep_time > 0:
                logger.info(
                    f"  ⏱ Rate limit reached ({len(self.request_times)}/{rate_limit} req/min), "
                    f"sleeping {sleep_time:.1f}s",
                    extra={"component": "rate_limiter", "throttled": True},
                )
                time.sleep(sleep_time)

        # Record this request
        self.request_times.append(time.time())

    def log_fetch_entry(
        self,
        symbol: str,
        interval: str,
        chunk_start: str,
        chunk_end: str,
        rows_fetched: int,
        source: str,
        status: str,
        retries: int = 0,
        warnings: int = 0,
        error: str | None = None,
    ) -> None:
        """Append fetch entry to fetch_log.jsonl (US-028 Phase 7 Initiative 1).

        Args:
            symbol: Stock symbol
            interval: Time interval
            chunk_start: Chunk start date (YYYY-MM-DD)
            chunk_end: Chunk end date (YYYY-MM-DD)
            rows_fetched: Number of rows fetched
            source: Data source ("cache" or "api")
            status: Fetch status ("success", "failed", "cached")
            retries: Number of retries attempted
            warnings: Number of data quality warnings
            error: Error message if failed
        """
        import json

        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "rows_fetched": rows_fetched,
            "source": source,
            "status": status,
            "retries": retries,
            "warnings": warnings,
        }

        if error:
            entry["error"] = error

        # Atomic append write
        with open(self.fetch_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

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
            # US-028 Phase 6i: Parse timestamp column to avoid type comparison errors
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            return len(df) > 0
        except Exception as e:
            logger.warning(f"Invalid cache file {cache_path}: {e}")
            return False

    def validate_ohlcv_data(
        self, df: pd.DataFrame, symbol: str, date: str
    ) -> tuple[bool, str | None, pd.DataFrame, list[str]]:
        """Validate OHLCV DataFrame and apply corrections for non-critical issues.

        US-028 Phase 6j: Refactored to treat data quality anomalies as warnings
        instead of fatal errors. Corrects negative volumes while preserving data.

        Args:
            df: DataFrame to validate
            symbol: Stock symbol (for logging)
            date: Date string (for logging)

        Returns:
            Tuple of (is_valid, error_message, corrected_df, warnings)
            - is_valid: False only for hard errors (missing columns, empty data, unsorted timestamps)
            - error_message: Critical error message if is_valid is False
            - corrected_df: DataFrame with corrections applied (e.g., negative volumes clipped)
            - warnings: List of warning messages for non-critical issues
        """
        warnings = []
        corrected_df = df.copy()

        # Check required columns (HARD ERROR)
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in corrected_df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}", corrected_df, warnings

        # Check non-empty (HARD ERROR)
        if len(corrected_df) == 0:
            return False, "Empty DataFrame", corrected_df, warnings

        # Check for null values (HARD ERROR - cannot safely correct)
        if corrected_df[required_cols].isnull().any().any():
            null_counts = corrected_df[required_cols].isnull().sum()
            null_cols = [col for col in null_counts.index if null_counts[col] > 0]
            return False, f"Contains null values in columns: {null_cols}", corrected_df, warnings

        # Validate OHLC relationships (WARNING - log but don't fail)
        invalid_ohlc = (
            (corrected_df["high"] < corrected_df["open"])
            | (corrected_df["high"] < corrected_df["close"])
            | (corrected_df["low"] > corrected_df["open"])
            | (corrected_df["low"] > corrected_df["close"])
        )
        if invalid_ohlc.any():
            count = invalid_ohlc.sum()
            warnings.append(f"Invalid OHLC relationships in {count} rows (retained)")
            logger.warning(
                f"Data quality issue: {symbol} {date} has {count} invalid OHLC rows",
                extra={"component": "validation", "symbol": symbol, "date": date, "issue": "invalid_ohlc"},
            )

        # Validate volume non-negative (WARNING - clip negatives to zero)
        negative_volume = corrected_df["volume"] < 0
        if negative_volume.any():
            count = negative_volume.sum()
            corrected_df.loc[negative_volume, "volume"] = 0
            warnings.append(f"Negative volume in {count} rows (clipped to 0)")
            logger.warning(
                f"Data quality issue: {symbol} {date} has {count} negative volume rows, clipped to 0",
                extra={"component": "validation", "symbol": symbol, "date": date, "issue": "negative_volume", "corrected_rows": count},
            )

        # Validate timestamp ordering (HARD ERROR)
        try:
            timestamps = pd.to_datetime(corrected_df["timestamp"])
            if not timestamps.is_monotonic_increasing:
                return False, "Timestamps not in ascending order", corrected_df, warnings
        except Exception as e:
            return False, f"Invalid timestamps: {e}", corrected_df, warnings

        return True, None, corrected_df, warnings

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
        """Save DataFrame to cache file with deduplication (US-028 Phase 7 Initiative 1).

        Args:
            df: DataFrame to save
            symbol: Stock symbol
            interval: Time interval
            date: Date string
        """
        cache_path = self.get_cache_path(symbol, interval, date)

        # Create directory
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # US-028 Phase 7 Initiative 1: Deduplication logic
        # If file exists, merge with existing data and remove duplicates
        existing_df = None
        if cache_path.exists():
            try:
                cache_path.chmod(0o644)
                # Load existing data
                existing_df = pd.read_csv(cache_path)
                logger.debug(f"Loaded {len(existing_df)} existing rows from {cache_path}")
            except Exception as e:
                logger.warning(f"Could not load existing cache file {cache_path}: {e}")
                existing_df = None

        # Merge with existing data if present
        if existing_df is not None and len(existing_df) > 0:
            # Make copies to avoid SettingWithCopyWarning
            existing_df = existing_df.copy()
            df = df.copy()

            # Normalize datetime columns to same type before concatenating
            for col in ["datetime", "timestamp"]:
                if col in existing_df.columns and col in df.columns:
                    # Convert both to datetime
                    existing_df[col] = pd.to_datetime(existing_df[col])
                    df[col] = pd.to_datetime(df[col])

            # Concatenate new and existing data
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            original_count = len(combined_df)

            # Remove duplicates based on timestamp (keep last occurrence to allow corrections)
            if "datetime" in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=["datetime"], keep="last")
            elif "timestamp" in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")

            duplicates_removed = original_count - len(combined_df)

            if duplicates_removed > 0:
                logger.warning(
                    f"Removed {duplicates_removed} duplicate rows for {symbol} {date}",
                    extra={"component": "deduplication", "duplicates": duplicates_removed},
                )
                self.stats["duplicates_removed"] += duplicates_removed

            # Sort by timestamp
            if "datetime" in combined_df.columns:
                combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
            elif "timestamp" in combined_df.columns:
                combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

            df = combined_df

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
            # US-028 Phase 7 Initiative 1: Log cache hit
            self.log_fetch_entry(
                symbol=symbol,
                interval=interval,
                chunk_start=date,
                chunk_end=date,
                rows_fetched=0,
                source="cache",
                status="cached",
            )
            return True

        # US-028 Phase 7 Initiative 1: Enforce rate limiting before API call
        self.enforce_rate_limit()

        # Fetch from API
        try:
            df = self.fetch_with_retry(symbol, date, interval)

            if df is None or len(df) == 0:
                logger.warning(f"No data returned for {symbol} {date} {interval}")
                self.stats["failures"] += 1
                # US-028 Phase 7 Initiative 1: Log empty result
                self.log_fetch_entry(
                    symbol=symbol,
                    interval=interval,
                    chunk_start=date,
                    chunk_end=date,
                    rows_fetched=0,
                    source="api",
                    status="failed",
                    error="No data returned",
                )
                return False

            # Validate and correct data quality issues (US-028 Phase 6j)
            is_valid, error_msg, corrected_df, warnings = self.validate_ohlcv_data(df, symbol, date)
            if not is_valid:
                logger.error(f"Validation failed for {symbol} {date}: {error_msg}")
                self.stats["failures"] += 1
                # US-028 Phase 7 Initiative 1: Log validation failure
                self.log_fetch_entry(
                    symbol=symbol,
                    interval=interval,
                    chunk_start=date,
                    chunk_end=date,
                    rows_fetched=len(df),
                    source="api",
                    status="failed",
                    error=error_msg,
                )
                return False

            # Track warnings
            warning_count = len(warnings) if warnings else 0
            if warnings:
                self.stats["warnings"] += warning_count

            # Save to cache (use corrected data)
            self.save_to_cache(corrected_df, symbol, interval, date)
            self.stats["downloads"] += 1

            # US-028 Phase 7 Initiative 1: Log successful fetch
            self.log_fetch_entry(
                symbol=symbol,
                interval=interval,
                chunk_start=date,
                chunk_end=date,
                rows_fetched=len(corrected_df),
                source="api",
                status="success",
                warnings=warning_count,
            )
            return True

        except tenacity.RetryError as e:
            logger.error(f"All retries exhausted for {symbol} {date}: {e}")
            self.stats["failures"] += 1
            # US-028 Phase 7 Initiative 1: Log retry exhaustion
            self.log_fetch_entry(
                symbol=symbol,
                interval=interval,
                chunk_start=date,
                chunk_end=date,
                rows_fetched=0,
                source="api",
                status="failed",
                retries=self.settings.breeze_max_retries,
                error=str(e),
            )
            return False
        except Exception as e:
            logger.error(f"Unexpected error for {symbol} {date}: {e}")
            self.stats["failures"] += 1
            # US-028 Phase 7 Initiative 1: Log unexpected error
            self.log_fetch_entry(
                symbol=symbol,
                interval=interval,
                chunk_start=date,
                chunk_end=date,
                rows_fetched=0,
                source="api",
                status="failed",
                error=str(e),
            )
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
                chunk_data = []
                for date_str in chunk_dates:
                    cache_path = self.get_cache_path(symbol, interval, date_str)
                    if cache_path.exists():
                        try:
                            # US-028 Phase 6i: Parse timestamp column to avoid type comparison errors
                            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
                            if len(df) > 0:
                                chunk_data.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to load cache {cache_path}: {e}")

                # US-028 Phase 6j: Validate and correct cached data
                if chunk_data:
                    combined_chunk = pd.concat(chunk_data, ignore_index=True)
                    is_valid, error_msg, corrected_chunk, warnings = self.validate_ohlcv_data(
                        combined_chunk, symbol, str(chunk_start.date())
                    )
                    if not is_valid:
                        logger.error(f"Cached chunk {i}/{len(chunks)} invalid: {error_msg}")
                        failed_chunks.append((chunk_start, chunk_end))
                        self.stats["chunks_failed"] += 1
                        continue

                    # Track warnings
                    if warnings:
                        self.stats["warnings"] += len(warnings)
                        logger.info(f"Cached chunk {i}/{len(chunks)} warnings: {', '.join(warnings)}")
                        # Re-save corrected data to cache
                        for date_str in corrected_chunk["timestamp"].dt.strftime("%Y-%m-%d").unique():
                            df_day = corrected_chunk[corrected_chunk["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
                            self.save_to_cache(df_day, symbol, interval, date_str)

                    all_data.append(corrected_chunk)
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

                # Validate chunk and correct data quality issues (US-028 Phase 6j)
                is_valid, error_msg, corrected_chunk, warnings = self.validate_ohlcv_data(
                    df_chunk, symbol, str(chunk_start.date())
                )
                if not is_valid:
                    logger.error(f"Validation failed for chunk {i}/{len(chunks)}: {error_msg}")
                    failed_chunks.append((chunk_start, chunk_end))
                    self.stats["chunks_failed"] += 1
                    continue

                # Track warnings
                if warnings:
                    self.stats["warnings"] += len(warnings)
                    logger.info(f"Chunk {i}/{len(chunks)} warnings: {', '.join(warnings)}")

                # Save chunk to cache (split by day for consistency with existing cache structure)
                # Use corrected data with fixes applied
                corrected_chunk["timestamp"] = pd.to_datetime(corrected_chunk["timestamp"])
                for date_str in corrected_chunk["timestamp"].dt.strftime("%Y-%m-%d").unique():
                    df_day = corrected_chunk[corrected_chunk["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
                    self.save_to_cache(df_day, symbol, interval, date_str)

                all_data.append(corrected_chunk)
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

        # US-028 Phase 6i: Ensure timestamp column is datetime type (normalize cached vs API data)
        try:
            if "timestamp" in combined_df.columns:
                # Convert to datetime if not already (handles string timestamps from cache)
                if not pd.api.types.is_datetime64_any_dtype(combined_df["timestamp"]):
                    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], utc=True)
                    logger.debug(f"Converted timestamp column to datetime for {symbol} {interval}")
        except Exception as e:
            error_msg = f"Failed to normalize timestamp column for {symbol} {interval}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Remove duplicates (in case of overlapping chunks)
        combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="first")

        # Sort by timestamp (now safe since all timestamps are same type)
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"✓ Combined {len(all_data)} chunk(s) into {len(combined_df)} rows for {symbol} {interval}"
        )

        # US-028 Phase 7 Initiative 1: Detect gaps in fetched data
        gaps = self.detect_gaps(combined_df, symbol, start_dt, end_dt)
        if gaps:
            logger.warning(
                f"Gap detection summary for {symbol}: {len(gaps)} gap(s) found",
                extra={"symbol": symbol, "gaps": gaps},
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

        # US-028 Phase 7 Initiative 4: Progress monitoring with tqdm
        total_tasks = len(symbols) * len(intervals)
        with tqdm(total=total_tasks, desc="Fetching historical data", unit="symbol-interval") as pbar:
            for symbol in symbols:
                for interval in intervals:
                    try:
                        df = self.fetch_symbol_date_range_chunked(
                            symbol, start_dt, end_dt, interval, force=force
                        )

                        if df is None or len(df) == 0:
                            logger.warning(f"No data fetched for {symbol} {interval}")
                            self.stats["failures"] += 1
                            pbar.set_postfix({
                                "symbol": symbol,
                                "status": "no_data",
                                "cached": self.stats["cached_hits"],
                                "fetched": self.stats["downloads"],
                            })
                        else:
                            pbar.set_postfix({
                                "symbol": symbol,
                                "status": "ok",
                                "rows": len(df),
                                "cached": self.stats["cached_hits"],
                                "fetched": self.stats["downloads"],
                            })

                    except Exception as e:
                        # US-028 Phase 6i: Include exception type for better debugging
                        logger.error(f"Failed to fetch {symbol} {interval}: {type(e).__name__}: {e}")
                        self.stats["failures"] += 1
                        pbar.set_postfix({
                            "symbol": symbol,
                            "status": "error",
                            "error": type(e).__name__,
                        })

                    # US-028 Phase 7 Initiative 4: Update progress bar
                    pbar.update(1)

        # Generate summary
        summary = {
            "total_requests": self.stats["total_requests"],
            "cached_hits": self.stats["cached_hits"],
            "downloads": self.stats["downloads"],
            "failures": self.stats["failures"],
            "total_rows": self.stats["total_rows"],
            "chunks_fetched": self.stats["chunks_fetched"],
            "chunks_failed": self.stats["chunks_failed"],
            "warnings": self.stats["warnings"],  # US-028 Phase 6j: Data quality warnings
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
        "--symbols-mode",
        type=str,
        choices=["pilot", "nifty100", "metals_etfs", "all"],
        help="Load symbols from metadata (US-028 Phase 7 Initiative 1): pilot (5 symbols), nifty100 (100), metals_etfs (2), all (102)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        help="Limit number of symbols from --symbols-mode (e.g., --symbols-mode nifty100 --max-symbols 20)",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Path to text file with one symbol per line (alternative to --symbols or --symbols-mode)",
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

    # US-028 Phase 7 Initiative 1: Load symbols from metadata if symbols_mode specified
    # US-028 Phase 7 CLI Hardening: Support --symbols-file and --max-symbols
    if args.symbols_file:
        # Load symbols from file (one symbol per line)
        symbols_file_path = Path(args.symbols_file)
        if not symbols_file_path.exists():
            logger.error(f"Symbols file not found: {symbols_file_path}")
            return 1

        try:
            with open(symbols_file_path) as f:
                symbols = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            logger.info(f"Loaded {len(symbols)} symbols from {symbols_file_path}")
            logger.info(f"  → Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        except Exception as e:
            logger.error(f"Failed to read symbols file {symbols_file_path}: {e}")
            return 1
    elif args.symbols_mode:
        logger.info(f"Loading symbols from metadata (mode={args.symbols_mode})")
        symbols = settings.get_symbols_for_mode(args.symbols_mode)
        logger.info(f"  → Loaded {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    else:
        # Use command-line args or settings defaults
        symbols = args.symbols if args.symbols else settings.historical_data_symbols

    # US-028 Phase 7 CLI Hardening: Apply --max-symbols limit
    if args.max_symbols and args.max_symbols > 0:
        original_count = len(symbols)
        symbols = symbols[:args.max_symbols]
        logger.info(f"Applied --max-symbols limit: {original_count} → {len(symbols)} symbols")
        logger.info(f"  → Limited to: {', '.join(symbols)}")

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
    if summary.get("warnings", 0) > 0:
        logger.warning(f"Data quality warnings: {summary['warnings']} (see logs for details)")  # US-028 Phase 6j
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
