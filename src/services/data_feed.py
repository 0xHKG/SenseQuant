"""Historical data feed service with CSV, Breeze API, and hybrid support."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pytz
from loguru import logger

from src.adapters.breeze_client import BreezeClient
from src.app.config import Settings

IntervalType = Literal["1minute", "5minute", "1day"]


class DataFeed(ABC):
    """Abstract interface for historical data sources."""

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: IntervalType = "1minute",
    ) -> pd.DataFrame:
        """Fetch historical bars for symbol/interval/daterange.

        Args:
            symbol: Stock symbol
            from_date: Start date (inclusive)
            to_date: End date (inclusive)
            interval: Bar interval

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            All timestamps in IST timezone

        Raises:
            ValueError: If data unavailable or invalid parameters
        """
        pass


class CSVDataFeed(DataFeed):
    """Data feed that loads historical bars from local CSV files."""

    def __init__(self, csv_directory: str | Path):
        """Initialize CSV data feed.

        Args:
            csv_directory: Base directory containing CSV files
        """
        self.csv_directory = Path(csv_directory)
        if not self.csv_directory.exists():
            logger.warning(
                f"CSV directory does not exist: {self.csv_directory}",
                extra={"component": "data_feed", "directory": str(self.csv_directory)},
            )

    def get_historical_bars(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: IntervalType = "1minute",
    ) -> pd.DataFrame:
        """Load historical bars from CSV files.

        Args:
            symbol: Stock symbol
            from_date: Start date (inclusive)
            to_date: End date (inclusive)
            interval: Bar interval

        Returns:
            DataFrame with IST timestamps

        Raises:
            ValueError: If CSV file not found or invalid format
        """
        # Find CSV files for symbol/interval
        # Try primary structure: csv_directory/symbol/interval/*.csv
        symbol_dir = self.csv_directory / symbol / interval

        # Also try alternate structure for minute data: csv_directory/SYMBOL_1m.csv (US-018)
        alternate_file = None
        if interval in ["1minute", "5minute", "15minute"]:
            resolution_map = {"1minute": "1m", "5minute": "5m", "15minute": "15m"}
            resolution = resolution_map[interval]
            alternate_file = self.csv_directory / f"{symbol}_{resolution}.csv"

        # Load all CSV files in date range (including gzipped)
        all_bars = []

        # Try alternate single-file structure first for minute data
        if alternate_file and alternate_file.exists():
            try:
                df = self._load_csv(alternate_file)
                if not df.empty:
                    all_bars.append(df)
                logger.info(
                    f"Loaded minute bars from alternate structure: {alternate_file}",
                    extra={"component": "data_feed", "file": str(alternate_file)},
                )
            except Exception as e:
                logger.error(
                    f"Failed to load alternate CSV {alternate_file}: {e}",
                    extra={"component": "data_feed", "file": str(alternate_file), "error": str(e)},
                )

        # Try primary directory structure
        if symbol_dir.exists():
            csv_files = sorted(list(symbol_dir.glob("*.csv")) + list(symbol_dir.glob("*.csv.gz")))

            if not csv_files and not all_bars:
                # Directory exists but no CSV files
                raise ValueError(f"No CSV files found in {symbol_dir}")

            for csv_file in csv_files:
                try:
                    df = self._load_csv(csv_file)
                    if not df.empty:
                        all_bars.append(df)
                except Exception as e:
                    logger.error(
                        f"Failed to load CSV {csv_file}: {e}",
                        extra={"component": "data_feed", "file": str(csv_file), "error": str(e)},
                    )
        elif not alternate_file or not alternate_file.exists():
            # Neither directory nor alternate file exists
            raise ValueError(
                f"CSV directory not found: {symbol_dir}. "
                f"Expected structure: {self.csv_directory}/{{symbol}}/{{interval}}/"
            )

        if not all_bars:
            # Files exist but failed to load any valid data
            raise ValueError(f"Failed to load any CSV files from {symbol_dir}")

        # Concatenate and filter by date range
        bars = pd.concat(all_bars, ignore_index=True)
        bars = self._filter_date_range(bars, from_date, to_date)

        # Validate minute bars for market hours and intervals (US-018)
        bars = self._validate_minute_bars(bars, interval)

        logger.info(
            f"Loaded {len(bars)} bars from CSV",
            extra={
                "component": "data_feed",
                "symbol": symbol,
                "interval": interval,
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
                "rows": len(bars),
            },
        )

        return bars

    def _load_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load and parse a single CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with standardized columns and IST timestamps

        Raises:
            ValueError: If CSV format is invalid
        """
        # Try gzip if .gz extension
        if csv_path.suffix == ".gz":
            df = pd.read_csv(csv_path, compression="gzip")
        else:
            df = pd.read_csv(csv_path)

        if df.empty:
            return df

        # Standardize column names (case-insensitive, handle aliases)
        df.columns = df.columns.str.lower().str.strip()

        # Map common column name variations
        column_mapping = {
            "time": "timestamp",
            "datetime": "timestamp",
            "date": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vol": "volume",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # Validate required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV {csv_path.name} missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Convert timestamp to IST datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ist = pytz.timezone("Asia/Kolkata")

        # Handle timezone-naive timestamps (assume UTC)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(ist)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(ist)

        # Sort by timestamp and remove duplicates (keep last)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN in OHLCV
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        return df[required_cols]

    def _filter_date_range(
        self, df: pd.DataFrame, from_date: datetime, to_date: datetime
    ) -> pd.DataFrame:
        """Filter DataFrame by date range.

        Args:
            df: DataFrame with timestamp column
            from_date: Start date (inclusive)
            to_date: End date (inclusive)

        Returns:
            Filtered DataFrame
        """
        ist = pytz.timezone("Asia/Kolkata")

        # Ensure from_date and to_date have IST timezone
        if from_date.tzinfo is None:
            from_date = ist.localize(from_date)
        else:
            from_date = from_date.astimezone(ist)

        if to_date.tzinfo is None:
            to_date = ist.localize(to_date)
        else:
            to_date = to_date.astimezone(ist)

        # Filter
        mask = (df["timestamp"] >= from_date) & (df["timestamp"] <= to_date)
        return df[mask].reset_index(drop=True)

    def _validate_minute_bars(
        self,
        df: pd.DataFrame,
        interval: IntervalType,
        market_hours_start: str = "09:15",
        market_hours_end: str = "15:30",
    ) -> pd.DataFrame:
        """Validate minute-resolution bars for market hours and intervals.

        Args:
            df: DataFrame with timestamp column
            interval: Bar interval (e.g., "1minute", "5minute")
            market_hours_start: Market opening time (HH:MM format)
            market_hours_end: Market closing time (HH:MM format)

        Returns:
            Validated DataFrame (filtered by market hours, duplicates removed)
        """
        if interval not in ["1minute", "5minute"]:
            # Not a minute interval, skip validation
            return df

        if df.empty:
            return df

        # Filter by market hours (IST)
        start_hour, start_minute = map(int, market_hours_start.split(":"))
        end_hour, end_minute = map(int, market_hours_end.split(":"))

        def is_market_hours(ts: pd.Timestamp) -> bool:
            time_of_day = ts.time()
            market_start = pd.Timestamp(f"2000-01-01 {market_hours_start}:00").time()
            market_end = pd.Timestamp(f"2000-01-01 {market_hours_end}:00").time()
            return market_start <= time_of_day <= market_end

        # Apply market hours filter
        original_len = len(df)
        df = df[df["timestamp"].apply(is_market_hours)].copy()

        if len(df) < original_len:
            logger.debug(
                f"Filtered {original_len - len(df)} bars outside market hours",
                extra={"component": "data_feed", "filtered": original_len - len(df)},
            )

        # Validate intervals (1minute = 60s, 5minute = 300s)
        expected_interval_seconds = {"1minute": 60, "5minute": 300}.get(interval, 60)

        if len(df) > 1:
            # Check time differences
            time_diffs = df["timestamp"].diff().dt.total_seconds()
            # Allow 5-second tolerance
            invalid_intervals = (time_diffs > expected_interval_seconds + 5) | (
                time_diffs < expected_interval_seconds - 5
            )

            # Log warnings for gaps
            if invalid_intervals.any():
                gap_count = invalid_intervals.sum()
                logger.warning(
                    f"Found {gap_count} irregular intervals in minute bars",
                    extra={"component": "data_feed", "gap_count": int(gap_count)},
                )

        return df


class BreezeDataFeed(DataFeed):
    """Data feed that fetches from Breeze API with automatic CSV caching."""

    def __init__(self, breeze_client: BreezeClient, settings: Settings):
        """Initialize Breeze data feed.

        Args:
            breeze_client: Breeze API client
            settings: Application settings
        """
        self.breeze = breeze_client
        self.settings = settings
        self.cache_dir = Path(settings.data_feed_csv_directory)
        self.enable_cache = settings.data_feed_enable_cache
        self.cache_compression = settings.data_feed_cache_compression

        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_historical_bars(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: IntervalType = "1minute",
    ) -> pd.DataFrame:
        """Fetch historical bars from Breeze API.

        Args:
            symbol: Stock symbol
            from_date: Start date (inclusive)
            to_date: End date (inclusive)
            interval: Bar interval

        Returns:
            DataFrame with IST timestamps

        Raises:
            ValueError: If API call fails
        """
        logger.info(
            f"Fetching {symbol} {interval} bars from Breeze API",
            extra={
                "component": "data_feed",
                "symbol": symbol,
                "interval": interval,
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
            },
        )

        try:
            # Fetch from Breeze API
            bars = self.breeze.historical_bars(
                symbol=symbol, from_date=from_date, to_date=to_date, interval=interval
            )

            if bars.empty:
                raise ValueError(f"Breeze API returned no data for {symbol}")

            # Cache to CSV if enabled
            if self.enable_cache:
                self._cache_to_csv(bars, symbol, interval, from_date, to_date)

            logger.info(
                f"Fetched {len(bars)} bars from Breeze API",
                extra={"component": "data_feed", "symbol": symbol, "rows": len(bars)},
            )

            return bars

        except Exception as e:
            logger.error(
                f"Breeze API fetch failed: {e}",
                extra={
                    "component": "data_feed",
                    "symbol": symbol,
                    "interval": interval,
                    "error": str(e),
                },
            )
            raise ValueError(f"Failed to fetch data from Breeze API: {e}") from e

    def _cache_to_csv(
        self,
        bars: pd.DataFrame,
        symbol: str,
        interval: IntervalType,
        from_date: datetime,
        to_date: datetime,
    ) -> None:
        """Cache fetched bars to CSV.

        Args:
            bars: DataFrame to cache
            symbol: Stock symbol
            interval: Bar interval
            from_date: Start date
            to_date: End date
        """
        try:
            # Create symbol/interval directory
            cache_dir = self.cache_dir / symbol / interval
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename based on date range
            if (to_date - from_date).days > 31:
                # Multi-month: use month-year format
                filename = f"{from_date.strftime('%Y-%m')}_to_{to_date.strftime('%Y-%m')}.csv"
            elif (to_date - from_date).days > 1:
                # Multi-day: use date range
                filename = f"{from_date.strftime('%Y-%m-%d')}_to_{to_date.strftime('%Y-%m-%d')}.csv"
            else:
                # Single day
                filename = f"{from_date.strftime('%Y-%m-%d')}.csv"

            if self.cache_compression:
                filename += ".gz"

            cache_file = cache_dir / filename

            # Write CSV
            if self.cache_compression:
                bars.to_csv(cache_file, index=False, compression="gzip")
            else:
                bars.to_csv(cache_file, index=False)

            # Update metadata
            self._update_metadata(cache_dir, cache_file, bars, from_date, to_date)

            logger.info(
                f"Cached {len(bars)} bars to {cache_file}",
                extra={"component": "data_feed", "file": str(cache_file), "rows": len(bars)},
            )

        except Exception as e:
            logger.error(
                f"Failed to cache data: {e}",
                extra={"component": "data_feed", "error": str(e)},
            )

    def _update_metadata(
        self,
        cache_dir: Path,
        cache_file: Path,
        bars: pd.DataFrame,
        from_date: datetime,
        to_date: datetime,
    ) -> None:
        """Update cache metadata file.

        Args:
            cache_dir: Cache directory path
            cache_file: Path to cached CSV
            bars: Cached DataFrame
            from_date: Start date
            to_date: End date
        """
        metadata_file = cache_dir / "metadata.json"

        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {"symbol": cache_dir.parent.name, "interval": cache_dir.name, "files": {}}

        # Add/update file metadata
        metadata["files"][cache_file.name] = {
            "date_range": f"{from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}",
            "rows": len(bars),
            "start": bars["timestamp"].min().isoformat(),
            "end": bars["timestamp"].max().isoformat(),
            "fetched_at": datetime.now().isoformat(),
            "source": "breeze_api",
        }

        # Write metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


class HybridDataFeed(DataFeed):
    """Data feed that tries Breeze API first, falls back to CSV cache."""

    def __init__(self, breeze_client: BreezeClient, settings: Settings):
        """Initialize hybrid data feed.

        Args:
            breeze_client: Breeze API client
            settings: Application settings
        """
        self.breeze_feed = BreezeDataFeed(breeze_client, settings)
        self.csv_feed = CSVDataFeed(settings.data_feed_csv_directory)
        self.settings = settings

    def get_historical_bars(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: IntervalType = "1minute",
    ) -> pd.DataFrame:
        """Fetch historical bars with intelligent fallback.

        Strategy:
        1. Check CSV cache for complete date range
        2. If cache miss or partial: fetch from Breeze API (with caching)
        3. If API fails: return cached data (even if partial) with warning

        Args:
            symbol: Stock symbol
            from_date: Start date (inclusive)
            to_date: End date (inclusive)
            interval: Bar interval

        Returns:
            DataFrame with IST timestamps

        Raises:
            ValueError: If no data available from any source
        """
        logger.debug(
            f"Hybrid feed request: {symbol} {interval} {from_date} to {to_date}",
            extra={"component": "data_feed"},
        )

        # Try CSV cache first
        try:
            cached_bars = self.csv_feed.get_historical_bars(symbol, from_date, to_date, interval)

            # Check if cache covers full date range
            if not cached_bars.empty:
                cache_start = cached_bars["timestamp"].min().to_pydatetime()
                cache_end = cached_bars["timestamp"].max().to_pydatetime()

                ist = pytz.timezone("Asia/Kolkata")
                if from_date.tzinfo is None:
                    from_date = ist.localize(from_date)
                if to_date.tzinfo is None:
                    to_date = ist.localize(to_date)

                # Allow small tolerance (1 day) for cache coverage
                tolerance = timedelta(days=1)
                if cache_start <= (from_date + tolerance) and cache_end >= (to_date - tolerance):
                    logger.info(
                        f"Cache hit: {symbol} {interval} ({len(cached_bars)} bars)",
                        extra={"component": "data_feed", "source": "cache"},
                    )
                    return cached_bars

                logger.debug(
                    f"Partial cache: {cache_start} to {cache_end} (requested {from_date} to {to_date})",
                    extra={"component": "data_feed"},
                )

        except ValueError as e:
            logger.debug(f"Cache miss: {e}", extra={"component": "data_feed"})
            cached_bars = pd.DataFrame()

        # Cache miss or partial: fetch from Breeze API
        try:
            api_bars = self.breeze_feed.get_historical_bars(symbol, from_date, to_date, interval)
            logger.info(
                f"API fetch: {symbol} {interval} ({len(api_bars)} bars)",
                extra={"component": "data_feed", "source": "api"},
            )
            return api_bars

        except Exception as api_error:
            logger.warning(
                f"API fetch failed: {api_error}",
                extra={"component": "data_feed", "error": str(api_error)},
            )

            # API failed: fallback to partial cache if available
            if not cached_bars.empty:
                logger.warning(
                    f"Falling back to partial cached data: {len(cached_bars)} bars",
                    extra={"component": "data_feed", "source": "cache_fallback"},
                )
                return cached_bars

            # No data available from any source
            raise ValueError(
                f"Failed to fetch data for {symbol} {interval}: "
                f"API error ({api_error}) and no cached data available"
            ) from api_error


def create_data_feed(settings: Settings, breeze_client: BreezeClient | None = None) -> DataFeed:
    """Factory function to create appropriate DataFeed based on settings.

    Args:
        settings: Application settings
        breeze_client: Optional Breeze client (required for breeze/hybrid modes)

    Returns:
        Configured DataFeed instance

    Raises:
        ValueError: If configuration is invalid
    """
    source = settings.data_feed_source

    if source == "csv":
        return CSVDataFeed(settings.data_feed_csv_directory)

    elif source == "breeze":
        if breeze_client is None:
            raise ValueError("breeze_client required for source='breeze'")
        return BreezeDataFeed(breeze_client, settings)

    elif source == "hybrid":
        if breeze_client is None:
            raise ValueError("breeze_client required for source='hybrid'")
        return HybridDataFeed(breeze_client, settings)

    else:
        raise ValueError(
            f"Invalid data_feed_source: {source}. Must be 'csv', 'breeze', or 'hybrid'"
        )


# =====================================================================
# US-029: Market Data Loaders (Order Book, Options, Macro)
# =====================================================================


def load_order_book_snapshots(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
    base_dir: str | Path = "data/order_book",
) -> list[dict]:
    """Load cached order book snapshots from disk (US-029).

    Args:
        symbol: Stock symbol
        from_date: Start date (inclusive)
        to_date: End date (inclusive)
        base_dir: Base directory for order book data

    Returns:
        List of order book snapshot dictionaries, sorted by timestamp

    Example:
        >>> snapshots = load_order_book_snapshots("RELIANCE", from_date, to_date)
        >>> for snap in snapshots:
        ...     print(snap["timestamp"], snap["bids"][0]["price"])
    """
    base_path = Path(base_dir) / symbol

    if not base_path.exists():
        logger.warning(
            f"Order book directory not found: {base_path}",
            extra={"symbol": symbol, "base_dir": str(base_dir)},
        )
        return []

    snapshots = []

    # Iterate through date directories
    current_date = from_date
    while current_date <= to_date:
        date_str = current_date.strftime("%Y-%m-%d")
        date_dir = base_path / date_str

        if date_dir.exists() and date_dir.is_dir():
            # Load all JSON snapshots in this date directory
            for snapshot_file in sorted(date_dir.glob("*.json")):
                try:
                    with open(snapshot_file) as f:
                        snapshot = json.load(f)
                        snapshots.append(snapshot)
                except Exception as e:
                    logger.error(
                        f"Failed to load order book snapshot: {snapshot_file}: {e}",
                        extra={"file": str(snapshot_file), "error": str(e)},
                    )

        current_date += timedelta(days=1)

    logger.info(
        f"Loaded {len(snapshots)} order book snapshots",
        extra={
            "symbol": symbol,
            "from_date": from_date.strftime("%Y-%m-%d"),
            "to_date": to_date.strftime("%Y-%m-%d"),
            "count": len(snapshots),
        },
    )

    return snapshots


def load_options_chain(
    symbol: str,
    date: datetime,
    base_dir: str | Path = "data/options",
) -> dict | None:
    """Load cached options chain from disk (US-029).

    Args:
        symbol: Stock symbol (e.g., "NIFTY", "BANKNIFTY")
        date: Date for options chain
        base_dir: Base directory for options data

    Returns:
        Options chain dictionary, or None if not found

    Example:
        >>> chain = load_options_chain("NIFTY", datetime(2025, 1, 15))
        >>> if chain:
        ...     print(f"Underlying: {chain['underlying_price']}")
        ...     for option in chain['options']:
        ...         print(option['strike'], option['call']['iv'])
    """
    date_str = date.strftime("%Y-%m-%d")
    chain_file = Path(base_dir) / symbol / f"{date_str}.json"

    if not chain_file.exists():
        logger.warning(
            f"Options chain file not found: {chain_file}",
            extra={"symbol": symbol, "date": date_str, "file": str(chain_file)},
        )
        return None

    try:
        with open(chain_file) as f:
            chain = json.load(f)

        logger.info(
            f"Loaded options chain: {symbol} {date_str}",
            extra={
                "symbol": symbol,
                "date": date_str,
                "strikes": len(chain.get("options", [])),
            },
        )

        return chain

    except Exception as e:
        logger.error(
            f"Failed to load options chain: {chain_file}: {e}",
            extra={"file": str(chain_file), "error": str(e)},
        )
        return None


def load_macro_data(
    indicator: str,
    from_date: datetime,
    to_date: datetime,
    base_dir: str | Path = "data/macro",
) -> pd.DataFrame:
    """Load cached macro economic indicator data from disk (US-029).

    Args:
        indicator: Macro indicator name (e.g., "NIFTY50", "INDIAVIX", "USDINR")
        from_date: Start date (inclusive)
        to_date: End date (inclusive)
        base_dir: Base directory for macro data

    Returns:
        DataFrame with columns: date, value, change, change_pct
        Empty DataFrame if no data found

    Example:
        >>> df = load_macro_data("NIFTY50", from_date, to_date)
        >>> print(df[['date', 'value', 'change_pct']])
    """
    indicator_dir = Path(base_dir) / indicator

    if not indicator_dir.exists():
        logger.warning(
            f"Macro data directory not found: {indicator_dir}",
            extra={"indicator": indicator, "base_dir": str(base_dir)},
        )
        return pd.DataFrame()

    data_points = []

    # Iterate through date range
    current_date = from_date
    while current_date <= to_date:
        date_str = current_date.strftime("%Y-%m-%d")
        data_file = indicator_dir / f"{date_str}.json"

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                    data_points.append(
                        {
                            "date": data["date"],
                            "value": data["value"],
                            "change": data["change"],
                            "change_pct": data["change_pct"],
                        }
                    )
            except Exception as e:
                logger.error(
                    f"Failed to load macro data: {data_file}: {e}",
                    extra={"file": str(data_file), "error": str(e)},
                )

        current_date += timedelta(days=1)

    if not data_points:
        logger.warning(
            f"No macro data found for {indicator}",
            extra={
                "indicator": indicator,
                "from_date": from_date.strftime("%Y-%m-%d"),
                "to_date": to_date.strftime("%Y-%m-%d"),
            },
        )
        return pd.DataFrame()

    df = pd.DataFrame(data_points)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} macro data points",
        extra={
            "indicator": indicator,
            "from_date": from_date.strftime("%Y-%m-%d"),
            "to_date": to_date.strftime("%Y-%m-%d"),
            "count": len(df),
        },
    )

    return df


# =============================================================================
# US-029 Phase 5b: Streaming DataFeed Integration
# =============================================================================


def get_latest_order_book(
    symbol: str,
    streaming_cache_dir: str | Path = "data/order_book/streaming",
    fallback_csv_dir: str | Path | None = None,
) -> dict | None:
    """Get latest order book snapshot from streaming cache or fallback to CSV.

    This function reads the latest order book snapshot written by the streaming
    script (scripts/stream_order_book.py). If streaming is disabled or the cache
    is stale, it can optionally fall back to cached CSV snapshots.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        streaming_cache_dir: Directory where streaming script writes latest.json
        fallback_csv_dir: Optional fallback to CSV cache if streaming unavailable

    Returns:
        Order book snapshot dict with keys:
            - symbol: str
            - timestamp: str (ISO format)
            - bids: list[dict] with price, quantity, orders
            - asks: list[dict] with price, quantity, orders
            - metadata: dict with source, dryrun, etc.
        Returns None if no data available.

    Example:
        >>> snapshot = get_latest_order_book("RELIANCE")
        >>> if snapshot:
        ...     best_bid = snapshot["bids"][0]["price"]
        ...     best_ask = snapshot["asks"][0]["price"]
        ...     spread = best_ask - best_bid
    """
    streaming_dir = Path(streaming_cache_dir)
    latest_file = streaming_dir / symbol / "latest.json"

    # Try streaming cache first
    if latest_file.exists():
        try:
            with open(latest_file) as f:
                data = json.load(f)

            logger.debug(
                "Loaded latest order book from streaming cache",
                extra={"component": "data_feed", "symbol": symbol, "source": "streaming"},
            )
            return data

        except Exception as e:
            logger.error(
                f"Failed to load streaming cache: {e}",
                extra={"component": "data_feed", "symbol": symbol, "error": str(e)},
            )

    # Fallback to CSV cache if configured
    if fallback_csv_dir:
        csv_dir = Path(fallback_csv_dir)
        symbol_dir = csv_dir / symbol

        if symbol_dir.exists():
            # Find most recent snapshot
            snapshot_files = sorted(symbol_dir.rglob("*.json"), reverse=True)
            if snapshot_files:
                try:
                    with open(snapshot_files[0]) as f:
                        data = json.load(f)

                    logger.debug(
                        "Loaded order book from CSV fallback",
                        extra={
                            "component": "data_feed",
                            "symbol": symbol,
                            "source": "csv_fallback",
                            "file": str(snapshot_files[0]),
                        },
                    )
                    return data

                except Exception as e:
                    logger.error(
                        f"Failed to load CSV fallback: {e}",
                        extra={"component": "data_feed", "symbol": symbol, "error": str(e)},
                    )

    logger.warning(
        f"No order book data available for {symbol}",
        extra={"component": "data_feed", "symbol": symbol},
    )
    return None


def get_order_book_history(
    symbol: str,
    limit: int = 100,
    streaming_cache_dir: str | Path = "data/order_book/streaming",
) -> list[dict]:
    """Get recent order book snapshots from streaming buffer.

    Note: This function only accesses the latest snapshot from the streaming cache.
    For full historical buffer access, use the OrderBookStreamer.get_buffer_snapshots()
    method directly, which maintains an in-memory circular buffer.

    Args:
        symbol: Stock symbol
        limit: Max snapshots to return (only 1 available from cache file)
        streaming_cache_dir: Directory where streaming script writes latest.json

    Returns:
        List of order book snapshots (newest first), limited to 1 from cache file.
        Empty list if no data available.

    Example:
        >>> history = get_order_book_history("RELIANCE", limit=10)
        >>> if history:
        ...     latest = history[0]  # Most recent
        ...     print(f"Latest snapshot at {latest['timestamp']}")
    """
    latest = get_latest_order_book(symbol, streaming_cache_dir, fallback_csv_dir=None)

    if latest:
        logger.debug(
            "Retrieved order book history (1 snapshot from cache)",
            extra={"component": "data_feed", "symbol": symbol, "count": 1},
        )
        return [latest]

    logger.debug(
        "No order book history available",
        extra={"component": "data_feed", "symbol": symbol},
    )
    return []
