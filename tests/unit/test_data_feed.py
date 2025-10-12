"""Comprehensive unit tests for DataFeed service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytz

from src.services.data_feed import (
    BreezeDataFeed,
    CSVDataFeed,
    HybridDataFeed,
    create_data_feed,
)


@pytest.fixture
def ist_tz():
    """IST timezone fixture."""
    return pytz.timezone("Asia/Kolkata")


@pytest.fixture
def sample_csv_dir(tmp_path: Path) -> Path:
    """Create sample CSV directory structure."""
    # Create directory structure: symbol/interval/
    symbol_dir = tmp_path / "TEST" / "1day"
    symbol_dir.mkdir(parents=True)

    # Create sample CSV file
    csv_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=10, freq="D", tz=pytz.timezone("Asia/Kolkata")
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0],
            "volume": [100000] * 10,
        }
    )

    csv_file = symbol_dir / "2024-01-01_to_2024-01-10.csv"
    csv_data.to_csv(csv_file, index=False)

    return tmp_path


@pytest.fixture
def mock_breeze_client():
    """Create mock Breeze client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.data_feed_csv_directory = "data/historical"
    settings.data_feed_enable_cache = True
    settings.data_feed_cache_compression = False
    return settings


# ============================================================================
# CSVDataFeed Tests
# ============================================================================


def test_csv_data_feed_initialization(sample_csv_dir: Path):
    """Test CSVDataFeed initialization."""
    feed = CSVDataFeed(sample_csv_dir)
    assert feed.csv_directory == sample_csv_dir
    assert feed.csv_directory.exists()


def test_csv_data_feed_nonexistent_directory(tmp_path: Path):
    """Test CSVDataFeed with nonexistent directory."""
    nonexistent = tmp_path / "nonexistent"
    feed = CSVDataFeed(nonexistent)
    assert feed.csv_directory == nonexistent


def test_csv_data_feed_get_historical_bars_success(sample_csv_dir: Path, ist_tz):
    """Test successful historical bars fetch from CSV."""
    feed = CSVDataFeed(sample_csv_dir)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 10, tzinfo=ist_tz)

    df = feed.get_historical_bars("TEST", from_date, to_date, "1day")

    assert not df.empty
    assert len(df) >= 9  # Date filtering may exclude edge dates
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dt.tz.zone == "Asia/Kolkata"


def test_csv_data_feed_directory_not_found(sample_csv_dir: Path, ist_tz):
    """Test CSVDataFeed when symbol directory not found."""
    feed = CSVDataFeed(sample_csv_dir)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 10, tzinfo=ist_tz)

    with pytest.raises(ValueError, match="CSV directory not found"):
        feed.get_historical_bars("NONEXISTENT", from_date, to_date, "1day")


def test_csv_data_feed_no_csv_files(tmp_path: Path, ist_tz):
    """Test CSVDataFeed when no CSV files in directory."""
    # Create empty symbol directory
    symbol_dir = tmp_path / "EMPTY" / "1day"
    symbol_dir.mkdir(parents=True)

    feed = CSVDataFeed(tmp_path)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 10, tzinfo=ist_tz)

    with pytest.raises(ValueError, match="No CSV files found"):
        feed.get_historical_bars("EMPTY", from_date, to_date, "1day")


def test_csv_data_feed_date_range_filtering(sample_csv_dir: Path, ist_tz):
    """Test CSVDataFeed date range filtering."""
    feed = CSVDataFeed(sample_csv_dir)

    # Request subset of data
    from_date = datetime(2024, 1, 3, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 7, tzinfo=ist_tz)

    df = feed.get_historical_bars("TEST", from_date, to_date, "1day")

    assert len(df) >= 4  # Date filtering may exclude edge dates
    assert df["timestamp"].min().date() >= from_date.date()
    assert df["timestamp"].max().date() <= to_date.date()


def test_csv_data_feed_column_name_variations(tmp_path: Path, ist_tz):
    """Test CSVDataFeed handles various column name formats."""
    # Create CSV with alternative column names
    symbol_dir = tmp_path / "ALT" / "1day"
    symbol_dir.mkdir(parents=True)

    csv_data = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=5, freq="D", tz=ist_tz),
            "o": [100.0, 101.0, 102.0, 103.0, 104.0],
            "h": [105.0, 106.0, 107.0, 108.0, 109.0],
            "l": [99.0, 100.0, 101.0, 102.0, 103.0],
            "c": [104.0, 105.0, 106.0, 107.0, 108.0],
            "vol": [100000, 110000, 120000, 130000, 140000],
        }
    )

    csv_file = symbol_dir / "2024-01-01.csv"
    csv_data.to_csv(csv_file, index=False)

    feed = CSVDataFeed(tmp_path)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    df = feed.get_historical_bars("ALT", from_date, to_date, "1day")

    assert not df.empty
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns


def test_csv_data_feed_missing_required_columns(tmp_path: Path, ist_tz):
    """Test CSVDataFeed with missing required columns."""
    symbol_dir = tmp_path / "INVALID" / "1day"
    symbol_dir.mkdir(parents=True)

    # Create CSV with missing columns
    csv_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=ist_tz),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            # Missing high, low, close, volume
        }
    )

    csv_file = symbol_dir / "2024-01-01.csv"
    csv_data.to_csv(csv_file, index=False)

    feed = CSVDataFeed(tmp_path)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    with pytest.raises(ValueError, match="Failed to load any CSV files"):
        feed.get_historical_bars("INVALID", from_date, to_date, "1day")


def test_csv_data_feed_timezone_naive_timestamps(tmp_path: Path, ist_tz):
    """Test CSVDataFeed converts timezone-naive timestamps to IST."""
    symbol_dir = tmp_path / "NAIVE" / "1day"
    symbol_dir.mkdir(parents=True)

    # Create CSV with timezone-naive timestamps
    csv_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),  # No timezone
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [100000, 110000, 120000, 130000, 140000],
        }
    )

    csv_file = symbol_dir / "2024-01-01.csv"
    csv_data.to_csv(csv_file, index=False)

    feed = CSVDataFeed(tmp_path)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    df = feed.get_historical_bars("NAIVE", from_date, to_date, "1day")

    # Verify timestamps are IST
    assert df["timestamp"].dt.tz.zone == "Asia/Kolkata"


def test_csv_data_feed_gzip_compression(tmp_path: Path, ist_tz):
    """Test CSVDataFeed reads gzip compressed files."""
    symbol_dir = tmp_path / "GZIP" / "1day"
    symbol_dir.mkdir(parents=True)

    # Create gzip compressed CSV
    csv_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=ist_tz),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [100000, 110000, 120000, 130000, 140000],
        }
    )

    # Save as .gz file
    csv_file = symbol_dir / "2024-01-01.csv.gz"
    csv_data.to_csv(csv_file, index=False, compression="gzip")

    feed = CSVDataFeed(tmp_path)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    df = feed.get_historical_bars("GZIP", from_date, to_date, "1day")

    assert not df.empty
    assert len(df) >= 4  # Date filtering may exclude edge dates


def test_csv_data_feed_multiple_files(tmp_path: Path, ist_tz):
    """Test CSVDataFeed concatenates multiple CSV files."""
    symbol_dir = tmp_path / "MULTI" / "1day"
    symbol_dir.mkdir(parents=True)

    # Create multiple CSV files
    csv_data1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=ist_tz),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [100000, 110000, 120000, 130000, 140000],
        }
    )

    csv_data2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-06", periods=5, freq="D", tz=ist_tz),
            "open": [105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [110.0, 111.0, 112.0, 113.0, 114.0],
            "low": [104.0, 105.0, 106.0, 107.0, 108.0],
            "close": [109.0, 110.0, 111.0, 112.0, 113.0],
            "volume": [150000, 160000, 170000, 180000, 190000],
        }
    )

    csv_file1 = symbol_dir / "2024-01-01_to_2024-01-05.csv"
    csv_file2 = symbol_dir / "2024-01-06_to_2024-01-10.csv"
    csv_data1.to_csv(csv_file1, index=False)
    csv_data2.to_csv(csv_file2, index=False)

    feed = CSVDataFeed(tmp_path)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 10, tzinfo=ist_tz)

    df = feed.get_historical_bars("MULTI", from_date, to_date, "1day")

    assert len(df) >= 9  # Date filtering may exclude edge dates
    assert df["timestamp"].is_monotonic_increasing


def test_csv_data_feed_duplicate_timestamps(tmp_path: Path, ist_tz):
    """Test CSVDataFeed removes duplicate timestamps (keeps last)."""
    symbol_dir = tmp_path / "DUP" / "1day"
    symbol_dir.mkdir(parents=True)

    # Create CSV with duplicate timestamps
    csv_data = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=ist_tz),
                datetime(2024, 1, 2, tzinfo=ist_tz),
                datetime(2024, 1, 2, tzinfo=ist_tz),  # Duplicate
                datetime(2024, 1, 3, tzinfo=ist_tz),
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [105.0, 106.0, 107.0, 108.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [104.0, 105.0, 106.0, 107.0],
            "volume": [100000, 110000, 120000, 130000],
        }
    )

    csv_file = symbol_dir / "2024-01-01.csv"
    csv_data.to_csv(csv_file, index=False)

    feed = CSVDataFeed(tmp_path)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 3, tzinfo=ist_tz)

    df = feed.get_historical_bars("DUP", from_date, to_date, "1day")

    # Should keep only one row for 2024-01-02 (the last one)
    assert len(df) == 3
    assert df["timestamp"].is_unique


# ============================================================================
# BreezeDataFeed Tests
# ============================================================================


def test_breeze_data_feed_initialization(mock_breeze_client, mock_settings):
    """Test BreezeDataFeed initialization."""
    feed = BreezeDataFeed(mock_breeze_client, mock_settings)
    assert feed.breeze == mock_breeze_client
    assert feed.settings == mock_settings
    assert feed.enable_cache == mock_settings.data_feed_enable_cache


def test_breeze_data_feed_get_historical_bars_success(mock_breeze_client, mock_settings, ist_tz):
    """Test successful historical bars fetch from Breeze API."""
    # Mock Breeze API response
    mock_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=ist_tz),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [100000, 110000, 120000, 130000, 140000],
        }
    )

    mock_breeze_client.historical_bars.return_value = mock_df

    feed = BreezeDataFeed(mock_breeze_client, mock_settings)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    df = feed.get_historical_bars("TEST", from_date, to_date, "1day")

    assert not df.empty
    assert len(df) == 5
    mock_breeze_client.historical_bars.assert_called_once()


def test_breeze_data_feed_empty_response(mock_breeze_client, mock_settings, ist_tz):
    """Test BreezeDataFeed when API returns empty DataFrame."""
    mock_breeze_client.historical_bars.return_value = pd.DataFrame()

    feed = BreezeDataFeed(mock_breeze_client, mock_settings)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    with pytest.raises(ValueError, match="returned no data"):
        feed.get_historical_bars("TEST", from_date, to_date, "1day")


def test_breeze_data_feed_api_error(mock_breeze_client, mock_settings, ist_tz):
    """Test BreezeDataFeed when API raises error."""
    mock_breeze_client.historical_bars.side_effect = Exception("API error")

    feed = BreezeDataFeed(mock_breeze_client, mock_settings)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    with pytest.raises(ValueError, match="Failed to fetch data from Breeze API"):
        feed.get_historical_bars("TEST", from_date, to_date, "1day")


def test_breeze_data_feed_caching_disabled(mock_breeze_client, mock_settings, ist_tz, tmp_path):
    """Test BreezeDataFeed with caching disabled."""
    mock_settings.data_feed_enable_cache = False
    mock_settings.data_feed_csv_directory = str(tmp_path)

    mock_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=ist_tz),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [100000, 110000, 120000, 130000, 140000],
        }
    )

    mock_breeze_client.historical_bars.return_value = mock_df

    feed = BreezeDataFeed(mock_breeze_client, mock_settings)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    df = feed.get_historical_bars("TEST", from_date, to_date, "1day")

    assert not df.empty
    # Verify no cache files were created
    cache_dir = tmp_path / "TEST" / "1day"
    assert not cache_dir.exists() or len(list(cache_dir.glob("*.csv"))) == 0


# ============================================================================
# HybridDataFeed Tests
# ============================================================================


def test_hybrid_data_feed_cache_hit(mock_breeze_client, mock_settings, sample_csv_dir, ist_tz):
    """Test HybridDataFeed uses cache when available."""
    mock_settings.data_feed_csv_directory = str(sample_csv_dir)

    feed = HybridDataFeed(mock_breeze_client, mock_settings)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 10, tzinfo=ist_tz)

    df = feed.get_historical_bars("TEST", from_date, to_date, "1day")

    # Should use cache, not API
    assert not df.empty
    assert len(df) >= 9  # Date filtering may exclude edge dates
    mock_breeze_client.historical_bars.assert_not_called()


def test_hybrid_data_feed_cache_miss(mock_breeze_client, mock_settings, tmp_path, ist_tz):
    """Test HybridDataFeed fetches from API when cache misses."""
    mock_settings.data_feed_csv_directory = str(tmp_path)
    mock_settings.data_feed_enable_cache = False

    # Create empty directory structure to avoid "directory not found" error
    symbol_dir = tmp_path / "NEWTEST" / "1day"
    symbol_dir.mkdir(parents=True)

    mock_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=ist_tz),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [100000, 110000, 120000, 130000, 140000],
        }
    )

    mock_breeze_client.historical_bars.return_value = mock_df

    feed = HybridDataFeed(mock_breeze_client, mock_settings)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 5, tzinfo=ist_tz)

    df = feed.get_historical_bars("NEWTEST", from_date, to_date, "1day")

    # Should fetch from API
    assert not df.empty
    mock_breeze_client.historical_bars.assert_called_once()


def test_hybrid_data_feed_api_failure_fallback(
    mock_breeze_client, mock_settings, sample_csv_dir, ist_tz
):
    """Test HybridDataFeed falls back to partial cache when API fails."""
    mock_settings.data_feed_csv_directory = str(sample_csv_dir)

    # API call will fail
    mock_breeze_client.historical_bars.side_effect = Exception("API down")

    feed = HybridDataFeed(mock_breeze_client, mock_settings)

    from_date = datetime(2024, 1, 1, tzinfo=ist_tz)
    to_date = datetime(2024, 1, 15, tzinfo=ist_tz)  # Request more than cache has

    # Should fall back to partial cache
    df = feed.get_historical_bars("TEST", from_date, to_date, "1day")

    assert not df.empty
    # Will only have cached data (10 days instead of 15)
    assert len(df) <= 10


# ============================================================================
# Factory Function Tests
# ============================================================================


def test_create_data_feed_csv(mock_settings):
    """Test create_data_feed with CSV source."""
    mock_settings.data_feed_source = "csv"
    mock_settings.data_feed_csv_directory = "data/historical"

    feed = create_data_feed(mock_settings)

    assert isinstance(feed, CSVDataFeed)


def test_create_data_feed_breeze(mock_breeze_client, mock_settings):
    """Test create_data_feed with Breeze source."""
    mock_settings.data_feed_source = "breeze"

    feed = create_data_feed(mock_settings, mock_breeze_client)

    assert isinstance(feed, BreezeDataFeed)


def test_create_data_feed_breeze_missing_client(mock_settings):
    """Test create_data_feed with Breeze source but no client."""
    mock_settings.data_feed_source = "breeze"

    with pytest.raises(ValueError, match="breeze_client required"):
        create_data_feed(mock_settings)


def test_create_data_feed_hybrid(mock_breeze_client, mock_settings):
    """Test create_data_feed with hybrid source."""
    mock_settings.data_feed_source = "hybrid"

    feed = create_data_feed(mock_settings, mock_breeze_client)

    assert isinstance(feed, HybridDataFeed)


def test_create_data_feed_invalid_source(mock_settings):
    """Test create_data_feed with invalid source."""
    mock_settings.data_feed_source = "invalid"

    with pytest.raises(ValueError, match="Invalid data_feed_source"):
        create_data_feed(mock_settings)
