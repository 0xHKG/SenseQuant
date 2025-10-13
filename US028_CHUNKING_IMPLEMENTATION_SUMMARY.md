# US-028 Phase 6 — Chunked Historical Data Ingestion Implementation Summary

**Date**: October 14, 2025
**Scope**: Add rate-limited chunking to historical data ingestion
**Status**: Implementation in progress

---

## ✅ Changes Completed

### 1. Configuration Settings Added ([src/app/config.py:502-511](src/app/config.py#L502-L511))

Added three new settings after existing historical_data settings:

```python
# US-028: Chunked Historical Data Ingestion with Rate Limiting
historical_chunk_days: int = Field(
    90, validation_alias="HISTORICAL_CHUNK_DAYS", ge=1, le=365
)  # Max days per API chunk request (prevents timeout/overload)

breeze_rate_limit_requests_per_minute: int = Field(
    30, validation_alias="BREEZE_RATE_LIMIT_REQUESTS_PER_MINUTE", ge=1, le=100
)  # Max Breeze API requests per minute (conservative default)

breeze_rate_limit_delay_seconds: float = Field(
    2.0, validation_alias="BREEZE_RATE_LIMIT_DELAY_SECONDS", ge=0.1, le=10.0
)  # Delay between chunk requests to respect rate limits
```

**Rationale**:
- `historical_chunk_days=90`: Default 90-day chunks balance API load vs request count
- `breeze_rate_limit_requests_per_minute=30`: Conservative to avoid throttling
- `breeze_rate_limit_delay_seconds=2.0`: Sleep between chunks

### 2. BreezeClient.fetch_historical_chunk() Added ([src/adapters/breeze_client.py:320-413](src/adapters/breeze_client.py#L320-L413))

New production-ready method for fetching historical data:

```python
def fetch_historical_chunk(
    self,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1day",
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch historical data for a date range chunk using v2 API.

    Features:
    - Uses v2 API with ISO8601 timestamps
    - Automatic stock code mapping (NSE → ISEC codes)
    - Retry logic with exponential backoff
    - Empty DataFrame on no data (not an error)
    """
```

**Key Features**:
1. Wraps existing `historical_bars()` which already has v2 API + stock mapping
2. Returns DataFrame (not Bar objects) for easier consumption
3. Handles dry_run mode gracefully
4. Clear logging for debugging

---

## 📋 Remaining Tasks

### 3. Update fetch_historical_data.py (IN PROGRESS)

**Required Changes**:

```python
# Add at top of HistoricalDataFetcher class:
from datetime import datetime, timedelta
import time

def _split_date_range_into_chunks(
    self,
    start_date: str,
    end_date: str,
    chunk_days: int,
) -> list[tuple[datetime, datetime]]:
    """Split date range into chunks of max chunk_days."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        chunks.append((current, chunk_end))
        current = chunk_end

    return chunks

def fetch_symbol_date_chunked(
    self,
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    """Fetch data for symbol across date range using chunks."""
    chunk_days = self.settings.historical_chunk_days
    delay = self.settings.breeze_rate_limit_delay_seconds

    chunks = self._split_date_range_into_chunks(start_date, end_date, chunk_days)
    all_data = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.info(
            f"Fetching chunk {i}/{len(chunks)} for {symbol} "
            f"({chunk_start.date()} to {chunk_end.date()})"
        )

        try:
            df = self.breeze_client.fetch_historical_chunk(
                symbol=symbol,
                start_date=chunk_start,
                end_date=chunk_end,
                interval=interval,
            )

            if not df.empty:
                all_data.append(df)
                logger.debug(f"  ✓ Chunk {i}: {len(df)} bars")
            else:
                logger.warning(f"  ⚠ Chunk {i}: No data")

        except Exception as e:
            logger.error(f"  ✗ Chunk {i} failed: {e}")
            raise

        # Rate limiting: sleep between chunks (except last)
        if i < len(chunks):
            logger.debug(f"  Sleeping {delay}s (rate limiting)...")
            time.sleep(delay)

    if not all_data:
        logger.warning(f"No data fetched for {symbol} {start_date} to {end_date}")
        return pd.DataFrame()

    # Combine all chunks and deduplicate
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"✓ Fetched {len(combined)} total bars for {symbol} "
        f"(from {len(chunks)} chunks)"
    )

    return combined
```

**Integration Point**:
- Replace `breeze_client.get_historical()` calls with `fetch_symbol_date_chunked()`
- Update `fetch_symbol_date()` to use new method
- Add `--force` flag to bypass cache and refetch

### 4. Audit Other Scripts

**Scripts to check**:
- ✅ `train_teacher.py` - Already uses `historical_bars()` via `BreezeClient` (updated in Phase 6)
- ✅ `fetch_options_data.py` - Uses different Breeze endpoint (options), not historical data
- ✅ `fetch_order_book.py` - Uses order book endpoint, not historical data
- ✅ `backtest.py` - Uses historical data from CSVs, not direct API calls
- ✅ `optimize.py` - Uses historical data from CSVs, not direct API calls

**Conclusion**: Only `fetch_historical_data.py` needs updating for chunked ingestion.

### 5. Tests

**Unit Tests** (`tests/unit/test_breeze_client.py`):

```python
def test_fetch_historical_chunk_success(client_live, mock_breeze_sdk):
    """Test fetch_historical_chunk with successful data."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.get_historical_data_v2.return_value = {
        "Status": 200,
        "Success": [
            {
                "datetime": "2024-01-01 09:15:00",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 10000,
            }
        ],
    }

    client_live._client = mock_instance
    df = client_live.fetch_historical_chunk(
        "RELIANCE",
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        interval="1day",
    )

    assert not df.empty
    assert "timestamp" in df.columns
    assert len(df) == 1
    assert df.iloc[0]["close"] == 103.0


def test_fetch_historical_chunk_empty(client_live, mock_breeze_sdk):
    """Test fetch_historical_chunk with no data."""
    mock_instance = mock_breeze_sdk.return_value
    mock_instance.get_historical_data_v2.return_value = {"Status": 200, "Success": []}

    client_live._client = mock_instance
    df = client_live.fetch_historical_chunk(
        "RELIANCE",
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
    )

    assert df.empty
```

**Integration Tests** (`tests/integration/test_historical_training.py`):

```python
def test_chunked_ingestion_multi_month():
    """Test chunked historical data ingestion across multiple months."""
    # Create fetcher with small chunk size for testing
    settings = Settings()
    settings.historical_chunk_days = 30  # Force multiple chunks

    fetcher = HistoricalDataFetcher(settings)

    # Fetch Q1 2024 (should create 3 chunks)
    df = fetcher.fetch_symbol_date_chunked(
        symbol="RELIANCE",
        start_date="2024-01-01",
        end_date="2024-03-31",
        interval="1day",
    )

    # Verify combined data
    assert not df.empty
    assert len(df) > 60  # ~90 days minus weekends
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].min().date() >= date(2024, 1, 1)
    assert df["timestamp"].max().date() <= date(2024, 3, 31)

    # Verify no duplicates
    assert len(df) == len(df["timestamp"].unique())
```

### 6. Documentation

**Update** `docs/stories/us-028-historical-run.md`:

```markdown
## Update (Oct 14 2025 — Phase 6b: Chunked Historical Data Ingestion)

### Context
Historical data fetching was failing for large date ranges due to:
- API timeouts on requests >90 days
- Rate limiting issues
- v1 API instability

### Changes
1. **Breeze API v2 Migration**: All historical data now uses `get_historical_data_v2`
2. **Stock Code Mapping**: Automatic NSE → ISEC code translation (RELIANCE → RELIND)
3. **Chunked Fetching**: Date ranges split into configurable chunks (default: 90 days)
4. **Rate Limiting**: Configurable delay between chunks to respect API limits
5. **Retry Logic**: Exponential backoff on transient errors

### Configuration
New settings in `.env`:
- `HISTORICAL_CHUNK_DAYS=90` - Max days per API request
- `BREEZE_RATE_LIMIT_REQUESTS_PER_MINUTE=30` - Rate limit threshold
- `BREEZE_RATE_LIMIT_DELAY_SECONDS=2.0` - Inter-chunk delay

### Minimum Data Requirements
Teacher model training requires **≥6 months** of trading data:
- 5-day forward label window removes last 5 days
- Feature generation requires lookback period
- Short ranges (<6 months) will fail with clear error messages

### API Endpoints
- **v1 (deprecated)**: `get_historical_data` - Returns `None` on errors
- **v2 (recommended)**: `get_historical_data_v2` - Proper error handling, 1-second intervals
```

---

## 📊 Benefits

1. **Reliability**: v2 API more stable than v1
2. **Scalability**: Can fetch years of data by chunking
3. **Rate Limit Compliance**: Configurable delays prevent throttling
4. **Error Handling**: Clear logging and retry logic
5. **Stock Code Mapping**: Automatic translation prevents lookup errors
6. **Performance**: Parallel-ready (can fetch multiple symbols concurrently)

---

## 🎯 Next Steps

1. Complete `fetch_historical_data.py` chunking implementation
2. Add comprehensive test coverage
3. Update documentation
4. Run quality gates
5. Execute end-to-end test with full year date range

---

**Status**: Configuration ✅ | BreezeClient ✅ | fetch_historical_data.py ⏳ | Tests ⏳ | Docs ⏳
