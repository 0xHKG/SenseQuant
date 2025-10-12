# US-014 — Historical Data Feed & Backtester CSV Support

## Problem Statement

Currently, the system relies heavily on direct Breeze API calls for historical data, which creates several issues:

1. **No offline backtesting**: Cannot run backtests without live API access
2. **Redundant API calls**: Repeatedly fetching the same historical data wastes rate limits
3. **No reproducibility**: Backtest results vary based on API availability and data changes
4. **Development friction**: Developers need API credentials and connectivity for local testing
5. **No data versioning**: Cannot freeze historical datasets for consistent model training

Production trading systems require:
- **Local CSV support**: Run backtests offline using pre-downloaded datasets
- **Intelligent caching**: Automatically cache Breeze API responses to avoid redundant fetches
- **Hybrid failover**: Fall back to cached CSV when API is unavailable
- **Data reproducibility**: Freeze datasets for model training and validation
- **Fast iteration**: Rapid backtest execution without API latency

## Objectives

1. **DataFeed Service**: Unified abstraction for historical bar data with multiple sources
2. **CSV Support**: Load OHLCV data from local CSV files with IST timezone handling
3. **Breeze Integration**: Fetch from API with automatic caching to CSV
4. **Hybrid Mode**: Intelligent failover (Breeze → CSV cache → error)
5. **Backtester Integration**: Dependency injection for pure CSV-driven simulations
6. **CLI Tool**: Standalone backtest script with CSV path arguments
7. **Caching Strategy**: Organized directory structure under `data/historical/`

## Requirements

### Functional Requirements

#### FR-1: DataFeed Service Interface
```python
class DataFeed(ABC):
    """Abstract interface for historical data sources."""

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: Literal["1minute", "5minute", "1day"],
    ) -> pd.DataFrame:
        """Fetch historical bars for symbol/interval/daterange.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            All timestamps in IST timezone
        """
        pass
```

#### FR-2: CSV Data Feed
- Parse CSV files with standard OHLCV format
- Support flexible column naming (case-insensitive, aliases)
- Convert timestamps to IST timezone
- Validate data integrity (no gaps, monotonic timestamps)
- Filter by date range
- Handle missing files gracefully

**CSV Format**:
```csv
timestamp,open,high,low,close,volume
2025-01-10 09:15:00,2456.50,2458.75,2455.00,2457.25,125000
2025-01-10 09:16:00,2457.25,2459.00,2456.50,2458.00,98000
```

#### FR-3: Breeze Data Feed
- Wrap existing BreezeClient.get_historical()
- Automatic CSV caching after successful fetch
- Organized directory structure: `data/historical/{symbol}/{interval}/{date}.csv`
- Append-only writes (never overwrite existing cache)
- Cache metadata (fetch timestamp, symbol, interval, date range)

#### FR-4: Hybrid Data Feed
- **Primary source**: Breeze API
- **Fallback**: Local CSV cache
- **Strategy**:
  1. Check CSV cache for requested date range
  2. If cache hit (complete range): return cached data
  3. If cache miss/partial: fetch from Breeze API
  4. Cache API response to CSV
  5. If API fails: use cached data (even if partial)

#### FR-5: Backtester Integration
- Replace direct BreezeClient dependency with DataFeed
- Constructor accepts DataFeed implementation
- Support pure CSV mode (no Breeze dependency)
- Maintain existing backtest logic (signal generation, fills, metrics)

#### FR-6: CLI Backtest Tool
```bash
# Run backtest with CSV files
python scripts/backtest.py \
    --symbol RELIANCE \
    --csv data/historical/RELIANCE/1minute/2025-01.csv \
    --from-date 2025-01-10 \
    --to-date 2025-01-20 \
    --strategy intraday

# Run backtest with Breeze API (cached)
python scripts/backtest.py \
    --symbol RELIANCE \
    --from-date 2025-01-10 \
    --to-date 2025-01-20 \
    --strategy intraday \
    --use-api
```

#### FR-7: Engine Integration
- Engine dryrun/backtest modes use DataFeed
- Live mode continues using direct Breeze WebSocket
- DataFeed selection based on mode:
  - `mode=backtest`: CSV DataFeed
  - `mode=dryrun`: Hybrid DataFeed (Breeze + cache)
  - `mode=live`: Direct Breeze (no DataFeed)

#### FR-8: Configuration
```python
# Settings extension
data_feed_source: Literal["csv", "breeze", "hybrid"] = "hybrid"
data_feed_enable_cache: bool = True
data_feed_csv_directory: str = "data/historical"
data_feed_cache_compression: bool = False
```

### Non-Functional Requirements

#### NFR-1: Performance
- CSV loading: <1s for 1 month of 1-minute data (~10k rows)
- Cache write: Non-blocking, async preferred
- Memory efficient: Stream CSV rows, don't load entire dataset

#### NFR-2: Data Integrity
- Validate CSV schema (required columns present)
- Check for timestamp gaps (warn, don't fail)
- Handle duplicate timestamps (keep last)
- Timezone consistency (always IST)

#### NFR-3: Robustness
- Graceful degradation when cache incomplete
- Clear error messages for missing/corrupt CSVs
- Retry logic for Breeze API failures
- Never crash on cache read errors

#### NFR-4: Maintainability
- Clean abstraction (Strategy pattern for DataFeed)
- Dependency injection (no global state)
- Comprehensive logging (cache hits/misses, API calls, errors)
- Unit testable (mock Breeze API, use temp CSV files)

## Architecture

### Component Design

```
┌─────────────────────────────────────────────────────────────┐
│                      DataFeed (ABC)                          │
│  + get_historical_bars(symbol, from, to, interval)          │
└─────────────────────────────────────────────────────────────┘
                            △
                            │ implements
            ┌───────────────┼───────────────┐
            │               │               │
┌───────────▼───────┐  ┌────▼──────┐  ┌────▼────────────┐
│  CSVDataFeed      │  │ BreezeDF  │  │  HybridDataFeed │
│  - csv_dir: Path  │  │ - breeze  │  │  - breeze_df    │
│  - load_csv()     │  │ - cache   │  │  - csv_df       │
│  - parse()        │  │ - fetch() │  │  - failover()   │
└───────────────────┘  └───────────┘  └─────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Backtester                                │
│  - data_feed: DataFeed                                       │
│  - run_backtest(symbol, from, to, strategy)                 │
│    → bars = data_feed.get_historical_bars(...)              │
│    → for bar in bars: generate signal, simulate fill        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Engine                                   │
│  - data_feed: DataFeed | None                                │
│  - if mode == "backtest": use data_feed                     │
│  - if mode == "live": use breeze WebSocket                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Backtest Mode (CSV Only)
```
scripts/backtest.py --csv data/historical/RELIANCE/1minute/2025-01.csv
        │
        ├──> CSVDataFeed.get_historical_bars()
        │    ├─ Parse CSV
        │    ├─ Filter date range
        │    └─ Return DataFrame
        │
        └──> Backtester.run_backtest(bars)
             ├─ Generate signals
             ├─ Simulate fills
             └─ Compute metrics (Sharpe, max DD, win rate)
```

#### Hybrid Mode (Breeze + Cache)
```
Engine.tick_intraday() [mode=dryrun]
        │
        ├──> HybridDataFeed.get_historical_bars()
        │    │
        │    ├─ Check CSV cache
        │    │  ├─ Cache hit → return cached DataFrame
        │    │  └─ Cache miss ↓
        │    │
        │    ├─ Fetch from BreezeClient.get_historical()
        │    │  ├─ Success → cache to CSV → return DataFrame
        │    │  └─ Failure → fallback to partial cache (warn)
        │    │
        │    └─ Return DataFrame
        │
        └──> Strategy.generate_signal(bars)
```

### Directory Structure

```
data/
└── historical/
    ├── RELIANCE/
    │   ├── 1minute/
    │   │   ├── 2025-01-10.csv
    │   │   ├── 2025-01-11.csv
    │   │   └── metadata.json
    │   ├── 5minute/
    │   │   └── 2025-01.csv
    │   └── 1day/
    │       └── 2025.csv
    ├── TCS/
    │   └── 1minute/
    │       └── 2025-01-10.csv
    └── cache_index.json  # Cache metadata (fetch times, ranges)
```

**Metadata Format** (`metadata.json`):
```json
{
  "symbol": "RELIANCE",
  "interval": "1minute",
  "files": {
    "2025-01-10.csv": {
      "date": "2025-01-10",
      "rows": 375,
      "start": "2025-01-10T09:15:00+05:30",
      "end": "2025-01-10T15:30:00+05:30",
      "fetched_at": "2025-01-11T08:00:00+05:30",
      "source": "breeze_api"
    }
  }
}
```

## Implementation Plan

### Phase 1: Core DataFeed Service
1. Create `src/services/data_feed.py` with ABC
2. Implement `CSVDataFeed`:
   - CSV parsing with pandas
   - Timezone conversion (UTC → IST)
   - Date range filtering
   - Schema validation
3. Implement `BreezeDataFeed`:
   - Wrap BreezeClient.get_historical()
   - Cache management (write to CSV)
   - Directory creation
   - Metadata tracking
4. Implement `HybridDataFeed`:
   - Cache lookup logic
   - API failover
   - Partial cache handling

### Phase 2: Configuration & Settings
1. Extend Settings with data feed options
2. Add CSV directory configuration
3. Add cache enable/disable flags

### Phase 3: Backtester Integration
1. Refactor Backtester constructor:
   - Accept DataFeed dependency
   - Remove direct BreezeClient usage (keep for backward compat)
2. Update backtest logic to use DataFeed
3. Add CSV-only mode tests

### Phase 4: CLI Tool
1. Create `scripts/backtest.py`:
   - Argument parsing (symbol, dates, CSV paths)
   - DataFeed instantiation
   - Backtester execution
   - Results output (table + CSV export)

### Phase 5: Engine Integration
1. Add DataFeed to Engine constructor
2. Update tick_intraday/run_swing_daily:
   - Use DataFeed in backtest/dryrun modes
   - Keep WebSocket for live mode
3. Ensure sentiment/risk flows unaffected

### Phase 6: Documentation & Testing
1. Update architecture.md data flow section
2. Unit tests:
   - CSV parsing (valid/invalid files)
   - Cache write/read
   - Hybrid failover scenarios
3. Integration tests:
   - End-to-end backtest with CSV
   - Engine dryrun with hybrid feed
   - Cache hit/miss scenarios

## Acceptance Criteria

### AC-1: CSV Data Feed
- [ ] CSVDataFeed.get_historical_bars() loads CSV and returns DataFrame
- [ ] Handles missing CSV files gracefully (clear error message)
- [ ] Converts timestamps to IST timezone
- [ ] Filters date range correctly
- [ ] Validates required columns (timestamp, open, high, low, close, volume)

### AC-2: Breeze Data Feed with Caching
- [ ] BreezeDataFeed.get_historical_bars() fetches from API
- [ ] Automatically caches API response to CSV
- [ ] Creates directory structure (`data/historical/{symbol}/{interval}/`)
- [ ] Writes metadata.json with fetch details
- [ ] Subsequent calls use cache (no redundant API fetches)

### AC-3: Hybrid Data Feed
- [ ] HybridDataFeed checks cache before API call
- [ ] Cache hit returns cached data (no API call)
- [ ] Cache miss triggers API fetch + cache write
- [ ] API failure falls back to partial cache (with warning)
- [ ] Logs cache hit/miss/API call events

### AC-4: Backtester Integration
- [ ] Backtester accepts DataFeed via constructor
- [ ] Can run pure CSV backtest (no Breeze dependency)
- [ ] Existing backtest logic unchanged (signals, fills, metrics)
- [ ] Results match previous implementation (validation test)

### AC-5: CLI Backtest Tool
- [ ] `scripts/backtest.py --csv` runs backtest from CSV
- [ ] `--from-date` and `--to-date` filter data correctly
- [ ] Outputs metrics table (Sharpe, max DD, win rate, total trades)
- [ ] Supports `--export` flag to save results CSV

### AC-6: Engine Integration
- [ ] Engine backtest mode uses DataFeed (no live WebSocket)
- [ ] Engine dryrun mode uses HybridDataFeed
- [ ] Live mode unaffected (continues using WebSocket)
- [ ] Sentiment and risk flows work correctly in all modes

### AC-7: Configuration
- [ ] Settings include data_feed_source, data_feed_enable_cache, data_feed_csv_directory
- [ ] Can disable caching via config (for testing)
- [ ] Can override CSV directory path

### AC-8: Tests & Quality
- [ ] Unit tests for CSV parsing, caching, failover (>20 tests)
- [ ] Integration test: backtest with CSV (no Breeze mock)
- [ ] Integration test: engine dryrun with hybrid feed
- [ ] All quality gates pass (ruff, mypy, pytest)

## Technical Risks & Mitigations

### Risk 1: CSV Schema Variations
**Risk**: Different CSV formats from various sources (broker exports, scrapers)
**Mitigation**:
- Flexible column name matching (case-insensitive, aliases)
- Clear validation errors with column names
- Support for optional columns (e.g., adj_close)

### Risk 2: Timezone Confusion
**Risk**: CSV timestamps in different timezones (UTC, IST, local)
**Mitigation**:
- Explicit timezone conversion in DataFeed
- Assume UTC if no timezone info
- Log timezone conversions
- Validate IST timestamps in tests

### Risk 3: Cache Staleness
**Risk**: Cached data becomes outdated, affecting backtest accuracy
**Mitigation**:
- Metadata tracks fetch timestamp
- CLI option to force refresh (`--no-cache`)
- Warning if cache > N days old

### Risk 4: Partial Cache Failures
**Risk**: Cache partially covers date range, API fails for remaining days
**Mitigation**:
- Return partial data with warning
- Log missing date ranges
- Option to fail-fast vs. best-effort

## Future Enhancements (Out of Scope for US-014)

- Real-time bar aggregation (tick → 1min bars)
- Multi-symbol bulk download
- Data quality checks (outlier detection, volume spikes)
- CSV compression (gzip) for storage efficiency
- Database backend (PostgreSQL/TimescaleDB) for large datasets
- WebSocket data feed for live tick simulation
- Synthetic data generation for stress testing

## Dependencies

- **pandas**: CSV parsing and DataFrame operations
- **pytz**: Timezone handling (IST)
- **pathlib**: File system operations
- **BreezeClient**: API integration (existing)

## Estimated Effort

- Phase 1 (DataFeed): 4-6 hours
- Phase 2 (Config): 1 hour
- Phase 3 (Backtester): 2-3 hours
- Phase 4 (CLI): 2-3 hours
- Phase 5 (Engine): 2-3 hours
- Phase 6 (Tests/Docs): 4-5 hours

**Total**: 15-21 hours (2-3 development days)

## Success Metrics

1. **Offline Development**: Developers can run backtests without Breeze API credentials
2. **API Efficiency**: 90% cache hit rate for repeated backtests (no redundant API calls)
3. **Fast Iteration**: Backtest execution time reduced by >50% (cached vs. API)
4. **Reproducibility**: Frozen CSV datasets produce identical backtest results
5. **Test Coverage**: >85% code coverage for DataFeed module
