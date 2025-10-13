# US-029: Order Book, Options, and Macro Data Integration (Phase 1)

## Problem Statement

The current SenseQuant system relies solely on OHLCV (price-volume) and sentiment data for trading signals. However, institutional-quality quantitative strategies often incorporate additional market microstructure and macro-economic signals:

1. **Order Book Depth**: Bid/ask spreads, depth at multiple price levels, market impact estimation
2. **Options Chain Data**: Implied volatility surface, put-call ratios, options volume/OI by strike/expiry
3. **Macro Indicators**: Interest rates, inflation metrics, sector indices, market breadth indicators

**Current Gap**: We have no infrastructure to ingest, store, or expose these advanced data sources.

**US-029 Phase 1 Goal**: Build foundation for advanced market data by implementing ingestion scripts, storage layout, configuration, and basic DataFeed extensions. This phase focuses on **data collection and storage** only; feature engineering will come in Phase 2.

## Acceptance Criteria

### AC1: Configuration for Advanced Data Sources (Safe Defaults)
- [ ] Add configuration settings in `src/app/config.py` for:
  - Order book depth snapshots (enabled/disabled, provider endpoints, snapshot intervals, retention days)
  - Options chain data (enabled/disabled, provider endpoints, refresh intervals, retention days)
  - Macro economic indicators (enabled/disabled, provider endpoints, indicator list, refresh intervals)
- [ ] **All advanced data sources disabled by default** (require explicit opt-in via environment variables)
- [ ] Include provider endpoint URLs, retry policies, and rate limits

### AC2: Order Book Ingestion Script
- [ ] Create `scripts/fetch_order_book.py` with:
  - Fetch L2 order book snapshots (best 5/10 bid/ask levels) for configured symbols
  - Support incremental mode (fetch only new snapshots since last run)
  - Support dryrun mode (mock data generation, no network calls)
  - Store snapshots as JSON under `data/order_book/<symbol>/<YYYY-MM-DD>/<HH-MM-SS>.json`
  - Update StateManager with last fetch timestamp and statistics (snapshots fetched, errors)
  - Respect retry limits with exponential backoff

### AC3: Options Chain Ingestion Script
- [ ] Create `scripts/fetch_options_data.py` with:
  - Fetch options chain (all strikes/expiries) for configured symbols
  - Support incremental mode (fetch only if chain updated since last run)
  - Support dryrun mode (mock options chain data)
  - Store chain snapshots as JSON under `data/options/<symbol>/<YYYY-MM-DD>.json`
  - Update StateManager with last fetch timestamp, statistics (chains fetched, strikes/expiries count)
  - Respect retry limits and rate limits

### AC4: Macro Data Ingestion Script
- [ ] Create `scripts/fetch_macro_data.py` with:
  - Fetch configured macro indicators (e.g., NIFTY 50, India VIX, 10Y bond yield, USD/INR)
  - Support incremental mode (fetch only new macro data since last run)
  - Support dryrun mode (mock macro data)
  - Store macro snapshots as JSON under `data/macro/<indicator>/<YYYY-MM-DD>.json`
  - Update StateManager with last fetch timestamp, statistics (indicators fetched, data points)
  - Respect retry limits

### AC5: StateManager Extensions for Market Data
- [ ] Extend `StateManager` with methods:
  - `record_market_data_fetch(data_type, symbol, timestamp, stats)` - Record fetch metadata
  - `get_last_market_data_fetch(data_type, symbol)` - Get last fetch timestamp
  - `get_market_data_fetch_stats(data_type)` - Get aggregated fetch statistics
- [ ] Track fetch statistics: snapshots_fetched, errors, last_error_message, avg_latency_ms

### AC6: DataFeed Extensions for Loading Market Data
- [ ] Extend `src/services/data_feed.py` with helper methods:
  - `load_order_book_snapshots(symbol, from_date, to_date)` - Load cached order book data
  - `load_options_chain(symbol, date)` - Load cached options chain for a date
  - `load_macro_data(indicator, from_date, to_date)` - Load cached macro indicator data
- [ ] Methods return normalized dictionaries/DataFrames (consistent schema)
- [ ] Methods gracefully handle missing data (return empty results, log warnings)

### AC7: Integration Tests for Market Data Ingestion
- [ ] Create `tests/integration/test_market_data_ingestion.py` with tests:
  - `test_order_book_ingestion_dryrun()` - Verify order book directory structure and state updates
  - `test_options_ingestion_dryrun()` - Verify options chain directory structure and state updates
  - `test_macro_ingestion_dryrun()` - Verify macro data directory structure and state updates
  - `test_market_data_state_manager()` - Verify StateManager tracking methods
  - `test_data_feed_market_data_loaders()` - Verify DataFeed can load cached market data
  - `test_incremental_mode_respects_lookback()` - Verify incremental mode skips existing data
- [ ] All tests mock external API calls (no real network requests)

### AC8: Documentation
- [ ] Update `docs/stories/us-029-market-data.md`:
  - Data schemas for order book, options chain, and macro indicators
  - Retry policies and rate limit specifications
  - Usage examples for each ingestion script
  - Instructions for enabling advanced data sources
- [ ] Update `docs/architecture.md` (Section 19):
  - Directory structure for market data storage
  - StateManager market data tracking design
  - DataFeed extension architecture
  - Provider stubs for Phase 1 (real providers in Phase 2)
  - Future Phase 2 roadmap (feature engineering, signal generation)

## Technical Design

### Configuration Schema (src/app/config.py)

```python
# Order Book Configuration
order_book_enabled: bool = Field(False, validation_alias="ORDER_BOOK_ENABLED")
order_book_provider: str = Field("stub", validation_alias="ORDER_BOOK_PROVIDER")  # "stub", "breeze", "websocket"
order_book_endpoint: str = Field("", validation_alias="ORDER_BOOK_ENDPOINT")
order_book_depth_levels: int = Field(5, validation_alias="ORDER_BOOK_DEPTH_LEVELS", ge=1, le=20)
order_book_snapshot_interval_seconds: int = Field(60, validation_alias="ORDER_BOOK_SNAPSHOT_INTERVAL_SECONDS", ge=1, le=3600)
order_book_retention_days: int = Field(7, validation_alias="ORDER_BOOK_RETENTION_DAYS", ge=1, le=90)
order_book_retry_limit: int = Field(3, validation_alias="ORDER_BOOK_RETRY_LIMIT", ge=1, le=10)
order_book_retry_backoff_seconds: int = Field(2, validation_alias="ORDER_BOOK_RETRY_BACKOFF_SECONDS", ge=1, le=60)

# Options Chain Configuration
options_enabled: bool = Field(False, validation_alias="OPTIONS_ENABLED")
options_provider: str = Field("stub", validation_alias="OPTIONS_PROVIDER")  # "stub", "breeze", "nse"
options_endpoint: str = Field("", validation_alias="OPTIONS_ENDPOINT")
options_refresh_interval_hours: int = Field(24, validation_alias="OPTIONS_REFRESH_INTERVAL_HOURS", ge=1, le=168)
options_retention_days: int = Field(30, validation_alias="OPTIONS_RETENTION_DAYS", ge=1, le=365)
options_retry_limit: int = Field(3, validation_alias="OPTIONS_RETRY_LIMIT", ge=1, le=10)
options_retry_backoff_seconds: int = Field(2, validation_alias="OPTIONS_RETRY_BACKOFF_SECONDS", ge=1, le=60)

# Macro Data Configuration
macro_enabled: bool = Field(False, validation_alias="MACRO_ENABLED")
macro_provider: str = Field("stub", validation_alias="MACRO_PROVIDER")  # "stub", "yfinance", "rbi"
macro_endpoint: str = Field("", validation_alias="MACRO_ENDPOINT")
macro_indicators: list[str] = Field(
    default=["NIFTY50", "INDIAVIX", "USDINR", "IN10Y"],
    validation_alias="MACRO_INDICATORS"
)
macro_refresh_interval_hours: int = Field(24, validation_alias="MACRO_REFRESH_INTERVAL_HOURS", ge=1, le=168)
macro_retention_days: int = Field(90, validation_alias="MACRO_RETENTION_DAYS", ge=1, le=730)
macro_retry_limit: int = Field(3, validation_alias="MACRO_RETRY_LIMIT", ge=1, le=10)
macro_retry_backoff_seconds: int = Field(2, validation_alias="MACRO_RETRY_BACKOFF_SECONDS", ge=1, le=60)
```

### Order Book Data Schema

```json
{
  "symbol": "RELIANCE",
  "timestamp": "2025-01-15T10:30:00+05:30",
  "exchange": "NSE",
  "bids": [
    {"price": 2450.50, "quantity": 1000, "orders": 5},
    {"price": 2450.00, "quantity": 1500, "orders": 7},
    ...
  ],
  "asks": [
    {"price": 2451.00, "quantity": 800, "orders": 4},
    {"price": 2451.50, "quantity": 1200, "orders": 6},
    ...
  ],
  "metadata": {
    "depth_levels": 5,
    "fetch_latency_ms": 120,
    "source": "stub"
  }
}
```

### Options Chain Data Schema

```json
{
  "symbol": "NIFTY",
  "date": "2025-01-15",
  "timestamp": "2025-01-15T15:30:00+05:30",
  "underlying_price": 21500.50,
  "options": [
    {
      "strike": 21000,
      "expiry": "2025-01-30",
      "call": {
        "last_price": 550.25,
        "bid": 549.00,
        "ask": 551.00,
        "volume": 15000,
        "oi": 125000,
        "iv": 0.18
      },
      "put": {
        "last_price": 45.50,
        "bid": 45.00,
        "ask": 46.00,
        "volume": 8000,
        "oi": 95000,
        "iv": 0.17
      }
    },
    ...
  ],
  "metadata": {
    "total_strikes": 50,
    "expiries": ["2025-01-30", "2025-02-27", "2025-03-27"],
    "source": "stub"
  }
}
```

### Macro Data Schema

```json
{
  "indicator": "NIFTY50",
  "date": "2025-01-15",
  "timestamp": "2025-01-15T15:30:00+05:30",
  "value": 21500.50,
  "change": 150.25,
  "change_pct": 0.70,
  "metadata": {
    "source": "stub",
    "fetch_latency_ms": 80
  }
}
```

### Directory Structure

```
data/
├── order_book/
│   ├── RELIANCE/
│   │   ├── 2025-01-15/
│   │   │   ├── 09-15-00.json
│   │   │   ├── 09-16-00.json
│   │   │   └── ...
│   │   └── metadata.json
│   └── TCS/
│       └── ...
├── options/
│   ├── NIFTY/
│   │   ├── 2025-01-15.json
│   │   ├── 2025-01-16.json
│   │   └── metadata.json
│   └── BANKNIFTY/
│       └── ...
└── macro/
    ├── NIFTY50/
    │   ├── 2025-01-15.json
    │   ├── 2025-01-16.json
    │   └── metadata.json
    └── INDIAVIX/
        └── ...
```

### Retry Policy

All ingestion scripts follow consistent retry policy:
- **Max Retries**: Configurable via `*_retry_limit` (default: 3)
- **Backoff Strategy**: Exponential backoff with base delay from `*_retry_backoff_seconds`
- **Retryable Errors**: ConnectionError, TimeoutError, HTTP 5xx, rate limit errors
- **Non-Retryable Errors**: Authentication failures, invalid parameters, HTTP 4xx (except 429)

### Rate Limiting

- Order book snapshots: Max 1 request per second per symbol (configurable)
- Options chains: Max 10 requests per minute (configurable)
- Macro indicators: Max 60 requests per hour (configurable)

## Usage Examples

### Order Book Ingestion

```bash
# Fetch order book snapshots for default symbols (dryrun mode)
python scripts/fetch_order_book.py --dryrun

# Fetch order book for specific symbols
python scripts/fetch_order_book.py \
  --symbols RELIANCE TCS INFY \
  --start-time 09:15:00 \
  --end-time 15:30:00 \
  --interval-seconds 60

# Incremental mode (fetch only new snapshots since last run)
python scripts/fetch_order_book.py --incremental

# Force re-fetch (ignore cache)
python scripts/fetch_order_book.py --force
```

### Options Chain Ingestion

```bash
# Fetch options chain for default symbols (dryrun mode)
python scripts/fetch_options_data.py --dryrun

# Fetch options for specific symbol and date
python scripts/fetch_options_data.py \
  --symbols NIFTY BANKNIFTY \
  --date 2025-01-15

# Incremental mode
python scripts/fetch_options_data.py --incremental
```

### Macro Data Ingestion

```bash
# Fetch macro indicators (dryrun mode)
python scripts/fetch_macro_data.py --dryrun

# Fetch specific indicators and date range
python scripts/fetch_macro_data.py \
  --indicators NIFTY50 INDIAVIX USDINR IN10Y \
  --start-date 2025-01-01 \
  --end-date 2025-01-31

# Incremental mode
python scripts/fetch_macro_data.py --incremental
```

## Testing Strategy

### Unit Tests
- Configuration validation (ensure defaults are safe)
- Data schema validation (JSON structure, required fields)
- StateManager market data tracking methods
- DataFeed loader methods with mock data

### Integration Tests
- End-to-end ingestion scripts in dryrun mode
- Directory structure creation and validation
- State file updates and tracking
- Incremental mode behavior (skip existing data)
- Error handling and retry logic

### Manual Testing Checklist
1. Run all three ingestion scripts in dryrun mode
2. Verify directory structures created correctly
3. Verify state files updated with fetch metadata
4. Verify incremental mode skips existing data
5. Verify DataFeed can load cached market data
6. Verify configuration defaults keep all advanced data disabled

## Phase 2: Feature Engineering (US-029 Phase 2)

**Goal**: Transform raw market data (order book, options, macro) into normalized feature frames ready for model training.

### Phase 2 Acceptance Criteria

#### AC1: Order Book Feature Module
- [ ] Create `src/features/order_book.py` with functions:
  - `calculate_spread_features()` - Bid-ask spread metrics (absolute, relative, time-weighted)
  - `calculate_depth_imbalance()` - Order book imbalance at multiple levels
  - `calculate_order_flow_metrics()` - Order flow ratios, market pressure indicators
  - `calculate_liquidity_features()` - Volume-weighted metrics, market impact estimation
- [ ] Functions accept DataFrames of snapshots, return feature-ready frames keyed by timestamp
- [ ] Handle missing data gracefully (fill NaN, log warnings)

#### AC2: Options Feature Module
- [ ] Create `src/features/options.py` with functions:
  - `calculate_iv_features()` - IV percentile, ATM/OTM IV, IV rank
  - `calculate_skew_features()` - Put-call IV skew, strike-to-strike skew
  - `calculate_volume_features()` - Options volume ratios, put-call ratio
  - `calculate_greeks_aggregates()` - Aggregate delta, gamma, vega across strikes
- [ ] Functions accept options chain DataFrames, return features keyed by date
- [ ] Support both index options (NIFTY, BANKNIFTY) and equity options

#### AC3: Macro Feature Module
- [ ] Create `src/features/macro.py` with functions:
  - `calculate_regime_features()` - Volatility regime (low/medium/high)
  - `calculate_correlation_features()` - Rolling correlation with indices
  - `calculate_momentum_features()` - Macro momentum indicators (MA crossovers, rate of change)
  - `calculate_breadth_features()` - Market breadth indicators (advance-decline ratios)
- [ ] Functions accept macro DataFrames, return features keyed by date
- [ ] Support configurable lookback windows

#### AC4: Integration with Feature Library
- [ ] Extend `src/domain/features.py` with:
  - `compute_order_book_features()` - Wrapper that loads order book data and computes features
  - `compute_options_features()` - Wrapper that loads options data and computes features
  - `compute_macro_features()` - Wrapper that loads macro data and computes features
  - `compute_all_market_features()` - Unified interface that computes all available market features
- [ ] Functions check data availability before computing (skip if no data)
- [ ] Log feature coverage (which symbols/dates have market features)

#### AC5: Batch Training Integration
- [ ] Update `scripts/train_teacher_batch.py`:
  - Add `--include-market-features` flag (default: False)
  - Load and merge market features into training data when flag enabled
  - Record feature set usage in teacher_runs.json metadata
- [ ] Update `scripts/train_student_batch.py`:
  - Add `--include-market-features` flag (default: False)
  - Load and merge market features into training data when flag enabled
  - Record feature set usage in student_runs.json metadata
- [ ] Ensure training gracefully handles missing market features (warn but don't fail)

#### AC6: StateManager Feature Coverage Logging
- [ ] Extend StateManager with:
  - `record_feature_coverage(symbol, date_range, feature_types)` - Track which features generated
  - `get_feature_coverage(symbol)` - Retrieve feature coverage stats
- [ ] DataFeed logs warnings when market data unavailable for requested symbol/date

#### AC7: Validation/Statistical Workflow Updates
- [ ] Update `scripts/run_model_validation.py`:
  - Include feature set metadata in validation reports
  - Log which features were used in validated models
- [ ] Update `scripts/run_statistical_tests.py`:
  - Include feature set in test result summaries
  - Compare performance with/without market features

#### AC8: Testing
- [ ] Create `tests/unit/test_market_features.py`:
  - Unit tests for each feature function with deterministic toy data
  - Test edge cases (missing data, single snapshot, zero volume)
- [ ] Create `tests/integration/test_market_features.py`:
  - End-to-end test: generate mock data → compute features → verify output shape/values
  - Test feature integration with training pipeline
  - Verify metadata tracking

#### AC9: Documentation
- [ ] Update `docs/stories/us-029-market-data.md`:
  - Phase 2 feature catalog with descriptions
  - Usage examples for feature generation
  - Configuration toggles and defaults
- [ ] Update `docs/architecture.md`:
  - Feature engineering workflow diagram
  - Feature naming conventions
  - Performance considerations

### Phase 2 Feature Catalog

#### Order Book Features (12 features)

**Spread Features** (4):
- `ob_spread_abs` - Absolute bid-ask spread (ask - bid)
- `ob_spread_rel` - Relative spread (spread / mid_price)
- `ob_spread_pct` - Spread as percentage of mid price
- `ob_time_weighted_spread` - Time-weighted average spread over window

**Depth Imbalance Features** (4):
- `ob_depth_imbalance_l1` - Level 1 imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
- `ob_depth_imbalance_l5` - Level 5 aggregate imbalance
- `ob_volume_weighted_price_bid` - Volume-weighted bid price
- `ob_volume_weighted_price_ask` - Volume-weighted ask price

**Order Flow Features** (4):
- `ob_order_flow_ratio` - Ratio of buy orders to sell orders
- `ob_market_pressure` - Net buying/selling pressure indicator
- `ob_liquidity_imbalance` - Liquidity difference between bid/ask sides
- `ob_effective_spread` - Effective spread based on trade-weighted prices

#### Options Features (15 features)

**IV Features** (5):
- `opt_iv_atm` - At-the-money implied volatility
- `opt_iv_percentile` - IV percentile (current IV vs historical IV range)
- `opt_iv_rank` - IV rank (0-100 scale)
- `opt_iv_otm_call` - OTM call IV (5% OTM)
- `opt_iv_otm_put` - OTM put IV (5% OTM)

**Skew Features** (4):
- `opt_skew_put_call` - Put IV - Call IV at ATM
- `opt_skew_25delta` - 25-delta put IV - 25-delta call IV
- `opt_skew_slope` - Linear slope of IV curve across strikes
- `opt_skew_curvature` - Convexity of IV smile

**Volume/OI Features** (3):
- `opt_put_call_ratio_volume` - Put volume / Call volume
- `opt_put_call_ratio_oi` - Put OI / Call OI
- `opt_total_volume_ratio` - Options volume / Underlying volume

**Greeks Aggregates** (3):
- `opt_aggregate_delta` - Net delta across all strikes
- `opt_aggregate_gamma` - Net gamma across all strikes
- `opt_aggregate_vega` - Net vega across all strikes

#### Macro Features (10 features)

**Regime Features** (3):
- `macro_volatility_regime` - Categorical: low/medium/high (based on VIX percentile)
- `macro_trend_regime` - Categorical: bull/bear/sideways (based on MA slopes)
- `macro_liquidity_regime` - Categorical: tight/normal/loose (based on spreads)

**Correlation Features** (3):
- `macro_correlation_nifty` - Rolling 30-day correlation with NIFTY
- `macro_correlation_vix` - Rolling 30-day correlation with India VIX
- `macro_beta_nifty` - Rolling beta with respect to NIFTY

**Momentum Features** (4):
- `macro_ma_crossover` - Moving average crossover signal (1=bullish, -1=bearish, 0=neutral)
- `macro_roc` - Rate of change over lookback period
- `macro_momentum_score` - Composite momentum indicator
- `macro_volatility_mom` - Volatility momentum (expanding/contracting)

### Phase 2 Technical Design

#### Feature Function Signature Pattern

```python
def calculate_spread_features(
    snapshots: pd.DataFrame,
    lookback_window: int = 60,
) -> pd.DataFrame:
    """Calculate spread features from order book snapshots.

    Args:
        snapshots: DataFrame with columns [timestamp, bids, asks, ...]
        lookback_window: Lookback window in seconds for time-weighted metrics

    Returns:
        DataFrame with columns [timestamp, ob_spread_abs, ob_spread_rel, ...]
        Indexed by timestamp, one row per unique timestamp
    """
    pass
```

#### Feature Integration Pattern

```python
# In src/domain/features.py
def compute_order_book_features(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """Compute order book features for symbol/date range.

    Returns empty DataFrame if no order book data available.
    Logs warning if data missing.
    """
    from src.services.data_feed import load_order_book_snapshots
    from src.features.order_book import (
        calculate_spread_features,
        calculate_depth_imbalance,
        calculate_order_flow_metrics,
        calculate_liquidity_features,
    )

    # Load raw data
    snapshots = load_order_book_snapshots(symbol, from_date, to_date)
    if not snapshots:
        logger.warning(f"No order book data for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(snapshots)

    # Compute features
    spread_features = calculate_spread_features(df)
    depth_features = calculate_depth_imbalance(df)
    flow_features = calculate_order_flow_metrics(df)
    liquidity_features = calculate_liquidity_features(df)

    # Merge all features
    features = spread_features.join([depth_features, flow_features, liquidity_features])

    return features
```

### Phase 2 Usage Examples

#### Standalone Feature Generation

```python
from datetime import datetime
from src.features.order_book import calculate_spread_features
from src.services.data_feed import load_order_book_snapshots

# Load raw data
snapshots = load_order_book_snapshots("RELIANCE", datetime(2025, 1, 15), datetime(2025, 1, 15))
df = pd.DataFrame(snapshots)

# Compute features
spread_features = calculate_spread_features(df)
print(spread_features.head())
```

#### Training with Market Features

```bash
# Teacher training with market features
python scripts/train_teacher_batch.py \
  --symbols RELIANCE TCS \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --include-market-features

# Student training with market features
python scripts/train_student_batch.py \
  --include-market-features
```

#### Feature Coverage Check

```python
from src.services.state_manager import StateManager

sm = StateManager()
coverage = sm.get_feature_coverage("RELIANCE")
print(f"Order book features: {coverage.get('order_book_dates', 0)} days")
print(f"Options features: {coverage.get('options_dates', 0)} days")
print(f"Macro features: {coverage.get('macro_dates', 0)} days")
```

### Phase 2 Configuration

**Feature Engineering Toggles** (add to config.py):

```python
# Feature Engineering Configuration
enable_order_book_features: bool = Field(False, validation_alias="ENABLE_ORDER_BOOK_FEATURES")
enable_options_features: bool = Field(False, validation_alias="ENABLE_OPTIONS_FEATURES")
enable_macro_features: bool = Field(False, validation_alias="ENABLE_MACRO_FEATURES")

# Feature Computation Parameters
order_book_feature_lookback_seconds: int = Field(60, validation_alias="ORDER_BOOK_FEATURE_LOOKBACK_SECONDS")
options_feature_iv_lookback_days: int = Field(30, validation_alias="OPTIONS_FEATURE_IV_LOOKBACK_DAYS")
macro_feature_correlation_window: int = Field(30, validation_alias="MACRO_FEATURE_CORRELATION_WINDOW")
```

### Phase 2 Testing Strategy

**Unit Tests** (`tests/unit/test_market_features.py`):
- Test each feature function with deterministic toy data
- Verify output shape and value ranges
- Test edge cases: missing data, single snapshot, zero volume
- Test numerical stability (no inf/nan)

**Integration Tests** (`tests/integration/test_market_features.py`):
- End-to-end: Mock data → Compute features → Verify metadata
- Test feature integration with training pipeline
- Verify feature coverage tracking
- Test graceful handling of missing data

### Phase 2 Performance Considerations

- **Caching**: Feature computation can be expensive; consider caching computed features
- **Vectorization**: Use pandas vectorized operations for performance
- **Memory**: Order book snapshots can be large; process in chunks if needed
- **Incremental**: Compute features incrementally (only new dates) where possible

---

## Phase 3: Strategy Integration of Advanced Market Features (US-029 Phase 3)

**Goal**: Integrate order book, options, and macro features into trading strategies with configurable gating rules and signal adjustments.

### Phase 3 Acceptance Criteria

#### AC1: Intraday Strategy Market Feature Integration
- [ ] Add spread filter: block signals if `ob_spread_pct > threshold` (config: `intraday_spread_filter_enabled`)
- [ ] Add market pressure adjustment: boost signal strength based on `ob_market_pressure` (config: `intraday_market_pressure_adjustment_enabled`)
- [ ] Add IV gate: block LONG if `opt_iv_percentile > 80`, SHORT if < 20 (config: `intraday_iv_adjustment_enabled`)
- [ ] Add macro regime filter: block LONG in downtrend, SHORT in uptrend, all in high vol regime (config: `intraday_macro_regime_filter_enabled`)
- [ ] All feature gates disabled by default (backward compatible)
- [ ] Record all feature checks in signal metadata

#### AC2: Swing Strategy Market Feature Integration
- [ ] Add IV position sizing: adjust size ±20-30% based on `opt_iv_rank` (config: `swing_iv_position_sizing_enabled`)
- [ ] Add macro correlation filter: block if `abs(macro_correlation_nifty) < 0.3` (config: `swing_macro_correlation_filter_enabled`)
- [ ] Reduce strength by 20% if `macro_beta_nifty > 1.5`
- [ ] All feature gates disabled by default
- [ ] Record feature adjustments in signal metadata

#### AC3: Optimizer Feature Toggle Integration
- [ ] Extend optimizer to include feature flags in parameter grid
- [ ] Generate combinations: baseline (all off) + individual features + combinations
- [ ] Track feature configuration in optimization results metadata
- [ ] Add `optimizer_test_feature_combinations` config flag

#### AC4: Backtester Feature Usage Tracking
- [ ] Add `feature_checks` counter (how many times each gate evaluated)
- [ ] Add `feature_blocks` counter (how many signals blocked by each feature)
- [ ] Add `feature_adjustments` tracker (signal strength adjustments)
- [ ] Include feature stats in backtest summary
- [ ] Ensure feature config recorded in backtest metadata

#### AC5: Teacher/Student Training Feature Integration
- [ ] Load market features when enabled via `compute_all_market_features()`
- [ ] Merge market features with OHLCV/sentiment: `X = ohlcv.join(sentiment).join(market_features)`
- [ ] Record feature columns in training metadata
- [ ] Handle missing features gracefully (fill zeros, log warning)
- [ ] Maintain backward compatibility (training works without market features)

#### AC6: Configuration Extensions
- [ ] Add Phase 3 flags to `src/app/config.py` (all default False):
  - Intraday: `intraday_spread_filter_enabled`, `intraday_max_spread_pct`, `intraday_market_pressure_adjustment_enabled`, `intraday_iv_adjustment_enabled`, `intraday_macro_regime_filter_enabled`
  - Swing: `swing_iv_position_sizing_enabled`, `swing_macro_correlation_filter_enabled`
  - Optimizer: `optimizer_test_feature_combinations`

#### AC7: Unit Tests for Strategy Feature Integration
- [ ] `test_spread_filter_blocks_wide_spread()`, `test_spread_filter_passes_narrow_spread()`
- [ ] `test_market_pressure_boosts_long()`, `test_iv_blocks_high_volatility_long()`
- [ ] `test_macro_regime_blocks_downtrend_long()`, `test_feature_gates_disabled_by_default()`
- [ ] Swing: `test_iv_reduces_position_size_high_vol()`, `test_macro_correlation_blocks_low_correlation()`

#### AC8: Integration Tests for Engine Feature Handling
- [ ] `test_engine_loads_market_features_when_enabled()`, `test_engine_signal_blocked_by_spread_filter()`
- [ ] `test_engine_metadata_records_feature_usage()`, `test_engine_works_without_market_features()`
- [ ] Swing: `test_engine_applies_iv_position_sizing()`

#### AC9: Documentation
- [ ] Update this story with Phase 3 specification (done)
- [ ] Update `docs/architecture.md` Section 21 with strategy integration architecture

### Phase 3 Technical Design

**Strategy Integration Pattern**:
```python
def signal(
    df: pd.DataFrame,
    settings: Settings,
    *,
    sentiment: float | None = None,
    market_features: dict[str, float] | None = None,  # NEW
) -> Signal:
    # 1. Generate base signal (existing logic)
    direction, strength = _generate_base_signal(df, settings, sentiment)

    # 2. Apply market feature gates (if enabled)
    if market_features and _any_feature_enabled(settings):
        direction, strength = _apply_feature_gates(direction, strength, market_features, settings)

    # 3. Build metadata with feature checks
    meta = {"base_signal": {...}, "feature_checks": {...}}
    return Signal(symbol="", direction=direction, strength=strength, meta=meta)
```

**Feature Gate Helpers**:
- `_check_spread_filter()`: Return (passed, metadata)
- `_check_iv_gate()`: Return (passed, metadata)
- `_check_macro_regime()`: Return (passed, metadata)
- `_apply_feature_adjustments()`: Adjust strength based on features

**Configuration Example**:
```bash
# Enable intraday spread filter
INTRADAY_SPREAD_FILTER_ENABLED=true
INTRADAY_MAX_SPREAD_PCT=0.5

# Enable IV adjustment
INTRADAY_IV_ADJUSTMENT_ENABLED=true
```

**Metadata Schema**:
```json
{
  "base_signal": {"direction": "LONG", "strength": 0.7},
  "feature_checks": {
    "spread_filter": {"enabled": true, "passed": true, "spread_pct": 0.35},
    "iv_gate": {"enabled": true, "passed": true, "iv_percentile": 45.2},
    "macro_regime": {"enabled": true, "passed": true, "trend_regime": 0}
  },
  "final_signal": {"direction": "LONG", "strength": 0.7}
}
```

---

## Phase 4: Real Market Data Provider Integration (Breeze-based) — IMPLEMENTED ✅

**Goal**: Replace stub providers with real Breeze API integrations while preserving dryrun mode and safety controls.

### Phase 4 Implementation Summary

**Completed**:
- ✅ Created `src/adapters/market_data_providers.py` with Breeze-based providers
- ✅ `BreezeOrderBookProvider`: Order book snapshots via Breeze REST API
- ✅ `BreezeOptionsProvider`: Options chain data via Breeze REST API
- ✅ `MacroIndicatorProvider`: Macro indicators via yfinance/public APIs
- ✅ Provider factory functions with safe defaults (dryrun unless explicitly enabled)
- ✅ Comprehensive unit tests (28 tests, 100% passing)
- ✅ Deterministic mock data for all providers in dryrun mode
- ✅ Retry logic with exponential backoff via tenacity

### Architecture

**Provider Hierarchy**:
```
MarketDataProvider (ABC)
├── BreezeOrderBookProvider
├── BreezeOptionsProvider
└── MacroIndicatorProvider
```

**Data Classes**:
- `OrderBookSnapshot`: Bid/ask levels with price/quantity/orders
- `OptionsChainSnapshot`: Options chain with strikes/expiries/greeks
- `MacroIndicatorSnapshot`: Indicator value with change/change_pct

**Factory Functions** (with safe defaults):
```python
create_order_book_provider(settings, client=None, dry_run=None)
create_options_provider(settings, client=None, dry_run=None)
create_macro_provider(settings, dry_run=None)
```

### Safety Controls

1. **Read-only Operations**: No order placement, only market data reads
2. **Dryrun by Default**: Providers use mock data unless explicitly enabled in config
3. **Retry Logic**: Exponential backoff for transient errors (3 attempts, 2-30s wait)
4. **Credential Management**: Uses existing `breeze_api_key`, `breeze_api_secret`, `breeze_session_token` from Settings
5. **Graceful Degradation**: Falls back to mock data if API unavailable

### Usage Examples

#### Order Book Provider

```python
from src.adapters.market_data_providers import create_order_book_provider
from src.app.config import settings

# Dryrun mode (default, uses mock data)
provider = create_order_book_provider(settings)
snapshot = provider.fetch("RELIANCE", depth_levels=5)

print(f"Bids: {snapshot.bids[:2]}")
print(f"Asks: {snapshot.asks[:2]}")
print(f"Spread: {snapshot.asks[0]['price'] - snapshot.bids[0]['price']}")
```

#### Options Provider

```python
from src.adapters.market_data_providers import create_options_provider
from src.app.config import settings

# Dryrun mode
provider = create_options_provider(settings)
snapshot = provider.fetch("NIFTY", date="2025-01-15")

print(f"Underlying: {snapshot.underlying_price}")
print(f"Strikes: {snapshot.metadata['total_strikes']}")
print(f"Expiries: {snapshot.metadata['expiries']}")

# Access options data
for option in snapshot.options[:3]:
    print(f"Strike {option['strike']}: Call IV={option['call']['iv']}, Put IV={option['put']['iv']}")
```

#### Macro Provider

```python
from src.adapters.market_data_providers import create_macro_provider
from src.app.config import settings

# Dryrun mode
provider = create_macro_provider(settings)
snapshot = provider.fetch("NIFTY50", date="2025-01-15")

print(f"NIFTY50: {snapshot.value}")
print(f"Change: {snapshot.change} ({snapshot.change_pct}%)")
```

### Configuration

Providers respect existing US-029 Phase 1 configuration:

```bash
# Enable live mode for order book
ORDER_BOOK_ENABLED=true

# Enable live mode for options
OPTIONS_ENABLED=true

# Enable live mode for macro data
MACRO_ENABLED=true

# Breeze credentials (already configured)
BREEZE_API_KEY=your_api_key
BREEZE_API_SECRET=your_api_secret
BREEZE_SESSION_TOKEN=your_session_token
```

**Safe Defaults**: All providers default to `dry_run=True` if corresponding `*_enabled` flag is `False`.

### Testing

**Unit Tests** (`tests/unit/test_market_data_providers.py`):
- 28 tests covering all providers
- Dryrun mode validation
- Deterministic mock data verification
- Factory function behavior
- Edge cases (zero depth, null dates, timestamps)

**Test Coverage**:
- Order Book: 9 tests
- Options: 8 tests
- Macro: 7 tests
- Factories: 3 tests
- Edge Cases: 6 tests

### Limitations and Future Work

**Current Limitations**:
1. **Order Book API**: Breeze may not support full L2 depth; implementation uses placeholder logic
2. **Options Chain API**: Breeze endpoint for options chain may differ from mock implementation
3. **WebSocket Streaming**: Not implemented (Phase 5 candidate)
4. **Real-time Updates**: Providers fetch snapshots on-demand, not streaming

**Phase 5 Candidates**:
- WebSocket streaming for real-time order book updates
- Background ingestion workers for continuous data collection
- Advanced caching and compression for storage optimization
- Live options greeks calculation (delta, gamma, vega)
- Integration with RBI API for bond yields (currently mocked)

### Quality Gates

All quality gates passed:
- ✅ Unit tests: 28/28 passing
- ✅ `python -m ruff check src/adapters/market_data_providers.py`
- ✅ `python -m ruff format src/adapters/market_data_providers.py`
- ✅ `python -m mypy src/adapters/market_data_providers.py`

---

## Phase 4 Completion Summary — Pipeline Integration & State Metrics ✅

**Completed**: All three ingestion scripts now use real Breeze-based providers with comprehensive state tracking.

### Implementation Highlights

**1. Provider Integration (fetch_order_book.py, fetch_options_data.py, fetch_macro_data.py)**:
- ✅ SecretsManager integration for credential loading (plain/encrypted modes)
- ✅ Provider factories create BreezeOrderBookProvider, BreezeOptionsProvider, MacroIndicatorProvider
- ✅ Dryrun mode enabled by default (controlled by config flags)
- ✅ Retry logic handled internally by providers (3 attempts, exponential backoff)
- ✅ Scripts fetch snapshots, normalize to JSON, and save to established directory layout

**2. StateManager Enhancements (src/services/state_manager.py)**:
- ✅ Added `record_provider_metrics()`: Tracks success/error counts, retries, latency per provider
- ✅ Added `get_provider_stats()`: Returns aggregated metrics (success rate, avg/max latency)
- ✅ Added `get_all_provider_stats()`: Returns metrics for all providers
- ✅ Running latency averages calculated correctly (only for requests with latency data)

**3. Integration Tests (tests/integration/test_market_data_ingestion.py)**:
- ✅ 6 new tests covering all three providers in dryrun mode
- ✅ Verified snapshots written to disk with correct structure
- ✅ Confirmed state metrics tracking works
- ✅ Ensured dryrun mode makes no network calls (mocked BreezeClient)

**4. Quality Gates**:
- ✅ Ruff: All files pass linting
- ✅ Format: All files formatted correctly
- ✅ Mypy: Type checking passes (expected pandas warnings)
- ✅ Pytest: 6/6 integration tests passing, 28/28 provider unit tests passing

### Safety Controls

1. **Dryrun by Default**: Unless `*_enabled=true` in config, providers use mock data
2. **SecretsManager**: Credentials loaded securely from .env (plain) or encrypted file
3. **No Network in Dryrun**: Verified via integration tests with mocked BreezeClient
4. **Retry Logic**: Handled transparently by providers (tenacity decorators)
5. **State Tracking**: Every fetch recorded with success/failure, latency, retry count

### Configuration

Providers respect existing configuration flags:
```bash
ORDER_BOOK_ENABLED=true  # Enable real Breeze order book provider
OPTIONS_ENABLED=true     # Enable real Breeze options provider
MACRO_ENABLED=true       # Enable real macro provider (yfinance, etc.)

SECRETS_MODE=plain       # Credential mode (plain or encrypted)
```

**Default**: All providers run in dryrun mode with deterministic mock data.

### Provider Metrics Example

```python
from src.services.state_manager import StateManager

manager = StateManager("data/state/provider_metrics.json")

# Get stats for order book provider
stats = manager.get_provider_stats("order_book")
print(f"Success rate: {stats['success_rate']}%")
print(f"Avg latency: {stats['avg_latency_ms']}ms")
print(f"Total retries: {stats['total_retries']}")
```

---

## Phase 5: Real-Time Streaming & Background Ingestion ✅

**Status**: **COMPLETED**
**Implementation Date**: 2025-10-13

### Overview
Phase 5 adds real-time market data streaming capabilities and foundational infrastructure for continuous data ingestion.

### Components Implemented

#### 1. Order Book Streaming Script (`scripts/stream_order_book.py`)
**Purpose**: Stream real-time order book updates via WebSocket with graceful shutdown and state tracking.

**Features**:
- MockWebSocketClient for dryrun mode (deterministic mock data)
- BreezeWebSocketClient placeholder for live streaming (pending SDK integration)
- Circular buffers for in-memory snapshot caching (configurable size)
- Graceful shutdown handlers (SIGINT/SIGTERM, thread-safe)
- State manager integration for heartbeat tracking
- Automatic cache file updates (latest.json per symbol)

**Usage**:
```bash
# Dryrun mode with mock WebSocket
python scripts/stream_order_book.py --dryrun --symbols RELIANCE TCS --interval 1

# Live mode (requires Breeze credentials)
STREAMING_ENABLED=true python scripts/stream_order_book.py --symbols RELIANCE --interval 1
```

**Key Classes**:
- `OrderBookStreamer`: Main streaming orchestrator with buffer management
- `MockWebSocketClient`: Deterministic mock for testing (round-robin updates)
- `BreezeWebSocketClient`: Placeholder for real Breeze WebSocket (TODO)

#### 2. Streaming Configuration (`src/app/config.py`)
**New Settings**:
```python
# Master Control
streaming_enabled: bool = False  # Disabled by default for safety

# Buffer Configuration
streaming_buffer_size: int = 100  # Max snapshots per symbol
streaming_update_interval_seconds: int = 1  # Update frequency

# Health Monitoring
streaming_heartbeat_timeout_seconds: int = 30  # Timeout for health alerts
streaming_max_consecutive_errors: int = 10  # Error threshold

# Background Ingestion
background_ingestion_enabled: bool = False  # Daemon mode (future)
background_ingestion_interval_seconds: int = 300  # 5 min fetch interval
```

#### 3. StateManager Heartbeat Tracking (`src/services/state_manager.py`)
**New Methods**:
- `record_streaming_heartbeat(stream_type, symbols, stats)`: Record heartbeat with metadata
- `get_streaming_health(stream_type)`: Check health status with timeout detection
- `get_all_streaming_health()`: Get health for all active streams

**Health Status Schema**:
```python
{
    "exists": True,
    "is_healthy": True,  # Based on heartbeat timeout
    "last_heartbeat": "2025-10-13T01:45:00",
    "time_since_heartbeat_seconds": 5.2,
    "timeout_threshold_seconds": 30,
    "symbols": ["RELIANCE", "TCS"],
    "update_count": 1250,
    "error_count": 3
}
```

#### 4. Integration Tests (`tests/integration/test_market_data_ingestion.py`)
**New Tests** (5 total):
- `test_streaming_order_book_dryrun`: End-to-end streaming with threading
- `test_streaming_heartbeat_tracking`: StateManager heartbeat recording/retrieval
- `test_streaming_health_timeout`: Timeout detection for stale streams
- `test_streaming_buffer_management`: Circular buffer behavior (maxlen=5)
- `test_streaming_mock_websocket_deterministic`: MockWebSocketClient data validation

### Implementation Highlights

**Circular Buffer Management**:
```python
from collections import deque

# Initialize buffers (maxlen enforces circular behavior)
self.buffers = {
    symbol: deque(maxlen=buffer_size) for symbol in symbols
}

# Retrieve snapshots (newest first)
def get_buffer_snapshots(self, symbol: str, limit: int | None = None) -> list:
    snapshots = list(self.buffers[symbol])
    snapshots.reverse()  # Newest first
    return snapshots[:limit] if limit else snapshots
```

**Graceful Shutdown**:
```python
def start(self) -> None:
    self.running = True

    # Register signal handlers (thread-safe)
    try:
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
    except ValueError:
        # Signals only work in main thread
        pass

    # Streaming loop checks self.running flag
    while self.running:
        update = self.ws_client.receive(timeout=5)
        if update:
            self._process_update(update)
```

**Heartbeat Tracking**:
```python
# Record heartbeat every loop iteration
self.state_manager.record_streaming_heartbeat(
    stream_type="order_book",
    symbols=self.symbols,
    stats=self.stats,  # {"updates": 100, "errors": 2}
)

# Monitor health (e.g., from MonitoringService)
health = state_manager.get_streaming_health("order_book")
if not health["is_healthy"]:
    alert(f"Stream timeout: {health['time_since_heartbeat_seconds']}s")
```

### Safety Controls
1. **Disabled by Default**: `STREAMING_ENABLED=false` in config
2. **Dryrun Required**: Must test with `--dryrun` before live use
3. **Buffer Limits**: Prevents memory exhaustion (maxlen on deque)
4. **Graceful Shutdown**: Clean disconnection on SIGTERM/SIGINT
5. **Heartbeat Monitoring**: Automatic health checks via StateManager
6. **Thread Safety**: Signal handlers conditionally registered (main thread only)

### Testing Results
- **5 new tests added** (100% pass rate)
- **Total tests**: 594 passing (up from 589)
- **Coverage**: Streaming logic, buffer management, health monitoring, mock WebSocket

### Known Limitations
- **BreezeWebSocketClient is a placeholder**: Real Breeze SDK WebSocket integration pending
- **Background ingestion not implemented**: Daemon mode deferred to future work

---

## Phase 5b: Streaming DataFeed Integration & Monitoring Alerts ✅

**Status**: **COMPLETED**
**Implementation Date**: 2025-10-13

### Overview
Phase 5b extends Phase 5 streaming capabilities with DataFeed integration and monitoring alerts, enabling live strategies to consume real-time market data and alerting on streaming health issues.

### Components Implemented

#### 1. DataFeed Streaming Helpers (`src/services/data_feed.py`)
**New Functions**:

**`get_latest_order_book(symbol, streaming_cache_dir, fallback_csv_dir)`**:
- Reads latest snapshot from streaming cache (`data/order_book/streaming/{symbol}/latest.json`)
- Falls back to CSV cache if streaming disabled or stale
- Returns order book dict with bids, asks, metadata

**`get_order_book_history(symbol, limit, streaming_cache_dir)`**:
- Returns recent snapshots from streaming cache (currently 1 from cache file)
- For full historical buffer, use `OrderBookStreamer.get_buffer_snapshots()` directly
- Returns list of snapshots (newest first)

**Usage Example**:
```python
from src.services.data_feed import get_latest_order_book

# Get latest order book from streaming cache
snapshot = get_latest_order_book("RELIANCE")
if snapshot:
    best_bid = snapshot["bids"][0]["price"]
    best_ask = snapshot["asks"][0]["price"]
    spread = best_ask - best_bid
```

#### 2. StateManager Buffer Metadata (`src/services/state_manager.py`)
**Enhanced Method**: `record_streaming_heartbeat(..., buffer_metadata)`
- Now accepts optional `buffer_metadata` dict with:
  - `buffer_lengths`: Dict[str, int] - Current buffer size per symbol
  - `last_snapshot_times`: Dict[str, str] - Last snapshot timestamp per symbol
  - `total_capacity`: int - Buffer capacity (maxlen)
- Calculates buffer utilization percentage automatically
- Persists metadata to state file for monitoring

**Enhanced Method**: `get_streaming_health(stream_type)`
- Returns buffer metadata in health dict:
  - `buffer_lengths`, `last_snapshot_times`, `total_capacity`, `buffer_utilization_pct`

**Example**:
```python
from src.services.state_manager import StateManager

manager = StateManager("data/state/streaming.json")
health = manager.get_streaming_health("order_book")

print(f"Buffer utilization: {health['buffer_utilization_pct']}%")
print(f"RELIANCE buffer: {health['buffer_lengths']['RELIANCE']} snapshots")
```

#### 3. MonitoringService Streaming Health Checks (`src/services/monitoring.py`)
**New Method**: `check_streaming_health(state_manager)`
- Monitors all active streaming connections
- Detects lag exceeding `streaming_heartbeat_timeout_seconds`
- Warns on high buffer utilization (>90%)
- Emits alerts with escalating severity (INFO → WARNING → CRITICAL)
- Returns list of `HealthCheckResult` objects

**Alert Conditions**:
- **ERROR**: Lag exceeds timeout threshold
- **WARNING**: Buffer utilization >90% or 2+ consecutive failures
- **CRITICAL**: 3+ consecutive failures
- **OK**: Healthy stream with normal buffer utilization

**Usage Example**:
```python
from src.services.monitoring import MonitoringService
from src.services.state_manager import StateManager

monitoring = MonitoringService(settings)
state_manager = StateManager("data/state/streaming.json")

results = monitoring.check_streaming_health(state_manager)
for result in results:
    if result.status == "ERROR":
        print(f"Alert: {result.message}")
```

#### 4. Streaming Script Buffer Metadata (`scripts/stream_order_book.py`)
**New Method**: `OrderBookStreamer._get_buffer_metadata()`
- Collects current buffer lengths per symbol
- Extracts last snapshot timestamps from buffers
- Returns metadata dict for heartbeat tracking

**Enhanced**: Heartbeat recording now includes buffer metadata:
```python
buffer_metadata = self._get_buffer_metadata()
self.state_manager.record_streaming_heartbeat(
    stream_type="order_book",
    symbols=self.symbols,
    stats=self.stats,
    buffer_metadata=buffer_metadata,
)
```

**Periodic Logging**: Logs buffer stats every 100 updates for monitoring

#### 5. Integration Tests (`tests/integration/test_market_streaming.py`)
**New Tests** (9 total, all passing):
- `test_data_feed_get_latest_order_book`: DataFeed reads streaming cache
- `test_data_feed_get_latest_order_book_fallback`: Falls back to CSV cache
- `test_data_feed_get_order_book_history`: Returns snapshot history
- `test_state_manager_buffer_metadata`: Buffer metadata persistence
- `test_monitoring_service_streaming_health_check`: Healthy stream detection
- `test_monitoring_service_streaming_lag_alert`: Lag detection and alerting
- `test_monitoring_service_streaming_high_buffer_utilization`: Buffer warnings
- `test_streaming_integration_with_buffer_metadata`: End-to-end integration
- `test_monitoring_service_no_active_streams`: Handles no active streams

### Implementation Highlights

**Buffer Metadata Collection**:
```python
def _get_buffer_metadata(self) -> dict[str, Any]:
    buffer_lengths = {}
    last_snapshot_times = {}

    for symbol, buffer in self.buffers.items():
        buffer_lengths[symbol] = len(buffer)
        if buffer:
            latest = buffer[-1]
            if isinstance(latest, dict) and "timestamp" in latest:
                last_snapshot_times[symbol] = latest["timestamp"]

    return {
        "buffer_lengths": buffer_lengths,
        "last_snapshot_times": last_snapshot_times,
        "total_capacity": self.buffer_size,
    }
```

**Streaming Health Check with Alerts**:
```python
def check_streaming_health(self, state_manager) -> list[HealthCheckResult]:
    all_streams = state_manager.get_all_streaming_health()

    for stream_type, health in all_streams.items():
        if not health.get("is_healthy", False):
            # Track consecutive failures
            consecutive_failures = self._get_consecutive_failures(check_name)

            # Escalate severity based on failure count
            if consecutive_failures >= 3:
                severity = "CRITICAL"
            elif consecutive_failures >= 2:
                severity = "WARNING"

            # Emit alert
            alert = Alert(
                timestamp=datetime.now().isoformat(),
                severity=severity,
                rule=f"streaming_lag_{stream_type}",
                message=f"Streaming lag: {lag_seconds:.1f}s > {threshold}s",
                context={...},
            )
            self.emit_alert(alert)
```

### Testing Results
- **9 new tests added** (100% pass rate)
- **Total tests**: 603 passing (up from 594)
- **Coverage**: DataFeed helpers, StateManager buffer metadata, MonitoringService alerts, end-to-end streaming

### Safety Controls
1. **Streaming disabled by default**: `STREAMING_ENABLED=false`
2. **Fallback to CSV cache**: DataFeed falls back when streaming unavailable
3. **Alert escalation**: Gradual severity increase prevents alert fatigue
4. **Buffer utilization monitoring**: Warns before buffers fill up
5. **Consecutive failure tracking**: Avoids spurious alerts

### Configuration
No new configuration added. Uses existing Phase 5 settings:
- `streaming_heartbeat_timeout_seconds` (default 30s)
- `streaming_max_consecutive_errors` (default 10)
- `streaming_buffer_size` (default 100)

### Future Work (Phase 5c - Optional)
- Real Breeze WebSocket SDK integration
- Background daemon mode for continuous ingestion
- WebSocket reconnection logic with exponential backoff
- Dashboard panels for streaming status visualization

---

## Future Roadmap (Phase 6+)

### Phase 6: Advanced Streaming Features
- DataFeed streaming buffer integration for live strategies
- MonitoringService real-time alerts for streaming health
- Advanced compression and retention policies
- Live options greeks calculation
- Integration with additional data sources (RBI, Bloomberg, etc.)

## Dependencies
- Existing: `src/app/config.py`, `src/services/state_manager.py`, `src/services/data_feed.py`
- Existing scripts: `scripts/fetch_historical_data.py`, `scripts/fetch_sentiment_snapshots.py`
- New libraries: None (use existing requests, tenacity, pandas, loguru)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Storage bloat from order book snapshots | Configurable retention policies, compression (Phase 2) |
| Rate limiting by providers | Built-in rate limiters, respect API quotas |
| Complex options chain data | Normalized schema, validation, graceful degradation |
| Missing macro data providers | Stub provider for Phase 1, real providers in Phase 2 |
| Feature engineering complexity | Deferred to Phase 2, focus on ingestion first |

## Success Metrics
- All 3 ingestion scripts run successfully in dryrun mode
- Integration tests pass (100% coverage of ingestion workflows)
- StateManager correctly tracks fetch metadata for all 3 data types
- DataFeed can load cached market data without errors
- Documentation complete and accurate
- Quality gates pass (ruff, mypy, pytest)
