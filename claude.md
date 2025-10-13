# Claude Code Onboarding Guide

**Role**: Claude Code agent in the SenseQuant BMAD workflow
**Mission**: Implement production-grade algorithmic trading features with strict quality gates and comprehensive testing

---

## Update (October 13, 2025)

### Latest Status Summary

**All Completed Phases**:
- ✅ **US-000 to US-024**: Core engine, strategies, risk management, ML pipeline, backtesting, historical data ingestion, batch training, student model integration
- ✅ **US-029 Phase 1-3**: Order book/options/macro data ingestion, feature engineering (15+ market microstructure features), strategy integration with market data filters
- ✅ **US-029 Phase 4**: Breeze provider integration, SecretsManager credentials, StateManager provider metrics, ingestion pipeline with retry/backoff
- ✅ **US-029 Phase 5**: Real-time order book streaming script, MockWebSocketClient for dryrun, StateManager heartbeat tracking, streaming health monitoring

**Current Safety Defaults** (All disabled for safety):
- `MODE=dryrun` - Engine runs without real orders
- `ORDER_BOOK_ENABLED=false` - Order book ingestion disabled
- `OPTIONS_ENABLED=false` - Options chain ingestion disabled
- `MACRO_ENABLED=false` - Macro indicators ingestion disabled
- `STREAMING_ENABLED=false` - Real-time streaming disabled
- `ENABLE_ORDER_BOOK_FEATURES=false` - Order book features disabled in strategies
- `ENABLE_OPTIONS_FEATURES=false` - Options features disabled in strategies
- `ENABLE_MACRO_FEATURES=false` - Macro features disabled in strategies

**Test Suite**: 594/594 tests passing (100% success rate)

### Next Suggested Actions

1. **Run Historical Training (No Dryrun)**:
   ```bash
   python scripts/run_historical_training.py \
     --start-date 2024-01-01 \
     --end-date 2024-12-31 \
     --symbols RELIANCE TCS INFY \
     --no-dryrun
   ```
   - Trains Teacher and Student models on real historical data
   - Outputs to `data/models/live_candidate_YYYYMMDD_HHMMSS/`
   - Creates promotion checklist for validation

2. **Perform Model Validation**:
   ```bash
   python scripts/run_model_validation.py \
     --model-dir data/models/live_candidate_20250113_120000 \
     --baseline-dir data/models/production
   ```
   - Validates precision, recall, Sharpe ratio against baseline
   - Generates validation report in `release/audit_*/`
   - Checks promotion criteria (min uplift thresholds)

3. **Run Statistical Tests**:
   ```bash
   python scripts/run_statistical_tests.py \
     --model-dir data/models/live_candidate_20250113_120000 \
     --output-dir release/audit_20250113
   ```
   - Performs cross-validation, permutation tests, calibration checks
   - Outputs statistical analysis reports
   - Verifies model robustness and generalization

4. **Consider US-029 Phase 5b/6**:
   - **Phase 5b**: Real Breeze WebSocket integration, DataFeed streaming buffers, MonitoringService streaming alerts
   - **Phase 6**: DataFeed live strategy integration, advanced compression, live Greeks calculation

5. **Integrate Market Data Features into Live Strategies**:
   - Enable order book features: `ENABLE_ORDER_BOOK_FEATURES=true`
   - Enable options features: `ENABLE_OPTIONS_FEATURES=true`
   - Enable macro features: `ENABLE_MACRO_FEATURES=true`
   - Configure thresholds (spread filters, IV gates, macro regime filters)
   - Run extensive backtests before live deployment

**Important**: All market data providers use SecretsManager with credentials from `.env`. No live trades occur without explicit enable flags (`MODE=live`, `ORDER_BOOK_ENABLED=true`, etc.). Always test with `--dryrun` first.

---

## 1. Overview

**SenseQuant** is a multi-strategy algorithmic trading system for Indian equities (NSE) via ICICI Breeze API.

**Core Architecture**:
- **Engine**: `src/services/engine.py` - Main trading loop (intraday + swing)
- **Strategies**: `src/domain/strategies/{intraday,swing}.py` - Signal generation with SMA/EMA crossovers
- **Risk Management**: `src/services/risk_manager.py` - Position sizing, circuit breakers, fees/slippage
- **ML Pipeline**: Teacher-Student learning for swing predictions
- **Backtesting**: Historical strategy validation with performance metrics

**Key Documentation**:
- Product Requirements: `docs/prd.md`
- System Architecture: `docs/architecture.md`
- User Stories: `docs/stories/` (US-000 through US-029)
- Market Data Integration: `docs/stories/us-029-market-data.md` (Phases 1-4 complete)

---

## 2. Roles & Workflow

**BMAD Workflow** (Planner → Claude → QA loop):
1. **Planner** (Business Analyst + Mad Architect) defines story, acceptance criteria
2. **Claude Code** implements feature with tests
3. **QA** validates quality gates, integration testing

**Configuration**: `bmad.project.yaml` defines active workflows and agent roles

**Active Process**:
- Implement story per spec in `docs/stories/us-*.md`
- Run all quality gates before marking complete
- Update story document with implementation notes
- Report PASS/FAIL status for each gate

---

## 3. Quality Gates

**Required Commands** (all must PASS before story completion):

```bash
# Linting
python -m ruff check .

# Formatting
python -m ruff format --check .

# Type checking
python -m mypy src/

# Tests
python -m pytest -q
```

**Expectations**:
- **ruff check**: 0 errors in `src/`, `tests/`, `scripts/` (external files ignored)
- **ruff format**: All files formatted correctly
- **mypy**: No type errors in `src/` and `scripts/`
- **pytest**: 100% pass rate on implemented features (skipped tests allowed for future work)

**Current Status** (as of US-029 Phase 5):
- ruff check: PASS (0 project errors, 22 pre-existing in external files)
- ruff format: PASS (115 files formatted)
- mypy: PASS (0 project errors, 95 pre-existing in external files)
- pytest: PASS (594/594 passing, 100% success rate)

---

## 4. Current Capabilities

**Completed Stories**: US-000 through US-029 Phase 4

**Key Modules**:

| Module | Path | Purpose |
|--------|------|---------|
| Engine | `src/services/engine.py` | Main trading orchestrator (intraday + swing) |
| Breeze Client | `src/adapters/breeze_client.py` | ICICI Breeze API wrapper with retry/rate limiting |
| Risk Manager | `src/services/risk_manager.py` | Position sizing, circuit breakers, capital tracking |
| Sentiment Cache | `src/services/sentiment_cache.py` | Rate-limited sentiment provider with caching |
| Feature Library | `src/domain/features.py` | Technical indicators (SMA, EMA, RSI, MACD, ATR, etc.) |
| Teacher Labeler | `src/services/teacher_student.py` | Generate labeled training data from historical bars |
| Student Model | `src/services/teacher_student.py` | Lightweight inference for swing predictions |
| Backtester | `src/services/backtester.py` | Historical strategy validation with metrics |
| Intraday Strategy | `src/domain/strategies/intraday.py` | Momentum-based intraday entries/exits |
| Swing Strategy | `src/domain/strategies/swing.py` | SMA crossover with TP/SL/max-hold logic |
| Market Data Providers | `src/adapters/market_data_providers.py` | Factory pattern for order book, options, macro providers |
| State Manager | `src/services/state_manager.py` | Persistent state tracking with provider metrics |
| Data Feed | `src/services/data_feed.py` | Market data normalization and caching layer |

**Market Data Integration (US-029 Phase 1-4)**:
- **Order Book Snapshots**: Real-time L2 depth data via Breeze API
- **Options Chain Data**: Strike prices, IV, Greeks, OI for equity options
- **Macro Indicators**: NIFTY50, INDIAVIX, USDINR, IN10Y yield tracking
- **Feature Engineering**: 15+ market microstructure features (spread, imbalance, IV skew, etc.)
- **Strategy Integration**: Order book filters, IV-based sizing, macro regime gates
- **Provider Pattern**: Pluggable data sources with dryrun mocking
- **Ingestion Scripts**: `scripts/fetch_order_book.py`, `fetch_options_data.py`, `fetch_macro_data.py`
- **Credentials**: SecretsManager with plain/encrypted mode support
- **Metrics Tracking**: Per-provider success rates, latency, retry counts

**Test Coverage**:
- **Total**: 594 tests passing (100% pass rate)
- Unit tests: `tests/unit/` (200+ tests)
- Integration tests: `tests/integration/` (85+ tests including market data pipeline and streaming)
- All critical paths covered with edge case testing

---

## 5. Key Commands & Scripts

**Make Targets**:
```bash
make test          # Run pytest with coverage
make lint          # Run ruff check
make format        # Auto-format with ruff
make typecheck     # Run mypy
make test-cov      # Generate HTML coverage report
```

**Training Teacher Model**:
```bash
# Train Teacher for swing prediction
python scripts/train_teacher.py \
  --symbols RELIANCE TCS INFY \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir data/models/teacher_v1

# Output: model.pkl, labels.csv, metadata.json in output-dir
```

**Running Backtests**:
```bash
# Backtest swing strategy
python scripts/backtest.py \
  --symbols RELIANCE TCS \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strategy swing \
  --initial-capital 1000000

# Outputs: summary.json, equity.csv, trades.csv in data/backtests/

# Multi-strategy backtest
python scripts/backtest.py \
  --symbols RELIANCE \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strategy both \
  --verbose
```

**Running Live Engine**:
```bash
# Dryrun mode (no real orders, no network calls except auth)
MODE=dryrun python -m src.services.engine

# Live mode (requires valid Breeze credentials)
MODE=live python -m src.services.engine
```

**Market Data Ingestion (US-029)**:
```bash
# Fetch order book snapshots (dryrun mode)
python scripts/fetch_order_book.py \
  --symbols RELIANCE TCS \
  --output-dir data/order_book \
  --depth-levels 5 \
  --dryrun

# Fetch options chain data
python scripts/fetch_options_data.py \
  --symbols RELIANCE TCS \
  --output-dir data/options \
  --dryrun

# Fetch macro indicators
python scripts/fetch_macro_data.py \
  --indicators NIFTY50 INDIAVIX USDINR \
  --output-dir data/macro \
  --dryrun

# Live mode (requires Breeze credentials in .env)
SECRETS_MODE=plain python scripts/fetch_order_book.py \
  --symbols RELIANCE \
  --output-dir data/order_book \
  --depth-levels 5
```

**Real-Time Streaming (US-029 Phase 5)**:
```bash
# Stream order book updates in dryrun mode
python scripts/stream_order_book.py \
  --dryrun \
  --symbols RELIANCE TCS \
  --interval 1 \
  --buffer-size 100

# Live streaming (requires Breeze credentials, placeholder implementation)
STREAMING_ENABLED=true python scripts/stream_order_book.py \
  --symbols RELIANCE \
  --interval 1
```

**Orchestration Scripts (End-to-End Workflows)**:
```bash
# 1. Historical Training Pipeline
# Fetches historical data, trains Teacher, trains Student, generates promotion checklist
python scripts/run_historical_training.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols RELIANCE TCS INFY \
  --no-dryrun  # Use real Breeze API for historical data

# Key flags:
#   --dryrun: Use mock data (testing only)
#   --no-dryrun: Fetch real historical data from Breeze API
#   --skip-fetch: Skip data fetch if already downloaded
#   --skip-teacher: Skip Teacher training (use existing)
#   --skip-student: Skip Student training
# Output: data/models/live_candidate_YYYYMMDD_HHMMSS/
#         - teacher/model.pkl, labels.csv, metadata.json
#         - student/model.pkl, metadata.json
#         - promotion_checklist.md

# 2. Model Validation Pipeline
# Validates new model against baseline (production model)
python scripts/run_model_validation.py \
  --model-dir data/models/live_candidate_20250113_120000 \
  --baseline-dir data/models/production \
  --test-window 90  # Days of test data

# Prerequisites:
#   - Trained model in --model-dir
#   - Baseline model in --baseline-dir (optional)
#   - Historical data available
# Output: release/audit_YYYYMMDD_HHMMSS/
#         - validation_report.md (precision, recall, Sharpe uplift)
#         - backtest_comparison.json
#         - PASS/FAIL verdict based on promotion criteria

# 3. Statistical Testing Pipeline
# Runs comprehensive statistical tests (cross-validation, permutation, calibration)
python scripts/run_statistical_tests.py \
  --model-dir data/models/live_candidate_20250113_120000 \
  --output-dir release/audit_20250113 \
  --n-folds 5  # K-fold cross-validation
  --n-permutations 100  # Permutation test iterations

# Prerequisites:
#   - Trained model with metadata
#   - Historical data with labels
# Output: release/audit_*/
#         - cross_validation_report.json (fold-wise metrics)
#         - permutation_test_report.json (p-values)
#         - calibration_report.json (reliability curves)
#         - statistical_summary.md
```

**Notes**:
- **Dryrun mode**: Generates deterministic order IDs, simulates fills, no Breeze SDK calls
- **Backtest mode**: Historical simulation with Breeze API data, no real trading
- **Live mode**: Real orders placed via Breeze API (use with caution)
- **Market Data Dryrun**: Uses deterministic mock providers, no network calls
- **Orchestration Scripts**: Combine multiple steps (fetch → train → validate → test) with checkpoints
- **Output Directories**:
  - Training: `data/models/live_candidate_*/` (timestamped candidates)
  - Validation: `release/audit_*/` (timestamped audit trails)
  - Streaming caches: `data/order_book/streaming/{symbol}/latest.json`

---

## 6. Coding Guidelines

**Type Safety**:
- All functions must have type hints (params + return types)
- Use `from __future__ import annotations` for forward references
- Run `mypy src/` before committing - must pass with 0 errors

**Docstrings**:
- All public functions/classes require docstrings
- Format: Google-style with Args, Returns, Raises sections
- Include usage examples for complex functions

**Logging**:
- Use `loguru` logger with structured `extra={"component": "..."}` tags
- Component tags: `"engine"`, `"risk"`, `"sentiment"`, `"teacher"`, `"student"`, `"backtest"`, etc.
- Example:
  ```python
  logger.info("Signal generated", extra={"component": "swing", "symbol": symbol, "action": action})
  ```

**Dryrun Network Restrictions**:
- **NEVER** make real network calls in dryrun mode (except authentication)
- BreezeClient automatically stubs API calls when `dry_run=True`
- Sentiment provider returns neutral score in dryrun
- Tests should mock all external dependencies

**Artifact Locations**:
- **Teacher models**: `data/models/teacher_*/` (model.pkl, labels.csv, metadata.json)
- **Student models**: `data/models/student_*/` (model.pkl, metadata.json)
- **Live candidate models**: `data/models/live_candidate_YYYYMMDD_HHMMSS/` (trained models awaiting validation)
  - `teacher/` - Teacher model artifacts
  - `student/` - Student model artifacts
  - `promotion_checklist.md` - Validation checklist for promotion
- **Production models**: `data/models/production/` (currently deployed models)
- **Audit trails**: `release/audit_YYYYMMDD_HHMMSS/` (validation reports, statistical tests)
  - `validation_report.md` - Model performance vs baseline
  - `cross_validation_report.json` - K-fold CV metrics
  - `permutation_test_report.json` - Feature importance p-values
  - `calibration_report.json` - Reliability curves
  - `statistical_summary.md` - Comprehensive test summary
- **Backtests**: `data/backtests/` (summary JSON, equity CSV, trades CSV)
- **Trade journals**: `data/journals/` (daily trade logs with timestamps)
- **Logs**: `logs/` (component-specific log files with rotation)
- **Order Book Snapshots**: `data/order_book/{symbol}/{date}/` (JSON snapshots with L2 depth)
- **Streaming caches**: `data/order_book/streaming/{symbol}/latest.json` (real-time cache)
- **Options Chain Data**: `data/options/{symbol}/{date}/` (chain.json with strikes, IV, Greeks)
- **Macro Indicators**: `data/macro/{indicator}/{date}/` (JSON time series)
- **State Files**: `data/state/` (provider metrics, streaming heartbeats)

**Data Schemas**:
- Use `@dataclass` for domain types in `src/domain/types.py`
- Bar schema: `ts`, `open`, `high`, `low`, `close`, `volume` (all required)
- Timestamps must be timezone-aware (`Asia/Kolkata`)
- Prices in float, quantities in int

**Testing Requirements**:
- Every new feature requires unit tests
- Integration tests for multi-component workflows
- Use `pytest.mark.skip` for unimplemented features with TODO comment
- Mock external dependencies (Breeze API, sentiment providers)
- Fixtures in `conftest.py` for reusable test data

**Error Handling**:
- Raise `ValueError` for invalid inputs with clear messages
- Raise `RuntimeError` for operational failures
- Log exceptions with `logger.exception()` before re-raising
- Never silence errors - propagate or handle explicitly

---

## 7. Troubleshooting

### Market Data Providers & Credentials

**Issue**: Provider fetch fails with "Missing credentials"
**Solution**:
- Ensure `.env` file exists in repo root with Breeze credentials:
  ```bash
  BREEZE_API_KEY=your_api_key
  BREEZE_API_SECRET=your_api_secret
  BREEZE_SESSION_TOKEN=your_session_token
  ```
- Set `SECRETS_MODE=plain` for plain text credentials (default)
- Set `SECRETS_MODE=encrypted` if using encrypted credentials vault
- Run with `--dryrun` flag to test without credentials (uses mock providers)

**Issue**: Rate limiting errors from Breeze API
**Solution**:
- Check `order_book_retry_limit`, `options_retry_limit`, `macro_retry_limit` in config
- Adjust `*_retry_backoff_seconds` for exponential backoff tuning
- Use `--incremental` flag to skip existing snapshots
- Enable caching: `data_feed_enable_cache=true`

### Real-Time Streaming

**Issue**: MockWebSocketClient vs Breeze credentials
**Context**:
- `--dryrun` mode uses `MockWebSocketClient` (deterministic, no network calls)
- Live mode requires `BreezeWebSocketClient` with valid credentials
- Current implementation: `BreezeWebSocketClient` is a placeholder (pending real Breeze SDK WebSocket support)

**Solution**:
- Always test with `--dryrun` first:
  ```bash
  python scripts/stream_order_book.py --dryrun --symbols RELIANCE
  ```
- For live streaming (when SDK available):
  ```bash
  STREAMING_ENABLED=true SECRETS_MODE=plain \
    python scripts/stream_order_book.py --symbols RELIANCE
  ```

**Issue**: Streaming health shows "unhealthy" or timeout
**Solution**:
- Check StateManager heartbeat tracking:
  ```python
  from src.services.state_manager import StateManager
  manager = StateManager("data/order_book/state/streaming.json")
  health = manager.get_streaming_health("order_book")
  print(health)  # Check time_since_heartbeat_seconds
  ```
- Verify streaming process is running (check logs in `logs/`)
- Check `streaming_heartbeat_timeout_seconds` setting (default 30s)
- Review MonitoringService dashboard for streaming alerts (if enabled)

**Issue**: Streaming buffer memory issues
**Solution**:
- Reduce `streaming_buffer_size` (default 100, min 10, max 1000)
- Circular buffers automatically evict old snapshots (no manual cleanup needed)
- Monitor buffer usage:
  ```python
  streamer.get_buffer_snapshots("RELIANCE", limit=5)  # Last 5 snapshots
  ```

### Model Training & Validation

**Issue**: Historical training fails with "No data found"
**Solution**:
- Verify historical data exists: `ls data/historical/`
- Run data fetch first: `--skip-teacher --skip-student` to test fetch only
- Check date range: `--start-date` and `--end-date` must have trading days
- Ensure symbols are valid NSE symbols

**Issue**: Validation fails with "Baseline model not found"
**Solution**:
- Baseline is optional for first run (compares to ideal metrics)
- Create production baseline:
  ```bash
  mkdir -p data/models/production
  cp -r data/models/live_candidate_*/student data/models/production/
  ```
- Or skip baseline comparison: omit `--baseline-dir` flag

**Issue**: Statistical tests show poor calibration
**Solution**:
- Review calibration report in `release/audit_*/calibration_report.json`
- Check if model is overconfident (predicted probabilities vs actual frequencies)
- Consider re-training with different hyperparameters
- Increase training data size or adjust feature engineering

### General Debugging

**Dryrun vs Live Confusion**:
- **Dryrun mode** (`--dryrun` or `MODE=dryrun`):
  - No real orders, no real network calls (except auth)
  - Deterministic mock data for reproducibility
  - Safe for testing and development
- **Live mode** (`--no-dryrun` or `MODE=live`):
  - Real API calls, real data, real orders (if enabled)
  - Requires valid credentials in `.env`
  - Use only after extensive testing

**No Live Trades Without Explicit Flags**:
- Engine requires `MODE=live` to place real orders
- Market data requires `ORDER_BOOK_ENABLED=true`, `OPTIONS_ENABLED=true`, etc.
- Streaming requires `STREAMING_ENABLED=true`
- All features are **disabled by default** for safety

**Logs and Monitoring**:
- Check `logs/` directory for component-specific logs
- Use `extra={"component": "..."}` tags in logs for filtering
- StateManager tracks provider metrics: `data/state/provider_metrics.json`
- Streaming heartbeats: `data/order_book/state/streaming.json`

---

## 8. Roadmap / Pending Work

**Completed** (US-000 through US-029 Phase 4):
- ✅ Core engine with intraday + swing strategies
- ✅ Risk management with position sizing and circuit breakers
- ✅ Sentiment integration with caching
- ✅ Feature library (25+ technical indicators + market microstructure)
- ✅ Teacher-Student ML pipeline
- ✅ Backtest engine with comprehensive metrics
- ✅ **Order Book L2 depth ingestion** - Real-time snapshots via Breeze API
- ✅ **Options chain data** - IV, Greeks, OI for equity options
- ✅ **Macro indicators** - Index tracking (NIFTY50, VIX, FX, yields)
- ✅ **Market data feature engineering** - Spread, imbalance, IV skew, macro regime
- ✅ **Strategy integration** - Order book filters, IV-based sizing, macro gates
- ✅ **Provider pattern** - Pluggable data sources with dryrun mocking
- ✅ **Ingestion pipeline** - Breeze provider integration with credentials + metrics

**Completed** (US-029 Phase 5):
- ✅ **Real-time WebSocket streaming** - Order book streaming script with mock/live modes
- ✅ **Streaming configuration** - Buffer size, update interval, heartbeat timeout settings
- ✅ **Heartbeat tracking** - StateManager methods for stream health monitoring
- ✅ **Integration tests** - 5 tests for streaming, buffers, and health checks

**Upcoming Focus** (US-030+):
- **Real-time monitoring dashboard** - Web UI for live positions, PnL, metrics
- **Advanced order types** - Bracket orders, trailing stops, iceberg orders
- **Multi-timeframe analysis** - Combine 1min, 5min, daily signals
- **Walk-forward optimization** - Automated parameter tuning with cross-validation
- **Alert system** - Telegram/Email notifications for signals, fills, risk events
- **Portfolio optimization** - Multi-asset allocation with correlation analysis

**Outstanding Enhancements**:
- **CSV data source**: Backtest with custom historical data (currently only Breeze API)
- **Teacher labels data source**: Backtest using pre-labeled Teacher data
- **Real sentiment providers**: Integrate NewsAPI, Twitter, FinBERT (currently mock/cache only)
- **Intraday backtest**: Full minute-level simulation (currently simplified/stub)
- **Performance optimization**: Vectorized feature calculations, parallel backtests
- **Monitoring & alerting**: Prometheus metrics, Grafana dashboards

**Known Limitations**:
- Intraday strategy uses simplified logic (needs minute-level OHLCV)
- Sentiment analysis is mocked (requires real API integrations)
- Backtester only supports swing strategy fully (intraday stub)
- No position rebalancing or portfolio-level risk management yet
- BreezeWebSocketClient is placeholder (pending real Breeze SDK WebSocket support)
- Background daemon ingestion not yet implemented (future Phase 5b)

**Next Steps**:
1. Review active story in `docs/stories/` for current sprint
2. Implement feature per acceptance criteria
3. Write comprehensive tests (unit + integration)
4. Run all quality gates and report status
5. Update story document with implementation notes
6. Commit with descriptive message referencing story ID

---

**Questions?** Refer to:
- PRD: `docs/prd.md` - Product requirements and business context
- Architecture: `docs/architecture.md` - System design and component interactions
- Stories: `docs/stories/` - Detailed feature specifications with acceptance criteria
- Code: Start with `src/services/engine.py` and follow imports

**Quality Mantra**: No code ships without passing all quality gates. Period.
