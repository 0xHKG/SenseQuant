# US-024: Historical Data Ingestion & Batch Training (Phases 1, 2 & 3)

## Problem Statement

The current system lacks automated historical data acquisition and batch training capabilities for the Teacher model. To enable comprehensive backtesting, model training across multiple symbols, and systematic parameter optimization, we need:

1. **Automated Historical Data Ingestion**: Download OHLCV + sentiment snapshots for configurable symbol lists and date ranges, with robust caching, retry logic, and error handling.

2. **Batch Teacher Training**: Execute Teacher model training across multiple symbols/windows, tracking artifacts and metrics for each batch run.

3. **Storage Organization**: Structured directory layout for historical data and training artifacts to support reproducibility and versioning.

**Scope for Phase 1**: Data acquisition infrastructure and batch teacher training. Student model training integration will be handled in a future phase.

## Acceptance Criteria

### AC-1: Historical Data Ingestion Script
- [ ] `scripts/fetch_historical_data.py` downloads OHLCV data for configured symbols and date ranges
- [ ] Stores data as CSVs under `data/historical/<symbol>/<interval>/YYYY-MM-DD.csv`
- [ ] Implements caching to avoid re-downloading existing data
- [ ] Uses exponential backoff retry logic for API failures (3 retries, 2s/4s/8s delays)
- [ ] Supports dryrun mode that logs intended actions without network calls
- [ ] Validates downloaded data (non-empty, expected columns, date range coverage)
- [ ] Logs progress with structured output (symbol, date range, file count, errors)

### AC-2: Batch Teacher Training Script
- [ ] `scripts/train_teacher_batch.py` iterates through configured symbols and training windows
- [ ] Invokes teacher training for each symbol/window combination
- [ ] Records batch metadata in `data/models/<timestamp>/teacher_runs.json`
- [ ] Logs: symbol, date range, artifacts path, training metrics, status (success/failure)
- [ ] Supports resuming partial batches (skip already-trained windows)
- [ ] Implements parallel execution option (--workers N) with default sequential
- [ ] Provides summary report with success/failure counts and aggregate metrics

### AC-3: Configuration Extensions
- [ ] Add `Settings.historical_data_*` fields:
  - `historical_data_symbols: list[str]` (default: ["RELIANCE", "TCS", "INFY"])
  - `historical_data_intervals: list[str]` (default: ["1minute", "5minute", "1day"])
  - `historical_data_start_date: str` (default: "2024-01-01")
  - `historical_data_end_date: str` (default: "2024-12-31")
  - `historical_data_output_dir: str` (default: "data/historical")
  - `historical_data_retry_limit: int` (default: 3)
  - `historical_data_retry_backoff_seconds: int` (default: 2)
- [ ] Add `Settings.batch_training_*` fields:
  - `batch_training_window_days: int` (default: 90)
  - `batch_training_forecast_horizon_days: int` (default: 7)
  - `batch_training_output_dir: str` (default: "data/models")
  - `batch_training_enabled: bool` (default: False)

### AC-4: Teacher Service Extensions
- [ ] Add `TeacherStudentService.log_batch_metadata()` method
- [ ] Accepts: symbol, date_range, artifacts_path, metrics, status
- [ ] Appends to batch run log file in JSON Lines format
- [ ] Includes timestamp, batch_id, and execution metadata

### AC-5: Integration Testing
- [ ] Test `test_historical_data_fetch()` verifies:
  - CSVs created under correct directory structure
  - Caching prevents duplicate downloads
  - Retry logic handles transient failures
  - Dryrun mode uses mocks without network calls
- [ ] Test `test_batch_teacher_training()` verifies:
  - Teacher training invoked for each symbol/window
  - Batch metadata logged to teacher_runs.json
  - Artifacts paths recorded correctly
  - Summary report generated

### AC-6: Documentation
- [ ] Story document describes Phase 1 scope and acceptance criteria
- [ ] Architecture appendix documents:
  - Data acquisition workflow (fetch → validate → cache)
  - Storage directory layout
  - Retry/backoff policy
  - Batch training procedure
  - CSV schema and validation rules

## Technical Design

### Directory Structure

```
data/
├── historical/                     # Historical OHLCV data
│   ├── RELIANCE/
│   │   ├── 1minute/
│   │   │   ├── 2024-01-01.csv
│   │   │   ├── 2024-01-02.csv
│   │   │   └── ...
│   │   ├── 5minute/
│   │   │   └── ...
│   │   └── 1day/
│   │       └── ...
│   ├── TCS/
│   └── INFY/
└── models/                         # Training artifacts
    ├── 20251012_190000/           # Batch timestamp
    │   ├── teacher_runs.json      # Batch metadata log
    │   ├── RELIANCE_2024Q1/       # Per-symbol artifacts
    │   │   ├── labels.csv
    │   │   ├── features.csv
    │   │   └── teacher_model.pkl
    │   └── TCS_2024Q1/
    │       └── ...
    └── 20251010_120000/
        └── ...
```

### CSV Schema

**OHLCV Files** (`data/historical/<symbol>/<interval>/YYYY-MM-DD.csv`):

```csv
timestamp,open,high,low,close,volume
2024-01-01T09:15:00+05:30,2450.00,2455.50,2448.75,2453.25,125000
2024-01-01T09:16:00+05:30,2453.50,2458.00,2452.00,2456.75,110000
...
```

**Required Columns**:
- `timestamp` (ISO 8601 with timezone)
- `open`, `high`, `low`, `close` (float, prices in INR)
- `volume` (int, share count)

**Validation Rules**:
- No missing values in required columns
- `high >= max(open, close)` and `low <= min(open, close)`
- `volume >= 0`
- Timestamps within expected date range
- Timestamps sorted in ascending order

### Batch Metadata Schema

**File**: `data/models/<timestamp>/teacher_runs.json` (JSON Lines format)

```jsonl
{"batch_id": "batch_20251012_190000", "symbol": "RELIANCE", "date_range": {"start": "2024-01-01", "end": "2024-03-31"}, "artifacts_path": "data/models/20251012_190000/RELIANCE_2024Q1", "metrics": {"precision": 0.72, "recall": 0.68, "f1": 0.70}, "status": "success", "timestamp": "2025-10-12T19:05:23+05:30"}
{"batch_id": "batch_20251012_190000", "symbol": "TCS", "date_range": {"start": "2024-01-01", "end": "2024-03-31"}, "artifacts_path": "data/models/20251012_190000/TCS_2024Q1", "metrics": {"precision": 0.75, "recall": 0.71, "f1": 0.73}, "status": "success", "timestamp": "2025-10-12T19:12:45+05:30"}
{"batch_id": "batch_20251012_190000", "symbol": "INFY", "date_range": {"start": "2024-01-01", "end": "2024-03-31"}, "artifacts_path": "data/models/20251012_190000/INFY_2024Q1", "metrics": null, "status": "failed", "error": "Insufficient data points", "timestamp": "2025-10-12T19:15:12+05:30"}
```

### Retry/Backoff Policy

**API Call Retry Logic**:
- **Retries**: 3 attempts (configurable via `historical_data_retry_limit`)
- **Backoff**: Exponential with base delay (2s, 4s, 8s) (configurable via `historical_data_retry_backoff_seconds`)
- **Retryable Errors**:
  - Connection errors (network timeout, DNS failure)
  - HTTP 429 (rate limit)
  - HTTP 5xx (server errors)
- **Non-Retryable Errors**:
  - HTTP 400 (bad request - malformed parameters)
  - HTTP 401/403 (authentication failure)
  - HTTP 404 (symbol not found)

**Implementation**:
```python
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(settings.historical_data_retry_limit),
    wait=tenacity.wait_exponential(
        multiplier=settings.historical_data_retry_backoff_seconds,
        min=settings.historical_data_retry_backoff_seconds,
        max=30
    ),
    retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number}/{settings.historical_data_retry_limit}"
    )
)
def fetch_with_retry(symbol: str, date: str, interval: str) -> pd.DataFrame:
    ...
```

### Data Acquisition Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. PARAMETER VALIDATION                                                  │
│    - Validate symbol list (non-empty, valid stock codes)                │
│    - Validate date range (start < end, reasonable bounds)               │
│    - Validate intervals (supported by Breeze API)                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. CACHE CHECK                                                           │
│    For each (symbol, interval, date):                                   │
│      - Check if data/historical/<symbol>/<interval>/YYYY-MM-DD.csv exists│
│      - If exists and valid: skip download                               │
│      - If missing or invalid: add to download queue                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. DATA DOWNLOAD (with retry/backoff)                                   │
│    For each item in download queue:                                     │
│      - Call BreezeClient.get_historical(symbol, date, interval)         │
│      - Apply retry logic on failures                                    │
│      - Log progress (symbol, date, success/failure)                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. DATA VALIDATION                                                       │
│    For each downloaded DataFrame:                                        │
│      - Check required columns present                                    │
│      - Validate OHLC relationships (high >= max(O,C), etc.)             │
│      - Check volume non-negative                                         │
│      - Verify timestamp ordering                                         │
│      - Ensure date range coverage                                        │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 5. CSV PERSISTENCE                                                       │
│    For each validated DataFrame:                                         │
│      - Create directory: data/historical/<symbol>/<interval>/          │
│      - Write CSV: YYYY-MM-DD.csv                                        │
│      - Set file permissions (read-only to prevent accidental edits)     │
│      - Log file path and row count                                      │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 6. SUMMARY REPORT                                                        │
│    - Total files downloaded: N                                           │
│    - Total files cached (skipped): M                                     │
│    - Failures: K (with error details)                                    │
│    - Total rows: X                                                       │
│    - Date range coverage: YYYY-MM-DD to YYYY-MM-DD                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Batch Training Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. BATCH INITIALIZATION                                                  │
│    - Generate batch_id: batch_<timestamp>                               │
│    - Create output directory: data/models/<timestamp>/                  │
│    - Initialize teacher_runs.json log file                              │
│    - Load configuration (symbols, windows, horizons)                    │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. WINDOW GENERATION                                                     │
│    For each symbol:                                                      │
│      - Split date range into windows (default: 90 days)                 │
│      - Generate training tasks: (symbol, start_date, end_date)          │
│      - Check resume: skip if artifacts already exist                    │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. TEACHER TRAINING (per window)                                        │
│    For each (symbol, window):                                           │
│      - Load historical data from data/historical/<symbol>/              │
│      - Generate teacher labels (using market context)                   │
│      - Compute features (SMA, RSI, VWAP, etc.)                          │
│      - Train teacher model                                              │
│      - Save artifacts to data/models/<timestamp>/<symbol>_<window>/     │
│      - Extract training metrics (precision, recall, F1)                 │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. METADATA LOGGING                                                      │
│    For each completed training:                                          │
│      - Call TeacherStudentService.log_batch_metadata()                  │
│      - Record: symbol, date_range, artifacts_path, metrics, status      │
│      - Append to teacher_runs.json (JSON Lines format)                  │
│      - Log errors for failed trainings                                  │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 5. SUMMARY REPORT                                                        │
│    - Total symbols processed: N                                          │
│    - Total windows trained: M                                            │
│    - Successful trainings: K                                             │
│    - Failed trainings: L (with reasons)                                 │
│    - Aggregate metrics: avg precision, recall, F1                       │
│    - Total execution time                                                │
└──────────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1.1: Configuration & Data Structures
1. Extend `Settings` with historical data configuration fields
2. Extend `Settings` with batch training configuration fields
3. Update `pyproject.toml` dependencies if needed (tenacity for retry logic)

### Phase 1.2: Historical Data Ingestion
1. Create `scripts/fetch_historical_data.py`:
   - Argument parsing (--symbols, --start-date, --end-date, --intervals, --dryrun)
   - Cache checking logic
   - Download queue management
   - Retry/backoff implementation using tenacity
   - Data validation functions
   - CSV persistence with proper directory structure
   - Progress logging and summary report
2. Add helper functions to `src/services/data_feed.py` if needed

### Phase 1.3: Batch Teacher Training
1. Create `scripts/train_teacher_batch.py`:
   - Argument parsing (--symbols, --window-days, --forecast-horizon, --workers)
   - Batch initialization and directory creation
   - Window generation logic
   - Teacher training invocation (reuse existing train_teacher.py logic)
   - Parallel execution support (optional, default sequential)
   - Summary report generation
2. Extend `src/services/teacher_student.py`:
   - Add `log_batch_metadata()` method for JSON Lines logging
   - Add helper for loading batch metadata

### Phase 1.4: Integration Testing
1. Create `tests/integration/test_historical_training.py`:
   - Mock BreezeClient for network calls
   - Test historical data fetch (caching, retry, validation)
   - Test batch teacher training (metadata logging, artifacts)
   - Test dryrun mode
   - Test resume functionality

### Phase 1.5: Documentation
1. Create `docs/stories/us-024-historical-data.md` (this document)
2. Update `docs/architecture.md`:
   - Add section on historical data ingestion
   - Document batch training workflow
   - Describe directory structure and schemas

## Usage Examples

### Fetch Historical Data

```bash
# Download data for default symbols (RELIANCE, TCS, INFY)
python scripts/fetch_historical_data.py

# Download specific symbols and date range
python scripts/fetch_historical_data.py \
  --symbols RELIANCE TCS INFY WIPRO HDFCBANK \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --intervals 1minute 5minute 1day

# Dryrun mode (no network calls)
python scripts/fetch_historical_data.py --dryrun

# Force re-download (ignore cache)
python scripts/fetch_historical_data.py --force
```

### Batch Teacher Training

```bash
# Train teacher models for all symbols with default 90-day windows
python scripts/train_teacher_batch.py

# Train with custom window and forecast horizon
python scripts/train_teacher_batch.py \
  --symbols RELIANCE TCS \
  --window-days 60 \
  --forecast-horizon 5

# Parallel execution with 4 workers
python scripts/train_teacher_batch.py --workers 4

# Resume partial batch (skip already-trained windows)
python scripts/train_teacher_batch.py --resume
```

### View Batch Results

```bash
# View batch metadata
cat data/models/20251012_190000/teacher_runs.json | jq '.'

# Summary statistics
cat data/models/20251012_190000/teacher_runs.json | \
  jq -s 'map(select(.status=="success")) | {count: length, avg_precision: (map(.metrics.precision) | add / length)}'
```

## Testing Strategy

### Unit Tests
- `test_validate_ohlc_data()` - OHLC validation rules
- `test_generate_training_windows()` - Window generation logic
- `test_batch_metadata_logging()` - JSON Lines format

### Integration Tests
- `test_historical_data_fetch()` - Full fetch workflow with mocks
- `test_batch_teacher_training()` - Full batch training with mocks
- `test_resume_partial_batch()` - Resume functionality
- `test_dryrun_mode()` - Verify no network calls in dryrun

### Manual Testing
1. Run fetch script with real API credentials
2. Verify CSV files created with correct structure
3. Run batch training on fetched data
4. Inspect teacher_runs.json for metadata
5. Check artifacts directory structure

## Success Metrics

- **Data Ingestion**:
  - Successfully download 100% of requested symbols/dates
  - Cache hit rate > 90% on re-runs
  - Retry success rate > 95% for transient failures
  - Download speed: > 1000 rows/second average

- **Batch Training**:
  - Successfully train > 95% of symbol/window combinations
  - Training time: < 5 minutes per window on average
  - Metadata accuracy: 100% of runs logged correctly
  - Resume functionality: skip 100% of already-trained windows

---

## Phase 2: Student Batch Retraining & Promotion Integration

### Phase 2 Overview

Phase 2 builds on Phase 1's teacher batch training by adding automated student model retraining and promotion workflow integration. This enables end-to-end batch training pipelines that produce promotion-ready student models.

### Phase 2 Acceptance Criteria

#### AC-7: Student Batch Training Script
- [ ] `scripts/train_student_batch.py` iterates over teacher batch metadata (teacher_runs.json)
- [ ] Trains student model for each successful teacher window
- [ ] Records student metadata in `data/models/<timestamp>/student_runs.json`
- [ ] Links student runs to corresponding teacher runs (via teacher_run_id)
- [ ] Logs: symbol, teacher_artifacts_path, student_artifacts_path, metrics, promotion_checklist_path
- [ ] Supports resume functionality (skip already-trained student models)
- [ ] Generates summary report with success/failure counts

#### AC-8: Non-Interactive train_student.py Mode
- [ ] Add `--batch-mode` flag to train_student.py
- [ ] Accept baseline metrics via `--baseline-precision` and `--baseline-recall` flags
- [ ] Auto-generate promotion checklist without CLI prompts in batch mode
- [ ] Skip interactive confirmation steps when batch-mode enabled
- [ ] Return exit code 0 on success, 1 on failure for automation compatibility

#### AC-9: Student Service Extensions
- [ ] Add `StudentModel.log_batch_metadata()` method for student runs
- [ ] Record student metadata: teacher_run_id, metrics, promotion_checklist_path, status
- [ ] Add `StudentModel.load_batch_metadata()` to retrieve student run logs
- [ ] Add helper method to summarize student batch results (success rate, avg metrics)

#### AC-10: Configuration Extensions
- [ ] Add `Settings.student_batch_enabled: bool` (default: False)
- [ ] Add `Settings.student_batch_baseline_precision: float` (default: 0.60)
- [ ] Add `Settings.student_batch_baseline_recall: float` (default: 0.55)
- [ ] Add `Settings.student_batch_output_dir: str` (default: "data/models")
- [ ] Add `Settings.student_batch_promotion_enabled: bool` (default: True)

#### AC-11: Integration Testing
- [ ] Extend test_historical_training.py with Phase 2 tests
- [ ] Test end-to-end teacher → student batch workflow
- [ ] Verify student_runs.json created and linked to teacher runs
- [ ] Verify promotion checklists generated for each student model
- [ ] Test batch mode flag skips interactive prompts
- [ ] Test resume functionality for student batch training

#### AC-12: Documentation
- [ ] Update US-024 story with Phase 2 completion details
- [ ] Add Phase 2 workflow diagram to architecture.md
- [ ] Document student batch directory structure
- [ ] Document promotion integration and criteria
- [ ] Document batch mode usage examples

### Phase 2 Directory Structure

```
data/models/20251012_190000/              # Batch timestamp
├── teacher_runs.json                     # Teacher batch metadata
├── student_runs.json                     # Student batch metadata (Phase 2)
├── RELIANCE_2024Q1/                      # Teacher artifacts
│   ├── labels.csv
│   ├── features.csv
│   └── teacher_model.pkl
├── RELIANCE_2024Q1_student/              # Student artifacts (Phase 2)
│   ├── student_model.pkl
│   ├── training_data.csv
│   ├── metrics.json
│   └── promotion_checklist.md           # Auto-generated checklist
└── TCS_2024Q1_student/
    └── ...
```

### Student Batch Metadata Schema

**File**: `data/models/<timestamp>/student_runs.json` (JSON Lines format)

```jsonl
{"batch_id": "batch_20251012_190000", "symbol": "RELIANCE", "teacher_run_id": "RELIANCE_2024Q1", "teacher_artifacts_path": "data/models/20251012_190000/RELIANCE_2024Q1", "student_artifacts_path": "data/models/20251012_190000/RELIANCE_2024Q1_student", "metrics": {"precision": 0.68, "recall": 0.65, "f1": 0.66}, "promotion_checklist_path": "data/models/20251012_190000/RELIANCE_2024Q1_student/promotion_checklist.md", "status": "success", "timestamp": "2025-10-12T19:15:23+05:30"}
{"batch_id": "batch_20251012_190000", "symbol": "TCS", "teacher_run_id": "TCS_2024Q1", "teacher_artifacts_path": "data/models/20251012_190000/TCS_2024Q1", "student_artifacts_path": "data/models/20251012_190000/TCS_2024Q1_student", "metrics": {"precision": 0.71, "recall": 0.68, "f1": 0.69}, "promotion_checklist_path": "data/models/20251012_190000/TCS_2024Q1_student/promotion_checklist.md", "status": "success", "timestamp": "2025-10-12T19:22:45+05:30"}
```

### Phase 2 Usage Examples

**Student Batch Training:**

```bash
# Train student models for all teacher runs
python scripts/train_student_batch.py

# Train with custom baseline criteria
python scripts/train_student_batch.py \
  --baseline-precision 0.65 \
  --baseline-recall 0.60

# Resume partial batch
python scripts/train_student_batch.py --resume

# View student batch results
cat data/models/20251012_190000/student_runs.json | jq '.'
```

**Non-Interactive Student Training:**

```bash
# Batch mode (no prompts)
python scripts/train_student.py \
  --teacher-artifacts data/models/20251012_190000/RELIANCE_2024Q1 \
  --batch-mode \
  --baseline-precision 0.60 \
  --baseline-recall 0.55
```

### Promotion Integration

Student models trained in batch mode automatically generate promotion checklists with:

- Baseline comparison (precision, recall, F1)
- Feature stability checks
- Data leakage verification
- Model size validation
- Recommended next steps (promote / reject / retrain)

**Auto-Promotion Criteria** (configurable):
- Precision >= baseline (default: 0.60)
- Recall >= baseline (default: 0.55)
- No data leakage detected
- Model size < 10MB

---

## Phase 3: Sentiment Snapshot Ingestion

### Phase 3 Overview

Phase 3 extends US-024 with daily sentiment snapshot ingestion alongside historical OHLCV data. This enables batch training pipelines to incorporate sentiment features and track sentiment data availability in metadata logs.

### Phase 3 Acceptance Criteria

#### AC-13: Sentiment Snapshot Fetcher Script
- [x] `scripts/fetch_sentiment_snapshots.py` downloads daily sentiment for symbols/date ranges
- [x] Uses sentiment provider registry (NewsAPI/Twitter/stub) with fallback support
- [x] Stores snapshots as JSON Lines under `data/sentiment/<symbol>/<YYYY-MM-DD>.jsonl`
- [x] Implements caching to skip already-fetched dates
- [x] Retry/backoff logic for transient API failures (3 retries, exponential backoff)
- [x] Dryrun mode uses stub provider without network calls
- [x] Progress logging and summary reports

#### AC-14: Metadata Extensions for Sentiment
- [x] `TeacherLabeler.log_batch_metadata()` accepts optional `sentiment_snapshot_path`
- [x] Records `sentiment_available` boolean flag in teacher metadata
- [x] `StudentModel.log_batch_metadata()` inherits sentiment reference from teacher
- [x] Both teacher and student metadata include sentiment snapshot directory path

#### AC-15: Batch Script Integration
- [x] `train_teacher_batch.py` checks for sentiment snapshots and records availability
- [x] Warns if sentiment enabled but snapshots missing for symbol
- [x] `train_student_batch.py` propagates sentiment path from teacher to student metadata
- [x] Batch summary reports include sentiment coverage statistics

#### AC-16: Configuration Extensions
- [x] Add `Settings.sentiment_snapshot_enabled: bool` (default: False)
- [x] Add `Settings.sentiment_snapshot_providers: list[str]` (default: ["stub"])
- [x] Add `Settings.sentiment_snapshot_output_dir: str` (default: "data/sentiment")
- [x] Add `Settings.sentiment_snapshot_retry_limit: int` (default: 3)
- [x] Add `Settings.sentiment_snapshot_retry_backoff_seconds: int` (default: 2)
- [x] Add `Settings.sentiment_snapshot_max_per_day: int` (default: 100)

#### AC-17: Integration Testing
- [x] Extend test_historical_training.py with Phase 3 tests
- [x] Test sentiment snapshot fetch creates JSONL files in correct directory
- [x] Test caching prevents duplicate fetches
- [x] Test dryrun mode uses stub provider without network calls
- [x] Test teacher/student metadata records sentiment availability
- [x] Verify sentiment references propagate from teacher to student

#### AC-18: Documentation
- [x] Update US-024 story with Phase 3 completion details
- [x] Add Phase 3 workflow to architecture.md
- [x] Document sentiment snapshot JSON schema
- [x] Document provider integration and fallback logic
- [x] Document configuration options and defaults

### Phase 3 Directory Structure

```
data/
├── sentiment/                         # Sentiment snapshots (Phase 3)
│   ├── RELIANCE/
│   │   ├── 2024-01-01.jsonl          # Daily sentiment snapshot
│   │   ├── 2024-01-02.jsonl
│   │   └── ...
│   ├── TCS/
│   └── INFY/
└── models/
    └── 20251012_190000/
        ├── teacher_runs.json          # Includes sentiment_snapshot_path
        ├── student_runs.json          # Inherits sentiment reference
        └── ...
```

### Sentiment Snapshot JSON Schema

**File**: `data/sentiment/<symbol>/<YYYY-MM-DD>.jsonl`

```jsonl
{"symbol": "RELIANCE", "date": "2024-01-01", "timestamp": "2025-10-12T19:05:23+05:30", "score": 0.65, "confidence": 0.82, "providers": ["newsapi", "twitter"], "metadata": {"article_count": 15, "tweet_count": 237}}
{"symbol": "RELIANCE", "date": "2024-01-02", "timestamp": "2025-10-12T19:10:15+05:30", "score": 0.58, "confidence": 0.79, "providers": ["newsapi"], "metadata": {"article_count": 12}}
```

**Fields**:
- `symbol`: Stock symbol
- `date`: Date (YYYY-MM-DD)
- `timestamp`: ISO 8601 timestamp when snapshot was fetched
- `score`: Sentiment score (-1.0 to 1.0, where -1=negative, 0=neutral, 1=positive)
- `confidence`: Confidence level (0.0 to 1.0)
- `providers`: List of providers used (e.g., ["newsapi", "twitter", "stub"])
- `metadata`: Provider-specific metadata (article counts, sources, etc.)

### Teacher Metadata Schema (Extended)

**File**: `data/models/<timestamp>/teacher_runs.json`

```jsonl
{"batch_id": "batch_20251012_190000", "symbol": "RELIANCE", "date_range": {"start": "2024-01-01", "end": "2024-03-31"}, "artifacts_path": "data/models/20251012_190000/RELIANCE_2024Q1", "metrics": {"precision": 0.72, "recall": 0.68, "f1": 0.70}, "status": "success", "sentiment_snapshot_path": "data/sentiment/RELIANCE", "sentiment_available": true, "timestamp": "2025-10-12T19:05:23+05:30"}
```

**New Fields (Phase 3)**:
- `sentiment_snapshot_path`: Path to sentiment snapshot directory (optional)
- `sentiment_available`: Boolean flag indicating if sentiment data exists

### Phase 3 Usage Examples

**Fetch Sentiment Snapshots:**

```bash
# Fetch sentiment for default symbols/dates
python scripts/fetch_sentiment_snapshots.py

# Fetch specific symbols and date range
python scripts/fetch_sentiment_snapshots.py \
  --symbols RELIANCE TCS INFY \
  --start-date 2024-01-01 \
  --end-date 2024-03-31

# Dryrun mode (no network calls, use stub provider)
python scripts/fetch_sentiment_snapshots.py --dryrun

# Force re-fetch (ignore cache)
python scripts/fetch_sentiment_snapshots.py --force

# Use specific providers
python scripts/fetch_sentiment_snapshots.py --providers newsapi twitter
```

**Enable Sentiment in Batch Training:**

Set environment variables in `.env`:
```bash
SENTIMENT_SNAPSHOT_ENABLED=true
SENTIMENT_SNAPSHOT_PROVIDERS=["newsapi", "twitter"]
SENTIMENT_SNAPSHOT_OUTPUT_DIR=data/sentiment
```

Then run batch training:
```bash
# Fetch sentiment snapshots first
python scripts/fetch_sentiment_snapshots.py \
  --symbols RELIANCE TCS \
  --start-date 2024-01-01 \
  --end-date 2024-03-31

# Run teacher batch training (will check for sentiment)
python scripts/train_teacher_batch.py

# Run student batch training (inherits sentiment reference)
python scripts/train_student_batch.py
```

### Sentiment Provider Integration

Phase 3 integrates with the existing sentiment provider registry (from US-004):

**Supported Providers**:
- **NewsAPI**: Fetches news articles and analyzes keyword sentiment
- **Twitter**: Analyzes tweet sentiment (if API credentials configured)
- **Stub**: Returns neutral sentiment (0.0) for testing/dryrun

**Fallback Logic**:
1. Try primary provider (highest priority)
2. On failure, fall back to secondary providers
3. Weighted averaging if multiple providers succeed
4. Circuit breaker disables unhealthy providers temporarily

**Configuration**:
```python
# settings
sentiment_snapshot_providers = ["newsapi", "twitter", "stub"]
# Priority: newsapi (0), twitter (1), stub (2)
```

### Retry/Backoff Policy

**Sentiment API Retry Logic** (same as Phase 1):
- **Retries**: 3 attempts (configurable via `sentiment_snapshot_retry_limit`)
- **Backoff**: Exponential 2s/4s/8s (configurable via `sentiment_snapshot_retry_backoff_seconds`)
- **Retryable Errors**: ConnectionError, TimeoutError, HTTP 429, HTTP 5xx
- **Non-Retryable Errors**: HTTP 400, 401, 403, 404

**Implementation**:
```python
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=2, min=2, max=30),
    retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
)
def fetch_sentiment_with_retry(symbol: str, date: datetime) -> dict:
    ...
```


---

## Phase 4: Incremental Daily Updates

### Phase 4 Overview

Phase 4 extends US-024 with incremental daily update capabilities, enabling efficient daily refreshes of historical OHLCV and sentiment data without re-downloading existing data. This phase introduces state tracking to remember the last fetch date per symbol and adds incremental modes to batch training scripts.

### Phase 4 Acceptance Criteria

#### AC-19: Incremental Historical Data Fetch
- [x] `fetch_historical_data.py` supports `--incremental` flag
- [x] Tracks last fetch date per symbol in `data/state/historical_fetch.json`
- [x] Fetches only dates after last successful fetch
- [x] Falls back to lookback window if no previous fetch exists
- [x] Updates state file after successful fetch
- [x] Lookback days configurable via `--lookback-days` or settings

#### AC-20: Incremental Sentiment Fetch
- [x] `fetch_sentiment_snapshots.py` supports `--incremental` flag
- [x] Tracks last fetch date per symbol in `data/state/sentiment_fetch.json`
- [x] Fetches only new sentiment snapshots since last run
- [x] Dryrun mode supports incremental (uses stub provider)
- [x] State tracking reuses same retry/backoff as full mode

#### AC-21: Incremental Batch Training
- [x] `train_teacher_batch.py` supports `--incremental` flag
- [x] Marks metadata entries with `incremental: true`
- [x] Appends new runs to existing `teacher_runs.json`
- [x] `train_student_batch.py` supports `--incremental` flag
- [x] Student metadata inherits incremental flag from teacher

#### AC-22: State Management Infrastructure
- [x] `StateManager` class tracks last fetch dates per symbol
- [x] State files stored as JSON in `data/state/` directory
- [x] Per-symbol last fetch date tracking
- [x] Last run metadata (timestamp, run_type, success, files_created, errors)
- [x] Supports clearing individual symbols or all state

#### AC-23: Configuration Extensions
- [x] Add `Settings.incremental_enabled: bool` (default: False)
- [x] Add `Settings.incremental_lookback_days: int` (default: 30)
- [x] Add `Settings.incremental_cron_schedule: str` (default: "0 18 * * 1-5")
- [x] All defaults keep incremental mode disabled unless explicitly enabled

#### AC-24: Integration Testing
- [x] Extend test_historical_training.py with Phase 4 tests
- [x] Test state manager tracks and persists last fetch dates
- [x] Test incremental config defaults are safe
- [x] Verify state files created in correct directory
- [x] Verify incremental flag appears in metadata

#### AC-25: Documentation
- [x] Update US-024 story with Phase 4 completion details
- [x] Add Phase 4 workflow to architecture.md
- [x] Document state file JSON schema
- [x] Document incremental mode usage examples
- [x] Document scheduling recommendations

### Phase 4 Directory Structure

```
data/
├── state/                             # State tracking (Phase 4)
│   ├── historical_fetch.json         # Last OHLCV fetch dates
│   └── sentiment_fetch.json          # Last sentiment fetch dates
├── sentiment/
│   └── ...
├── historical/
│   └── ...
└── models/
    └── 20251012_190000/
        ├── teacher_runs.json          # Includes incremental: true/false
        └── student_runs.json          # Includes incremental: true/false
```

### State File JSON Schema

**File**: `data/state/historical_fetch.json` or `sentiment_fetch.json`

```json
{
  "symbols": {
    "RELIANCE": {
      "last_fetch_date": "2024-03-15T00:00:00",
      "last_updated": "2024-03-16T18:30:45+05:30"
    },
    "TCS": {
      "last_fetch_date": "2024-03-15T00:00:00",
      "last_updated": "2024-03-16T18:30:45+05:30"
    }
  },
  "last_run": {
    "timestamp": "2024-03-16T18:30:45+05:30",
    "run_type": "incremental",
    "success": true,
    "symbols_processed": ["RELIANCE", "TCS"],
    "files_created": 6,
    "errors": 0
  }
}
```

**Fields**:
- `symbols`: Per-symbol state
  - `last_fetch_date`: ISO 8601 date of last successful fetch
  - `last_updated`: ISO 8601 timestamp when state was updated
- `last_run`: Metadata about the last run
  - `timestamp`: When the run completed
  - `run_type`: "full" or "incremental"
  - `success`: Whether run completed without errors
  - `symbols_processed`: List of symbols in the run
  - `files_created`: Number of files created/updated
  - `errors`: Number of errors encountered

### Metadata Extensions (Phase 4)

**Teacher Metadata** (`teacher_runs.json`):
```jsonl
{
  ...
  "incremental": true,
  "timestamp": "2024-03-16T18:35:12+05:30"
}
```

**Student Metadata** (`student_runs.json`):
```jsonl
{
  ...
  "incremental": true,
  "timestamp": "2024-03-16T18:45:23+05:30"
}
```

### Phase 4 Usage Examples

**Incremental Historical Data Fetch:**

```bash
# Fetch only new days since last run
python scripts/fetch_historical_data.py --incremental

# Incremental with custom lookback (if no previous fetch)
python scripts/fetch_historical_data.py --incremental --lookback-days 7

# First run: no state exists, uses 30-day lookback (default)
# Subsequent runs: fetches only days after last fetch date
```

**Incremental Sentiment Fetch:**

```bash
# Fetch only new sentiment snapshots
python scripts/fetch_sentiment_snapshots.py --incremental

# Dryrun incremental mode
python scripts/fetch_sentiment_snapshots.py --incremental --dryrun
```

**Incremental Batch Training:**

```bash
# Train only windows with new data
python scripts/train_teacher_batch.py --incremental

# Student training with incremental flag
python scripts/train_student_batch.py --incremental
```

**Complete Incremental Pipeline:**

```bash
#!/bin/bash
# Daily incremental update script (run via cron)

# Fetch new OHLCV data
python scripts/fetch_historical_data.py --incremental

# Fetch new sentiment snapshots
python scripts/fetch_sentiment_snapshots.py --incremental --dryrun

# Train teacher models (incremental)
python scripts/train_teacher_batch.py --incremental

# Train student models (incremental)
python scripts/train_student_batch.py --incremental
```

### Scheduling Recommendations

**Cron Schedule** (Mon-Fri at 6PM IST):
```cron
0 18 * * 1-5 cd /path/to/SenseQuant && ./scripts/incremental_update.sh >> logs/incremental.log 2>&1
```

**Configuration** (`.env`):
```bash
INCREMENTAL_ENABLED=true
INCREMENTAL_LOOKBACK_DAYS=30
INCREMENTAL_CRON_SCHEDULE="0 18 * * 1-5"
```

**Best Practices**:
- Run incremental updates daily after market close
- Keep lookback window at 30+ days for safety
- Monitor state files for corruption
- Periodically run full fetch to ensure completeness
- Check logs for fetch failures and retry manually

### Incremental Mode Behavior

**First Run** (no state exists):
1. Checks state file → not found
2. Falls back to lookback window (default: 30 days)
3. Fetches last 30 days of data
4. Creates state file with last fetch date

**Subsequent Runs** (state exists):
1. Loads state file
2. Gets last fetch date for each symbol
3. Calculates date range: (last_fetch_date + 1 day) to today
4. Fetches only new data
5. Updates state file with new last fetch date

**Resume After Failure**:
- State file only updated on successful fetch
- Failed runs don't update state
- Next run will retry from last successful fetch date

### Error Handling

**Retries**:
- Same retry/backoff policy as full mode (3 attempts, exponential backoff)
- Retryable: ConnectionError, TimeoutError, HTTP 429, 5xx
- Non-retryable: HTTP 400, 401, 403, 404

**State File Corruption**:
- Invalid JSON → falls back to empty state (full lookback)
- Missing fields → uses defaults
- Logged as warning but doesn't fail

**Missing Data Detection**:
- Gap detection not implemented (future enhancement)
- Assumes continuous daily fetch cadence
- Manual full fetch recommended periodically




---

## Phase 5: Distributed Training & Scheduled Automation (Detailed Documentation)

### ✅ Phase 5: Distributed Training & Scheduled Automation (Completed)

**Status**: Completed  
**Completion Date**: 2025-10-12

Phase 5 adds parallel execution, retry logic, batch status tracking, and orchestration scripts for scheduled daily updates. This phase provides the foundation for scaling to distributed training systems.

---

#### Phase 5 Acceptance Criteria

**AC-26: Parallel Worker Execution** ✅
- [x] `train_teacher_batch.py` supports `--workers N` flag for parallel execution
- [x] Uses Python's `ProcessPoolExecutor` for multi-core parallelism
- [x] Thread-safe metadata logging with locks
- [x] State updates remain atomic across parallel workers
- [x] Default remains sequential (workers=1) for safety

**AC-27: Retry/Resume Logic** ✅
- [x] Failed training windows automatically retry up to configurable limit
- [x] Exponential backoff between retries (default: 5 seconds)
- [x] Failed tasks logged to state file with reason and attempt count
- [x] Tasks exceeding retry limit marked for manual review
- [x] State manager tracks retry attempts per task

**AC-28: Batch Execution Status Tracking** ✅
- [x] StateManager extended with batch status methods
- [x] Tracks batch status: running, completed, failed, partial
- [x] Records completed/failed/pending retry counts
- [x] Failed tasks stored with symbol, window, reason, attempts
- [x] Batch status queryable for monitoring/alerting

**AC-29: Scheduled Pipeline Orchestration** ✅
- [x] `scripts/run_incremental_update.sh` chains fetch + training
- [x] Configurable phase skipping via environment variables
- [x] Logging to timestamped log files
- [x] Exit codes suitable for cron (0=success, 1=failure)
- [x] Clear failure reporting for manual intervention

**AC-30: Distributed Training Stub** ✅
- [x] `scripts/distributed_training_worker.py` stub created
- [x] Protocol defined for future distributed executors
- [x] Documentation for Kubernetes, Airflow, Celery integration
- [x] Current implementation uses local ProcessPoolExecutor

**AC-31: Phase 5 Configuration** ✅
- [x] `parallel_workers`: Number of parallel workers (1-32, default: 1)
- [x] `parallel_retry_limit`: Max retry attempts (1-10, default: 3)
- [x] `parallel_retry_backoff_seconds`: Retry backoff (1-60s, default: 5s)
- [x] `scheduled_pipeline_skip_*`: Phase skip flags (default: false)
- [x] All defaults keep system safe and sequential

**AC-32: Integration Testing** ✅
- [x] Test batch status tracking with StateManager
- [x] Test parallel configuration defaults
- [x] Test orchestration script exists and is executable
- [x] Test distributed worker stub imports correctly
- [x] All 22 tests pass (9 Phase 1 + 5 Phase 2 + 2 Phase 3 + 2 Phase 4 + 4 Phase 5)

---

#### Directory Structure (Extended)

```
data/
├── state/                             # State tracking (Phases 4-5)
│   ├── historical_fetch.json         # Last OHLCV fetch dates
│   ├── sentiment_fetch.json          # Last sentiment fetch dates
│   ├── teacher_batch.json            # Teacher batch execution status ⭐ NEW
│   └── student_batch.json            # Student batch execution status ⭐ NEW
├── sentiment/
├── historical/
└── models/
    └── 20251012_190000/
        ├── teacher_runs.json          # Includes attempts field ⭐ NEW
        └── student_runs.json          # Includes retries tracking

scripts/
├── train_teacher_batch.py             # Parallel execution support ⭐ UPDATED
├── train_student_batch.py             # Parallel execution support ⭐ UPDATED
├── run_incremental_update.sh          # Orchestration script ⭐ NEW
└── distributed_training_worker.py     # Distributed executor stub ⭐ NEW

logs/
└── incremental_update_*.log           # Timestamped pipeline logs ⭐ NEW
```

---

#### Batch Status JSON Schema

**File**: `data/state/teacher_batch.json` or `student_batch.json`

```json
{
  "batches": {
    "batch_20250112_180000": {
      "status": "partial",
      "total_tasks": 10,
      "completed": 8,
      "failed": 2,
      "pending_retries": 0,
      "last_updated": "2025-10-12T18:45:23+05:30",
      "failed_tasks": [
        {
          "task_id": "RELIANCE_2024Q1",
          "symbol": "RELIANCE",
          "window_label": "RELIANCE_2024Q1",
          "reason": "Training timeout (10 minutes exceeded)",
          "attempts": 3,
          "timestamp": "2025-10-12T18:42:15+05:30"
        }
      ]
    }
  }
}
```

**Fields**:
- `status`: Batch status (running, completed, failed, partial)
- `total_tasks`: Total number of training windows
- `completed`: Successfully completed tasks
- `failed`: Failed tasks (after all retries)
- `pending_retries`: Tasks pending retry
- `failed_tasks`: List of failed task details for manual review

---

#### Phase 5 Usage

**Parallel Teacher Training:**
```bash
# Train with 4 parallel workers
python scripts/train_teacher_batch.py --workers 4 --incremental

# Sequential with retries (default)
python scripts/train_teacher_batch.py --incremental
```

**Orchestration Script:**
```bash
# Full incremental pipeline
./scripts/run_incremental_update.sh

# Skip data fetch (training only)
SKIP_FETCH=true ./scripts/run_incremental_update.sh

# Dry run mode for sentiment
DRY_RUN=true ./scripts/run_incremental_update.sh

# Parallel training with 8 workers
PARALLEL_WORKERS=8 ./scripts/run_incremental_update.sh

# Skip specific phases
SKIP_TEACHER=true SKIP_STUDENT=true ./scripts/run_incremental_update.sh
```

**Cron Schedule (Mon-Fri at 6PM IST):**
```cron
0 18 * * 1-5 cd /path/to/SenseQuant && ./scripts/run_incremental_update.sh >> logs/cron.log 2>&1
```

---

#### Configuration

**Environment Variables (.env):**
```bash
# Phase 5: Distributed Training & Scheduled Automation
PARALLEL_WORKERS=4              # Number of parallel workers
PARALLEL_RETRY_LIMIT=3          # Max retry attempts per task
PARALLEL_RETRY_BACKOFF_SECONDS=5  # Backoff between retries
SCHEDULED_PIPELINE_SKIP_FETCH=false
SCHEDULED_PIPELINE_SKIP_TEACHER=false
SCHEDULED_PIPELINE_SKIP_STUDENT=false
```

**Config Validation:**
- `parallel_workers`: 1-32 (default: 1, sequential)
- `parallel_retry_limit`: 1-10 (default: 3)
- `parallel_retry_backoff_seconds`: 1-60 (default: 5)

---

#### Retry Behavior

**Sequential Mode (workers=1):**
1. Attempt 1: Train window
2. If failed → wait 5s → Attempt 2
3. If failed → wait 5s → Attempt 3
4. If failed → record in state file, mark as failed
5. Continue to next window

**Parallel Mode (workers>1):**
1. Submit N tasks to ProcessPoolExecutor
2. Each worker retries failed tasks independently
3. Metadata logging uses lock for thread safety
4. State manager records failures atomically
5. All results collected before summary

**Failed Task Handling:**
- Failed tasks logged to state file with reason
- Summary report shows failed tasks count
- Exit code 1 if any failures (alerts cron)
- Manual review required for tasks exceeding retry limit

---

#### Distributed Execution (Future)

The `distributed_training_worker.py` stub provides a protocol for future distributed executors:

**Kubernetes Integration (Example):**
```python
# Create K8s Job per training window
apiVersion: batch/v1
kind: Job
metadata:
  name: teacher-training-reliance-2024q1
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: sensequant/teacher-trainer:latest
        args:
          - --symbol=RELIANCE
          - --start=2024-01-01
          - --end=2024-03-31
      restartPolicy: OnFailure
  backoffLimit: 3  # Auto-retry
```

**Airflow Integration (Example):**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

# Dynamic task generation
for window in training_windows:
    task = PythonOperator(
        task_id=f"train_{window['label']}",
        python_callable=train_teacher_window,
        op_kwargs={'window': window},
        retries=3,
    )
    dag >> task
```

**Scaling Recommendations:**
- **1-4 workers**: Single machine, multi-core
- **5-16 workers**: Consider Celery with Redis
- **17+ workers**: Use Kubernetes or Ray for distributed execution
- **Enterprise**: Airflow for orchestration + Kubernetes for compute

---

#### Monitoring & Alerts

**Check Batch Status:**
```bash
# View batch status
cat data/state/teacher_batch.json | jq '.batches'

# Count failed tasks
cat data/state/teacher_batch.json | jq '.batches[].failed_tasks | length'

# List failed windows
cat data/state/teacher_batch.json | jq '.batches[].failed_tasks[] | "\(.symbol)/\(.window_label): \(.reason)"'
```

**Alerting (Example):**
```bash
#!/bin/bash
# Alert on failures
FAILED=$(cat data/state/teacher_batch.json | jq '.batches[].failed' | paste -sd+ | bc)
if [ "$FAILED" -gt 0 ]; then
    echo "ALERT: $FAILED training windows failed" | mail -s "SenseQuant Training Failure" ops@example.com
fi
```

---

#### Operational Best Practices

1. **Start Sequential**: Use default workers=1 until stable
2. **Monitor Logs**: Check `logs/incremental_update_*.log` for errors
3. **Review Failures**: Inspect state files for failed tasks after each run
4. **Retry Limits**: Keep retry_limit at 3 (balance between resilience and speed)
5. **Parallel Workers**: Scale gradually (1 → 2 → 4 → 8)
6. **Timeout Tuning**: Adjust training timeout if legitimate jobs fail
7. **Disk Space**: Monitor `data/models/` growth, clean old batches
8. **Resource Limits**: Parallel workers share CPU/memory, avoid oversubscription

---

#### Phase 5 Summary

Phase 5 completes US-024 with production-ready parallel execution, retry logic, and scheduled automation. The system now supports:

- ✅ Multi-core parallelism with ProcessPoolExecutor
- ✅ Automatic retry with exponential backoff
- ✅ Comprehensive batch status tracking
- ✅ Cron-ready orchestration script
- ✅ Foundation for distributed execution (K8s, Airflow, etc.)
- ✅ Safe defaults (sequential, moderate retries)
- ✅ Complete test coverage (22 tests pass)

**Next Steps** (Beyond US-024):
- Implement Kubernetes executor for true distributed training
- Add Prometheus metrics for real-time monitoring
- Create Grafana dashboard for batch pipeline visualization
- Integrate with alerting system (PagerDuty, Slack)
- Add gap detection for missing data between incremental runs



---

## Phase Roadmap

### ✅ Phase 1: Historical Data & Teacher Batch Training (Completed)
- Historical OHLCV data ingestion with caching and retry logic
- Batch teacher training across multiple symbols/windows
- Teacher metadata logging (JSON Lines format)

### ✅ Phase 2: Student Batch Retraining & Promotion Integration (Completed)
- Batch student training from teacher outputs
- Non-interactive training mode for automation
- Auto-generated promotion checklists
- Student metadata logging linked to teacher runs

### ✅ Phase 3: Sentiment Snapshot Ingestion (Completed)
- Daily sentiment snapshot downloads for historical windows
- Provider registry integration (NewsAPI/Twitter/stub)
- Sentiment metadata recording in teacher/student batch logs
- Caching and retry/backoff for sentiment API calls
- Dryrun mode support for testing without API calls

### ✅ Phase 4: Incremental Daily Updates (Completed)
- Incremental OHLCV and sentiment downloads (fetch only new days)
- State tracking for last fetch dates per symbol
- Incremental batch training (process only windows with new data)
- Metadata flags for incremental runs

### ✅ Phase 5: Distributed Training & Scheduled Automation (Completed)

**Status**: Completed
**Completion Date**: 2025-10-12

Phase 5 adds parallel execution, retry logic, batch status tracking, and orchestration scripts for scheduled daily updates. This phase provides the foundation for scaling to distributed training systems.

See full Phase 5 documentation below (after Phase 4 section).

### ✅ Phase 6: Data Quality Dashboard & Alerts (Completed)

**Status**: Completed
**Completion Date**: 2025-10-12

Phase 6 adds a Streamlit dashboard for data quality monitoring, automated quality scanning, alert thresholds, and operational playbooks. This phase provides visibility into data coverage, validation issues, and batch execution status.

**Key Features:**
- Streamlit dashboard with 5 pages (Overview, Quality Metrics, Alerts, Batch Status, Fetch History)
- DataQualityService for scanning historical OHLCV and sentiment data
- Automated quality metrics tracking in StateManager
- Configurable alert thresholds for missing files, duplicates, zero-volume bars
- Integration with MonitoringService for alerts
- Comprehensive test coverage (27 tests pass)

## Related User Stories

- US-020: Teacher/Student Model Training Automation
- US-021: Student Model Promotion & Live Scoring
- US-022: Release Audit Workflow
- US-023: Release Deployment Automation

## Acceptance Sign-Off

- [ ] Engineering Lead: Code review passed, quality gates green
- [ ] Data Engineer: Storage layout approved, CSV schema validated
- [ ] QA: Integration tests pass, manual testing complete
- [ ] Product Owner: Acceptance criteria verified
