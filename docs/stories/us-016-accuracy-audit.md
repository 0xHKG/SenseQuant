# US-016 — Strategy Accuracy Audit & Enhancement

**Status**: In Progress
**Priority**: High
**Complexity**: High
**Estimated Effort**: 10-14 hours

---

## Problem Statement

The trading system currently lacks systematic accuracy diagnostics and performance auditing capabilities. While we have strategies (intraday, swing), a backtester, and optimization tools, we have no automated way to:

1. **Measure prediction accuracy** across strategies (precision, recall, confusion matrices)
2. **Track signal quality** over time (hit ratio, win rate, average returns)
3. **Analyze feature importance** from teacher/student models
4. **Audit parameter sensitivity** across different market conditions
5. **Generate reproducible reports** for strategy performance analysis
6. **Export telemetry data** for offline analysis without impacting live trading

This makes it difficult to:
- Identify which strategies/parameters work best
- Debug strategy failures (false positives, false negatives)
- Justify parameter changes with data
- Improve model accuracy systematically
- Audit compliance and risk metrics

---

## Objectives

1. **Automated Accuracy Diagnostics**: Run batch backtests across symbols/intervals capturing comprehensive metrics
2. **Prediction-Level Telemetry**: Track every signal vs realized outcome for detailed analysis
3. **Reusable Report Notebook**: Generate analysis reports with visualizations (no live API dependency)
4. **CLI Integration**: Add flags to scripts for dumping accuracy traces and summaries
5. **Configurable Telemetry**: Enable/disable telemetry capture without code changes
6. **Integration Testing**: Verify audit pipeline with mocked data
7. **Documentation**: Comprehensive workflow documentation in architecture guide

---

## Requirements

### FR-1: Prediction-Level Telemetry Capture

**Description**: Extend backtester and optimizer to emit detailed prediction telemetry

**Acceptance Criteria**:
- [ ] Backtester emits `PredictionTrace` records for each signal:
  - `timestamp`: Signal generation time
  - `symbol`: Stock symbol
  - `strategy`: Strategy name (intraday/swing)
  - `predicted_direction`: LONG/SHORT/NOOP
  - `actual_direction`: Realized direction (based on forward returns)
  - `predicted_confidence`: Model confidence score
  - `entry_price`: Entry price
  - `exit_price`: Actual exit price
  - `holding_period_minutes`: Time held
  - `realized_return_pct`: Actual return %
  - `features`: Feature values at signal time (dict)
  - `metadata`: Additional context (SL/TP hit, max_hold, etc.)

- [ ] Optimizer tracks parameter sensitivity:
  - Parameter combination → Performance metrics mapping
  - Grid search results with statistical significance
  - Feature importance from best models

- [ ] Telemetry written to structured storage:
  - CSV format: `data/analytics/{timestamp}/predictions.csv`
  - JSON format: `data/analytics/{timestamp}/predictions.jsonl` (line-delimited)
  - Metadata: `data/analytics/{timestamp}/metadata.json`

### FR-2: Accuracy Metrics Computation

**Description**: Compute comprehensive accuracy metrics from telemetry data

**Acceptance Criteria**:
- [ ] Classification metrics:
  - **Precision**: TP / (TP + FP) per direction
  - **Recall**: TP / (TP + FN) per direction
  - **F1 Score**: Harmonic mean of precision/recall
  - **Confusion Matrix**: 3x3 (LONG/SHORT/NOOP vs actual)
  - **Accuracy**: (TP + TN) / Total

- [ ] Trading metrics:
  - **Hit Ratio**: Winning trades / Total trades
  - **Win Rate**: % of profitable trades
  - **Average Return**: Mean of realized_return_pct
  - **Sharpe Ratio**: Risk-adjusted returns
  - **Max Drawdown**: Peak-to-trough decline
  - **Profit Factor**: Gross profit / Gross loss

- [ ] Holding period analysis:
  - Average holding time by direction
  - Exit reason distribution (TP/SL/Max Hold/Signal Reversal)
  - Time-to-profit distribution

- [ ] Feature importance:
  - Top 10 features by model weight
  - Feature correlation with returns
  - Feature stability over time

### FR-3: Batch Backtesting Framework

**Description**: Automate backtests across multiple symbols/intervals

**Acceptance Criteria**:
- [ ] Batch configuration format (YAML/JSON):
  ```yaml
  batch_backtest:
    symbols: ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
    intervals: ["1minute", "5minute", "1day"]
    strategies: ["intraday", "swing"]
    date_range:
      from: "2024-01-01"
      to: "2024-12-31"
    data_source: "csv"  # Use cached data
    telemetry_enabled: true
  ```

- [ ] Batch executor:
  - Parallel execution (configurable workers)
  - Progress tracking (tqdm progress bar)
  - Failure handling (continue on error, log failures)
  - Results aggregation across runs

- [ ] Output structure:
  ```
  data/analytics/{batch_id}/
  ├── metadata.json           # Batch configuration + runtime info
  ├── summary.csv             # Aggregated metrics per symbol/strategy
  ├── predictions/
  │   ├── RELIANCE_1minute_intraday.csv
  │   ├── RELIANCE_1day_swing.csv
  │   └── ...
  ├── confusion_matrices/
  │   ├── RELIANCE_intraday.json
  │   └── ...
  └── feature_importance/
      ├── RELIANCE_intraday.json
      └── ...
  ```

### FR-4: Report Notebook

**Description**: Jupyter notebook for reproducible analysis and visualization

**Acceptance Criteria**:
- [ ] Notebook location: `notebooks/accuracy_report.ipynb`

- [ ] Sections:
  1. **Executive Summary**: High-level metrics across all runs
  2. **Strategy Performance**: Per-strategy breakdown with charts
  3. **Confusion Matrix Analysis**: Heatmaps showing prediction accuracy
  4. **Feature Importance**: Bar charts of top features
  5. **Temporal Analysis**: Accuracy trends over time
  6. **Parameter Sensitivity**: Grid search results visualization
  7. **Recommendations**: Data-driven parameter update suggestions

- [ ] Visualizations (matplotlib/seaborn):
  - Confusion matrix heatmaps
  - Precision/recall curves
  - Return distribution histograms
  - Cumulative P&L curves
  - Feature importance bar charts
  - Sharpe ratio comparison
  - Holding period distributions

- [ ] Data loading:
  - Load from `data/analytics/{batch_id}/`
  - No live API calls (use cached data only)
  - Parameterized batch_id selection

- [ ] Export:
  - Notebook can export to HTML report
  - All charts saveable as PNG/SVG

### FR-5: Configuration Settings

**Description**: Add telemetry configuration to Settings

**Acceptance Criteria**:
- [ ] New settings in `src/app/config.py`:
  ```python
  # Accuracy Audit & Telemetry
  telemetry_enabled: bool = False
  telemetry_storage_path: str = "data/analytics"
  telemetry_sample_rate: float = 1.0  # 0.0-1.0 (100% = capture all)
  telemetry_include_features: bool = True
  telemetry_max_file_size_mb: int = 100
  telemetry_compression: bool = False

  # Batch Backtesting
  batch_parallel_workers: int = 4
  batch_progress_bar: bool = True
  ```

- [ ] Validation:
  - `telemetry_sample_rate` must be between 0.0 and 1.0
  - `telemetry_storage_path` must be writable directory
  - `batch_parallel_workers` must be >= 1

### FR-6: CLI Enhancements

**Description**: Add accuracy audit flags to existing scripts

**Acceptance Criteria**:
- [ ] `scripts/backtest.py` additions:
  ```bash
  python scripts/backtest.py \
    --symbols RELIANCE TCS \
    --strategy intraday \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --enable-telemetry \           # NEW: Enable telemetry capture
    --telemetry-dir data/audit \   # NEW: Custom output directory
    --export-metrics               # NEW: Export accuracy metrics CSV
  ```

- [ ] `scripts/optimize.py` additions:
  ```bash
  python scripts/optimize.py \
    --symbol RELIANCE \
    --strategy intraday \
    --param-grid config/param_grid.yaml \
    --export-sensitivity \          # NEW: Export parameter sensitivity analysis
    --export-features               # NEW: Export feature importance
  ```

- [ ] New script: `scripts/batch_audit.py`:
  ```bash
  python scripts/batch_audit.py \
    --config config/batch_audit.yaml \
    --workers 4 \
    --output-dir data/analytics/batch_20240115
  ```

### FR-7: Accuracy Analysis Module

**Description**: Create reusable module for computing accuracy metrics

**Acceptance Criteria**:
- [ ] New module: `src/services/accuracy_analyzer.py`

- [ ] Key classes:
  ```python
  @dataclass
  class PredictionTrace:
      """Single prediction record for analysis."""
      timestamp: datetime
      symbol: str
      strategy: str
      predicted_direction: str
      actual_direction: str
      predicted_confidence: float
      entry_price: float
      exit_price: float
      holding_period_minutes: int
      realized_return_pct: float
      features: dict[str, float]
      metadata: dict[str, Any]

  @dataclass
  class AccuracyMetrics:
      """Computed accuracy metrics."""
      precision: dict[str, float]      # Per direction
      recall: dict[str, float]          # Per direction
      f1_score: dict[str, float]        # Per direction
      confusion_matrix: np.ndarray      # 3x3 matrix
      hit_ratio: float
      win_rate: float
      avg_return: float
      sharpe_ratio: float
      max_drawdown: float
      profit_factor: float
      total_trades: int

  class AccuracyAnalyzer:
      """Compute accuracy metrics from prediction traces."""

      def load_traces(self, path: Path) -> list[PredictionTrace]:
          """Load prediction traces from CSV/JSON."""

      def compute_metrics(self, traces: list[PredictionTrace]) -> AccuracyMetrics:
          """Compute all accuracy metrics."""

      def export_report(self, metrics: AccuracyMetrics, output_path: Path):
          """Export metrics to JSON/CSV."""

      def plot_confusion_matrix(self, metrics: AccuracyMetrics) -> Figure:
          """Generate confusion matrix heatmap."""
  ```

### FR-8: Telemetry Integration

**Description**: Integrate telemetry capture into backtester and optimizer

**Acceptance Criteria**:
- [ ] Backtester modifications:
  - Add `enable_telemetry: bool` parameter to constructor
  - Add `telemetry_writer: TelemetryWriter` for output
  - Capture prediction trace for every signal
  - Write traces incrementally (no memory buildup)
  - Minimal performance impact when disabled (< 1% overhead)

- [ ] Optimizer modifications:
  - Track parameter grid search results
  - Export feature importance from best models
  - Log optimization convergence metrics

- [ ] TelemetryWriter class:
  - Support CSV and JSONL formats
  - Buffered writes (configurable batch size)
  - Compression support (gzip)
  - Rotation when file size exceeds limit

### FR-9: Integration Testing

**Description**: Test accuracy audit pipeline end-to-end

**Acceptance Criteria**:
- [ ] Test file: `tests/integration/test_accuracy_audit.py`

- [ ] Test scenarios:
  - **test_telemetry_capture**: Verify traces captured during backtest
  - **test_batch_backtest**: Run mini batch with 2 symbols
  - **test_metrics_computation**: Compute accuracy metrics from traces
  - **test_report_generation**: Generate report from batch results
  - **test_telemetry_disabled**: Verify no overhead when disabled
  - **test_file_rotation**: Verify rotation at max file size
  - **test_parallel_execution**: Verify parallel batch execution

- [ ] Assertions:
  - Output directories created
  - CSV/JSON files exist and parseable
  - Metrics computed correctly (known test data)
  - No errors in logs
  - Performance within tolerance

### FR-10: Documentation

**Description**: Document accuracy audit workflow

**Acceptance Criteria**:
- [ ] Update `docs/architecture.md` with Section 14: Accuracy Audit & Telemetry
- [ ] Include:
  - Architecture diagram
  - Data flow explanation
  - Configuration options
  - CLI usage examples
  - Notebook usage guide
  - Performance considerations
  - Best practices

---

## Architecture Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Batch Audit Pipeline                      │
└────────────────┬────────────────────────────────────────────┘
                 │ orchestrates
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backtester                             │
│  + run(symbol, strategy, date_range)                        │
│  + enable_telemetry: bool                                   │
└────────────────┬────────────────────────────────────────────┘
                 │ emits
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   TelemetryWriter                           │
│  + write_prediction_trace(trace: PredictionTrace)          │
│  + flush()                                                  │
│  + rotate_if_needed()                                       │
└────────────────┬────────────────────────────────────────────┘
                 │ writes to
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              data/analytics/{batch_id}/                     │
│  ├── predictions.csv                                        │
│  ├── predictions.jsonl                                      │
│  └── metadata.json                                          │
└────────────────┬────────────────────────────────────────────┘
                 │ analyzed by
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  AccuracyAnalyzer                           │
│  + load_traces(path)                                        │
│  + compute_metrics(traces)                                  │
│  + export_report(metrics)                                   │
└────────────────┬────────────────────────────────────────────┘
                 │ feeds
                 ▼
┌─────────────────────────────────────────────────────────────┐
│            notebooks/accuracy_report.ipynb                  │
│  - Load data from analytics directory                       │
│  - Compute visualizations                                   │
│  - Generate recommendations                                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Phase 1: Telemetry Capture**
1. User runs batch audit or backtest with `--enable-telemetry`
2. Backtester processes historical data
3. For each strategy signal:
   - Capture prediction (LONG/SHORT/NOOP)
   - Track entry price, exit price, holding period
   - Compute realized return
   - Determine actual direction (forward return)
   - Record feature values at signal time
4. TelemetryWriter buffers and writes traces to CSV/JSONL
5. Metadata written at completion (symbol, date range, runtime, etc.)

**Phase 2: Metrics Computation**
1. AccuracyAnalyzer loads prediction traces
2. Computes classification metrics (precision, recall, F1)
3. Builds confusion matrix (predicted vs actual direction)
4. Calculates trading metrics (hit ratio, Sharpe, max DD)
5. Analyzes feature importance and correlations
6. Exports metrics to JSON/CSV

**Phase 3: Report Generation**
1. Jupyter notebook loads batch results
2. Aggregates metrics across symbols/strategies
3. Generates visualizations (heatmaps, charts, distributions)
4. Identifies best/worst performers
5. Suggests parameter adjustments based on data
6. Exports HTML report for stakeholders

---

## Implementation Plan

### Phase 1: Core Telemetry Infrastructure (3-4 hours)

1. **Create data types** (`src/services/accuracy_analyzer.py`):
   - `PredictionTrace` dataclass
   - `AccuracyMetrics` dataclass
   - `TelemetryWriter` class

2. **Implement TelemetryWriter**:
   - CSV writer with buffering
   - JSONL writer (line-delimited JSON)
   - File rotation at max size
   - Compression support (gzip)

3. **Add settings** (`src/app/config.py`):
   - Telemetry enable/disable flags
   - Storage paths and limits
   - Sample rate configuration

### Phase 2: Backtester Integration (2-3 hours)

1. **Modify Backtester** (`src/services/backtester.py`):
   - Add `enable_telemetry` parameter
   - Inject `TelemetryWriter` instance
   - Capture prediction trace after each signal
   - Track realized outcomes (exit price, return)
   - Write traces incrementally

2. **Determine actual direction logic**:
   - Compute forward return over holding period
   - Classify as LONG (positive), SHORT (negative), NOOP (neutral)
   - Use threshold (e.g., |return| < 0.5% = NOOP)

### Phase 3: Accuracy Analyzer (2-3 hours)

1. **Implement AccuracyAnalyzer**:
   - `load_traces()`: Parse CSV/JSONL
   - `compute_metrics()`: Calculate all metrics
   - `export_report()`: Write JSON/CSV summaries

2. **Metrics computations**:
   - Scikit-learn: precision_score, recall_score, confusion_matrix
   - Custom: hit_ratio, Sharpe, max_drawdown, profit_factor

3. **Visualization helpers**:
   - `plot_confusion_matrix()`: Seaborn heatmap
   - `plot_return_distribution()`: Histogram
   - `plot_feature_importance()`: Bar chart

### Phase 4: Batch Auditing (2-3 hours)

1. **Create batch executor** (`src/services/batch_auditor.py`):
   - Load batch configuration (YAML)
   - Iterate symbols × intervals × strategies
   - Run backtests in parallel (multiprocessing)
   - Aggregate results
   - Export summary CSV

2. **CLI script** (`scripts/batch_audit.py`):
   - Parse command-line arguments
   - Load batch config
   - Execute batch auditor
   - Display progress bar (tqdm)
   - Print summary statistics

### Phase 5: Report Notebook (1-2 hours)

1. **Create notebook** (`notebooks/accuracy_report.ipynb`):
   - Parameterized batch_id input
   - Load data from analytics directory
   - Generate all visualizations
   - Compute aggregate statistics
   - Export recommendations

2. **Visualizations**:
   - Confusion matrices (per strategy)
   - Precision/recall bars
   - Return distributions
   - Cumulative P&L curves
   - Feature importance
   - Parameter sensitivity heatmaps

### Phase 6: CLI Integration (1 hour)

1. **Update `scripts/backtest.py`**:
   - Add `--enable-telemetry` flag
   - Add `--telemetry-dir` option
   - Add `--export-metrics` flag

2. **Update `scripts/optimize.py`**:
   - Add `--export-sensitivity` flag
   - Add `--export-features` flag

### Phase 7: Testing & Documentation (2-3 hours)

1. **Integration tests** (`tests/integration/test_accuracy_audit.py`):
   - Test telemetry capture
   - Test batch execution
   - Test metrics computation
   - Test report generation

2. **Update documentation** (`docs/architecture.md`):
   - Add Section 14
   - Document workflow
   - Provide usage examples

---

## Acceptance Criteria

### Functional Requirements

- [ ] FR-1: Prediction-level telemetry capture (CSV/JSONL)
- [ ] FR-2: Comprehensive accuracy metrics (precision, recall, confusion matrix, trading metrics)
- [ ] FR-3: Batch backtesting framework (parallel execution, progress tracking)
- [ ] FR-4: Jupyter notebook report (visualizations, recommendations)
- [ ] FR-5: Configuration settings (enable/disable, paths, limits)
- [ ] FR-6: CLI enhancements (telemetry flags, batch audit script)
- [ ] FR-7: AccuracyAnalyzer module (load, compute, export)
- [ ] FR-8: Telemetry integration (Backtester, Optimizer)
- [ ] FR-9: Integration testing (7 test scenarios)
- [ ] FR-10: Documentation (architecture section, usage guide)

### Quality Gates

- [ ] **Code Quality**: `ruff check .` passes (zero errors)
- [ ] **Formatting**: `ruff format .` passes
- [ ] **Type Safety**: `mypy src/` passes (zero errors)
- [ ] **Tests**: All tests pass (100% success rate)
- [ ] **Performance**: Telemetry overhead < 5% when enabled, < 1% when disabled

### Performance Requirements

- [ ] Telemetry capture: < 1ms per trace
- [ ] Batch backtest: Process 1000 trades/second
- [ ] Metrics computation: < 5 seconds for 10K traces
- [ ] Report generation: < 30 seconds for full batch
- [ ] Memory usage: < 500MB for 100K traces

---

## Risks and Mitigations

### Risk 1: Performance Degradation

**Impact**: High
**Probability**: Medium
**Mitigation**:
- Implement buffered writes (batch size: 100 traces)
- Use generator patterns for large datasets
- Add telemetry disable flag (default: off)
- Profile and optimize hot paths
- Lazy load features (only when requested)

### Risk 2: Storage Consumption

**Impact**: Medium
**Probability**: High
**Mitigation**:
- Implement file rotation at max size
- Add compression support (gzip reduces by 70%+)
- Configurable sample rate (capture only N% of traces)
- Auto-cleanup old audit runs (retention policy)

### Risk 3: Data Leakage in Reports

**Impact**: High
**Probability**: Low
**Mitigation**:
- Reports use only cached historical data
- No live API calls from notebooks
- Clearly document data sources
- Add validation checks for date ranges

### Risk 4: Incorrect Metrics

**Impact**: High
**Probability**: Medium
**Mitigation**:
- Use well-tested libraries (scikit-learn)
- Comprehensive unit tests with known data
- Cross-validate against manual calculations
- Document metric definitions clearly

---

## Future Enhancements (Post-MVP)

1. **Real-Time Dashboard**: Grafana/Plotly dashboard for live accuracy monitoring
2. **A/B Testing**: Compare strategy variants with statistical significance tests
3. **Auto-Optimization**: Automatically adjust parameters based on audit results
4. **Anomaly Detection**: Flag unusual accuracy drops for investigation
5. **Multi-Asset Analysis**: Cross-asset correlation and sector analysis
6. **ML Model Retraining**: Trigger retraining when accuracy drops below threshold

---

## References

- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Confusion Matrix Interpretation](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)
- [Maximum Drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)

---

**End of US-016 Story Document**
