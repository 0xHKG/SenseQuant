# US-008 — Teacher Label Generator

**Status:** ✅ Completed
**Priority:** High
**Estimated Effort:** 8 story points
**Dependencies:** US-007 (Multi-Indicator Feature Library)

---

## Overview

Build an offline "Teacher" service that ingests historical OHLCV data, generates features using the shared feature library, fits a gradient boosted classifier (LightGBM), and emits training artifacts (trained model, labels, feature importance) for downstream consumption by a Student model.

This establishes the foundation for a Teacher-Student learning framework where:
- **Teacher**: Trains on historical data with full feature set, generates high-quality labels
- **Student**: (Future) Trains on Teacher's labels with simplified features for real-time inference

---

## Business Value

1. **Data-Driven Signals**: Replace hand-coded signal logic with ML-based predictions
2. **Feature Validation**: Identify which technical indicators contribute most to profitable trades
3. **Reproducible Training**: Deterministic training pipeline with versioned artifacts
4. **Foundation for Student**: Prepare labeled data and distillation targets for lightweight Student model

---

## Acceptance Criteria

### AC-1: Historical Data Loading
- [x] Load historical OHLCV bars from BreezeClient or CSV
- [x] Support configurable date ranges and symbols
- [x] Handle missing data and holidays gracefully
- [x] Log data quality metrics (rows, missing values, date range)

### AC-2: Feature Matrix Generation
- [x] Use shared feature library (src/domain/features.py) to compute all 9 indicators
- [x] Generate feature DataFrame with SMA, EMA, RSI, ATR, VWAP, Bollinger Bands, MACD, ADX, OBV
- [x] Drop rows with NaN values (post warm-up period)
- [x] Validate feature matrix shape and types

### AC-3: Label Generation
- [x] Define labeling strategy: forward-looking N-day returns
- [x] Generate binary labels (1 = profitable trade, 0 = unprofitable/neutral)
- [x] Support configurable labeling window (e.g., 5-day forward return > threshold)
- [x] Log label distribution (class balance)

### AC-4: Model Training
- [x] Train LightGBM classifier on feature matrix and labels
- [x] Use train/validation split (80/20) with stratification
- [x] Log training metrics (accuracy, precision, recall, F1, AUC)
- [x] Support reproducible seeds for deterministic results
- [x] Export feature importance

### AC-5: Artifact Persistence
- [x] Save trained model to data/models/teacher_model_YYYYMMDD.pkl
- [x] Save labels to data/models/teacher_labels_YYYYMMDD.csv
- [x] Save feature importance to data/models/teacher_importance_YYYYMMDD.csv
- [x] Save training metadata (config, metrics) to data/models/teacher_metadata_YYYYMMDD.json

### AC-6: Type Safety and Logging
- [x] Add TrainingConfig and TrainingResult DTOs in src/domain/types.py
- [x] Use structured logging with component="teacher"
- [x] Full type hints with mypy validation

### AC-7: Testing
- [x] Unit tests for data prep, label generation, persistence
- [x] Integration test that trains on sample data and validates artifacts
- [x] All tests passing with 100% success rate

### AC-8: CLI Entry Point
- [x] Add teacher training command or script
- [x] Support command-line arguments for config (symbol, date range, seed)
- [x] Log training progress and final metrics

---

## Technical Design

### Data Flow

```
Historical Bars → Feature Generation → Label Generation → Train/Val Split → Model Training → Artifacts
     ↓                    ↓                    ↓                  ↓                ↓            ↓
BreezeClient      feature library    Forward returns    Stratified      LightGBM      data/models/
                  (9 indicators)     (binary labels)      80/20                      - model.pkl
                                                                                    - labels.csv
                                                                                    - importance.csv
                                                                                    - metadata.json
```

### Labeling Strategy

**Forward-Looking Returns Approach:**
1. For each bar at time `t`, calculate forward return over next `N` days:
   ```
   forward_return = (close[t+N] - close[t]) / close[t]
   ```

2. Generate binary label:
   ```
   label = 1 if forward_return > threshold else 0
   ```

3. Configuration:
   - `label_window_days`: N-day forward window (default: 5)
   - `label_threshold_pct`: Minimum return to classify as profitable (default: 2%)

**Alternative (Advanced):** Multi-class labels (LONG, SHORT, FLAT) based on directional returns

### Feature Set

Using all 9 indicators from feature library:
1. **SMA** (Simple Moving Average) - Trend following
2. **EMA** (Exponential Moving Average) - Recent trend
3. **RSI** (Relative Strength Index) - Momentum
4. **ATR** (Average True Range) - Volatility
5. **VWAP** (Volume-Weighted Average Price) - Intraday benchmark
6. **Bollinger Bands** (Upper, Middle, Lower) - Volatility bands
7. **MACD** (Line, Signal, Histogram) - Trend momentum
8. **ADX** (Average Directional Index) - Trend strength
9. **OBV** (On-Balance Volume) - Volume flow

### Model Architecture

**LightGBM Classifier Configuration:**
```python
lgb.LGBMClassifier(
    objective='binary',
    num_leaves=31,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42,
    verbose=-1,
)
```

**Training Pipeline:**
1. Load historical data (1+ years for robust training)
2. Compute features using feature library
3. Generate forward-looking labels
4. Drop NaN rows (warm-up period + forward window)
5. Split train/validation (80/20 stratified)
6. Train LightGBM classifier
7. Evaluate on validation set
8. Export artifacts

### Artifact Schema

**teacher_model_YYYYMMDD.pkl:**
- Serialized LightGBM model (pickle format)

**teacher_labels_YYYYMMDD.csv:**
```csv
timestamp,symbol,label,forward_return
2024-01-01,RELIANCE,1,0.0324
2024-01-02,RELIANCE,0,-0.0012
...
```

**teacher_importance_YYYYMMDD.csv:**
```csv
feature,importance,rank
rsi14,0.234,1
macd_histogram,0.189,2
atr14,0.143,3
...
```

**teacher_metadata_YYYYMMDD.json:**
```json
{
  "training_date": "2024-10-12",
  "symbol": "RELIANCE",
  "date_range": ["2023-01-01", "2024-10-11"],
  "total_rows": 5234,
  "train_rows": 4187,
  "val_rows": 1047,
  "label_distribution": {"0": 2891, "1": 2343},
  "config": {
    "label_window_days": 5,
    "label_threshold_pct": 0.02,
    "random_seed": 42
  },
  "metrics": {
    "train_accuracy": 0.732,
    "val_accuracy": 0.689,
    "val_precision": 0.671,
    "val_recall": 0.702,
    "val_f1": 0.686,
    "val_auc": 0.745
  }
}
```

### Type Definitions

**src/domain/types.py additions:**

```python
@dataclass
class TrainingConfig:
    """Configuration for Teacher model training."""
    symbol: str
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    label_window_days: int = 5
    label_threshold_pct: float = 0.02
    train_split: float = 0.8
    random_seed: int = 42
    model_params: dict[str, Any] | None = None

@dataclass
class TrainingResult:
    """Results from Teacher model training."""
    model_path: str
    labels_path: str
    importance_path: str
    metadata_path: str
    metrics: dict[str, float]
    feature_count: int
    train_samples: int
    val_samples: int
```

### Implementation Modules

**src/services/teacher_student.py:**
```python
class TeacherLabeler:
    """Offline Teacher service for label generation and model training."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize with training configuration."""

    def load_historical_data(self) -> pd.DataFrame:
        """Load historical OHLCV bars."""

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate feature matrix using feature library."""

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """Generate forward-looking binary labels."""

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train LightGBM model and export artifacts."""

    def save_artifacts(
        self,
        model: Any,
        labels: pd.DataFrame,
        importance: pd.DataFrame,
        metadata: dict[str, Any]
    ) -> TrainingResult:
        """Save all training artifacts to data/models/."""
```

---

## Testing Strategy

### Unit Tests (test_teacher_labeler.py)

1. **test_load_historical_data_valid_range** - Verifies data loading
2. **test_generate_features_all_indicators** - Validates feature generation
3. **test_generate_labels_forward_return** - Tests label calculation
4. **test_generate_labels_threshold** - Tests labeling threshold
5. **test_train_validation_split** - Validates stratified split
6. **test_save_artifacts_creates_files** - Tests file persistence
7. **test_load_saved_model** - Tests model serialization/deserialization
8. **test_feature_importance_export** - Validates importance CSV

### Integration Tests (test_teacher_pipeline.py)

1. **test_full_training_pipeline** - End-to-end training on sample data
2. **test_trained_model_predictions** - Validates model inference
3. **test_artifact_completeness** - Ensures all files created
4. **test_reproducible_training** - Same seed produces same results

---

## Implementation Checklist

- [x] Create US-008 story document
- [x] Add TrainingConfig and TrainingResult to src/domain/types.py
- [x] Implement TeacherLabeler class in src/services/teacher_student.py
- [x] Implement historical data loading
- [x] Implement feature matrix generation
- [x] Implement label generation logic
- [x] Implement LightGBM training pipeline
- [x] Implement artifact persistence (model, labels, importance, metadata)
- [x] Add CLI entry point (script or engine command)
- [x] Create unit tests (8+ test cases)
- [x] Create integration test (full pipeline)
- [x] Run quality gates (ruff, mypy, pytest)
- [x] Update story document with implementation summary

---

## Future Enhancements (Out of Scope)

1. **Multi-Symbol Training**: Train on portfolio of symbols simultaneously
2. **Feature Selection**: Automated feature selection based on importance
3. **Hyperparameter Tuning**: Grid search or Bayesian optimization
4. **Student Model**: Lightweight model distilled from Teacher predictions
5. **Online Learning**: Incremental model updates with new data
6. **Ensemble Models**: Combine multiple Teacher models
7. **Alternative Labels**: Multi-class (LONG/SHORT/FLAT) or regression (expected return)

---

## References

- LightGBM Documentation: https://lightgbm.readthedocs.io/
- Teacher-Student Learning: https://arxiv.org/abs/1503.02531
- Feature Importance: Gain-based importance from tree models
- Financial ML: Advances in Financial Machine Learning (de Prado)

---

## Story Completion Checklist

- [x] Story document created
- [x] Type definitions added (TrainingConfig, TrainingResult)
- [x] TeacherLabeler service implemented
- [x] Unit tests created (15 tests)
- [x] Integration tests created (6 tests)
- [x] CLI entry point added (scripts/train_teacher.py)
- [x] All quality gates pass (ruff, mypy, pytest)
- [x] Code review complete

---

## Implementation Summary

**Status:** ✅ All Implementation Complete

**Test Results:**
- ruff check: All checks passed ✓
- ruff format: All files formatted ✓
- mypy: Success, no type errors ✓
- pytest: 149/149 tests passing (100% pass rate) ✓

**New Capabilities:**
- Offline Teacher training service with LightGBM/sklearn support
- Historical data loading and feature generation using shared library
- Forward-looking label generation (binary classification)
- Train/validation split with stratification
- Model training with reproducible seeds
- Comprehensive artifact persistence (model, labels, importance, metadata)
- CLI entry point for easy training execution
- Full type safety and structured logging

**Files Created:**
1. `src/services/teacher_student.py` (520+ lines) - TeacherLabeler service
   - Historical data loading
   - Feature matrix generation (all 9 indicators)
   - Forward return label generation
   - LightGBM/sklearn model training
   - Artifact persistence to data/models/
   - Full pipeline orchestration

2. `tests/unit/test_teacher_labeler.py` (470+ lines) - Comprehensive unit tests
   - 15 unit tests covering all Teacher functionality
   - Tests for data loading, feature generation, label generation
   - Tests for model training, feature importance, reproducibility
   - Tests for artifact persistence and metadata structure

3. `tests/integration/test_teacher_pipeline.py` (280+ lines) - Integration tests
   - 6 end-to-end integration tests
   - Full pipeline test with artifact validation
   - Model prediction tests
   - Reproducibility validation
   - Configuration flexibility tests

4. `scripts/train_teacher.py` (200+ lines) - CLI entry point
   - Command-line arguments for all training parameters
   - Date range validation
   - BreezeClient integration
   - Structured logging and progress reporting
   - Results summary with metrics

**Type Definitions Added** (src/domain/types.py):
- `TrainingConfig`: Training configuration dataclass
- `TrainingResult`: Training results dataclass

**Test Coverage:**
- From 128 tests to 149 tests (21 new tests added)
- Unit tests: 15 tests for Teacher functionality
- Integration tests: 6 end-to-end tests
- 100% pass rate maintained
- No regressions in existing tests

**Quality Gates:**
- ✓ ruff check: All checks passed
- ✓ ruff format: 36 files formatted correctly
- ✓ mypy: Success, no type errors (20 source files)
- ✓ pytest: 149/149 tests passing

**CLI Usage:**
```bash
# Basic usage
python scripts/train_teacher.py --symbol RELIANCE --start 2023-01-01 --end 2024-10-01

# Advanced usage with custom parameters
python scripts/train_teacher.py \
  --symbol TCS \
  --start 2023-01-01 \
  --end 2024-10-01 \
  --window 10 \
  --threshold 0.03 \
  --split 0.75 \
  --seed 123 \
  --estimators 200 \
  --max-depth 7
```

**Artifact Output Structure:**
```
data/models/
├── teacher_model_YYYYMMDD_HHMMSS.pkl          # Trained model (pickle)
├── teacher_labels_YYYYMMDD_HHMMSS.csv         # Generated labels
├── teacher_importance_YYYYMMDD_HHMMSS.csv     # Feature importance rankings
└── teacher_metadata_YYYYMMDD_HHMMSS.json      # Training metadata & metrics
```

**Key Features:**
1. **Flexible Model Backend**: Automatic fallback from LightGBM to sklearn GradientBoosting
2. **Robust Label Generation**: Forward-looking returns with configurable window and threshold
3. **Stratified Splitting**: Maintains class balance in train/val split (when possible)
4. **Reproducible Training**: Fixed random seeds for deterministic results
5. **Comprehensive Logging**: Structured logging with component tags
6. **Type-Safe**: Full type hints with mypy validation
7. **Production-Ready**: CLI interface, error handling, artifact versioning
