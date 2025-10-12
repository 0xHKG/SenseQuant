# US-020: Teacher/Student Model Training Automation

## Status
**Status**: Complete
**Priority**: High
**Assignee**: Development Team
**Sprint**: Current
**Completed**: 2025-10-12
**Dependencies**: US-008 (Teacher Labeler), US-009 (Student Inference)

## Problem Statement

Following US-008 (Teacher Labeler) and US-009 (Student Inference), we have the foundational components for teacher/student modeling but lack automated training pipelines. Currently:

1. **No Automated Teacher Training**: Teacher model training requires manual execution with no standardized pipeline for telemetry capture, label generation, and artifact versioning.

2. **Missing Student Training Workflow**: While student inference exists, there's no automated pipeline for training the student model from teacher-generated labels with incremental updates.

3. **No Post-Training Validation**: No automated workflow to validate new models via backtesting before promotion to production.

4. **Artifact Versioning Gaps**: Model artifacts, labels, and metadata are not systematically versioned with linkage to optimization runs and config hashes.

5. **Unclear Promotion Process**: No formal checklist or evaluation criteria for promoting trained models to production.

**Current Gaps**:
- No `scripts/train_teacher.py` for automated teacher training pipeline
- No `scripts/train_student.py` for automated student training pipeline
- Missing artifact versioning under `data/models/<timestamp>/`
- No integration between training and backtest validation
- Unclear rollback procedures for model deployments

## Objectives

1. **Automated Teacher Training Pipeline**:
   - CLI script for teacher training across multiple symbols/date ranges
   - Telemetry capture during backtesting for label generation
   - Teacher model training with feature engineering
   - Artifact versioning (model, labels, metadata, feature stats)
   - Dataset statistics and quality metrics

2. **Automated Student Training Pipeline**:
   - CLI script consuming teacher-generated labels
   - Support for incremental updates (append new training data)
   - Student model training with hyperparameter tuning
   - Evaluation metrics (accuracy, precision/recall, calibration, AUC)
   - Artifact versioning linked to teacher run

3. **Post-Training Validation**:
   - Automated backtest with new student model
   - Telemetry accuracy computation
   - Comparison with baseline model
   - Promotion checklist generation
   - Human review workflow

4. **Artifact Management**:
   - Versioned storage: `data/models/<timestamp>/teacher/` and `.../student/`
   - Metadata linking to optimization run, config hash, dataset stats
   - Model provenance tracking
   - Rollback capability

5. **Documentation & Testing**:
   - Integration tests for full teacher→student→backtest workflow
   - Training workflow documentation (prerequisites, steps, evaluation)
   - Rollback procedures

## Requirements

### FR-1: Teacher Training Pipeline

**Description**: Automated pipeline for training teacher model from backtest telemetry.

**Script**: `scripts/train_teacher.py`

**Usage**:
```bash
# Train teacher model on historical data
python scripts/train_teacher.py \
  --symbols RELIANCE TCS INFY \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --strategy intraday \
  --output-dir data/models/$(date +%Y%m%d_%H%M%S) \
  --min-holding-return 0.005 \
  --lookback-days 20 \
  --features rsi sma volume sentiment \
  --run-backtest

# Incremental training (append new data)
python scripts/train_teacher.py \
  --symbols RELIANCE TCS \
  --start-date 2024-07-01 \
  --end-date 2024-09-30 \
  --strategy intraday \
  --baseline-model data/models/20250112_143000/teacher/teacher_model.pkl \
  --append-labels
```

**Teacher Training Workflow**:
1. Run backtest with telemetry enabled
2. Collect prediction traces (entry/exit prices, holding periods, returns)
3. Apply teacher labeling logic (label LONG/SHORT/NOOP based on future returns)
4. Engineer features (technical indicators, sentiment, market regime)
5. Train teacher model (LightGBM/RandomForest)
6. Save artifacts (model, labels, features, metadata)
7. Compute and save evaluation metrics

**Artifacts Generated**:
```
data/models/<timestamp>/teacher/
├── teacher_model.pkl           # Trained teacher model
├── labels.csv.gz                # Teacher-generated labels
├── features.csv.gz              # Engineered features
├── metadata.json                # Training metadata
├── dataset_stats.json           # Dataset statistics
├── evaluation_metrics.json     # Model evaluation (accuracy, AUC, etc.)
└── feature_importance.json     # Feature importance scores
```

**Metadata Structure** (`metadata.json`):
```json
{
  "timestamp": "2025-01-12T14:30:00Z",
  "symbols": ["RELIANCE", "TCS", "INFY"],
  "date_range": {"start": "2024-01-01", "end": "2024-06-30"},
  "strategy": "intraday",
  "config_hash": "a3f5d8c2",
  "optimization_run": "data/optimization/run_20250110_120000",
  "teacher_params": {
    "min_holding_return": 0.005,
    "lookback_days": 20,
    "features": ["rsi", "sma", "volume", "sentiment"]
  },
  "dataset_stats": {
    "total_samples": 15234,
    "label_distribution": {"LONG": 4521, "SHORT": 3812, "NOOP": 6901},
    "missing_values": 0,
    "date_range_actual": {"start": "2024-01-02", "end": "2024-06-28"}
  },
  "model_type": "LightGBM",
  "model_params": {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05
  }
}
```

**Acceptance Criteria**:
- CLI accepts symbols, date range, strategy, output directory
- Backtest runs with telemetry enabled
- Teacher labels generated based on configurable thresholds
- Features engineered and saved
- Teacher model trained and saved as pickle
- Metadata includes config hash and dataset stats
- Evaluation metrics computed (accuracy, precision/recall per class)

### FR-2: Student Training Pipeline

**Description**: Automated pipeline for training student model from teacher labels.

**Script**: `scripts/train_student.py`

**Usage**:
```bash
# Train student from teacher labels
python scripts/train_student.py \
  --teacher-dir data/models/20250112_143000/teacher \
  --output-dir data/models/20250112_143000/student \
  --model-type logistic \
  --hyperparameter-tuning \
  --cv-folds 5 \
  --test-size 0.2

# Incremental student training
python scripts/train_student.py \
  --teacher-dir data/models/20250112_143000/teacher \
  --baseline-student data/models/20250110_100000/student/student_model.pkl \
  --output-dir data/models/20250112_143000/student \
  --incremental
```

**Student Training Workflow**:
1. Load teacher-generated labels and features
2. Split into train/validation/test sets
3. Optional: hyperparameter tuning via cross-validation
4. Train student model (Logistic Regression/SGD/LightGBM)
5. Evaluate on test set (accuracy, precision/recall, F1, AUC, calibration)
6. Save artifacts (model, evaluation metrics, metadata)
7. Generate calibration plot and confusion matrix

**Artifacts Generated**:
```
data/models/<timestamp>/student/
├── student_model.pkl           # Trained student model
├── metadata.json                # Student training metadata
├── evaluation_metrics.json     # Test set evaluation
├── confusion_matrix.json       # Confusion matrix
├── calibration_plot.png        # Calibration curve
└── roc_curves.png              # ROC curves per class
```

**Evaluation Metrics** (`evaluation_metrics.json`):
```json
{
  "timestamp": "2025-01-12T15:00:00Z",
  "teacher_dir": "data/models/20250112_143000/teacher",
  "test_size": 0.2,
  "cv_folds": 5,
  "model_type": "logistic",
  "hyperparameters": {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs"
  },
  "test_metrics": {
    "accuracy": 0.687,
    "precision": {"LONG": 0.72, "SHORT": 0.65, "NOOP": 0.68},
    "recall": {"LONG": 0.68, "SHORT": 0.61, "NOOP": 0.73},
    "f1_score": {"LONG": 0.70, "SHORT": 0.63, "NOOP": 0.70},
    "auc": {"LONG": 0.78, "SHORT": 0.74, "NOOP": 0.76},
    "macro_avg_f1": 0.677
  },
  "calibration_metrics": {
    "expected_calibration_error": 0.042,
    "brier_score": 0.215
  },
  "confusion_matrix": {
    "LONG": {"LONG": 612, "SHORT": 98, "NOOP": 194},
    "SHORT": {"LONG": 102, "SHORT": 465, "NOOP": 195},
    "NOOP": {"LONG": 156, "SHORT": 214, "NOOP": 1010}
  }
}
```

**Acceptance Criteria**:
- CLI accepts teacher directory, output directory, model type
- Loads teacher labels and features
- Splits data into train/validation/test
- Supports hyperparameter tuning via cross-validation
- Trains student model and saves as pickle
- Computes comprehensive evaluation metrics
- Generates calibration plot and confusion matrix
- Metadata links to teacher run

### FR-3: Post-Training Validation

**Description**: Automated validation of trained student model via backtesting.

**Integrated into**: `scripts/train_student.py --validate`

**Usage**:
```bash
# Train and validate student model
python scripts/train_student.py \
  --teacher-dir data/models/20250112_143000/teacher \
  --output-dir data/models/20250112_143000/student \
  --validate \
  --validation-symbols RELIANCE TCS \
  --validation-start 2024-07-01 \
  --validation-end 2024-09-30 \
  --baseline-model data/models/baseline/student_model.pkl
```

**Validation Workflow**:
1. Load trained student model
2. Run backtest on validation date range (out-of-sample)
3. Enable telemetry to capture predictions
4. Compute accuracy metrics from telemetry
5. Compare with baseline model (if provided)
6. Generate promotion checklist

**Validation Artifacts**:
```
data/models/<timestamp>/student/
├── validation_results.json     # Backtest results
├── validation_telemetry/        # Prediction traces
├── promotion_checklist.md      # Human review checklist
└── comparison_baseline.json    # Comparison with baseline
```

**Promotion Checklist** (`promotion_checklist.md`):
```markdown
# Student Model Promotion Checklist

**Model**: `data/models/20250112_143000/student/student_model.pkl`
**Generated**: 2025-01-12 15:30:00

## Model Performance (Validation Period: 2024-07-01 to 2024-09-30)

### Accuracy Metrics
- **Precision (LONG)**: 71.2% (baseline: 68.5%, +2.7%)
- **Hit Ratio**: 66.8% (baseline: 63.2%, +3.6%)
- **Overall Accuracy**: 68.7% (baseline: 65.4%, +3.3%)

### Financial Metrics (Backtest)
- **Sharpe Ratio**: 1.85 (baseline: 1.72, +7.6%)
- **Total Return**: 14.2% (baseline: 11.8%, +2.4%)
- **Max Drawdown**: 8.3% (baseline: 9.1%, -0.8%)
- **Win Rate**: 58.3% (baseline: 55.7%, +2.6%)

## Validation Criteria

✅ **Accuracy Improvement**: Precision (LONG) ≥ baseline + 2%
✅ **Financial Improvement**: Sharpe Ratio ≥ baseline + 5%
✅ **Calibration Quality**: Expected Calibration Error < 0.05
✅ **Robustness**: Performance consistent across all validation symbols
⚠️ **Statistical Significance**: p-value = 0.08 (marginally significant)

## Pre-Deployment Checklist

- [ ] Model performance meets all validation criteria
- [ ] Out-of-sample validation period representative
- [ ] Calibration plot reviewed (no over/under-confidence)
- [ ] Feature drift analysis completed
- [ ] Baseline model archived with timestamp
- [ ] Rollback procedure documented and tested
- [ ] Monitoring alerts configured for production deployment

## Approval Sign-offs

- [ ] **ML Lead**: _______________ Date: ___________
- [ ] **Quant Team Lead**: _______________ Date: ___________
- [ ] **Risk Manager**: _______________ Date: ___________

## Deployment Instructions

1. **Archive Baseline Model**:
   ```bash
   cp data/models/production/student_model.pkl \
      data/models/archive/student_baseline_$(date +%Y%m%d).pkl
   ```

2. **Deploy New Model**:
   ```bash
   cp data/models/20250112_143000/student/student_model.pkl \
      data/models/production/student_model.pkl
   ```

3. **Update Metadata**:
   ```bash
   echo "20250112_143000" > data/models/production/model_version.txt
   ```

4. **Monitor for 1 Week**: Daily review of telemetry accuracy

## Rollback Procedure

If production telemetry accuracy < 90% of validation accuracy for 3 consecutive days:

1. **Halt Trading**: Disable strategy execution
2. **Restore Baseline**:
   ```bash
   cp data/models/archive/student_baseline_YYYYMMDD.pkl \
      data/models/production/student_model.pkl
   ```
3. **Restart Trading**: Re-enable strategy execution
4. **Root Cause Analysis**: Investigate performance degradation
```

**Acceptance Criteria**:
- Validation backtest runs successfully on out-of-sample data
- Telemetry accuracy metrics computed
- Comparison with baseline model (if provided)
- Promotion checklist generated with all criteria
- Statistical significance testing included
- Rollback procedure documented

### FR-4: Artifact Versioning & Metadata

**Description**: Systematic versioning of training artifacts with metadata linkage.

**Directory Structure**:
```
data/models/
├── <timestamp>/                # Training run directory
│   ├── teacher/
│   │   ├── teacher_model.pkl
│   │   ├── labels.csv.gz
│   │   ├── features.csv.gz
│   │   ├── metadata.json
│   │   ├── dataset_stats.json
│   │   ├── evaluation_metrics.json
│   │   └── feature_importance.json
│   ├── student/
│   │   ├── student_model.pkl
│   │   ├── metadata.json
│   │   ├── evaluation_metrics.json
│   │   ├── confusion_matrix.json
│   │   ├── calibration_plot.png
│   │   ├── roc_curves.png
│   │   ├── validation_results.json
│   │   ├── validation_telemetry/
│   │   ├── promotion_checklist.md
│   │   └── comparison_baseline.json
│   └── training_pipeline.json  # Links teacher→student→validation
├── production/                  # Production models (symlinks or copies)
│   ├── student_model.pkl
│   └── model_version.txt
└── archive/                     # Archived baselines
    └── student_baseline_YYYYMMDD.pkl
```

**Pipeline Metadata** (`training_pipeline.json`):
```json
{
  "timestamp": "2025-01-12T14:30:00Z",
  "pipeline_version": "1.0",
  "stages": {
    "teacher_training": {
      "start": "2025-01-12T14:30:00Z",
      "end": "2025-01-12T14:45:00Z",
      "status": "completed",
      "artifacts": "data/models/20250112_143000/teacher/"
    },
    "student_training": {
      "start": "2025-01-12T14:45:00Z",
      "end": "2025-01-12T15:00:00Z",
      "status": "completed",
      "artifacts": "data/models/20250112_143000/student/"
    },
    "validation": {
      "start": "2025-01-12T15:00:00Z",
      "end": "2025-01-12T15:30:00Z",
      "status": "completed",
      "artifacts": "data/models/20250112_143000/student/validation_results.json"
    }
  },
  "config_hash": "a3f5d8c2",
  "optimization_run": "data/optimization/run_20250110_120000",
  "git_commit": "8b965e5d",
  "dependencies": {
    "python": "3.12.2",
    "scikit-learn": "1.5.2",
    "lightgbm": "4.5.0"
  }
}
```

**Acceptance Criteria**:
- Artifacts stored under timestamped directories
- Metadata includes config hash, optimization run link, git commit
- Pipeline metadata tracks all stages
- Production directory contains current model
- Archive directory preserves baseline models

### FR-5: Integration Testing

**Description**: End-to-end integration test for teacher→student→validation workflow.

**Test**: `tests/integration/test_model_training.py`

**Test Cases**:

1. **test_teacher_training_pipeline**: Train teacher model, verify artifacts
2. **test_student_training_pipeline**: Train student from teacher labels, verify artifacts
3. **test_full_training_workflow**: Teacher→student→validation end-to-end
4. **test_incremental_teacher_training**: Append new labels to existing teacher
5. **test_model_promotion_checklist**: Verify checklist generation

**Example Test**:
```python
def test_full_training_workflow(tmp_path):
    """Test full teacher→student→validation workflow (US-020).

    Verifies:
    - Teacher training generates all artifacts
    - Student training consumes teacher labels
    - Validation backtest runs successfully
    - Promotion checklist generated
    - All metadata properly linked
    """
    from scripts.train_teacher import train_teacher
    from scripts.train_student import train_student

    output_dir = tmp_path / "models" / "test_run"

    # Step 1: Train teacher
    teacher_result = train_teacher(
        symbols=["RELIANCE"],
        start_date="2024-01-02",
        end_date="2024-01-05",  # Short for speed
        strategy="intraday",
        output_dir=output_dir / "teacher",
        data_source="csv",
    )

    # Verify teacher artifacts
    assert (output_dir / "teacher" / "teacher_model.pkl").exists()
    assert (output_dir / "teacher" / "labels.csv.gz").exists()
    assert (output_dir / "teacher" / "metadata.json").exists()

    # Step 2: Train student
    student_result = train_student(
        teacher_dir=output_dir / "teacher",
        output_dir=output_dir / "student",
        model_type="logistic",
        test_size=0.2,
    )

    # Verify student artifacts
    assert (output_dir / "student" / "student_model.pkl").exists()
    assert (output_dir / "student" / "evaluation_metrics.json").exists()
    assert (output_dir / "student" / "metadata.json").exists()

    # Step 3: Validate with backtest
    validation_result = student_result.validate(
        symbols=["RELIANCE"],
        start_date="2024-01-06",
        end_date="2024-01-08",
        data_source="csv",
    )

    # Verify validation artifacts
    assert (output_dir / "student" / "validation_results.json").exists()
    assert (output_dir / "student" / "promotion_checklist.md").exists()

    # Verify metadata linkage
    student_metadata = json.load((output_dir / "student" / "metadata.json").open())
    assert student_metadata["teacher_dir"] == str(output_dir / "teacher")

    print("\n✓ Full training workflow validated")
```

**Acceptance Criteria**:
- Integration test covers teacher→student→validation workflow
- Test runs with sample CSV data
- All artifacts verified to exist
- Metadata linkage validated
- Test completes in < 30 seconds

## Architecture Design

### Training Pipeline Workflow

```
┌──────────────────────────────────────────────────────────┐
│         US-020 Training Pipeline Workflow                │
└──────────────────────────────────────────────────────────┘

User Input
  ├─ Symbols (list)
  ├─ Date Range
  ├─ Strategy
  └─ Training Parameters
  │
  ▼
┌────────────────────────────────────────┐
│  scripts/train_teacher.py              │
│  - Run backtest with telemetry         │
│  - Collect prediction traces           │
│  - Apply teacher labeling logic        │
│  - Engineer features                   │
│  - Train teacher model                 │
│  - Save artifacts                      │
└────────────────────────────────────────┘
  │
  │ teacher_model.pkl, labels.csv.gz, features.csv.gz
  │
  ▼
┌────────────────────────────────────────┐
│  scripts/train_student.py              │
│  - Load teacher labels & features      │
│  - Split train/test sets               │
│  - Optional: hyperparameter tuning     │
│  - Train student model                 │
│  - Evaluate on test set                │
│  - Save artifacts                      │
└────────────────────────────────────────┘
  │
  │ student_model.pkl, evaluation_metrics.json
  │
  ▼
┌────────────────────────────────────────┐
│  Validation (--validate flag)          │
│  - Run backtest with new model         │
│  - Capture telemetry                   │
│  - Compute accuracy metrics            │
│  - Compare with baseline               │
│  - Generate promotion checklist        │
└────────────────────────────────────────┘
  │
  │ promotion_checklist.md
  │
  ▼
┌────────────────────────────────────────┐
│  Human Review & Approval               │
│  - Review promotion checklist          │
│  - Verify validation criteria          │
│  - Approve/reject deployment           │
│  - Manual model promotion              │
└────────────────────────────────────────┘
```

### Component Interactions

```
┌─────────────────┐
│  Backtester     │──┐
└─────────────────┘  │
                     │ Telemetry
┌─────────────────┐  │
│  TeacherLabeler │◄─┘
└─────────────────┘
        │
        │ Labels
        ▼
┌─────────────────┐
│  Teacher Model  │
│  Training       │
└─────────────────┘
        │
        │ Labels + Features
        ▼
┌─────────────────┐
│  Student Model  │
│  Training       │
└─────────────────┘
        │
        │ Model
        ▼
┌─────────────────┐
│  Validation     │◄─ Backtester
│  (Backtest)     │
└─────────────────┘
        │
        │ Metrics
        ▼
┌─────────────────┐
│  Promotion      │
│  Checklist      │
└─────────────────┘
```

## Implementation Plan

### Phase 1: Teacher Training Pipeline (Day 1)
- [ ] Create `scripts/train_teacher.py` CLI
- [ ] Implement backtest integration with telemetry
- [ ] Implement teacher label generation
- [ ] Implement feature engineering
- [ ] Implement teacher model training
- [ ] Implement artifact saving and metadata generation
- [ ] Create sample teacher artifacts

### Phase 2: Student Training Pipeline (Day 2)
- [ ] Create `scripts/train_student.py` CLI
- [ ] Implement teacher artifact loading
- [ ] Implement train/test split
- [ ] Implement hyperparameter tuning (optional)
- [ ] Implement student model training
- [ ] Implement evaluation metrics computation
- [ ] Create sample student artifacts

### Phase 3: Post-Training Validation (Day 3)
- [ ] Implement `--validate` flag in train_student.py
- [ ] Integrate backtest for validation
- [ ] Implement accuracy metrics computation from telemetry
- [ ] Implement baseline comparison
- [ ] Implement promotion checklist generation
- [ ] Generate sample validation artifacts

### Phase 4: Integration Testing (Day 4)
- [ ] Create `tests/integration/test_model_training.py`
- [ ] Implement teacher training test
- [ ] Implement student training test
- [ ] Implement full workflow test
- [ ] Implement incremental training test
- [ ] Verify all artifacts and metadata

### Phase 5: Documentation & Quality Gates (Day 5)
- [ ] Update `docs/architecture.md` with Section 16
- [ ] Document training workflow prerequisites
- [ ] Document evaluation criteria
- [ ] Document rollback procedures
- [ ] Run quality gates (ruff, mypy, pytest)

## Success Metrics

- ✅ Teacher training pipeline generates all required artifacts
- ✅ Student training pipeline consumes teacher labels successfully
- ✅ Post-training validation runs and generates promotion checklist
- ✅ Artifacts properly versioned under `data/models/<timestamp>/`
- ✅ Metadata includes config hash and optimization run linkage
- ✅ Integration tests pass for full workflow
- ✅ All quality gates pass (ruff, mypy, pytest)

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Teacher labels noisy | High | Medium | Apply filtering thresholds, require min holding return |
| Student overfits training data | High | High | Cross-validation, out-of-sample validation, calibration checks |
| Validation period unrepresentative | Critical | Medium | Use recent data, multiple validation windows, walk-forward |
| Model promotion without adequate testing | Critical | Low | Require human approval, multi-stage checklist, rollback procedure |
| Artifact versioning inconsistent | Medium | Low | Automated timestamping, metadata validation in tests |

## References

- US-008: Teacher Labeler (foundation for label generation)
- US-009: Student Inference (foundation for student model)
- US-019: Strategy Accuracy Optimization (parameter tuning context)
- Architecture Doc: Section 16 - Model Training Automation

---

**Document History**:
- 2025-10-12: Initial draft created
- Last Updated: 2025-10-12
