# US-025: Historical Backtest & Model Validation Run

## Problem Statement

Before promoting new models to live trading, we need a comprehensive validation workflow that:
1. Runs full teacher & student batch training on historical data
2. Evaluates strategy configurations using the optimizer
3. Generates accuracy and optimization reports
4. Produces consolidated validation summaries with promotion recommendations
5. Tracks validation runs for audit trail

This workflow ensures models meet quality gates before live deployment.

## Acceptance Criteria

### AC-1: Full Batch Training Execution
- [ ] Execute teacher batch training on configured historical window
- [ ] Execute student batch training from teacher outputs
- [ ] Store artifacts under `data/models/<run_id>/`
- [ ] Populate `teacher_runs.json` and `student_runs.json` with metadata
- [ ] Support dryrun mode for testing without real data

### AC-2: Optimizer Validation Mode
- [ ] Run optimizer in read-only mode (no config changes)
- [ ] Evaluate current strategy configurations
- [ ] Store optimization results in `data/optimization/<run_id>/`
- [ ] Include parameter sweep results and best configurations
- [ ] Generate optimization metrics summary

### AC-3: Report Generation
- [ ] Execute accuracy_report.ipynb using nbconvert
- [ ] Execute optimization_report.ipynb using nbconvert
- [ ] Export reports as HTML to `release/audit_<run_id>/reports/`
- [ ] Include all visualizations and metrics tables
- [ ] Handle notebook execution errors gracefully

### AC-4: Validation Summary
- [ ] Generate consolidated validation summary (Markdown + JSON)
- [ ] Include key accuracy metrics (precision, recall, F1)
- [ ] List best strategy configurations from optimizer
- [ ] Provide student promotion recommendations
- [ ] Highlight outstanding risks and required actions
- [ ] Store summary in `release/audit_<run_id>/validation_summary.md`

### AC-5: State Management
- [ ] Record validation run metadata in StateManager
- [ ] Track timestamp, symbols, metrics, and outcomes
- [ ] Enable querying of past validation runs
- [ ] Support rerun capability with same configuration

### AC-6: Documentation
- [ ] Create US-025 story document
- [ ] Document validation workflow in architecture.md
- [ ] Include operational playbook for validation runs
- [ ] Document next steps for live promotion
- [ ] Ensure defaults prevent accidental live promotion

### AC-7: Integration Testing
- [ ] Test end-to-end validation workflow
- [ ] Verify teacher/student batch execution
- [ ] Confirm optimizer artifacts generated
- [ ] Validate report export and summary creation
- [ ] Ensure dryrun mode works correctly

## Technical Design

### Validation Workflow

```
1. Data Preparation
   ├─> Verify historical data exists
   ├─> Check sentiment snapshots available
   └─> Validate date ranges

2. Batch Training
   ├─> Teacher batch training
   │   ├─> Generate labels for each symbol/window
   │   ├─> Store teacher artifacts
   │   └─> Record teacher_runs.json
   └─> Student batch training
       ├─> Train student models from teacher labels
       ├─> Generate promotion checklists
       └─> Record student_runs.json

3. Optimizer Evaluation
   ├─> Load latest telemetry (if available)
   ├─> Run parameter sweep (read-only)
   ├─> Identify best configurations
   └─> Store optimization results

4. Report Generation
   ├─> Execute accuracy_report.ipynb
   ├─> Execute optimization_report.ipynb
   ├─> Export HTML reports
   └─> Copy to release/audit_<run_id>/reports/

5. Summary Generation
   ├─> Aggregate accuracy metrics
   ├─> Extract best optimizer configs
   ├─> Generate promotion recommendations
   ├─> Identify risks and blockers
   └─> Create validation_summary.md + .json

6. State Recording
   ├─> Record validation run metadata
   ├─> Store in StateManager
   └─> Enable audit trail queries
```

### Directory Structure

```
data/
├── models/
│   └── <run_id>/
│       ├── teacher_runs.json
│       ├── student_runs.json
│       └── <symbol>_<window>/
│           ├── labels.csv
│           ├── features.csv
│           └── model.pkl
├── optimization/
│   └── <run_id>/
│       ├── parameter_sweep.json
│       ├── best_configs.json
│       └── optimization_metrics.csv
└── state/
    └── validation_runs.json

release/
└── audit_<run_id>/
    ├── validation_summary.md
    ├── validation_summary.json
    └── reports/
        ├── accuracy_report.html
        └── optimization_report.html
```

### Validation Summary Schema

**validation_summary.json:**
```json
{
  "run_id": "validation_20251012_180000",
  "timestamp": "2025-10-12T18:00:00+05:30",
  "status": "completed",
  "symbols": ["RELIANCE", "TCS", "INFY"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-12-31"
  },
  "accuracy_metrics": {
    "RELIANCE": {
      "teacher_precision": 0.82,
      "teacher_recall": 0.76,
      "teacher_f1": 0.79,
      "student_accuracy": 0.84,
      "student_precision": 0.81,
      "student_recall": 0.78
    }
  },
  "best_configs": {
    "RELIANCE": {
      "rsi_period": 14,
      "bb_period": 20,
      "profit_target_pct": 2.5
    }
  },
  "promotion_recommendations": [
    {
      "symbol": "RELIANCE",
      "student_model": "data/models/validation_20251012_180000/RELIANCE_2024Q1/student_model.pkl",
      "promotion_status": "approved",
      "reason": "Meets all quality gates"
    }
  ],
  "risks": [
    "TCS student model underperforming on recent data",
    "Backtest period does not include market crash scenario"
  ],
  "next_steps": [
    "Review promotion recommendations",
    "Run additional validation on recent data",
    "Approve student model promotion"
  ]
}
```

## Configuration

```python
# US-025: Model Validation Configuration
model_validation_enabled: bool = False  # Disabled by default
model_validation_dryrun: bool = True  # Always dryrun unless explicitly disabled
model_validation_symbols: list[str] = ["RELIANCE", "TCS", "INFY"]
model_validation_start_date: str = "2024-01-01"
model_validation_end_date: str = "2024-12-31"
model_validation_export_reports: bool = True
model_validation_optimizer_enabled: bool = True
```

## Usage

### Run Full Validation

```bash
# Full validation run (dryrun mode by default)
python scripts/run_model_validation.py

# Real validation with actual data
python scripts/run_model_validation.py --no-dryrun

# Specify symbols and date range
python scripts/run_model_validation.py \
  --symbols RELIANCE TCS \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Skip optimizer (faster)
python scripts/run_model_validation.py --skip-optimizer

# Skip report generation
python scripts/run_model_validation.py --skip-reports
```

### Query Past Validation Runs

```python
from pathlib import Path
from src.services.state_manager import StateManager

state_manager = StateManager(Path("data/state/validation_runs.json"))

# Get all validation runs
runs = state_manager.get_validation_runs()

# Get specific run
run = state_manager.get_validation_run("validation_20251012_180000")
```

## Operational Playbook

### Pre-Validation Checklist
- [ ] Verify historical data fetched for validation period
- [ ] Confirm sentiment snapshots available
- [ ] Check sufficient disk space for artifacts
- [ ] Review validation configuration (symbols, dates)
- [ ] Ensure dryrun mode enabled for testing

### Validation Execution
1. Start validation run
2. Monitor batch training progress
3. Review optimizer results
4. Check report generation
5. Inspect validation summary

### Post-Validation Actions
1. Review accuracy metrics
2. Evaluate promotion recommendations
3. Identify and address risks
4. Approve or reject model promotion
5. Document decision in audit trail

### Promotion Approval Criteria
- Student accuracy >= 80%
- Student precision >= 75%
- Student recall >= 70%
- No critical risks identified
- Backtest covers representative market conditions

## Related User Stories

- US-020: Teacher Training Automation
- US-021: Student Model Promotion
- US-022: Release Audit Workflow
- US-024: Historical Data Ingestion

## Next Steps

After validation completes:
1. Review validation summary
2. Check promotion recommendations
3. Run additional validation if needed
4. Approve student model promotion (US-021)
5. Deploy to live trading (US-023)

## Acceptance Sign-Off

- [ ] Engineering Lead: Code review passed, quality gates green
- [ ] Data Scientist: Accuracy metrics validated, models approved
- [ ] QA: Integration tests pass, manual validation complete
- [ ] Product Owner: Acceptance criteria verified

---

## Phase 2: Optimizer & Report Integration (COMPLETED)

### Overview

Phase 2 completes the validation workflow by integrating optimizer evaluation and notebook report generation, providing comprehensive validation summaries with actionable recommendations.

### Implementation Status

#### ✅ Optimizer Integration (AC-2)
- **Implemented**: Full optimizer execution in read-only mode
- **Location**: `scripts/run_model_validation.py::_run_optimizer()`
- **Features**:
  - Executes `scripts/optimize.py` with validation context
  - Creates default search space if missing
  - Captures best configurations and metrics
  - Stores results in `data/optimization/<run_id>/`
  - Handles timeouts and errors gracefully
  - Skips in dryrun mode with placeholder results

**Usage**:
```bash
# Run with optimizer
python scripts/run_model_validation.py --no-dryrun

# Skip optimizer
python scripts/run_model_validation.py --skip-optimizer
```

#### ✅ Report Generation (AC-3)
- **Implemented**: Notebook export via nbconvert
- **Location**: `scripts/run_model_validation.py::_generate_reports()`
- **Features**:
  - Exports `accuracy_report.ipynb` to HTML
  - Exports `optimization_report.ipynb` to HTML
  - Executes notebooks with `--execute --no-input` flags
  - Handles missing notebooks and export failures
  - Stores HTML reports in `release/audit_<run_id>/reports/`
  - Skips in dryrun mode

**Dependencies**:
- `jupyter nbconvert` required for HTML export
- Install: `pip install nbconvert`

#### ✅ Enhanced Validation Summary (AC-4)
- **Implemented**: Comprehensive summaries with optimizer results
- **Location**: `scripts/run_model_validation.py::_generate_summary()`
- **Features**:
  - Loads teacher/student metrics from JSON files
  - Includes optimizer best configurations
  - Calculates accuracy deltas vs baseline
  - Generates promotion recommendations based on thresholds
  - Creates both JSON and Markdown summaries
  - Tracks generated report paths

**Promotion Approval Criteria**:
- Student Accuracy >= 80%
- Student Precision >= 75%
- Real validation (not dryrun)
- No critical errors

**Summary Structure**:
```json
{
  "run_id": "validation_20251012_180000",
  "status": "completed",
  "teacher_results": {
    "status": "success",
    "metrics": {
      "runs_completed": 3,
      "avg_precision": 0.82,
      "avg_recall": 0.78,
      "avg_f1": 0.80
    }
  },
  "student_results": {
    "status": "success",
    "metrics": {
      "runs_completed": 3,
      "avg_accuracy": 0.84,
      "avg_precision": 0.81,
      "avg_recall": 0.78
    }
  },
  "optimizer_results": {
    "status": "success",
    "best_config": {
      "config_id": "config_0001",
      "parameters": {"rsi_overbought": 70, "rsi_oversold": 30},
      "score": 1.45,
      "metrics": {"sharpe_ratio": 1.45, "total_return": 12.5}
    },
    "output_dir": "data/optimization/validation_20251012_180000"
  },
  "reports": [
    "release/audit_validation_20251012_180000/reports/accuracy_report.html",
    "release/audit_validation_20251012_180000/reports/optimization_report.html"
  ],
  "promotion_recommendation": {
    "approved": true,
    "reason": "Accuracy thresholds met (accuracy=84.0%, precision=81.0%)",
    "next_steps": [
      "Review validation summary",
      "Check optimizer best configurations",
      "Verify accuracy metrics meet thresholds",
      "Promote models if approved"
    ]
  }
}
```

#### ✅ Integration Tests (AC-7)
- **Added**: 3 new tests for Phase 2 features
- **Location**: `tests/integration/test_model_validation.py`
- **Tests**:
  1. `test_validation_optimizer_integration`: Verifies optimizer execution and results structure
  2. `test_validation_report_generation`: Tests notebook export workflow
  3. `test_validation_summary_with_metrics`: Validates summary includes teacher/student metrics and promotion recommendations

**Test Coverage**:
- Optimizer status tracking (success/failed/skipped)
- Best config extraction from optimizer artifacts
- Report generation with dryrun handling
- Teacher/student metrics aggregation
- Promotion recommendation logic

### Safety Controls

#### Dryrun Mode Behavior
- **Teacher/Student Training**: Skipped, status = "skipped"
- **Optimizer**: Skipped, status = "skipped", best_configs = {}
- **Reports**: Skipped, reports = []
- **Summary**: Generated with dryrun flag, promotion = false

#### Error Handling
- Optimizer timeout: 2 hours max execution time
- Notebook export errors: Logged as warnings, workflow continues
- Missing files: Graceful degradation with placeholder results
- State recording: Always executed even on failure

### File Changes

#### Modified Files
- `scripts/run_model_validation.py`:
  - `_run_optimizer()`: Complete implementation with subprocess execution
  - `_generate_reports()`: nbconvert integration
  - `_generate_summary()`: Enhanced with optimizer/metrics
  - Added `_load_teacher_metrics()` helper
  - Added `_load_student_metrics()` helper

#### New Tests
- `test_validation_optimizer_integration()`
- `test_validation_report_generation()`
- `test_validation_summary_with_metrics()`

#### Documentation Updated
- `docs/stories/us-025-model-validation.md` (this file)
- `docs/architecture.md` (Section 15: Phase 2 details)

### Usage Examples

#### Full Validation (All Features)
```bash
python scripts/run_model_validation.py \
    --symbols RELIANCE TCS \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --no-dryrun
```

**Output**:
- `data/models/validation_*/` - Teacher/student artifacts
- `data/optimization/validation_*/` - Optimizer results
- `release/audit_validation_*/reports/*.html` - HTML reports
- `release/audit_validation_*/validation_summary.{json,md}` - Summaries

#### Fast Validation (Skip Optimizer & Reports)
```bash
python scripts/run_model_validation.py \
    --symbols RELIANCE \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --skip-optimizer \
    --skip-reports
```

**Use Case**: Quick model accuracy check without full optimization sweep.

#### Dryrun Validation (Test Workflow)
```bash
python scripts/run_model_validation.py \
    --symbols RELIANCE TCS
```

**Behavior**: 
- No external commands executed
- Mock artifacts created
- Summary generated with dryrun flag
- Promotion always rejected in dryrun

### Operational Notes

#### Prerequisites
- Historical data available for date range
- Sentiment snapshots (if using sentiment features)
- `jupyter nbconvert` installed for report generation
- Search space config (auto-created if missing)

#### Monitoring
- Check logs for optimizer/notebook errors
- Verify summary files generated
- Review promotion recommendations
- Validate artifact directories exist

#### Troubleshooting

**Optimizer Fails**:
- Check search space configuration
- Verify historical data quality
- Review optimizer logs in stderr
- Consider `--skip-optimizer` flag

**Report Generation Fails**:
- Install nbconvert: `pip install nbconvert`
- Check notebook paths exist
- Review notebook execution errors
- Consider `--skip-reports` flag

**Summary Missing Metrics**:
- Verify `teacher_runs.json` exists
- Verify `student_runs.json` exists
- Check JSON format (one record per line)
- Ensure metrics keys are present

### Next Steps

1. **Production Deployment**:
   - Schedule weekly validation runs
   - Set up automated validation pipeline
   - Configure alerts for failed validations

2. **Enhanced Reporting**:
   - Add historical comparison reports
   - Include A/B test results
   - Generate performance dashboards

3. **Automation**:
   - Automated model promotion on approval
   - CI/CD integration for validation runs
   - Slack/email notifications

4. **Optimization**:
   - Parallel optimizer execution
   - Caching of intermediate results
   - Incremental validation support

### Related Documentation

- [US-019: Strategy Accuracy Optimization](us-019-optimizer.md) - Optimizer implementation
- [US-020: Teacher Model Training](us-020-teacher-training.md) - Batch training
- [US-021: Student Model Promotion](us-021-student-promotion.md) - Promotion workflow
- [US-024: Historical Data Ingestion](us-024-historical-data.md) - Data preparation
- [Architecture: Section 15](../architecture.md#15-model-validation-workflow-us-025) - Technical details

### Phase 2 Checklist

- [x] Complete optimizer execution in `_run_optimizer()`
- [x] Implement notebook export in `_generate_reports()`
- [x] Update validation summaries with optimizer best configs
- [x] Add teacher/student metrics aggregation
- [x] Implement promotion recommendation logic
- [x] Extend integration tests (3 new tests, 8 total)
- [x] Update US-025 story documentation
- [x] Update architecture.md with Phase 2 details
- [x] Run quality gates (ruff, mypy, pytest)
- [x] Verify dryrun mode safety controls

**Phase 2 Status**: ✅ **COMPLETED**
**Date**: 2025-10-12
**Tests**: 8/8 passing
**Safety**: Dryrun defaults maintained
