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
