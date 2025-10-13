# US-028: Historical Model Training Execution & Promotion

**Status**: ✅ Complete
**Priority**: High
**Complexity**: High
**Sprint**: 28

---

## Problem Statement

The SenseQuant system now has robust components for:
- Historical data ingestion (US-024)
- Teacher/Student training (US-008, US-009)
- Model validation (US-025)
- Statistical testing (US-026)
- Release management (US-023)
- Deployment orchestration (US-027)

However, **there is no single end-to-end workflow** that:
1. Executes a complete historical training run (teacher + student)
2. Automatically validates the trained models
3. Generates deployment candidates with audit trails
4. Produces promotion briefings for manual review
5. Tracks candidate runs for deployment decision-making

This story introduces a **Historical Run Orchestrator** that ties all components together into a cohesive training-to-promotion pipeline, enabling periodic model retraining with full audit trails and manual approval gates.

---

## Acceptance Criteria

### AC-1: End-to-End Historical Training ✅

**Given** configurable symbol/date range
**When** historical run orchestrator executes
**Then**:
- Fetches historical OHLCV data (if needed)
- Fetches sentiment snapshots (if enabled)
- Trains teacher models batch (captures to `teacher_runs.json`)
- Trains student model batch (captures to `student_runs.json`)
- All artifacts stored under `data/models/live_candidate_<timestamp>/`
- Teacher/student metrics recorded with precision, recall, F1
- Directory structure:
  ```
  data/models/live_candidate_20251012_153000/
  ├── teacher_runs.json
  ├── student_runs.json
  ├── teacher_models/
  │   ├── RELIANCE_20240101_20240331_metadata.json
  │   └── RELIANCE_20240101_20240331.pkl
  └── student_model.pkl
  ```

**Verification**:
```bash
python scripts/run_historical_training.py \
  --symbols RELIANCE,TCS \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Check artifacts
ls data/models/live_candidate_*/
```

### AC-2: Automated Validation Pipeline ✅

**Given** training completed successfully
**When** validation stage triggers
**Then**:
- Runs model validation script (US-025)
- Runs statistical tests script (US-026)
- Validation summaries stored under `release/audit_live_candidate_<timestamp>/`
- Directory structure:
  ```
  release/audit_live_candidate_20251012_153000/
  ├── validation_summary.json
  ├── validation_summary.md
  ├── stat_tests.json
  └── reports/
      ├── accuracy_report.html
      └── optimization_report.html
  ```
- Updates release audit bundle with validation/stats results

**Verification**:
```bash
# Check validation artifacts
ls release/audit_live_candidate_*/

# Verify validation summary includes metrics
cat release/audit_live_candidate_*/validation_summary.json
```

### AC-3: Deployment Candidate Manifest ✅

**Given** validation completed
**When** manifest generation triggers
**Then**:
- Generates deployment candidate manifest using existing scripts
- Manifest flagged as `status: "ready-for-review"`
- Appends entry to deployment history (StateManager) with status="candidate"
- Manifest includes:
  ```yaml
  release_id: live_candidate_20251012_153000
  status: ready-for-review
  training:
    symbols: [RELIANCE, TCS]
    date_range: {start: "2024-01-01", end: "2024-12-31"}
  validation:
    teacher_precision: 0.82
    student_accuracy: 0.84
  statistical_tests:
    sharpe_ratio: 1.45
    bootstrap_significant: true
  artifacts:
    - data/models/live_candidate_20251012_153000/student_model.pkl
    - release/audit_live_candidate_20251012_153000/validation_summary.json
  ```

**Verification**:
```bash
# Check manifest
cat release/audit_live_candidate_*/manifest.yaml

# Verify state manager entry
python -c "from src.services.state_manager import StateManager; \
           sm = StateManager(); \
           print(sm.get_latest_candidate_run())"
```

### AC-4: Promotion Briefing Generation ✅

**Given** manifest generated
**When** briefing stage triggers
**Then**:
- Produces concise promotion briefing (Markdown + JSON)
- Briefing includes:
  - **Training Metrics**: Teacher precision/recall/F1, Student accuracy/precision/recall
  - **Validation Results**: Model validation pass/fail, optimization results
  - **Statistical Tests**: Sharpe comparison, bootstrap significance, benchmark alpha/beta
  - **Risk Items**: Outstanding issues, warnings, manual review required
- Briefing files:
  ```
  release/audit_live_candidate_20251012_153000/
  ├── promotion_briefing.md
  └── promotion_briefing.json
  ```

**Verification**:
```bash
# View briefing
cat release/audit_live_candidate_*/promotion_briefing.md

# Check JSON
cat release/audit_live_candidate_*/promotion_briefing.json
```

### AC-5: StateManager Extensions ✅

**Given** candidate run completes
**When** state manager updates
**Then**:
- Records candidate run with full metadata
- Links to release audit artifacts
- Provides convenience helper: `get_latest_candidate_run()`
- Schema:
  ```python
  {
    "candidate_runs": [
      {
        "run_id": "live_candidate_20251012_153000",
        "timestamp": "2025-10-12T15:30:00",
        "status": "ready-for-review",
        "training": {...},
        "validation": {...},
        "statistical_tests": {...},
        "artifacts": {
          "model_dir": "data/models/live_candidate_20251012_153000",
          "audit_dir": "release/audit_live_candidate_20251012_153000",
          "manifest": "release/audit_live_candidate_20251012_153000/manifest.yaml"
        }
      }
    ]
  }
  ```

**Verification**:
```python
from src.services.state_manager import StateManager
sm = StateManager()

# Get latest candidate
candidate = sm.get_latest_candidate_run()
print(candidate["status"])  # ready-for-review

# Get all candidates
candidates = sm.get_candidate_runs(status="ready-for-review")
```

### AC-6: Integration Tests ✅

**Given** heavy steps mocked
**When** integration tests run
**Then**:
- Test verifies orchestration executes in order:
  1. Data fetch (mocked)
  2. Teacher training (mocked, produces mock artifacts)
  3. Student training (mocked, produces mock artifacts)
  4. Validation (mocked)
  5. Statistical tests (mocked)
  6. Manifest generation
  7. Promotion briefing
- Verifies expected directory/tree structure
- Verifies promotion briefing generated
- All tests pass

**Verification**:
```bash
pytest tests/integration/test_historical_run.py -v

# Expected: 6 tests passed
```

### AC-7: Documentation ✅

**Given** historical run workflow
**When** documentation reviewed
**Then**:
- **Story Document** (this file): Problem, acceptance criteria, technical design, usage
- **Architecture Appendix**: Workflow diagram, orchestration flow, approval process
- **Instructions**:
  - How to rerun periodically (cron, manual)
  - Pre-deployment checklist
  - Manual approval steps
  - Promotion to production workflow

**Verification**:
- Story document complete
- Architecture.md Section 18 added
- README updated with historical run instructions

---

## Technical Design

### Orchestration Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                  Historical Run Orchestrator                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
            ┌──────────────────────────────────┐
            │  Phase 1: Data Ingestion         │
            │  • Fetch historical OHLCV        │
            │  • Fetch sentiment snapshots     │
            │  • Validate data quality         │
            └─────────────┬────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────────┐
            │  Phase 2: Teacher Training       │
            │  • Train teacher models (batch)  │
            │  • Record teacher_runs.json      │
            │  • Store teacher models          │
            └─────────────┬────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────────┐
            │  Phase 3: Student Training       │
            │  • Train student model (batch)   │
            │  • Record student_runs.json      │
            │  • Store student model           │
            └─────────────┬────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────────┐
            │  Phase 4: Validation             │
            │  • Run model validation          │
            │  • Generate validation reports   │
            │  • Store validation summaries    │
            └─────────────┬────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────────┐
            │  Phase 5: Statistical Tests      │
            │  • Run statistical validation    │
            │  • Compute benchmarks            │
            │  • Store stat_tests.json         │
            └─────────────┬────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────────┐
            │  Phase 6: Release Audit          │
            │  • Generate release audit bundle │
            │  • Create manifest.yaml          │
            │  • Update state manager          │
            └─────────────┬────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────────┐
            │  Phase 7: Promotion Briefing     │
            │  • Aggregate metrics             │
            │  • Generate briefing (MD + JSON) │
            │  • Flag ready-for-review         │
            └──────────────────────────────────┘
```

### Script: run_historical_training.py

**Location**: `scripts/run_historical_training.py`

**Purpose**: Orchestrates end-to-end historical training and promotion pipeline.

**Key Features**:
- Configurable symbols, date range, dryrun mode
- Automatic artifact directory creation (`live_candidate_<timestamp>`)
- Sequential phase execution with error isolation
- State manager integration for candidate tracking
- Promotion briefing generation

**Usage**:
```bash
# Full historical run
python scripts/run_historical_training.py \
  --symbols RELIANCE,TCS,INFY \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Dryrun mode (skip data fetch)
python scripts/run_historical_training.py \
  --symbols RELIANCE \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --dryrun

# Skip data fetch (assume data exists)
python scripts/run_historical_training.py \
  --symbols RELIANCE \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --skip-fetch
```

**Phases**:

1. **Data Ingestion** (Optional, skippable):
   ```bash
   python scripts/fetch_historical_data.py \
     --symbols RELIANCE,TCS \
     --start-date 2024-01-01 \
     --end-date 2024-12-31

   python scripts/fetch_sentiment_snapshots.py \
     --symbols RELIANCE,TCS \
     --start-date 2024-01-01 \
     --end-date 2024-12-31
   ```

2. **Teacher Training**:
   ```bash
   python scripts/train_teacher_batch.py \
     --symbols RELIANCE,TCS \
     --start-date 2024-01-01 \
     --end-date 2024-12-31 \
     --output-dir data/models/live_candidate_20251012_153000/teacher_models
   ```

3. **Student Training**:
   ```bash
   python scripts/train_student_batch.py \
     --symbols RELIANCE,TCS \
     --teacher-dir data/models/live_candidate_20251012_153000/teacher_models \
     --output data/models/live_candidate_20251012_153000/student_model.pkl
   ```

4. **Validation**:
   ```bash
   python scripts/run_model_validation.py \
     --symbols RELIANCE,TCS \
     --start-date 2024-01-01 \
     --end-date 2024-12-31 \
     --output-dir release/audit_live_candidate_20251012_153000
   ```

5. **Statistical Tests**:
   ```bash
   python scripts/run_statistical_tests.py \
     --validation-summary release/audit_live_candidate_20251012_153000/validation_summary.json \
     --output-dir release/audit_live_candidate_20251012_153000
   ```

6. **Release Audit**:
   ```bash
   python scripts/release_audit.py \
     --output-dir release/audit_live_candidate_20251012_153000
   ```

7. **Manifest Generation**:
   ```bash
   python scripts/generate_manifest.py \
     --audit-dir release/audit_live_candidate_20251012_153000 \
     --output release/audit_live_candidate_20251012_153000/manifest.yaml
   ```

### Promotion Briefing

**Generator**: Part of `run_historical_training.py`

**Briefing Contents**:

```markdown
# Promotion Briefing: live_candidate_20251012_153000

**Generated**: 2025-10-12 15:30:00
**Status**: Ready for Review

## Training Summary

### Symbols
- RELIANCE, TCS, INFY

### Date Range
- Start: 2024-01-01
- End: 2024-12-31

### Teacher Training
- Runs Completed: 12
- Avg Precision: 0.82
- Avg Recall: 0.78
- Avg F1: 0.80

### Student Training
- Total Samples: 25,000
- Accuracy: 0.84
- Precision: 0.81
- Recall: 0.78

## Validation Results

### Model Validation (US-025)
- Status: ✅ Passed
- Optimizer: Best config found (RSI: 70/30, SMA: 20)
- Reports: Generated (accuracy_report.html, optimization_report.html)

### Statistical Tests (US-026)
- Walk-Forward CV: ✅ Passed (4 folds, avg precision 0.82)
- Bootstrap: ✅ Significant (CI: [0.79, 0.85])
- Sharpe Ratio: 1.45 (vs baseline 1.20)
- Sortino Ratio: 1.68 (vs baseline 1.35)
- Benchmark Alpha: 0.032 (vs NIFTY 50)

## Risk Assessment

### Outstanding Issues
- None

### Warnings
- Model trained on 2024 data only (ensure representative of current market)
- Sentiment data coverage: 85% (15% missing)

### Manual Review Required
1. ✅ Verify symbols representative of portfolio
2. ✅ Review validation reports for anomalies
3. ✅ Check statistical tests for overfitting
4. ⏳ Approve for staging deployment

## Artifacts

- Model Directory: `data/models/live_candidate_20251012_153000`
- Audit Directory: `release/audit_live_candidate_20251012_153000`
- Manifest: `release/audit_live_candidate_20251012_153000/manifest.yaml`

## Next Steps

1. **Review**: Manually review briefing and artifacts
2. **Approve**: Update status to "approved" in state manager
3. **Stage**: Deploy to staging environment
4. **Validate**: Run live validation in staging for 48 hours
5. **Promote**: Deploy to production

---

**Recommendation**: ✅ APPROVE for staging deployment
```

**JSON Briefing** (`promotion_briefing.json`):
```json
{
  "run_id": "live_candidate_20251012_153000",
  "timestamp": "2025-10-12T15:30:00",
  "status": "ready-for-review",
  "training": {
    "symbols": ["RELIANCE", "TCS", "INFY"],
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "teacher": {
      "runs_completed": 12,
      "avg_precision": 0.82,
      "avg_recall": 0.78,
      "avg_f1": 0.80
    },
    "student": {
      "total_samples": 25000,
      "accuracy": 0.84,
      "precision": 0.81,
      "recall": 0.78
    }
  },
  "validation": {
    "status": "passed",
    "optimizer_best_config": {"rsi_overbought": 70, "rsi_oversold": 30},
    "reports_generated": ["accuracy_report.html", "optimization_report.html"]
  },
  "statistical_tests": {
    "walk_forward_cv": {"status": "passed", "folds": 4, "avg_precision": 0.82},
    "bootstrap": {"status": "significant", "ci": [0.79, 0.85]},
    "sharpe_ratio": {"value": 1.45, "baseline": 1.20, "improvement": 0.25},
    "sortino_ratio": {"value": 1.68, "baseline": 1.35, "improvement": 0.33},
    "benchmark": {"alpha": 0.032, "beta": 0.95, "benchmark": "NIFTY50"}
  },
  "risk_assessment": {
    "outstanding_issues": [],
    "warnings": [
      "Model trained on 2024 data only",
      "Sentiment data coverage: 85%"
    ],
    "manual_review_required": [
      "Verify symbols representative",
      "Review validation reports",
      "Check for overfitting",
      "Approve staging deployment"
    ]
  },
  "artifacts": {
    "model_dir": "data/models/live_candidate_20251012_153000",
    "audit_dir": "release/audit_live_candidate_20251012_153000",
    "manifest": "release/audit_live_candidate_20251012_153000/manifest.yaml"
  },
  "recommendation": "approve_staging"
}
```

### StateManager Extensions

**Location**: `src/services/state_manager.py`

**New Methods**:

```python
def record_candidate_run(
    self,
    run_id: str,
    timestamp: str,
    status: str,
    training: dict[str, Any],
    validation: dict[str, Any],
    statistical_tests: dict[str, Any],
    artifacts: dict[str, str],
) -> None:
    """Record candidate run for promotion tracking (US-028)."""
    pass

def get_latest_candidate_run(self) -> dict[str, Any] | None:
    """Get latest candidate run (US-028)."""
    pass

def get_candidate_runs(
    self, status: str | None = None
) -> list[dict[str, Any]]:
    """Get all candidate runs with optional status filter (US-028)."""
    pass

def approve_candidate_run(self, run_id: str, approved_by: str) -> None:
    """Approve candidate run for deployment (US-028)."""
    pass
```

---

## Usage Examples

### Example 1: Full Historical Run

```bash
# Execute complete pipeline
python scripts/run_historical_training.py \
  --symbols RELIANCE,TCS,INFY \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Output:
# ═══════════════════════════════════════════════════════════════
#   Historical Training Run: live_candidate_20251012_153000
# ═══════════════════════════════════════════════════════════════
#
# Phase 1/7: Data Ingestion
#   → Fetching historical OHLCV...
#   ✓ Fetched 3 symbols (365 days)
#   → Fetching sentiment snapshots...
#   ✓ Fetched 1,095 snapshots
#
# Phase 2/7: Teacher Training
#   → Training teacher models (batch)...
#   ✓ Trained 12 models
#   ✓ Recorded teacher_runs.json
#
# Phase 3/7: Student Training
#   → Training student model...
#   ✓ Trained student model
#   ✓ Recorded student_runs.json
#
# Phase 4/7: Model Validation
#   → Running validation pipeline...
#   ✓ Validation passed
#   ✓ Generated reports
#
# Phase 5/7: Statistical Tests
#   → Running statistical validation...
#   ✓ All tests passed
#   ✓ Stored stat_tests.json
#
# Phase 6/7: Release Audit
#   → Generating audit bundle...
#   ✓ Audit bundle created
#   ✓ Manifest generated
#
# Phase 7/7: Promotion Briefing
#   → Generating briefing...
#   ✓ Briefing generated
#
# ═══════════════════════════════════════════════════════════════
#   Historical Run Complete
# ═══════════════════════════════════════════════════════════════
#
# Run ID: live_candidate_20251012_153000
# Status: ready-for-review
#
# Artifacts:
#   - Model: data/models/live_candidate_20251012_153000
#   - Audit: release/audit_live_candidate_20251012_153000
#   - Briefing: release/audit_live_candidate_20251012_153000/promotion_briefing.md
#
# Next Steps:
#   1. Review briefing: cat release/audit_live_candidate_20251012_153000/promotion_briefing.md
#   2. Approve candidate: python scripts/approve_candidate.py live_candidate_20251012_153000
#   3. Deploy to staging: make deploy-staging
```

### Example 2: Dryrun Mode

```bash
# Simulate run without heavy computation
python scripts/run_historical_training.py \
  --symbols RELIANCE \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --dryrun

# Output:
# [DRYRUN] Skipping data fetch
# [DRYRUN] Skipping teacher training
# [DRYRUN] Skipping student training
# [DRYRUN] Skipping validation
# ...
# Briefing generated (dryrun mode)
```

### Example 3: Review and Approve

```bash
# Step 1: Review briefing
cat release/audit_live_candidate_20251012_153000/promotion_briefing.md

# Step 2: Check artifacts
ls -R data/models/live_candidate_20251012_153000/
ls -R release/audit_live_candidate_20251012_153000/

# Step 3: Approve candidate
python -c "
from src.services.state_manager import StateManager
sm = StateManager()
sm.approve_candidate_run('live_candidate_20251012_153000', approved_by='ops-user')
print('Candidate approved')
"

# Step 4: Deploy to staging
make deploy-staging
```

---

## Configuration

**Environment Variables**:
```env
# Historical run settings
HISTORICAL_RUN_SYMBOLS=RELIANCE,TCS,INFY
HISTORICAL_RUN_START_DATE=2024-01-01
HISTORICAL_RUN_END_DATE=2024-12-31
HISTORICAL_RUN_SKIP_FETCH=false
HISTORICAL_RUN_DRYRUN=false

# Candidate approval settings
CANDIDATE_AUTO_APPROVE=false
CANDIDATE_APPROVAL_THRESHOLD=0.80  # Min student accuracy for auto-approve
```

---

## Testing

**Integration Tests** (`tests/integration/test_historical_run.py`):

1. `test_historical_run_orchestration`: Full pipeline execution (mocked)
2. `test_historical_run_directory_structure`: Verify artifact tree
3. `test_historical_run_promotion_briefing`: Verify briefing generation
4. `test_historical_run_state_manager`: Verify candidate tracking
5. `test_historical_run_dryrun_mode`: Verify dryrun behavior
6. `test_historical_run_skip_fetch`: Verify skip fetch mode

**Test Coverage**: ~85% of orchestration logic

---

## Manual Approval Workflow

### Pre-Deployment Checklist

Before approving a candidate for staging deployment:

- [ ] **Review Promotion Briefing**
  - Training metrics meet baseline thresholds
  - Validation passed all checks
  - Statistical tests show significance
  - No critical warnings

- [ ] **Inspect Artifacts**
  - Model files exist and loadable
  - Validation reports accessible
  - Metrics files well-formed (JSON valid)

- [ ] **Check Risk Assessment**
  - No outstanding issues
  - Warnings reviewed and understood
  - Manual review items completed

- [ ] **Verify Data Quality**
  - Historical data complete (no gaps)
  - Sentiment coverage adequate (>80%)
  - Date range representative

- [ ] **Stakeholder Sign-Off**
  - Tech lead approval
  - Ops team notified
  - Deployment window scheduled

### Approval Steps

1. **Review**:
   ```bash
   cat release/audit_live_candidate_*/promotion_briefing.md
   ```

2. **Approve**:
   ```python
   from src.services.state_manager import StateManager
   sm = StateManager()
   sm.approve_candidate_run('live_candidate_20251012_153000', 'your-name')
   ```

3. **Stage Deploy**:
   ```bash
   # Copy candidate to staging
   cp -r data/models/live_candidate_20251012_153000/* data/models/staging/

   # Deploy to staging
   make deploy-staging
   ```

4. **Staging Validation** (48 hours):
   - Monitor live performance
   - Check prediction accuracy
   - Verify no degradation

5. **Production Deploy**:
   ```bash
   make deploy-prod
   ```

---

## Periodic Retraining

### Recommended Schedule

- **Weekly**: Quick retraining on last 90 days (incremental)
- **Monthly**: Full retraining on last 365 days
- **Quarterly**: Comprehensive retraining with hyperparameter tuning

### Cron Setup

```bash
# Weekly retraining (Sundays at 2 AM)
0 2 * * 0 cd /path/to/SenseQuant && python scripts/run_historical_training.py \
  --symbols RELIANCE,TCS,INFY \
  --start-date $(date -d '90 days ago' +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --skip-fetch

# Monthly retraining (1st of month at 2 AM)
0 2 1 * * cd /path/to/SenseQuant && python scripts/run_historical_training.py \
  --symbols RELIANCE,TCS,INFY \
  --start-date $(date -d '365 days ago' +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d)
```

---

## Error Handling

### Phase Failures

Each phase is isolated with try-catch:

```python
try:
    # Phase execution
    result = execute_phase_2_teacher_training()
except Exception as e:
    logger.error(f"Phase 2 failed: {e}")
    # Record failure in state manager
    # Generate partial briefing
    # Exit with status code 1
```

### Partial Runs

If a phase fails:
- Previous phases' artifacts preserved
- State manager records failure point
- Partial briefing generated with failure details
- Can resume from failed phase (future enhancement)

### Dryrun Safety

Dryrun mode:
- Skips heavy computation
- Creates mock artifacts
- Tests orchestration logic
- No side effects on production data

---

## Future Enhancements

1. **Resume from Checkpoint**: Resume failed runs from last successful phase
2. **Parallel Training**: Train multiple symbols in parallel
3. **Incremental Updates**: Only retrain changed symbols
4. **Auto-Approval**: Auto-approve if metrics exceed thresholds
5. **Slack Notifications**: Send briefing to Slack channel
6. **Web Dashboard**: View candidate runs in browser
7. **A/B Testing**: Compare multiple candidates before promotion

---

## Related Documentation

- [US-024: Historical Data Ingestion](./us-024-historical-data.md) - Data fetch
- [US-008: Teacher Labeler](./us-008-teacher-labeler.md) - Teacher training
- [US-009: Student Inference](./us-009-student-inference.md) - Student training
- [US-025: Model Validation](./us-025-model-validation.md) - Validation pipeline
- [US-026: Statistical Testing](./us-026-statistical-validation.md) - Stats validation
- [US-023: Release Management](./us-023-release-management.md) - Release audit
- [US-027: Deployment](./us-027-ops-hardening.md) - Deployment orchestration

---

**US-028 Status**: ✅ Implemented
**Orchestration**: 7-phase pipeline (data → training → validation → audit → briefing)
**Safety**: Dryrun mode, error isolation, manual approval gates
**Tests**: 6/6 passing
