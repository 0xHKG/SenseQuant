# US-028 Phase 6o — Align Teacher/Student Artifact Paths

## Implementation Plan

### Root Cause (from Phase 6n Diagnostics)

Teacher batch trainer claims artifacts are in subdirectories (`data/models/20251014_221222/RELIANCE_2023-01-01_to_2023-06-30/`) but actually saves flat files (`data/models/teacher_model_20251014_221225.pkl`), causing 100% student training failure.

---

## Implementation Strategy

### Step 1: Add `--output-dir` to `train_teacher.py`

**File**: `scripts/train_teacher.py`

**Changes**:

1. Add argument parser option:
```python
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for model artifacts (default: data/models/)",
)
```

2. Pass `output_dir` to `TeacherLabeler`:
```python
# Line ~209: Update TeacherLabeler instantiation
teacher = TeacherLabeler(
    config,
    client=client,
    output_dir=Path(args.output_dir) if args.output_dir else None
)
```

3. Update logging to show output directory:
```python
logger.info(f"Output Directory: {args.output_dir or 'data/models/ (default)'}")
```

---

### Step 2: Update `TeacherLabeler` to Accept `output_dir`

**File**: `src/services/teacher_student.py`

**Changes**:

1. Update `TeacherLabeler.__init__` signature:
```python
def __init__(
    self,
    config: TrainingConfig,
    client: BreezeClient | None = None,
    output_dir: Path | None = None,  # NEW parameter
):
    self.config = config
    self.client = client
    self.output_dir = output_dir or Path("data/models")  # Default fallback
```

2. Update file path generation in `run_full_pipeline()` or wherever models/labels are saved:
```python
# Instead of:
model_path = Path("data/models") / f"teacher_model_{timestamp}.pkl"

# Use:
self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
model_path = self.output_dir / "model.pkl"
labels_path = self.output_dir / "labels.csv.gz"
importance_path = self.output_dir / "feature_importance.csv"
metadata_path = self.output_dir / "metadata.json"
```

3. Search for all hardcoded `Path("data/models")` references and replace with `self.output_dir`

**Testing**: Run single teacher training with `--output-dir /tmp/test_teacher` to verify artifacts land in correct location.

---

### Step 3: Update `train_teacher_batch.py` to Create Directories and Pass `--output-dir`

**File**: `scripts/train_teacher_batch.py`

**Changes**:

1. **Update `train_window()` method** (around line 272):

```python
def train_window(
    self,
    task: dict[str, Any],
    forecast_horizon: int,
) -> dict[str, Any]:
    """Train teacher model for a single window."""
    symbol = task["symbol"]
    start_date = task["start_date"]
    end_date = task["end_date"]
    window_label = task["window_label"]

    # US-028 Phase 6o: Create artifacts directory
    artifacts_path = Path(task["artifacts_path"])
    artifacts_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {window_label}: {start_date} to {end_date}")
    logger.info(f"  Artifacts directory: {artifacts_path}")

    cmd = [
        sys.executable,
        "scripts/train_teacher.py",
        "--symbol",
        symbol,
        "--start",
        start_date,
        "--end",
        end_date,
        "--window",
        str(forecast_horizon),
        "--output-dir",           # NEW: Pass output directory
        str(artifacts_path),      # NEW
    ]

    # ... rest of subprocess execution
```

2. **Also update parallel training method** (around line 865) with same changes

3. **Update `is_already_trained()` check** (around line 169):

```python
def is_already_trained(self, task: dict[str, Any]) -> bool:
    """Check if window already trained (US-028 Phase 6o: updated for subdirectories)."""
    artifacts_path = Path(task["artifacts_path"])

    # Check for key artifacts in the subdirectory
    required_files = [
        artifacts_path / "model.pkl",
        artifacts_path / "labels.csv.gz",
        artifacts_path / "metadata.json",
    ]

    return all(f.exists() for f in required_files)
```

**Testing**: Run batch training with 1-2 windows, verify subdirectories created with correct structure.

---

### Step 4: Verify Student Training Works

**No code changes needed** - student training already expects artifacts at `task["artifacts_path"]`. Just need to verify:

1. `train_student_batch.py` reads `teacher_runs.json`
2. Extracts `artifacts_path` from each entry
3. Passes to `train_student.py --teacher-dir {artifacts_path}`
4. Student script finds `labels.csv.gz`, `model.pkl`, `metadata.json` in that directory

**Testing**: After teacher batch completes, run student batch and verify it succeeds.

---

### Step 5: Update Integration Tests

**File**: `tests/integration/test_teacher_pipeline.py`

**New Test**:
```python
def test_teacher_batch_artifacts_in_subdirectories(tmp_path, mock_settings):
    """Test that teacher batch creates subdirectories for artifacts (US-028 Phase 6o)."""
    # Override batch_training_output_dir to tmp_path
    mock_settings.batch_training_output_dir = tmp_path

    # Run teacher batch with 1 window
    from scripts.train_teacher_batch import TeacherBatchTrainer
    trainer = TeacherBatchTrainer(
        symbols=["RELIANCE"],
        start_date="2023-01-01",
        end_date="2023-06-30",
        window_days=180,
        forecast_horizon=7,
        output_dir=tmp_path,
    )

    summary = trainer.run_batch()

    # Verify batch directory exists
    batch_dirs = list(tmp_path.glob("202*"))
    assert len(batch_dirs) == 1
    batch_dir = batch_dirs[0]

    # Verify teacher_runs.json exists
    teacher_runs = batch_dir / "teacher_runs.json"
    assert teacher_runs.exists()

    # Parse first entry
    with open(teacher_runs) as f:
        first_run = json.loads(f.readline())

    # Verify artifacts_path is a subdirectory
    artifacts_path = Path(first_run["artifacts_path"])
    assert artifacts_path.parent == batch_dir
    assert artifacts_path.name.startswith("RELIANCE_")

    # Verify artifacts exist in subdirectory
    assert (artifacts_path / "model.pkl").exists()
    assert (artifacts_path / "labels.csv.gz").exists()
    assert (artifacts_path / "metadata.json").exists()
```

**File**: `tests/integration/test_student_pipeline.py`

**Update existing test or add new**:
```python
def test_student_consumes_teacher_subdirectory_artifacts(tmp_path, mock_settings):
    """Test student training can consume teacher artifacts from subdirectories (US-028 Phase 6o)."""
    # 1. Run teacher training with subdirectory output
    # 2. Verify artifacts in subdirectory
    # 3. Run student training pointing to that subdirectory
    # 4. Verify student succeeds
    pass  # Implement based on existing test patterns
```

---

### Step 6: Update Documentation

**File**: `docs/stories/us-028-historical-run.md`

**Add Phase 6o section**:

```markdown
## Phase 6o: Align Teacher/Student Artifact Paths (2025-10-14)

### Problem

Phase 6n diagnostics revealed that teacher batch trainer recorded incorrect `artifacts_path` in `teacher_runs.json`, pointing to subdirectories that were never created. Teacher models were saved as flat files in `data/models/`, causing 100% student training failure (6/6 windows failed with "Teacher labels not found" errors).

### Solution

**Artifacts Directory Structure**:

**Before (Phase 6n)**:
```
data/models/
├── teacher_model_20251014_221225.pkl  (flat file)
├── teacher_model_20251014_221228.pkl
└── 20251014_221222/
    ├── teacher_runs.json  (claims subdirectories exist)
    └── student_runs.json
```

**After (Phase 6o)**:
```
data/models/
└── 20251014_221222/
    ├── teacher_runs.json
    ├── student_runs.json
    ├── RELIANCE_2023-01-01_to_2023-06-30/
    │   ├── model.pkl
    │   ├── labels.csv.gz
    │   ├── metadata.json
    │   └── feature_importance.csv
    ├── RELIANCE_2023-06-30_to_2023-12-27/
    │   └── ...
    └── TCS_2023-01-01_to_2023-06-30/
        └── ...
```

### Implementation

**1. Added `--output-dir` to `train_teacher.py`** ([train_teacher.py:XX](scripts/train_teacher.py#LXX))
   - Accepts optional output directory parameter
   - Defaults to `data/models/` for backward compatibility
   - Passes to `TeacherLabeler` for artifact saving

**2. Updated `TeacherLabeler` class** ([teacher_student.py:XX](src/services/teacher_student.py#LXX))
   - Accepts `output_dir` parameter in constructor
   - Creates output directory if it doesn't exist
   - Saves all artifacts (model, labels, metadata, importance) in specified directory

**3. Updated `train_teacher_batch.py`** ([train_teacher_batch.py:XX](scripts/train_teacher_batch.py#LXX))
   - Creates artifacts subdirectory before teacher training: `{batch_dir}/{window_label}/`
   - Passes `--output-dir {artifacts_path}` to `train_teacher.py`
   - Updated `is_already_trained()` to check for artifacts in subdirectories

**4. Student Training** (no changes needed)
   - Already expects artifacts at `task["artifacts_path"]`
   - Now works correctly with new directory structure

### Benefits

- ✅ **Clean isolation**: Each window's artifacts in separate directory
- ✅ **Easy cleanup**: Delete entire batch directory to remove all artifacts
- ✅ **Scalability**: Supports hundreds of windows without flat-file pollution
- ✅ **Debuggability**: Clear 1:1 mapping between window and artifacts
- ✅ **Student training unblocked**: 0% → 100% success rate expected

### Migration

**Backward Compatibility**:
- `train_teacher.py` still works without `--output-dir` (uses default `data/models/`)
- Existing flat-file teacher models remain functional
- No migration of historical artifacts required

**New Behavior**:
- All batch training now creates subdirectories automatically
- `teacher_runs.json` correctly references existing paths
- Student training works out-of-the-box

### Validation

**Quality Gates**:
```bash
python -m ruff check scripts/train_teacher*.py src/services/teacher_student.py
python -m pytest tests/integration/test_teacher_pipeline.py::test_teacher_batch_artifacts_in_subdirectories -v
python -m pytest tests/integration/test_student_pipeline.py -v
```

**End-to-End Test**:
```bash
# Run full pipeline (Phase 6m retry)
conda run -n sensequant python scripts/run_historical_training.py \
  --symbols RELIANCE,TCS \
  --start-date 2023-01-01 \
  --end-date 2024-06-23

# Expected:
# - Phase 1: ✅ Data ingestion complete
# - Phase 2: ✅ 6/6 teacher windows trained
# - Phase 3: ✅ 6/6 student models trained (previously 0/6)
# - Phase 4-7: Complete successfully
```

### Artifacts

- **Code changes**:
  - `scripts/train_teacher.py` (+10 lines: --output-dir argument)
  - `src/services/teacher_student.py` (+15 lines: output_dir parameter)
  - `scripts/train_teacher_batch.py` (+20 lines: directory creation, --output-dir passing)
- **Tests**:
  - `tests/integration/test_teacher_pipeline.py` (+1 test for subdirectory structure)
  - `tests/integration/test_student_pipeline.py` (+1 test for student consuming subdirectories)
- **Documentation**: This section

---
```

---

## Quality Gates

### Ruff (Linting)
```bash
python -m ruff check scripts/train_teacher.py
python -m ruff check scripts/train_teacher_batch.py
python -m ruff check scripts/train_student_batch.py
python -m ruff check src/services/teacher_student.py
```

### Pytest (Integration Tests)
```bash
python -m pytest tests/integration/test_teacher_pipeline.py -q
python -m pytest tests/integration/test_student_pipeline.py -q
```

### End-to-End Validation
```bash
# Clean start
rm -rf data/models/202*

# Run full pipeline
conda run -n sensequant python scripts/run_historical_training.py \
  --symbols RELIANCE,TCS \
  --start-date 2023-01-01 \
  --end-date 2024-06-23

# Verify artifacts structure
ls -la data/models/202*/
ls -la data/models/202*/RELIANCE_2023-01-01_to_2023-06-30/
```

---

## Implementation Checklist

- [ ] Step 1: Add `--output-dir` to `train_teacher.py`
- [ ] Step 2: Update `TeacherLabeler` to accept and use `output_dir`
- [ ] Step 3: Update `train_teacher_batch.py` to create directories and pass `--output-dir`
- [ ] Step 4: Test single teacher training with `--output-dir`
- [ ] Step 5: Test teacher batch training with 2 windows
- [ ] Step 6: Verify student training works with new structure
- [ ] Step 7: Update `is_already_trained()` check
- [ ] Step 8: Add integration tests
- [ ] Step 9: Update documentation
- [ ] Step 10: Run quality gates
- [ ] Step 11: Run end-to-end Phase 6m pipeline retry

---

## Risk Mitigation

**Risk**: Breaking existing workflows that depend on flat-file teacher models

**Mitigation**:
- Keep default behavior unchanged when `--output-dir` not specified
- Add deprecation warning for flat-file mode
- Document migration path

**Risk**: Disk space usage with subdirectories

**Mitigation**:
- Each window artifacts < 10MB typically
- Batch cleanup scripts can delete entire batch directories
- Monitor disk usage in production

**Risk**: Path length issues on Windows

**Mitigation**:
- Keep window labels concise
- Test on Windows before deployment
- Document path length limits

---

## Success Criteria

1. ✅ Teacher batch training creates subdirectories for each window
2. ✅ All artifacts (model, labels, metadata, importance) saved in subdirectories
3. ✅ `teacher_runs.json` correctly references subdirectory paths
4. ✅ Student training succeeds with 100% rate (6/6 windows)
5. ✅ Integration tests pass
6. ✅ Quality gates pass (ruff, pytest)
7. ✅ End-to-end pipeline completes Phases 1-7
8. ✅ Documentation updated and accurate

---

## Next Steps (Phase 6p)

After Phase 6o implementation and validation:

1. Resume Phase 6m full pipeline execution
2. Capture evidence for Phases 3-7
3. Generate promotion briefing
4. Deliver final handoff with complete pipeline results
