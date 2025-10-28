# Phase 7 Batch 4 - Teacher Training Results

**Date**: October 28, 2025
**Run ID**: `live_candidate_20251028_154400`
**Batch Directory**: `data/models/20251028_154400/`
**Status**: ✅ **SUCCESS**

---

## Executive Summary

Phase 7 Batch 4 teacher training completed successfully on October 28, 2025, training all 36 symbols across 252 training windows with an 85.7% success rate (216 success, 36 skipped). The orchestrator completed all 7 phases in 18 minutes with exit code 0.

**Key Achievements:**
- ✅ 36/36 symbols trained (100% coverage)
- ✅ 216/252 windows succeeded (85.7%)
- ✅ 36/252 windows skipped (insufficient future data - expected)
- ✅ 0 failures
- ✅ All validation and audit phases passed

---

## Training Timeline

| Phase | Start | End | Duration | Status |
|-------|-------|-----|----------|--------|
| Phase 1: Data Ingestion | 15:44:00 | 15:44:00 | <1 sec | ⏭️ Skipped (--skip-fetch) |
| Phase 2: Teacher Training | 15:44:00 | 15:50:00 | 6 min | ✅ Success |
| Phase 3: Student Training | 15:50:00 | 15:53:00 | 3 min | ✅ Success |
| Phase 4: Model Validation | 15:53:00 | 16:01:51 | 9 min | ✅ Success |
| Phase 5: Statistical Tests | 16:01:51 | 16:01:51 | <1 sec | ✅ Success |
| Phase 6: Release Audit | 16:01:51 | 16:01:51 | <1 sec | ✅ Success |
| Phase 7: Promotion Briefing | 16:01:51 | 16:01:51 | <1 sec | ✅ Success |
| **Total** | **15:44:00** | **16:02:15** | **18 min** | **✅ Success** |

---

## Phase 2: Teacher Training Results

### Summary Statistics

```
Total Training Windows: 252
├─ Success: 216 (85.7%)
├─ Skipped: 36 (14.3%)
└─ Failed: 0 (0.0%)

Symbols Trained: 36/36 (100%)
Average Windows per Symbol: 7 (6 success + 1 skipped)
```

### Skipped Windows Analysis

All 36 skipped windows represent the final training window (2024-12-16 to 2024-12-31) for each of the 36 symbols. These were skipped due to insufficient future data for label generation (forecast horizon extends beyond available data).

**Pattern:**
- Each symbol: 6 successful windows + 1 skipped window = 7 total
- Skipped windows: All end on 2024-12-31 (last available date)
- **Conclusion**: Expected behavior, not an error

### Artifacts Validation

**Teacher Window Directories:**
- Count: 221 directories (includes teacher windows)
- Each contains: `model.pkl`, `labels.csv.gz`, `metadata.json`, `feature_importance.csv`

**Labels Files:**
- Count: 216 files (one per successful window)
- Format: Compressed CSV with OHLCV + features + labels
- Sample columns: `ts, symbol, open, high, low, close, volume, sma_20, sma_50, ema_12, ema_26, rsi_14, atr_14, vwap, bb_upper, bb_middle, bb_lower, macd_line, macd_signal, macd_histogram, adx_14, obv, label, forward_return`

---

## Phase 3: Student Training Results

### Summary Statistics

```
Total Student Runs: 216
├─ Success: 216 (100%)
└─ Failed: 0 (0.0%)

Reward Metrics (Aggregated):
├─ Mean Reward: 0.0161
└─ Positive Samples: 4957
```

### Student Artifacts

- Student window directories: 216 (one per teacher success)
- Each contains student model artifacts corresponding to teacher labels

---

## Critical Fixes Applied

During this training run, two critical bugs were discovered and fixed:

### Fix #1: Timezone Comparison Issue

**File**: `src/services/teacher_student.py` (lines 119-120)

**Problem**: In DRYRUN mode, historical CSV data contains timezone-aware timestamps (UTC+05:30), but the teacher service created timezone-naive filter bounds, causing pandas comparison errors.

**Fix**:
```python
# Before:
start_ts = pd.Timestamp(self.config.start_date)
end_ts = pd.Timestamp(self.config.end_date)

# After:
start_ts = pd.Timestamp(self.config.start_date, tz="Asia/Kolkata")
end_ts = pd.Timestamp(self.config.end_date, tz="Asia/Kolkata")
```

**Root Cause**: BreezeClient DRYRUN loader reads CSV timestamps that already contain `+05:30` info, then explicitly adds `tz='Asia/Kolkata'` at line 330, creating tz-aware timestamps. These were compared against tz-naive bounds from teacher service.

### Fix #2: Bar Initialization Error

**File**: `src/adapters/breeze_client.py` (lines 333-340)

**Problem**: Bar() constructor was called with `symbol` parameter, but the Bar dataclass only accepts 6 fields: `ts, open, high, low, close, volume`.

**Fix**:
```python
# Before:
bars.append(Bar(
    ts=ts,
    open=float(row['open']),
    high=float(row['high']),
    low=float(row['low']),
    close=float(row['close']),
    volume=int(row['volume']),
    symbol=symbol,  # ← Invalid parameter
))

# After:
bars.append(Bar(
    ts=ts,
    open=float(row['open']),
    high=float(row['high']),
    low=float(row['low']),
    close=float(row['close']),
    volume=int(row['volume']),
))
```

---

## Phase 4-7: Validation & Release Audit

### Phase 4: Model Validation
- **Validation Run ID**: `validation_20251028_155300`
- **Status**: ✅ Passed
- **Duration**: 9 minutes
- **Reports**: Generated in `release/audit_live_candidate_20251028_154400/validation_results/`

### Phase 5: Statistical Tests
- **Status**: ✅ All tests passed
- **Duration**: <1 second
- **Artifacts**: Statistical test results stored

### Phase 6: Release Audit
- **Status**: ✅ Passed (with expected warnings for historical training)
- **Audit Bundle**: `release/audit_live_candidate_20251028_154400/`
- **Artifacts**:
  - `metrics.json` - Training and validation metrics
  - `summary.md` - Audit summary
  - `promotion_briefing.md` - Release briefing
  - `plots/` - Performance visualizations
  - `configs/` - Training configurations
  - `telemetry_summaries/` - Telemetry data (empty - telemetry not enabled)

### Phase 7: Promotion Briefing
- **Briefing**: `release/audit_live_candidate_20251028_154400/promotion_briefing.md`
- **Status**: ready-for-review

---

## Known Limitations & Future Work

### 1. Telemetry Not Enabled ❌

**Issue**: Training was run without `--enable-telemetry` flag.

**Impact**:
- No real-time metrics collected in `data/analytics/`
- Telemetry dashboard remained blank during training
- No granular per-window performance tracking

**TODO for Future Runs**:
```bash
python scripts/run_historical_training.py \
  --symbols-file data/historical/metadata/batch4_training_symbols.txt \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --skip-fetch \
  --enable-telemetry  # ← Add this flag
```

### 2. Single GPU Training

**Current**: Training used 1 GPU (GPU 1, ~30% utilization, 540 MiB peak)

**Observation**: GPU 0 remained idle throughout training

**TODO**: Investigate multi-GPU training support for faster batch processing
- Check if `train_teacher_batch.py` supports `--gpus` flag
- Potential speedup: 2x with both GPUs utilized
- Could reduce Phase 2 duration from 6 min → 3 min

### 3. Future Data Skips

**Expected Behavior**: Last window for each symbol skipped (2024-12-16 to 2024-12-31)

**Consideration**: When retraining with newer data (e.g., 2025 data), these windows will become trainable

---

## Commands for Next Steps

### 1. Review Promotion Briefing
```bash
cat release/audit_live_candidate_20251028_154400/promotion_briefing.md
```

### 2. Approve Candidate Run
```bash
python scripts/approve_candidate.py live_candidate_20251028_154400
```

### 3. Deploy to Staging (if approved)
```bash
make deploy-staging
```

---

## Artifacts Reference

| Artifact | Path | Description |
|----------|------|-------------|
| Training Log | `logs/batch4_teacher_training_20251028_154357.log` | Full orchestrator log |
| Batch Directory | `data/models/20251028_154400/` | All training windows and artifacts |
| Teacher Runs | `data/models/20251028_154400/teacher_runs.json` | Per-window teacher training results |
| Student Runs | `data/models/20251028_154400/student_runs.json` | Per-window student training results |
| Run Metadata | `data/models/live_candidate_20251028_154400/` | Run-level metadata and state |
| Release Audit | `release/audit_live_candidate_20251028_154400/` | Validation reports and promotion briefing |

---

## Conclusion

Phase 7 Batch 4 teacher training completed successfully with 100% symbol coverage and 0 failures. The two critical bugs discovered during this run have been fixed and documented. The training artifacts are ready for review and approval.

**Recommended Actions:**
1. ✅ Review promotion briefing
2. ✅ Approve candidate run (if metrics acceptable)
3. ⏳ Plan re-run with telemetry enabled for performance monitoring
4. ⏳ Investigate multi-GPU support for faster training
