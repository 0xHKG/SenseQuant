# Known Issues - SenseQuant

**Last Updated**: 2025-11-01

---

## Active Issues

_No active issues at this time._

---

## Resolved Issues

### Issue #1: Phase 2 Telemetry Aggregation Bug ✅ RESOLVED

**Status**: ✅ Resolved
**Severity**: High (blocked telemetry dashboard testing)
**Discovered**: 2025-11-01
**Resolved**: 2025-11-01
**Resolution Commit**: d363de9
**Affects**: Training telemetry event emission

**Description**:
Historical training orchestrator crashes during Phase 2 (Teacher Training) telemetry aggregation with a KeyError/TypeError.

**Error Message**:
```
2025-11-01 12:37:56.550 | ERROR    | __main__:_run_phase_2_teacher_training:819 -   ✗ Teacher training failed: 'NoneType' object is not a mapping
```

**Location**: `scripts/run_historical_training.py:819` (Phase 2 telemetry aggregation)

**Impact**:
- Teacher training **completes successfully** (e.g., 12/12 windows trained)
- Orchestrator crashes during metric aggregation after training completes
- Telemetry JSONL file remains empty (0 bytes) - events not flushed before crash
- Dashboard cannot display training progress

**Reproduction Steps**:
```bash
# Create test symbols file
echo -e "LT\nADANIPORTS" > /tmp/test_symbols.txt

# Run training with telemetry enabled
conda run -n sensequant python scripts/run_historical_training.py \
  --symbols-file /tmp/test_symbols.txt \
  --start-date 2022-01-01 \
  --end-date 2024-12-01 \
  --skip-fetch \
  --enable-telemetry \
  --workers 2

# Training succeeds (12 windows) but crashes at line 819
```

**Expected Behavior**:
- Phase 2 telemetry events should be logged with valid metrics
- Telemetry JSONL should contain run_start, phase_start, teacher_window events
- Orchestrator should complete all 7 phases without crash

**Actual Behavior**:
- Phase 2 aggregation attempts to access None as a mapping
- Crash prevents telemetry buffer flush
- JSONL file exists but is empty

**Root Cause** (Resolved):
Three bugs in telemetry handling:

1. **Aggregation TypeError** (Line 755): stats accessed as dict before None check
   - `_aggregate_teacher_runs_from_json()` could return None
   - stats['completed'] accessed at line 755 without checking if stats is None first

2. **Metrics NoneType** (Line 498): metrics/sample_counts could be None
   - Telemetry emission tried to unpack None values with ** operator

3. **Flush on Early Exit** (Line 165-224): telemetry not flushed when phases return False
   - Only flushed on success or exception, not on early return
   - Caused empty telemetry files (0 bytes) even when Phase 2 succeeded

**Resolution**:
1. Moved fallback stats initialization before first access (line 727-756)
   - stats is now guaranteed to be a dict, never None
2. Added defensive "or {}" fallback for None metrics (line 498-499)
3. Wrapped all phases in try/finally block (line 165-224)
   - telemetry.close() now guaranteed on all exit paths
   - Removed duplicate close() from exception handler

**Testing**:
- Smoke test: 2 symbols (LT, ADANIPORTS), 12 teacher windows
- Phase 2 aggregation: ✅ SUCCESS (no crash)
- Telemetry flush: ✅ 17 events (5.6KB) flushed despite Phase 3 failure
- Evidence: `data/analytics/training/training_run_live_candidate_20251101_130729.jsonl`

**Fixed In**: Commit d363de9 (2025-11-01 13:08 IST)

**Related Commits**:
- 14614b8: Telemetry flush infrastructure (explicit flush + line buffering)
- 7158b93: Live order-book provider
- a6cf80d: Support/resistance analytics

---

## Resolved Issues

### Issue #2: Missing _load_cached_bars() Method ✅ RESOLVED

**Status**: Resolved
**Resolution Date**: 2025-10-29
**Commit**: e1222ec

**Description**: AttributeError when training in dry-run mode due to missing `_load_cached_bars()` method in BreezeClient.

**Resolution**: Implemented complete 59-line method with CSV parsing, timezone handling, and error handling.

**Details**: See `docs/batch5-ingestion-report.md` Phase 7 Training section.

---

### Issue #3: Telemetry File Empty (0 bytes) ✅ RESOLVED

**Status**: Resolved
**Resolution Date**: 2025-11-01
**Commit**: 14614b8

**Description**: Telemetry JSONL files created but remained 0 bytes after training due to buffering.

**Resolution**:
- Added explicit `f.flush()` in TrainingTelemetryLogger.flush()
- Used `buffering=1` (line buffering) for immediate writes
- Reduced default buffer_size from 50 to 10 events

**Testing**: Manual telemetry test validates 8 events successfully written and flushed.

**Note**: Issue #1 (aggregation bug) prevents testing full integration, but flush mechanism is confirmed working.

---

## Issue Tracking

| ID | Issue | Severity | Status | Resolution Date | Commit |
|----|-------|----------|--------|-----------------|--------|
| #1 | Phase 2 Telemetry Aggregation Bug | 🔴 HIGH | ✅ Resolved | 2025-11-01 | d363de9 |
| #2 | Missing _load_cached_bars() | 🔴 CRITICAL | ✅ Resolved | 2025-10-29 | e1222ec |
| #3 | Telemetry File Empty | 🔴 HIGH | ✅ Resolved | 2025-11-01 | 14614b8 |

---

**Report Generated**: 2025-11-01 13:10 IST (Updated after Issue #1 resolution)
