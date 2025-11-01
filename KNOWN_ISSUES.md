# Known Issues - SenseQuant

**Last Updated**: 2025-11-01

---

## Active Issues

### Issue #1: Phase 2 Telemetry Aggregation Bug 🔴 HIGH

**Status**: Open
**Severity**: High (blocks telemetry dashboard testing)
**Discovered**: 2025-11-01
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

**Root Cause** (Preliminary):
The error occurs when trying to aggregate teacher metrics for telemetry emission. The code at line 819 expects a dict but receives None. This suggests:

1. Teacher training doesn't return metrics in expected format, OR
2. Metrics extraction from `teacher_runs.json` returns None, OR
3. Telemetry event creation expects non-None metrics

**Code Context**:
```python
# scripts/run_historical_training.py:819 (approx)
# Phase 2 telemetry aggregation after teacher training
# Likely trying to access metrics['some_key'] where metrics is None
```

**Workaround**:
- Disable telemetry (`--enable-telemetry` flag removed) for production runs
- Training pipeline functions correctly without telemetry

**Fix Priority**: HIGH - Required for dashboard integration and production monitoring

**Investigation Needed**:
1. Check `_aggregate_teacher_runs_from_json()` return value at line 475
2. Verify Phase 2 telemetry emission code expects optional metrics
3. Add None-checking before metric dictionary access
4. Ensure buffer flush on error (try/finally pattern)

**Related Commits**:
- 14614b8: Fix telemetry flushing and output buffering issues (flush mechanism works)
- Telemetry infrastructure is functional (manual tests pass)
- Bug is in orchestrator's event emission, not telemetry logger

**Testing Notes**:
- Telemetry logger manual test passes (8 events validated)
- Flush improvements work correctly (explicit flush + line buffering)
- Issue is isolated to orchestrator Phase 2 code

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

| ID | Issue | Severity | Status | Assignee |
|----|-------|----------|--------|----------|
| #1 | Phase 2 Telemetry Aggregation Bug | 🔴 HIGH | Open | - |
| #2 | Missing _load_cached_bars() | 🔴 CRITICAL | ✅ Resolved | - |
| #3 | Telemetry File Empty | 🔴 HIGH | ✅ Resolved | - |

---

**Report Generated**: 2025-11-01 12:45 IST
