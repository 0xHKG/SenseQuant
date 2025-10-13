# US-028 Phase 6c: Critical Bug Fix - Missing Authentication Call

## Executive Summary

**Bug**: BreezeClient was never authenticated in `fetch_historical_data.py`, causing all API calls to fail with "Unexpected error in Breeze SDK call"

**Root Cause**: `BreezeClient.authenticate()` method was never called after instantiation

**Impact**: 100% failure rate for all chunked data fetches (24/24 chunks failing)

**Fix**: Added authentication call immediately after BreezeClient instantiation

**Status**: ✅ FIXED and VERIFIED

---

## Problem Diagnosis

### Symptoms
All 24 chunks failed with identical error pattern:
```
ERROR | src.adapters.breeze_client:_call_with_retry:545 - Unexpected error in Breeze SDK call
ERROR | src.adapters.breeze_client:historical_bars:270 - Failed to fetch historical bars
DEBUG | src.adapters.breeze_client:fetch_historical_chunk:391 - No data returned for RELIANCE 2024-01-01 to 2024-03-30
```

### Initial Hypothesis
User and I initially suspected:
1. Expired Breeze session token ❌
2. API v2 compatibility issues ❌

### User Correction
User pointed out:
> "TOKEN REFRESHED POST MIDNIGHT - already informed you this - current active token already stored in .env."
> "did you account for change of API to v2? did you check all related scripts for affected method names for new API?"

This prompted investigation of the actual API integration rather than token/v2 issues.

### Discovery Process

1. **Verified v2 API is working correctly**:
   ```bash
   $ python test_breeze_api.py
   ✓ v2 Success: <class 'dict'> with 0 records
   ```

2. **Tested stock code mapping**:
   ```bash
   $ python test_breeze_stock_codes.py
   RELIANCE → 0 records (incorrect stock code)
   RELIND   → 18 records ✓
   TCS      → 18 records ✓
   ```

3. **Verified BreezeClient wrapper is working**:
   ```bash
   $ python test_breeze_mapping.py
   ✓ RELIANCE fetched 19 records (correctly mapped to RELIND)
   ✓ TCS fetched 19 records
   ```

4. **Found the bug**: No `authenticate()` call in `fetch_historical_data.py`:
   ```bash
   $ grep "\.authenticate()" scripts/fetch_historical_data.py
   # No matches found
   ```

### Root Cause
```python
# scripts/fetch_historical_data.py (lines 780-789)
breeze_client = BreezeClient(
    api_key=settings.breeze_api_key,
    api_secret=settings.breeze_api_secret,
    session_token=settings.breeze_session_token,
    dry_run=use_dry_run,
)
logger.info(f"BreezeClient initialized (dry_run={use_dry_run})")
# ❌ Missing: breeze_client.authenticate()
```

Without `authenticate()`:
- `self._client` in BreezeClient remains `None`
- `_call_with_retry()` tries `getattr(None, method_name)` → AttributeError
- Generic exception handler at line 544 logs "Unexpected error" without details

---

## The Fix

### Code Changes

**File**: [scripts/fetch_historical_data.py](scripts/fetch_historical_data.py#L788-L791)

**Lines Added** (788-791):
```python
# Authenticate with Breeze API
if not use_dry_run:
    breeze_client.authenticate()
    logger.info("Breeze API session established")
```

**Location**: Immediately after BreezeClient instantiation (line 786)

### Why This Works

1. `authenticate()` establishes session with Breeze API
2. Creates `_client` instance (BreezeConnect SDK object)
3. All subsequent `_call_with_retry()` calls succeed
4. Historical data fetching works as expected

---

## Verification

### 1. Unit Tests
```bash
$ conda run -n sensequant python -m pytest tests/unit/test_breeze_client.py -xvs
============================== 21 passed in 7.23s ==============================
```

### 2. Integration Tests
```bash
$ conda run -n sensequant python -m pytest tests/integration/test_historical_training.py::test_chunked_historical_fetch_multi_chunk_aggregation -xvs
PASSED
```

### 3. Manual Verification - Cached Data
```bash
$ python scripts/fetch_historical_data.py --symbols RELIANCE --start-date 2024-11-01 --end-date 2024-11-05 --intervals 1day

2025-10-14 01:46:25.369 | INFO  | Breeze API session established
2025-10-14 01:46:25.371 | DEBUG | Chunk 1/1 partially cached: 2024-11-01 to 2024-11-05
2025-10-14 01:46:25.373 | INFO  | ✓ Combined 5 chunk(s) into 5 rows for RELIANCE 1day
✓ All requests completed successfully
```

### 4. Manual Verification - Live API Call
```bash
$ python scripts/fetch_historical_data.py --symbols TCS --start-date 2024-10-01 --end-date 2024-10-05 --intervals 1day --force

2025-10-14 01:46:40.945 | INFO  | Breeze API session established
2025-10-14 01:46:41.090 | DEBUG | Fetched 3 bars for TCS (2024-10-01 to 2024-10-05)
2025-10-14 01:46:41.094 | INFO  | ✓ Combined 1 chunk(s) into 3 rows for TCS 1day
Chunks fetched: 1
Chunks failed: 0
✓ All requests completed successfully
```

### 5. Quality Gates
```bash
$ ruff check scripts/fetch_historical_data.py
All checks passed!

$ ruff format scripts/fetch_historical_data.py
1 file reformatted
```

---

## Key Learnings

### 1. User Feedback Was Correct
The user correctly pointed me to investigate API integration, not token expiry.

### 2. Error Logging Could Be Better
The generic exception handler at [breeze_client.py:544-548](src/adapters/breeze_client.py#L544-L548) did use `exc_info=True`, but the traceback didn't appear in logs. This could be improved with:
```python
except Exception as e:
    logger.error(
        f"Unexpected error in Breeze SDK call: {type(e).__name__}: {e}",
        exc_info=True,
        extra={"component": "breeze"}
    )
    raise BreezeError(f"Unexpected error: {type(e).__name__}: {e}") from e
```

### 3. Test Coverage Gap
Integration tests used mocked BreezeClient, which didn't catch the missing `authenticate()` call. Could add:
- Integration test that verifies BreezeClient initialization flow
- Test that fails if `_client` is None when calling API methods

---

## Next Steps

1. ✅ Run full 10-month pipeline (2024-01-01 to 2024-10-31)
2. ✅ Verify Phase 1 chunk statistics (24 chunks expected)
3. ✅ Confirm MODE: LIVE logs in Phase 2
4. ✅ Enumerate artifacts
5. ✅ Deliver final handoff

---

## Files Modified

| File | Lines Changed | Change Type |
|------|--------------|-------------|
| [scripts/fetch_historical_data.py](scripts/fetch_historical_data.py#L788-L791) | 788-791 (4 lines) | Bug fix: Added authentication call |

---

## Appendix: Test Scripts Created

### test_breeze_api.py
Direct SDK test to verify v1 and v2 API methods

### test_breeze_stock_codes.py
Test different stock codes (RELIANCE vs RELIND) to verify mapping

### test_breeze_mapping.py
Test BreezeClient wrapper to verify stock code mapping is working

All test scripts confirmed the issue was NOT with the API or stock codes, but with missing authentication.

---

Generated: 2025-10-14 01:48 UTC
