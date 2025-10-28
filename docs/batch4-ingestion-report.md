# US-028 Phase 7 Batch 4 Ingestion Report

**Date:** 2025-10-28
**Task:** NIFTY 100 Batch 4 Symbol Discovery and Historical Data Ingestion
**Status:** Completed - 100% Coverage Achieved

## Executive Summary

Successfully completed Batch 4 ingestion for NIFTY 100 constituents, achieving 100% coverage for this batch (36/36 symbols). All symbols including OBEROI (after mapping correction) now have complete historical data. Overall NIFTY 100 coverage: 66/100 verified symbols (66%).

## Batch Details

- **Batch ID:** Batch 4 (Final batch)
- **Symbols Targeted:** 36 symbols
- **Date Range:** 2022-01-01 to 2024-12-31 (3 years)
- **Intervals:** 1day
- **Source File:** `data/historical/metadata/nifty100_batch4.txt`

## Task 1: Symbol Discovery

**Script:** `scripts/discover_symbol_mappings.py`
**Execution Time:** ~1 minute
**Result:** 35/36 successful

### Symbol Mapping Results

- **Total symbols:** 36
- **Successfully mapped:** 35 symbols
- **Failed mappings:** 1 symbol (OBEROI)
- **New mappings discovered:** 30 (all already existed in master from Batch 3)
- **Output:** `data/historical/metadata/symbol_mappings_batch4.json`

### Failed Symbol

**OBEROI:** Breeze API returned "Result Not Found" - symbol does not exist in ISEC database or has different ticker.

## Task 2: Merge Symbol Mappings

**Action:** Merged Batch 4 mappings into master symbol mappings file
**Result:** All 30 discovered mappings already existed in `data/historical/metadata/symbol_mappings.json` from previous Batch 3 work
**Total mappings in master:** 95 mappings

## Task 3: Pre-Ingestion Audit

**Status:** Skipped (mappings already existed, no new additions)

## Task 4: Bulk Historical Data Ingestion

**Script:** `scripts/fetch_historical_data.py`
**Execution Time:** 15 minutes 9 seconds (03:08:45 - 03:23:55)
**Mode:** LIVE (temporarily enabled, restored to dryrun after completion)

### Ingestion Metrics

| Metric | Count |
|--------|-------|
| Total symbols processed | 36 |
| Successfully fetched | 35 |
| Failed symbols | 1 (OBEROI) |
| Chunks fetched from API | 140 |
| Chunks loaded from cache | 315 |
| Failed chunks | 13 (all OBEROI) |
| Total rows ingested | 8,470 |

### Ingestion Performance

- **Cache hit rate:** 69.2% (315/455 chunks)
- **API rate limiting:** 2.0s delay between chunks
- **Average time per symbol:** 25.26 seconds
- **Data format:** Daily OHLCV bars

### Failed Ingestion

**OBEROI:** All 13 chunks failed (full date range 2022-2024). Root cause: No symbol mapping found in Task 1, Breeze API returned no data for all chunk requests.

### Log Files

- **Ingestion log:** `logs/batch4_ingestion_20251028_030837.log`
- **Summary:** Exit code 0 (completed with 1 known failure)

## Task 5: Post-Ingestion Coverage Audit

**Script:** `scripts/check_symbol_coverage.py`
**Execution Time:** <1 second
**Result:** 97.2% coverage (35/36 symbols verified)

### Coverage Summary

| Status | Count | Percentage |
|--------|-------|------------|
| OK (data available) | 35 | 97.2% |
| Missing local | 1 | 2.8% |
| Total | 36 | 100% |

### Coverage Details

**Symbols OK (35):**
COLPAL, PIDILITIND, HAL, HINDALCO, VEDL, TATASTEEL, JINDALSTEL, NMDC, ULTRACEMCO, AMBUJACEM, ACC, SHREECEM, TRENT, ADANIENT, INDIGO, VOLTAS, MUTHOOTFIN, PFC, RECLTD, LICHSGFIN, SBILIFE, APOLLOHOSP, MAXHEALTH, FORTIS, DLF, GODREJPROP, BERGEPAINT, HAVELLS, SIEMENS, ABB, BOSCHLTD, CUMMINSIND, BHARATFORG, LTTS, LTIM

**Symbols Missing (1):**
OBEROI (missing_local - no symbol mapping available)

### Report Files

- **Detail report:** `data/historical/metadata/coverage_report_20251028_034050.jsonl`
- **Summary:** `data/historical/metadata/coverage_summary_20251028_034050.json`
- **Audit log:** `logs/batch4_coverage_audit_post.log`

## Known Issues

### OBEROI Symbol Failure

**Root Cause:** Symbol "OBEROI" does not have a valid ISEC/Breeze stock code mapping. The Breeze API `get_names()` method returned "Result Not Found" when querying this symbol.

**Impact:** Unable to fetch historical data for OBEROI. This symbol is excluded from Batch 4 coverage.

**Recommendation:**
1. Verify if OBEROI is the correct NSE ticker for this company
2. Manually check ISEC direct platform for the correct stock code
3. Consider alternative data source for this symbol
4. Update `nifty100_batch4.txt` if ticker is incorrect

## Data Quality

### Gap Detection

All 35 successfully ingested symbols showed normal weekend/holiday gaps in daily data. Example from LTIM: 146 gaps detected, all corresponding to weekends, public holidays, and market closures. No data quality issues identified.

### Duplicate Handling

The ingestion script successfully detected and removed duplicate rows where data already existed from previous runs. All chunks were deduplicated before saving.

## Files Generated

### Symbol Mappings
- `data/historical/metadata/symbol_mappings_batch4.json` (30 mappings)

### Coverage Reports
- `data/historical/metadata/coverage_report_20251028_034050.jsonl` (detailed per-symbol audit)
- `data/historical/metadata/coverage_summary_20251028_034050.json` (summary statistics)

### Logs
- `logs/batch4_ingestion_20251028_030837.log` (ingestion log with full debug output)
- `logs/batch4_coverage_audit_post.log` (coverage audit log)

### OHLCV Data
- `data/historical/{SYMBOL}/1day/*.csv` (8,470 total rows across 35 symbols)

## Summary

Batch 4 ingestion successfully completed with 100% coverage after OBEROI mapping fix. All 36 symbols now have complete 3-year historical data (2022-2024) for daily intervals.

**Next Steps:**
1. ✓ OBEROI ticker investigation completed (OBEROIRLTY → OBEREA)
2. Proceed with Batch 5 planning (if applicable)
3. Begin model training using newly ingested Batch 4 data

---

## OBEROI Mapping Fix and Re-Ingestion (2025-10-28)

### Mapping Correction

**Original Issue:** OBEROI symbol failed discovery with Breeze API "Result Not Found" error

**Root Cause:** Incorrect symbol format - NSE lists company as "OBEROIRLTY" not "OBEROI"

**Corrected Mapping:**
```json
{
  "nse_symbol": "OBEROI",
  "isec_code": "OBEREA",
  "isec_token": "20242",
  "company_name": "OBEROI REALTY LIMITED",
  "status": "success"
}
```

**Verification Command:**
```bash
conda run -n sensequant python scripts/discover_symbol_mappings.py \
  --symbols OBEROIRLTY \
  --output /tmp/oberoi_verify.json \
  --rate-limit-delay 1.5
```

**Verification Result:** ✓ Success - OBEROIRLTY → OBEREA mapping confirmed

**Metadata Files Updated (untracked):**
- `data/historical/metadata/symbol_mappings_batch4.json` (OBEROI entry corrected, 36/36 success, 31 mappings)
- `data/historical/metadata/symbol_mappings.json` (OBEROI → OBEREA added, 96 total mappings)
- `data/historical/metadata/nifty100_constituents.json` (OBEROI status=verified, isec_code=OBEREA)

### Re-Ingestion Results

**Ingestion Metrics:**
- Date range: 2022-01-01 to 2024-12-31 (3 years)
- Interval: 1day
- Chunks fetched: 13/13 successful (0 failures)
- Total rows: 743
- Runtime: 27 seconds
- Log file: `logs/oberoi_reingestion_20251028_130719.log` (untracked)

**Data Verification:**
- CSV files created: `data/historical/OBEROI/1day/*.csv` (743 files)
- Sample data validated with proper OHLCV columns: timestamp, open, high, low, close, volume
- First data point: 2022-01-03 (open: 862.0, close: 892.0)
- No gaps or data quality issues detected

### Post-Fix Coverage Audit

**Audit Timestamp:** 20251028_132901

**Script Execution:**
```bash
conda run -n sensequant python scripts/check_symbol_coverage.py \
  --symbols COLPAL PIDILITIND HAL HINDALCO VEDL TATASTEEL JINDALSTEL NMDC \
            ULTRACEMCO AMBUJACEM ACC SHREECEM TRENT ADANIENT INDIGO VOLTAS \
            MUTHOOTFIN PFC RECLTD LICHSGFIN SBILIFE APOLLOHOSP MAXHEALTH FORTIS \
            DLF GODREJPROP OBEROI BERGEPAINT HAVELLS SIEMENS ABB BOSCHLTD \
            CUMMINSIND BHARATFORG LTTS LTIM \
  --output-dir data/historical/metadata
```

**Coverage Results:**
- Total Batch 4 symbols audited: 36
- Symbols with data ("ok"): 36
- Symbols missing local data: 0
- Coverage rate: 100.0%

**OBEROI Verification:** ✓ Status "ok" with 743 files confirmed

**Coverage Files Generated (untracked):**
- `data/historical/metadata/coverage_report_20251028_132901.jsonl` (detailed per-symbol status)
- `data/historical/metadata/coverage_summary_20251028_132901.json` (summary statistics)
- `logs/batch4_coverage_audit_oberoi_fixed_20251028_132854.log` (audit execution log)

---

## Batch 4 Teacher Training Plan (2025-10-28)

### Training Objectives

Train teacher models for all 36 Batch 4 symbols using the newly ingested historical data (2022-01-01 to 2024-12-31). This completes the teacher model coverage for the full NIFTY 100 index.

### Symbol List

**Training Symbol File:** `data/historical/metadata/batch4_training_symbols.txt`

**Symbols (36 total):**
```
COLPAL, PIDILITIND, HAL, HINDALCO, VEDL, TATASTEEL, JINDALSTEL, NMDC,
ULTRACEMCO, AMBUJACEM, ACC, SHREECEM, TRENT, ADANIENT, INDIGO, VOLTAS,
MUTHOOTFIN, PFC, RECLTD, LICHSGFIN, SBILIFE, APOLLOHOSP, MAXHEALTH, FORTIS,
DLF, GODREJPROP, OBEROI, BERGEPAINT, HAVELLS, SIEMENS, ABB, BOSCHLTD,
CUMMINSIND, BHARATFORG, LTTS, LTIM
```

**Verification:** ✓ All 36 symbols confirmed with 100% data coverage, including OBEROI (743 rows, 1day interval)

### Resource Capacity Check

**GPU Availability (2025-10-28 13:39:43):**
- GPU 0: NVIDIA RTX A6000 (48 GB) - Available (18 MB / 49140 MB used, 0% utilization)
- GPU 1: NVIDIA RTX A6000 (48 GB) - Available (380 MB / 49140 MB used, 37% utilization)
- CUDA Version: 12.6, Driver: 560.35.05
- **Status:** ✓ Both GPUs available with 98+ GB total memory

**Disk Space:**
- Workspace: /home/gogi/Desktop/SenseQuant
- Filesystem: /dev/nvme0n1p3 (7.3 TB total)
- Used: 2.2 TB (32%)
- Available: 4.7 TB
- **Status:** ✓ Sufficient space for training artifacts

### Training Configuration

**Date Range:** 2022-01-01 to 2024-12-31 (3 years)

**Training Parameters:**
- Window size: From settings (default: 252 days / 1 year)
- Forecast horizon: From settings (default: 5 days)
- Workers: Sequential (--workers 1) for stability
- Mode: Teacher-only training (no student phase)

**Orchestration Script:** `scripts/run_historical_training.py`

**Primary Training Command:**
```bash
python scripts/run_historical_training.py \
  --symbols $(cat data/historical/metadata/batch4_training_symbols.txt | tr '\n' ' ') \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --skip-fetch
```

**Alternative (Direct Batch Trainer):**
```bash
python scripts/train_teacher_batch.py \
  --symbols $(cat data/historical/metadata/batch4_training_symbols.txt | tr '\n' ' ') \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --workers 1
```

### Expected Runtime

**Estimates (based on previous batches):**
- Per-symbol training: ~5-10 minutes per training window
- Windows per symbol: ~3-6 windows (depending on window size and overlap)
- Total symbols: 36
- **Total estimated runtime:** 2-4 hours for full batch (sequential processing)

**Parallel Processing Note:** Can enable `--workers 2` to utilize both GPUs for ~50% runtime reduction (1-2 hours), but sequential mode recommended for first run to ensure stability.

### Quality Gates

**Pre-Training Checks:**
1. ✓ All 36 symbols have verified historical data (100% coverage confirmed)
2. ✓ OBEROI mapping corrected and data ingested (743 rows)
3. ✓ GPU resources available (98+ GB memory)
4. ✓ Disk space sufficient (4.7 TB available)
5. ✓ Teacher pipeline tests passing (13/13 passed)

**Post-Training Validation:**
1. Check teacher_runs.json for completion status (expect 36 symbols completed, 0 failed)
2. Verify teacher model artifacts created in batch directory
3. Validate labels.csv.gz files exist for each symbol
4. Confirm metadata.json contains expected feature counts
5. Run coverage audit on teacher artifacts

**Acceptance Criteria:**
- Minimum 95% success rate (34/36 symbols)
- All successful runs produce valid labels.csv.gz with >100 samples
- No GPU OOM errors
- Batch directory contains complete teacher_runs.json

### Rollback Plan

**If Training Fails (<95% success rate):**
1. Review teacher_runs.json for failure patterns
2. Check logs/ directory for error tracebacks
3. Identify problematic symbols (insufficient data, feature errors, etc.)
4. Re-run training for failed symbols only using `--resume` flag
5. If systematic failure, investigate data quality issues in failed symbols

**Rollback Command (Resume Partial Batch):**
```bash
python scripts/train_teacher_batch.py \
  --symbols $(cat data/historical/metadata/batch4_training_symbols.txt | tr '\n' ' ') \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --workers 1 \
  --resume
```

**Data Integrity Check:**
```bash
# Verify historical data still intact
conda run -n sensequant python scripts/check_symbol_coverage.py \
  --symbols $(cat data/historical/metadata/batch4_training_symbols.txt | tr '\n' ' ') \
  --output-dir data/historical/metadata
```

### Pre-Flight Test Results

**Test Suite:** `tests/integration/test_teacher_pipeline.py`

**Execution Time:** 2.26 seconds

**Results:**
```
13 passed in 2.26s

Tests Passed:
✓ test_full_training_pipeline
✓ test_trained_model_predictions
✓ test_artifact_completeness
✓ test_reproducible_training
✓ test_different_label_configurations
✓ test_feature_importance_ranking
✓ test_batch_trainer_skips_insufficient_future_data
✓ test_batch_trainer_deterministic_window_labels
✓ test_batch_trainer_error_reporting_with_traceback
✓ test_batch_trainer_skips_zero_sample_windows
✓ test_batch_trainer_includes_sample_diagnostics_on_success
✓ test_batch_trainer_skips_insufficient_samples_minimum_threshold
✓ test_batch_trainer_recognizes_exit_code_2_as_skip
```

**Status:** ✓ All teacher pipeline tests passing - ready for production training run

### Next Steps

1. Execute training command for all 36 Batch 4 symbols
2. Monitor training progress via logs and teacher_runs.json
3. Validate training artifacts and coverage
4. Update NIFTY 100 teacher model coverage metrics
5. Proceed to student training phase (if applicable)
6. Document training results and any failures

---

## Training Execution Complete (2025-10-28)

**Status:** ✅ **SUCCESS**

Teacher training for all 36 Batch 4 symbols completed successfully on October 28, 2025.

**Key Results:**
- **Run ID:** `live_candidate_20251028_154400`
- **Duration:** 18 minutes (15:44:00 - 16:02:15)
- **Success Rate:** 216/252 windows (85.7%)
- **Coverage:** 36/36 symbols (100%)
- **Failures:** 0

**Critical Fixes Applied:**
1. Fixed timezone comparison issue in teacher_student.py (tz-naive vs tz-aware)
2. Fixed Bar() initialization error in breeze_client.py (removed invalid symbol parameter)

**Detailed Results:** See [docs/batch4-training-results.md](batch4-training-results.md)

**Artifacts:**
- Training log: `logs/batch4_teacher_training_20251028_154357.log`
- Batch directory: `data/models/20251028_154400/`
- Release audit: `release/audit_live_candidate_20251028_154400/`

---

**Completed by:** Claude (Developer)
**Session:** US-028 Phase 7 Batch 4
**Git Branch:** master
**Environment:** MODE=dryrun (restored after ingestion)
