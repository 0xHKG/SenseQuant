#!/bin/bash
# US-024 Phase 5: Incremental Daily Update Orchestration Script
#
# This script chains historical data fetch, sentiment fetch, and batch training
# with incremental flags. Suitable for cron scheduling.
#
# Usage:
#   ./scripts/run_incremental_update.sh
#
# Environment variables (optional):
#   SKIP_FETCH=true           - Skip data fetch phase
#   SKIP_TEACHER=true         - Skip teacher training phase
#   SKIP_STUDENT=true         - Skip student training phase
#   PARALLEL_WORKERS=1        - Number of parallel workers (default: 1)
#   DRY_RUN=true              - Use dryrun mode for sentiment fetch
#
# Exit codes:
#   0  - All phases completed successfully
#   1  - One or more phases failed
#   2  - Configuration error

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Logging
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/incremental_update_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Configuration from environment or defaults
SKIP_FETCH="${SKIP_FETCH:-false}"
SKIP_TEACHER="${SKIP_TEACHER:-false}"
SKIP_STUDENT="${SKIP_STUDENT:-false}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-false}"

# Exit status tracking
EXIT_CODE=0

# Banner
log "======================================================================"
log "US-024 Phase 5: Incremental Daily Update Pipeline"
log "======================================================================"
log "Configuration:"
log "  Skip fetch: ${SKIP_FETCH}"
log "  Skip teacher: ${SKIP_TEACHER}"
log "  Skip student: ${SKIP_STUDENT}"
log "  Parallel workers: ${PARALLEL_WORKERS}"
log "  Dry run: ${DRY_RUN}"
log "  Log file: ${LOG_FILE}"
log "======================================================================"

# Change to project root
cd "$PROJECT_ROOT"

# Check Python availability
if ! command -v python &> /dev/null; then
    log_error "Python not found in PATH"
    exit 2
fi

# Phase 1: Data Fetch
if [ "$SKIP_FETCH" = "false" ]; then
    log "Phase 1: Incremental Data Fetch"
    log "----------------------------------------------------------------------"

    # Historical OHLCV fetch
    log "Fetching historical OHLCV data..."
    if python scripts/fetch_historical_data.py --incremental >> "$LOG_FILE" 2>&1; then
        log "✓ Historical data fetch completed"
    else
        log_error "✗ Historical data fetch failed (exit code: $?)"
        EXIT_CODE=1
    fi

    # Sentiment snapshot fetch
    log "Fetching sentiment snapshots..."
    if [ "$DRY_RUN" = "true" ]; then
        if python scripts/fetch_sentiment_snapshots.py --incremental --dryrun >> "$LOG_FILE" 2>&1; then
            log "✓ Sentiment fetch completed (dryrun mode)"
        else
            log_error "✗ Sentiment fetch failed (exit code: $?)"
            EXIT_CODE=1
        fi
    else
        if python scripts/fetch_sentiment_snapshots.py --incremental >> "$LOG_FILE" 2>&1; then
            log "✓ Sentiment fetch completed"
        else
            log_error "✗ Sentiment fetch failed (exit code: $?)"
            EXIT_CODE=1
        fi
    fi
else
    log "Phase 1: Data Fetch (SKIPPED)"
fi

# Phase 2: Teacher Training
if [ "$SKIP_TEACHER" = "false" ]; then
    log "======================================================================"
    log "Phase 2: Incremental Teacher Training"
    log "----------------------------------------------------------------------"

    if python scripts/train_teacher_batch.py \
        --incremental \
        --workers "$PARALLEL_WORKERS" \
        --resume >> "$LOG_FILE" 2>&1; then
        log "✓ Teacher training completed"
    else
        TEACHER_EXIT=$?
        log_error "✗ Teacher training failed (exit code: $TEACHER_EXIT)"
        EXIT_CODE=1
    fi
else
    log "======================================================================"
    log "Phase 2: Teacher Training (SKIPPED)"
fi

# Phase 3: Student Training
if [ "$SKIP_STUDENT" = "false" ]; then
    log "======================================================================"
    log "Phase 3: Incremental Student Training"
    log "----------------------------------------------------------------------"

    if python scripts/train_student_batch.py \
        --resume >> "$LOG_FILE" 2>&1; then
        log "✓ Student training completed"
    else
        STUDENT_EXIT=$?
        log_error "✗ Student training failed (exit code: $STUDENT_EXIT)"
        EXIT_CODE=1
    fi
else
    log "======================================================================"
    log "Phase 3: Student Training (SKIPPED)"
fi

# Summary
log "======================================================================"
log "Pipeline Summary"
log "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    log "✓ All phases completed successfully"
else
    log_error "✗ Pipeline completed with errors (exit code: $EXIT_CODE)"
    log_error "Check logs for details: $LOG_FILE"
    log_error "Check state files for failed tasks:"
    log_error "  - data/state/teacher_batch.json"
    log_error "  - data/state/student_batch.json"
fi
log "======================================================================"

exit $EXIT_CODE
