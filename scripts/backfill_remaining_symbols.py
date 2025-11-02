#!/usr/bin/env python3
"""
Full-Universe Backfill Script
==============================
Executes historical data backfill for all remaining symbols.

IMPORTANT: This script operates under Breeze API limitations:
- 1-minute data: API provides sparse coverage (~39 files/symbol, not full range)
- 5-minute data: API provides data from 2022-03-09 onwards (not 2022-01-01)
- 1-day data: Full historical coverage available

Usage:
    python scripts/backfill_remaining_symbols.py --interval 1minute --dry-run
    python scripts/backfill_remaining_symbols.py --interval 5minute --execute
    python scripts/backfill_remaining_symbols.py --interval all --execute

Generated: 2025-11-02
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "historical"
METADATA_DIR = DATA_DIR / "metadata"
FETCH_SCRIPT = BASE_DIR / "scripts" / "fetch_historical_data.py"

# Load backfill tasks
TASKS_FILE = Path("/tmp/backfill_tasks.json")

# Date ranges (adjusted for API limitations)
DATE_RANGES = {
    "1minute": {
        "start": "2022-01-01",  # Will fetch sparse data from API
        "end": "2024-12-30",
        "expected_files_per_symbol": 39  # Based on priority symbol results
    },
    "5minute": {
        "start": "2022-01-01",  # API will return from 2022-03-09
        "end": "2025-03-09",
        "expected_files_per_symbol": 238  # Based on priority symbol results
    }
}

# Symbols completed in priority phase
PRIORITY_SYMBOLS = [
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "SBIN", "SUNPHARMA", "TATAMOTORS", "WIPRO"
]


def load_backfill_tasks() -> Dict:
    """Load backfill tasks from JSON file."""
    if not TASKS_FILE.exists():
        print(f"ERROR: Tasks file not found: {TASKS_FILE}")
        sys.exit(1)

    with open(TASKS_FILE) as f:
        return json.load(f)


def get_remaining_symbols(interval: str, tasks: Dict) -> List[str]:
    """Get list of symbols needing backfill for given interval."""
    if interval == "1minute":
        all_symbols = tasks["tasks"]["1minute_backfill"]["symbols"]
    elif interval == "5minute":
        all_symbols = tasks["tasks"]["5minute_backfill"]["all_symbols"]
    else:
        return []

    # Exclude priority symbols already completed
    remaining = [s for s in all_symbols if s not in PRIORITY_SYMBOLS]
    return remaining


def execute_fetch(symbol: str, interval: str, dry_run: bool = True) -> bool:
    """
    Execute historical data fetch for a single symbol.

    Returns True if successful, False otherwise.
    """
    date_range = DATE_RANGES[interval]

    cmd = [
        "conda", "run", "-n", "sensequant",
        "python", str(FETCH_SCRIPT),
        "--symbols", symbol,
        "--start-date", date_range["start"],
        "--end-date", date_range["end"],
        "--intervals", interval,
        "--force"
    ]

    env = {"PYTHONPATH": str(BASE_DIR)}

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Fetching {interval} data for {symbol}...")
    print(f"  Date range: {date_range['start']} → {date_range['end']}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print(f"  [DRY-RUN] Would execute fetch command")
        return True

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"  ✓ SUCCESS ({elapsed:.1f}s)")
            # Extract row count from output if available
            for line in result.stdout.split('\n'):
                if 'rows saved' in line.lower() or 'total:' in line.lower():
                    print(f"    {line.strip()}")
            return True
        else:
            print(f"  ✗ FAILED ({elapsed:.1f}s)")
            print(f"  Error: {result.stderr[-500:]}")  # Last 500 chars
            return False

    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        return False


def verify_coverage(symbol: str, interval: str) -> Dict:
    """Verify data coverage for a symbol after fetch."""
    interval_dir = DATA_DIR / symbol / interval

    if not interval_dir.exists():
        return {"exists": False, "file_count": 0}

    files = sorted(interval_dir.glob("*.csv"))
    if not files:
        return {"exists": True, "file_count": 0}

    return {
        "exists": True,
        "file_count": len(files),
        "first_date": files[0].stem,
        "last_date": files[-1].stem
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical data for remaining symbols"
    )
    parser.add_argument(
        "--interval",
        choices=["1minute", "5minute", "all"],
        required=True,
        help="Which interval to backfill"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without actual API calls"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute actual backfill (required for production run)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of symbols to process before checkpoint (default: 10)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Delay in seconds between symbols (default: 5)"
    )

    args = parser.parse_args()

    # Safety check
    if not args.dry_run and not args.execute:
        print("ERROR: Must specify either --dry-run or --execute")
        sys.exit(1)

    if args.dry_run and args.execute:
        print("ERROR: Cannot specify both --dry-run and --execute")
        sys.exit(1)

    # Load tasks
    print("Loading backfill tasks...")
    tasks = load_backfill_tasks()

    # Determine intervals to process
    intervals = ["1minute", "5minute"] if args.interval == "all" else [args.interval]

    # Execution summary
    summary = {
        "start_time": datetime.now().isoformat(),
        "mode": "dry-run" if args.dry_run else "execute",
        "results": {}
    }

    for interval in intervals:
        print(f"\n{'='*80}")
        print(f"Processing {interval.upper()} backfill")
        print(f"{'='*80}")

        remaining = get_remaining_symbols(interval, tasks)
        print(f"Symbols to process: {len(remaining)}")
        print(f"Priority symbols (already done): {len(PRIORITY_SYMBOLS)}")
        print(f"Expected files per symbol: ~{DATE_RANGES[interval]['expected_files_per_symbol']}")

        if args.dry_run:
            print(f"\n[DRY-RUN MODE] Simulating fetch for first 3 symbols...")
            remaining = remaining[:3]  # Limit dry-run to first 3

        results = {
            "total": len(remaining),
            "successful": 0,
            "failed": 0,
            "symbols": {}
        }

        for idx, symbol in enumerate(remaining, 1):
            print(f"\n[{idx}/{len(remaining)}] Processing {symbol}...")

            success = execute_fetch(symbol, interval, dry_run=args.dry_run)

            # Verify coverage
            coverage = verify_coverage(symbol, interval)

            results["symbols"][symbol] = {
                "fetch_success": success,
                "coverage": coverage
            }

            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1

            # Checkpoint every batch_size symbols
            if idx % args.batch_size == 0 and not args.dry_run:
                checkpoint_file = METADATA_DIR / f"backfill_{interval}_checkpoint_{idx}.json"
                METADATA_DIR.mkdir(exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n  Checkpoint saved: {checkpoint_file}")

            # Rate limiting delay
            if idx < len(remaining):  # Don't delay after last symbol
                if not args.dry_run:
                    print(f"  Waiting {args.delay}s before next symbol...")
                    time.sleep(args.delay)

        summary["results"][interval] = results

        # Print interval summary
        print(f"\n{'-'*80}")
        print(f"{interval.upper()} Backfill Summary:")
        print(f"  Total: {results['total']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"{'-'*80}")

    # Save final summary
    summary["end_time"] = datetime.now().isoformat()
    summary_file = METADATA_DIR / f"backfill_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    METADATA_DIR.mkdir(exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"Mode: {summary['mode']}")
    print(f"Summary saved: {summary_file}")

    # Calculate estimated time for full execution
    if args.dry_run:
        total_symbols = sum(len(get_remaining_symbols(i, tasks)) for i in intervals)
        avg_time_per_symbol = 45  # seconds (based on priority symbol timings)
        total_time_seconds = total_symbols * (avg_time_per_symbol + args.delay)
        total_hours = total_time_seconds / 3600

        print(f"\nEstimated time for full execution:")
        print(f"  Total symbols: {total_symbols}")
        print(f"  Avg time/symbol: ~{avg_time_per_symbol}s fetch + {args.delay}s delay")
        print(f"  Total estimated time: ~{total_hours:.1f} hours")


if __name__ == "__main__":
    main()
