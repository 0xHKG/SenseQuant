#!/usr/bin/env python3
"""Live training progress monitor for teacher batch runs.

Monitors teacher_runs.json files and displays real-time progress statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def analyze_teacher_runs(teacher_runs_file: Path) -> dict[str, Any]:
    """Analyze teacher_runs.json and extract statistics.

    Args:
        teacher_runs_file: Path to teacher_runs.json

    Returns:
        Dictionary with progress statistics
    """
    if not teacher_runs_file.exists():
        return {
            "exists": False,
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }

    try:
        with open(teacher_runs_file) as f:
            lines = f.readlines()

        total = len(lines)
        success = 0
        failed = 0
        skipped = 0
        failed_windows = []

        for line in lines:
            try:
                entry = json.loads(line)
                status = entry.get("status", "unknown")

                if status == "success":
                    success += 1
                elif status == "failed":
                    failed += 1
                    failed_windows.append({
                        "window": entry.get("window_label", "unknown"),
                        "symbol": entry.get("symbol", "unknown"),
                        "error": entry.get("error", "No error message"),
                    })
                elif status == "skipped":
                    skipped += 1
            except json.JSONDecodeError:
                continue

        return {
            "exists": True,
            "total": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "failed_windows": failed_windows,
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e),
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }


def find_latest_run(models_dir: Path) -> Path | None:
    """Find the latest teacher training run directory.

    Args:
        models_dir: Path to models directory

    Returns:
        Path to latest run directory or None
    """
    if not models_dir.exists():
        return None

    # Find directories with timestamp format YYYYMMDD_HHMMSS
    run_dirs = [
        d for d in models_dir.iterdir()
        if d.is_dir() and d.name.replace("_", "").isdigit() and len(d.name) == 15
    ]

    if not run_dirs:
        return None

    # Sort by directory name (timestamp) and return latest
    return sorted(run_dirs, reverse=True)[0]


def display_progress(stats: dict[str, Any], run_dir: Path, expected_windows: int) -> None:
    """Display formatted progress statistics.

    Args:
        stats: Statistics dictionary from analyze_teacher_runs
        run_dir: Path to run directory
        expected_windows: Expected total number of windows
    """
    print("\n" + "=" * 70)
    print(f"Training Progress Monitor - Run: {run_dir.name}")
    print("=" * 70)

    if not stats["exists"]:
        print("⚠️  teacher_runs.json not found or not readable")
        if "error" in stats:
            print(f"   Error: {stats['error']}")
        return

    total = stats["total"]
    success = stats["success"]
    failed = stats["failed"]
    skipped = stats["skipped"]

    progress_pct = (total / expected_windows * 100) if expected_windows > 0 else 0
    success_rate = (success / total * 100) if total > 0 else 0
    failure_rate = (failed / total * 100) if total > 0 else 0

    print("\n📊 Overall Progress:")
    print(f"   Total windows processed: {total}/{expected_windows} ({progress_pct:.1f}%)")
    print(f"   ✅ Success: {success} ({success_rate:.1f}%)")
    print(f"   ⊘  Skipped: {skipped}")
    print(f"   ❌ Failed:  {failed} ({failure_rate:.2f}%)")

    # Progress bar
    bar_width = 50
    filled = int(bar_width * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n   [{bar}] {progress_pct:.1f}%")

    # Failure threshold check
    threshold = 15.0  # Default threshold
    if failure_rate <= threshold:
        status_emoji = "✅"
        status_msg = f"within threshold ({threshold}%)"
    else:
        status_emoji = "⚠️"
        status_msg = f"EXCEEDS threshold ({threshold}%)"

    print(f"\n🎯 Failure Rate: {failure_rate:.2f}% {status_emoji} {status_msg}")

    # Show failed windows if any
    if failed > 0 and stats.get("failed_windows"):
        print(f"\n❌ Failed Windows ({failed}):")
        for fw in stats["failed_windows"][:5]:  # Show first 5
            print(f"   • {fw['symbol']}: {fw['window']}")
            if fw['error']:
                error_preview = fw['error'][:80]
                print(f"     └─ {error_preview}...")
        if failed > 5:
            print(f"   ... and {failed - 5} more")

    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor live training progress from teacher_runs.json"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Specific run directory to monitor (default: latest)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Base models directory (default: data/models)",
    )
    parser.add_argument(
        "--expected-windows",
        type=int,
        default=768,
        help="Expected total number of windows (default: 768 for 96 symbols × 8 windows)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch and update (refresh every 10 seconds)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Refresh interval in seconds for watch mode (default: 10)",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(models_dir)
        if run_dir is None:
            print(f"❌ No training runs found in {models_dir}")
            return 1

    teacher_runs_file = run_dir / "teacher_runs.json"

    try:
        if args.watch:
            print(f"👁️  Watching {teacher_runs_file}")
            print(f"   Refresh interval: {args.interval}s")
            print("   Press Ctrl+C to stop")

            while True:
                # Clear screen (works on Unix/Linux/Mac)
                print("\033[2J\033[H", end="")

                stats = analyze_teacher_runs(teacher_runs_file)
                display_progress(stats, run_dir, args.expected_windows)

                print(f"\n⏱️  Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Next refresh in {args.interval}s...")

                time.sleep(args.interval)
        else:
            # Single snapshot
            stats = analyze_teacher_runs(teacher_runs_file)
            display_progress(stats, run_dir, args.expected_windows)

            return 0

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
