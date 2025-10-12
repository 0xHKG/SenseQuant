"""Batch Teacher training script for multiple symbols and time windows (US-024).

This script orchestrates Teacher model training across multiple symbols and training windows,
recording metadata and artifacts for each batch run.

Usage:
    # Train with defaults (90-day windows)
    python scripts/train_teacher_batch.py

    # Train specific symbols with custom window
    python scripts/train_teacher_batch.py \\
        --symbols RELIANCE TCS \\
        --window-days 60 \\
        --forecast-horizon 5

    # Parallel execution with 4 workers
    python scripts/train_teacher_batch.py --workers 4

    # Resume partial batch
    python scripts/train_teacher_batch.py --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.config import Settings
from src.services.state_manager import StateManager


class BatchTrainer:
    """Manages batch Teacher training across multiple symbols and windows."""

    def __init__(
        self,
        settings: Settings,
        output_dir: Path,
        resume: bool = False,
        incremental: bool = False,
        workers: int = 1,
    ):
        """Initialize batch trainer.

        Args:
            settings: Application settings
            output_dir: Directory for batch artifacts
            resume: If True, skip already-trained windows
            incremental: If True, incremental mode (US-024 Phase 4)
            workers: Number of parallel workers (US-024 Phase 5)
        """
        self.settings = settings
        self.output_dir = output_dir
        self.resume = resume
        self.incremental = incremental
        self.workers = workers

        # Generate batch ID
        self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create batch directory
        self.batch_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        # Batch metadata file (JSON Lines format)
        self.metadata_file = self.batch_dir / "teacher_runs.json"

        # Statistics
        self.stats = {
            "total_windows": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "retries": 0,
        }

        # US-024 Phase 5: State management for batch tracking
        state_file = Path("data/state/teacher_batch.json")
        self.state_manager = StateManager(state_file)

        # Thread-safe metadata logging (for parallel execution)
        self.metadata_lock = Lock()

        # Task retry tracking
        self.task_attempts: dict[str, int] = {}  # task_id -> attempt count

    def generate_training_windows(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        window_days: int,
    ) -> list[dict[str, Any]]:
        """Generate training window tasks.

        Args:
            symbols: List of stock symbols
            start_date: Overall start date (YYYY-MM-DD)
            end_date: Overall end date (YYYY-MM-DD)
            window_days: Size of each training window in days

        Returns:
            List of training task dicts with symbol, start, end
        """
        tasks = []

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        for symbol in symbols:
            current_start = start_dt

            while current_start < end_dt:
                current_end = current_start + timedelta(days=window_days)

                # Don't exceed overall end date
                if current_end > end_dt:
                    current_end = end_dt

                # Only add if window is at least 7 days
                if (current_end - current_start).days >= 7:
                    # Generate window label (e.g., "RELIANCE_2024Q1")
                    quarter = (current_start.month - 1) // 3 + 1
                    window_label = f"{symbol}_{current_start.year}Q{quarter}"

                    tasks.append(
                        {
                            "symbol": symbol,
                            "start_date": current_start.strftime("%Y-%m-%d"),
                            "end_date": current_end.strftime("%Y-%m-%d"),
                            "window_label": window_label,
                            "artifacts_path": str(self.batch_dir / window_label),
                        }
                    )

                # Move to next window
                current_start = current_end

        return tasks

    def is_already_trained(self, task: dict[str, Any]) -> bool:
        """Check if a window has already been trained.

        Args:
            task: Training task dict

        Returns:
            True if artifacts exist for this window
        """
        artifacts_path = Path(task["artifacts_path"])
        # Check for common teacher artifacts
        return (artifacts_path / "labels.csv").exists()

    def train_window(
        self,
        task: dict[str, Any],
        forecast_horizon: int,
    ) -> dict[str, Any]:
        """Train teacher model for a single window.

        Args:
            task: Training task dict
            forecast_horizon: Forecast horizon in days

        Returns:
            Result dict with status, metrics, error (if failed)
        """
        symbol = task["symbol"]
        start_date = task["start_date"]
        end_date = task["end_date"]
        window_label = task["window_label"]

        logger.info(f"Training {window_label}: {start_date} to {end_date}")

        # Build command to invoke train_teacher.py
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
        ]

        try:
            # Execute training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per window
            )

            if result.returncode == 0:
                logger.info(f"✓ {window_label} completed successfully")
                return {
                    "status": "success",
                    "metrics": self._extract_metrics_from_output(result.stdout),
                }
            else:
                logger.error(f"✗ {window_label} failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr[:500],  # Truncate long errors
                    "metrics": None,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {window_label} timed out")
            return {
                "status": "failed",
                "error": "Training timeout (10 minutes exceeded)",
                "metrics": None,
            }
        except Exception as e:
            logger.error(f"✗ {window_label} error: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "metrics": None,
            }

    def _extract_metrics_from_output(self, stdout: str) -> dict[str, float] | None:
        """Extract metrics from training output.

        Args:
            stdout: Training script stdout

        Returns:
            Dict with precision, recall, f1 or None if not found
        """
        # Simple parsing - look for common metrics patterns
        # This is a placeholder; real implementation would parse structured output
        metrics = {}

        for line in stdout.split("\n"):
            if "precision" in line.lower():
                try:
                    # Try to extract number
                    parts = line.split(":")
                    if len(parts) >= 2:
                        value = float(parts[-1].strip().rstrip("%"))
                        if value > 1:  # If percentage
                            value /= 100.0
                        metrics["precision"] = value
                except (ValueError, IndexError):
                    pass

        # Return metrics if we found any, otherwise None
        return metrics if metrics else None

    def train_window_with_retry(
        self,
        task: dict[str, Any],
        forecast_horizon: int,
    ) -> dict[str, Any]:
        """Train window with retry logic (US-024 Phase 5).

        Args:
            task: Training task dict
            forecast_horizon: Forecast horizon in days

        Returns:
            Result dict with status, metrics, error, attempts
        """
        task_id = task["window_label"]
        max_attempts = self.settings.parallel_retry_limit
        backoff_seconds = self.settings.parallel_retry_backoff_seconds

        for attempt in range(1, max_attempts + 1):
            self.task_attempts[task_id] = attempt

            if attempt > 1:
                self.stats["retries"] += 1
                logger.info(
                    f"Retry {attempt}/{max_attempts} for {task_id} (waiting {backoff_seconds}s...)"
                )
                time.sleep(backoff_seconds)

            result = self.train_window(task, forecast_horizon)

            if result["status"] == "success":
                result["attempts"] = attempt
                return result

            # Log retry if not the last attempt
            if attempt < max_attempts:
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed for {task_id}: "
                    f"{result.get('error', 'Unknown error')}"
                )

        # All retries exhausted
        result["attempts"] = max_attempts
        logger.error(f"All {max_attempts} attempts failed for {task_id}")

        # Record failure in state manager
        self.state_manager.record_task_failure(
            batch_id=self.batch_id,
            task_id=task_id,
            symbol=task["symbol"],
            window_label=task["window_label"],
            reason=result.get("error", "Unknown error"),
            attempts=max_attempts,
        )

        return result

    def log_metadata(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Log batch metadata to JSON Lines file (thread-safe for Phase 5).

        Args:
            task: Training task dict
            result: Training result dict
        """
        metadata = {
            "batch_id": self.batch_id,
            "symbol": task["symbol"],
            "date_range": {
                "start": task["start_date"],
                "end": task["end_date"],
            },
            "window_label": task["window_label"],
            "artifacts_path": task["artifacts_path"],
            "metrics": result.get("metrics"),
            "status": result["status"],
            "timestamp": datetime.now().isoformat(),
            "incremental": getattr(self, "incremental", False),  # US-024 Phase 4
            "attempts": result.get("attempts", 1),  # US-024 Phase 5
        }

        if result["status"] == "failed":
            metadata["error"] = result.get("error")

        # US-024 Phase 3: Check for sentiment snapshots
        sentiment_dir = Path(self.settings.sentiment_snapshot_output_dir) / task["symbol"]
        if sentiment_dir.exists() and any(sentiment_dir.glob("*.jsonl")):
            metadata["sentiment_snapshot_path"] = str(sentiment_dir)
            metadata["sentiment_available"] = True
        else:
            metadata["sentiment_available"] = False
            if self.settings.sentiment_snapshot_enabled:
                logger.warning(
                    f"Sentiment snapshots not found for {task['symbol']} at {sentiment_dir}",
                    extra={
                        "symbol": task["symbol"],
                        "sentiment_dir": str(sentiment_dir),
                        "warning": "sentiment_missing",
                    },
                )

        # Append to JSON Lines file (thread-safe with lock)
        with self.metadata_lock:
            with open(self.metadata_file, "a") as f:
                f.write(json.dumps(metadata) + "\n")

    def run_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        window_days: int,
        forecast_horizon: int,
    ) -> dict[str, Any]:
        """Run batch training (US-024 Phases 1-5).

        Args:
            symbols: List of stock symbols
            start_date: Overall start date (YYYY-MM-DD)
            end_date: Overall end date (YYYY-MM-DD)
            window_days: Size of each training window in days
            forecast_horizon: Forecast horizon in days

        Returns:
            Summary dict with statistics
        """
        # Generate training tasks
        tasks = self.generate_training_windows(symbols, start_date, end_date, window_days)
        self.stats["total_windows"] = len(tasks)

        logger.info(f"Generated {len(tasks)} training windows")

        # Initialize metadata file
        self.metadata_file.write_text("")  # Clear file

        # Initialize batch status in state manager
        self.state_manager.set_batch_status(
            batch_id=self.batch_id,
            status="running",
            total_tasks=len(tasks),
        )

        # Filter out already-trained windows if resume
        if self.resume:
            tasks_to_run = [t for t in tasks if not self.is_already_trained(t)]
            skipped_count = len(tasks) - len(tasks_to_run)
            self.stats["skipped"] = skipped_count
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} already-trained windows (resume mode)")
        else:
            tasks_to_run = tasks

        # US-024 Phase 5: Parallel or sequential execution
        if self.workers > 1 and len(tasks_to_run) > 1:
            logger.info(f"Running in parallel mode with {self.workers} workers")
            self._run_parallel(tasks_to_run, forecast_horizon)
        else:
            logger.info("Running in sequential mode")
            self._run_sequential(tasks_to_run, forecast_horizon)

        # Update final batch status
        final_status = "completed" if self.stats["failed"] == 0 else "partial"
        self.state_manager.set_batch_status(
            batch_id=self.batch_id,
            status=final_status,
            total_tasks=len(tasks),
            completed=self.stats["completed"],
            failed=self.stats["failed"],
            pending_retries=0,
            failed_tasks=self.state_manager.get_failed_tasks(self.batch_id),
        )

        # Generate summary
        summary = {
            "batch_id": self.batch_id,
            "batch_dir": str(self.batch_dir),
            "total_windows": self.stats["total_windows"],
            "completed": self.stats["completed"],
            "failed": self.stats["failed"],
            "skipped": self.stats["skipped"],
            "retries": self.stats["retries"],
            "success_rate": (
                self.stats["completed"] / self.stats["total_windows"]
                if self.stats["total_windows"] > 0
                else 0.0
            ),
        }

        return summary

    def _run_sequential(
        self,
        tasks: list[dict[str, Any]],
        forecast_horizon: int,
    ) -> None:
        """Run tasks sequentially.

        Args:
            tasks: List of training tasks
            forecast_horizon: Forecast horizon in days
        """
        for i, task in enumerate(tasks, 1):
            logger.info(f"[{i}/{len(tasks)}] Processing {task['window_label']}")

            # Train window with retry
            result = self.train_window_with_retry(task, forecast_horizon)

            # Log metadata
            self.log_metadata(task, result)

            # Update stats
            if result["status"] == "success":
                self.stats["completed"] += 1
            else:
                self.stats["failed"] += 1

    def _run_parallel(
        self,
        tasks: list[dict[str, Any]],
        forecast_horizon: int,
    ) -> None:
        """Run tasks in parallel using ProcessPoolExecutor.

        Args:
            tasks: List of training tasks
            forecast_horizon: Forecast horizon in days
        """
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._train_window_worker,
                    task,
                    forecast_horizon,
                    self.settings,
                ): task
                for task in tasks
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[future]
                try:
                    result = future.result()
                    logger.info(
                        f"[{i}/{len(tasks)}] Completed {task['window_label']}: {result['status']}"
                    )

                    # Log metadata
                    self.log_metadata(task, result)

                    # Update stats
                    if result["status"] == "success":
                        self.stats["completed"] += 1
                    else:
                        self.stats["failed"] += 1

                except Exception as e:
                    logger.error(f"Worker exception for {task['window_label']}: {e}")
                    result = {
                        "status": "failed",
                        "error": f"Worker exception: {str(e)}",
                        "metrics": None,
                        "attempts": 1,
                    }
                    self.log_metadata(task, result)
                    self.stats["failed"] += 1

    @staticmethod
    def _train_window_worker(
        task: dict[str, Any],
        forecast_horizon: int,
        settings: Settings,
    ) -> dict[str, Any]:
        """Worker function for parallel execution.

        This is a static method so it can be pickled for ProcessPoolExecutor.

        Args:
            task: Training task dict
            forecast_horizon: Forecast horizon in days
            settings: Application settings

        Returns:
            Result dict with status, metrics, error, attempts
        """
        max_attempts = settings.parallel_retry_limit
        backoff_seconds = settings.parallel_retry_backoff_seconds

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(backoff_seconds)

            # Build command
            cmd = [
                sys.executable,
                "scripts/train_teacher.py",
                "--symbol",
                task["symbol"],
                "--start",
                task["start_date"],
                "--end",
                task["end_date"],
                "--window",
                str(forecast_horizon),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                )

                if result.returncode == 0:
                    return {
                        "status": "success",
                        "metrics": None,  # Could parse from stdout
                        "attempts": attempt,
                    }
                else:
                    error = result.stderr[:500]
                    if attempt < max_attempts:
                        continue  # Retry
                    return {
                        "status": "failed",
                        "error": error,
                        "metrics": None,
                        "attempts": attempt,
                    }

            except subprocess.TimeoutExpired:
                if attempt < max_attempts:
                    continue  # Retry
                return {
                    "status": "failed",
                    "error": "Training timeout (10 minutes exceeded)",
                    "metrics": None,
                    "attempts": attempt,
                }
            except Exception as e:
                if attempt < max_attempts:
                    continue  # Retry
                return {
                    "status": "failed",
                    "error": str(e),
                    "metrics": None,
                    "attempts": attempt,
                }

        # Should never reach here
        return {
            "status": "failed",
            "error": "Unknown error after all retries",
            "metrics": None,
            "attempts": max_attempts,
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Teacher training across multiple symbols and windows"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Stock symbols to train (default: from settings)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Overall start date (YYYY-MM-DD, default: from settings)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Overall end date (YYYY-MM-DD, default: from settings)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        help="Training window size in days (default: from settings)",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        help="Forecast horizon in days (default: from settings)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partial batch (skip already-trained windows)",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode (train only windows with new data)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load settings
    settings = Settings()  # type: ignore[call-arg]

    # Use command-line args or settings defaults
    symbols = args.symbols if args.symbols else settings.historical_data_symbols
    start_date = args.start_date if args.start_date else settings.historical_data_start_date
    end_date = args.end_date if args.end_date else settings.historical_data_end_date
    window_days = args.window_days if args.window_days else settings.batch_training_window_days
    forecast_horizon = (
        args.forecast_horizon
        if args.forecast_horizon
        else settings.batch_training_forecast_horizon_days
    )
    output_dir = Path(settings.batch_training_output_dir)

    logger.info("=" * 70)
    logger.info("BATCH TEACHER TRAINING (US-024 Phase 5)")
    logger.info("=" * 70)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Window size: {window_days} days")
    logger.info(f"Forecast horizon: {forecast_horizon} days")
    logger.info(f"Output dir: {output_dir}")
    logger.info(
        f"Workers: {args.workers} ({('parallel' if args.workers > 1 else 'sequential')} mode)"
    )
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Incremental: {args.incremental}")
    logger.info(f"Max retries: {settings.parallel_retry_limit}")
    logger.info("=" * 70)

    # Create batch trainer
    trainer = BatchTrainer(
        settings,
        output_dir,
        resume=args.resume,
        incremental=args.incremental,
        workers=args.workers,
    )

    # Run batch
    try:
        summary = trainer.run_batch(symbols, start_date, end_date, window_days, forecast_horizon)
    except Exception as e:
        logger.error(f"Fatal error during batch training: {e}")
        return 1

    # Print summary
    logger.info("=" * 70)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Batch ID: {summary['batch_id']}")
    logger.info(f"Batch directory: {summary['batch_dir']}")
    logger.info(f"Total windows: {summary['total_windows']}")
    logger.info(f"Completed: {summary['completed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Skipped: {summary['skipped']}")
    logger.info(f"Retries: {summary['retries']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info("=" * 70)
    logger.info(f"Metadata: {trainer.metadata_file}")

    # Log failed tasks for manual review (US-024 Phase 5)
    failed_tasks = trainer.state_manager.get_failed_tasks(summary["batch_id"])
    if failed_tasks:
        logger.info("=" * 70)
        logger.warning(f"FAILED TASKS REQUIRING MANUAL REVIEW: {len(failed_tasks)}")
        for task in failed_tasks:
            logger.warning(
                f"  - {task['symbol']}/{task['window_label']}: "
                f"{task['reason']} (after {task['attempts']} attempts)"
            )
        logger.info("=" * 70)

    logger.info("=" * 70)

    # Return exit code based on failures
    if summary["failed"] > 0:
        logger.warning(
            f"{summary['failed']} training windows failed after {settings.parallel_retry_limit} retries"
        )
        logger.warning("Check batch status in data/state/teacher_batch.json for details")
        return 1

    logger.info("All training windows completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
