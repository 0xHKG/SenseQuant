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
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from tqdm import tqdm

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

        # Create batch directory (use batch_id to ensure consistency)
        self.batch_dir = output_dir / self.batch_id
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

        # US-028 Phase 7 Batch 4: Multi-GPU support
        self.available_gpus = self._detect_gpus()
        self.num_gpus = len(self.available_gpus)

        if self.num_gpus > 0:
            logger.info(f"Detected {self.num_gpus} GPU(s): {self.available_gpus}")
        else:
            logger.warning("No GPUs detected - training will use CPU (slow)")

        if self.workers > 1 and self.num_gpus == 0:
            logger.warning(
                "Multi-worker mode requested but no GPUs detected. "
                "Training will use CPU (significantly slower)."
            )
        elif self.workers > self.num_gpus > 0:
            logger.info(
                f"Workers ({self.workers}) > GPUs ({self.num_gpus}). "
                f"Multiple workers will share GPUs via round-robin assignment."
            )

    def _detect_gpus(self) -> list[int]:
        """Detect available CUDA GPUs (US-028 Phase 7 Batch 4).

        Returns:
            List of GPU device IDs (e.g., [0, 1] for 2 GPUs)
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_ids = [
                int(line.strip())
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            return gpu_ids
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-smi not found or failed, assuming no GPUs")
            return []
        except Exception as e:
            logger.error(f"Failed to detect GPUs: {e}")
            return []

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
                    # Generate deterministic window label with explicit dates
                    # Format: SYMBOL_YYYY-MM-DD_to_YYYY-MM-DD
                    # This ensures uniqueness even when windows don't align with quarters
                    window_label = (
                        f"{symbol}_{current_start.strftime('%Y-%m-%d')}_to_"
                        f"{current_end.strftime('%Y-%m-%d')}"
                    )

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
        # US-028 Phase 6o: Check for new standardized artifact filenames
        required_files = [
            artifacts_path / "model.pkl",
            artifacts_path / "labels.csv.gz",
            artifacts_path / "metadata.json",
        ]
        return all(f.exists() for f in required_files)

    def get_latest_available_timestamp(self, symbol: str) -> datetime | None:
        """Get the latest available timestamp for a symbol from cached data.

        Args:
            symbol: Stock symbol

        Returns:
            Latest timestamp or None if no data found
        """
        data_dir = Path(self.settings.historical_data_output_dir) / symbol / "1day"

        if not data_dir.exists():
            return None

        # Find all CSV files
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            return None

        # Load the last file (sorted by name = sorted by date)
        try:
            df = pd.read_csv(csv_files[-1])
            if len(df) == 0 or "timestamp" not in df.columns:
                return None

            # Parse timestamp (handles both formats)
            last_ts = pd.to_datetime(df["timestamp"].iloc[-1], format="ISO8601")
            return last_ts.to_pydatetime()  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(
                f"Failed to read latest timestamp for {symbol}: {e}",
                extra={"component": "batch_trainer"},
            )
            return None

    def should_skip_window_insufficient_data(
        self, task: dict[str, Any], forecast_horizon: int
    ) -> tuple[bool, str]:
        """Check if a window should be skipped due to insufficient future data.

        Args:
            task: Training task dict
            forecast_horizon: Forward-looking label window in days

        Returns:
            Tuple of (should_skip, reason)
        """
        symbol = task["symbol"]
        end_date = datetime.strptime(task["end_date"], "%Y-%m-%d")

        # Get latest available data
        latest_ts = self.get_latest_available_timestamp(symbol)

        if latest_ts is None:
            reason = f"No historical data found for {symbol}"
            return True, reason

        # Remove timezone info for comparison
        if latest_ts.tzinfo is not None:
            latest_ts = latest_ts.replace(tzinfo=None)

        # Calculate required future date for label generation
        required_future_date = end_date + timedelta(days=forecast_horizon)

        # Check if we have enough future data
        if required_future_date > latest_ts:
            days_short = (required_future_date - latest_ts).days
            reason = (
                f"Insufficient future data: need data through {required_future_date.date()}, "
                f"but only have through {latest_ts.date()} ({days_short} days short)"
            )
            return True, reason

        return False, ""

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

        # US-028 Phase 6o: Create artifacts subdirectory
        artifacts_path = Path(task["artifacts_path"])
        artifacts_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training {window_label}: {start_date} to {end_date}")
        logger.info(f"  Artifacts directory: {artifacts_path}")

        # US-028 Phase 6o: Build command with --output-dir
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
            "--output-dir",
            str(artifacts_path),
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
                # Extract sample diagnostics (US-028 Phase 6f)
                sample_counts = self._extract_sample_diagnostics(result.stdout)

                # Check for zero-sample case (US-028 Phase 6f)
                if sample_counts and sample_counts.get("total_samples", 0) == 0:
                    skip_reason = (
                        f"Insufficient samples: 0 total samples after filtering "
                        f"(train={sample_counts.get('train_samples', 0)}, "
                        f"val={sample_counts.get('val_samples', 0)})"
                    )
                    logger.warning(
                        f"⊘ {window_label} skipped: {skip_reason}",
                        extra={
                            "component": "batch_trainer",
                            "status": "skipped",
                            "reason": "insufficient_samples",
                        },
                    )
                    return {
                        "status": "skipped",
                        "reason": skip_reason,
                        "sample_counts": sample_counts,
                        "metrics": None,
                    }

                logger.info(f"✓ {window_label} completed successfully")
                return {
                    "status": "success",
                    "metrics": self._extract_metrics_from_output(result.stdout),
                    "sample_counts": sample_counts,  # Include diagnostics in success case
                }
            elif result.returncode == 2:
                # US-028 Phase 6h: Exit code 2 indicates skip condition (insufficient samples)
                skip_info = self._extract_skip_info(result.stdout)
                skip_reason = skip_info.get("reason", "Unknown skip reason")
                logger.warning(
                    f"⊘ {window_label} skipped: {skip_reason}",
                    extra={
                        "component": "batch_trainer",
                        "status": "skipped",
                        "reason": "insufficient_samples",
                    },
                )
                return {
                    "status": "skipped",
                    "reason": skip_reason,
                    "sample_counts": None,
                    "metrics": None,
                }
            else:
                # Capture full error details from subprocess
                error_message = result.stderr.strip() if result.stderr else "No stderr output"
                stdout_tail = result.stdout[-500:] if result.stdout else ""

                # Log detailed error with window context
                logger.error(
                    f"✗ {window_label} failed (exit code {result.returncode})",
                    extra={
                        "component": "batch_trainer",
                        "window_label": window_label,
                        "symbol": symbol,
                        "exit_code": result.returncode,
                    },
                )
                logger.error(f"  stderr: {error_message}")
                if stdout_tail:
                    logger.error(f"  stdout (last 500 chars): {stdout_tail}")

                return {
                    "status": "failed",
                    "error": error_message,
                    "error_detail": {
                        "exit_code": result.returncode,
                        "stderr": error_message,
                        "stdout_tail": stdout_tail,
                    },
                    "metrics": None,
                }

        except subprocess.TimeoutExpired:
            error_message = f"Training timeout ({600}s exceeded)"
            logger.error(
                f"✗ {window_label} timed out",
                extra={
                    "component": "batch_trainer",
                    "window_label": window_label,
                    "symbol": symbol,
                },
            )
            return {
                "status": "failed",
                "error": error_message,
                "error_detail": {"timeout_seconds": 600},
                "metrics": None,
            }
        except Exception as e:
            # Capture full traceback for unexpected exceptions
            tb_str = traceback.format_exc()
            error_message = f"{type(e).__name__}: {str(e)}"

            logger.error(
                f"✗ {window_label} error: {error_message}",
                extra={
                    "component": "batch_trainer",
                    "window_label": window_label,
                    "symbol": symbol,
                },
            )
            logger.error(f"  Traceback:\n{tb_str}")

            return {
                "status": "failed",
                "error": error_message,
                "error_detail": {
                    "exception_type": type(e).__name__,
                    "traceback": tb_str,
                },
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

    def _extract_skip_info(self, stdout: str) -> dict[str, Any]:
        """Extract skip information from training output (US-028 Phase 6h).

        Args:
            stdout: Training script stdout

        Returns:
            Dict with skip reason, or empty dict if not found
        """
        import json

        for line in stdout.split("\n"):
            if "TEACHER_SKIP:" in line:
                try:
                    json_str = line.split("TEACHER_SKIP:", 1)[1].strip()
                    skip_info = json.loads(json_str)
                    return skip_info  # type: ignore[no-any-return]
                except (ValueError, IndexError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse teacher skip info: {e}")
        return {}

    def _extract_sample_diagnostics(self, stdout: str) -> dict[str, Any] | None:
        """Extract sample count diagnostics from training output (US-028 Phase 6f).

        Args:
            stdout: Training script stdout

        Returns:
            Dict with sample counts or None if not found
        """
        import json

        # Look for TEACHER_DIAGNOSTICS line in stdout
        for line in stdout.split("\n"):
            if "TEACHER_DIAGNOSTICS:" in line:
                try:
                    # Extract JSON after the prefix
                    json_str = line.split("TEACHER_DIAGNOSTICS:", 1)[1].strip()
                    diagnostics = json.loads(json_str)
                    return diagnostics.get("sample_counts")  # type: ignore[no-any-return]
                except (ValueError, IndexError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse teacher diagnostics: {e}")
                    pass

        return None

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

        # US-028 Phase 6f: Include sample diagnostics if available
        if "sample_counts" in result and result["sample_counts"]:
            metadata["sample_counts"] = result["sample_counts"]

        if result["status"] == "failed":
            metadata["error"] = result.get("error")

        # US-028 Phase 6f: Include skip reason for skipped windows
        if result["status"] == "skipped":
            metadata["skip_reason"] = result.get("reason", "Unknown")

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

        # US-029 Phase 2: Track market feature configuration
        metadata["feature_set"] = {
            "order_book_enabled": self.settings.enable_order_book_features,
            "options_enabled": self.settings.enable_options_features,
            "macro_enabled": self.settings.enable_macro_features,
        }

        # US-028 Phase 7: Track LightGBM GPU hyperparameters for profiling experiments
        metadata["teacher_hyperparameters"] = {
            "num_leaves": getattr(self.settings, "teacher_num_leaves", 127),
            "max_depth": getattr(self.settings, "teacher_max_depth", 9),
            "n_estimators": getattr(self.settings, "teacher_n_estimators", 500),
            "learning_rate": getattr(self.settings, "teacher_learning_rate", 0.01),
            "min_child_samples": getattr(self.settings, "teacher_min_child_samples", 20),
            "subsample": getattr(self.settings, "teacher_subsample", 0.8),
            "colsample_bytree": getattr(self.settings, "teacher_colsample_bytree", 0.8),
            "gpu_use_dp": getattr(self.settings, "teacher_gpu_use_dp", False),
            "gpu_platform_id": getattr(self.settings, "teacher_gpu_platform_id", 0),
            "gpu_device_id": getattr(self.settings, "teacher_gpu_device_id", 0),
        }

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
        # US-028 Phase 7 Initiative 4: Progress monitoring with tqdm
        with tqdm(total=len(tasks), desc="Training teacher models", unit="window") as pbar:
            for i, task in enumerate(tasks, 1):
                logger.info(f"[{i}/{len(tasks)}] Processing {task['window_label']}")

                # Check if window should be skipped due to insufficient future data
                should_skip, skip_reason = self.should_skip_window_insufficient_data(
                    task, forecast_horizon
                )

                if should_skip:
                    logger.warning(
                        f"⊘ {task['window_label']} skipped: {skip_reason}",
                        extra={
                            "component": "batch_trainer",
                            "status": "skipped",
                            "reason": "insufficient_future_data",
                        },
                    )

                    # Log skip metadata
                    skip_result = {
                        "status": "skipped",
                        "reason": skip_reason,
                        "metrics": None,
                    }
                    self.log_metadata(task, skip_result)

                    # Update stats
                    self.stats["skipped"] += 1
                    # US-028 Phase 7 Initiative 4: Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "window": task["window_label"][:20],
                        "status": "skipped",
                        "trained": self.stats["completed"],
                        "skipped": self.stats["skipped"],
                        "failed": self.stats["failed"],
                    })
                    continue

                # Train window with retry
                result = self.train_window_with_retry(task, forecast_horizon)

                # Log metadata
                self.log_metadata(task, result)

                # Update stats
                if result["status"] == "success":
                    self.stats["completed"] += 1
                elif result["status"] == "skipped":
                    self.stats["skipped"] += 1
                else:
                    self.stats["failed"] += 1

                # US-028 Phase 7 Initiative 4: Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "window": task["window_label"][:20],
                    "status": result["status"],
                    "trained": self.stats["completed"],
                    "skipped": self.stats["skipped"],
                    "failed": self.stats["failed"],
                })

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
        # Pre-filter tasks to skip those with insufficient data
        tasks_to_run = []
        for task in tasks:
            should_skip, skip_reason = self.should_skip_window_insufficient_data(
                task, forecast_horizon
            )

            if should_skip:
                logger.warning(
                    f"⊘ {task['window_label']} skipped: {skip_reason}",
                    extra={
                        "component": "batch_trainer",
                        "status": "skipped",
                        "reason": "insufficient_future_data",
                    },
                )

                # Log skip metadata
                skip_result = {
                    "status": "skipped",
                    "reason": skip_reason,
                    "metrics": None,
                }
                self.log_metadata(task, skip_result)
                self.stats["skipped"] += 1
            else:
                tasks_to_run.append(task)

        if not tasks_to_run:
            logger.warning("No tasks to run after skipping windows with insufficient data")
            return

        # US-028 Phase 7 Initiative 4: Progress monitoring for parallel execution
        # US-028 Phase 7 Batch 4: Multi-GPU support with round-robin assignment
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit filtered tasks with GPU assignment
            future_to_task = {}
            gpu_assignments = {}  # Track GPU distribution for logging

            for task_idx, task in enumerate(tasks_to_run):
                # Round-robin GPU assignment
                if self.num_gpus > 0:
                    gpu_id = self.available_gpus[task_idx % self.num_gpus]
                    gpu_assignments[gpu_id] = gpu_assignments.get(gpu_id, 0) + 1
                else:
                    gpu_id = None  # CPU fallback

                future = executor.submit(
                    self._train_window_worker,
                    task,
                    forecast_horizon,
                    self.settings,
                    gpu_id,  # GPU assignment
                )
                future_to_task[future] = (task, gpu_id)

            # Log GPU distribution
            if self.num_gpus > 0:
                logger.info(f"GPU task distribution: {dict(gpu_assignments)}")

            # Process completed tasks with progress bar
            with tqdm(total=len(tasks_to_run), desc="Training teacher models (parallel)", unit="window") as pbar:
                for i, future in enumerate(as_completed(future_to_task), 1):
                    task, assigned_gpu = future_to_task[future]
                    try:
                        result = future.result()
                        gpu_info = f" (GPU{assigned_gpu})" if assigned_gpu is not None else " (CPU)"
                        logger.info(
                            f"[{i}/{len(tasks_to_run)}] Completed {task['window_label']}: "
                            f"{result['status']}{gpu_info}"
                        )

                        # Log metadata
                        self.log_metadata(task, result)

                        # Update stats
                        if result["status"] == "success":
                            self.stats["completed"] += 1
                        elif result["status"] == "skipped":
                            self.stats["skipped"] += 1
                        else:
                            self.stats["failed"] += 1

                        # US-028 Phase 7 Initiative 4: Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            "window": task["window_label"][:20],
                            "status": result["status"],
                            "trained": self.stats["completed"],
                            "skipped": self.stats["skipped"],
                            "failed": self.stats["failed"],
                        })

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

                        # US-028 Phase 7 Initiative 4: Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            "window": task["window_label"][:20],
                            "status": "failed",
                            "trained": self.stats["completed"],
                            "skipped": self.stats["skipped"],
                            "failed": self.stats["failed"],
                        })

    @staticmethod
    def _train_window_worker(
        task: dict[str, Any],
        forecast_horizon: int,
        settings: Settings,
        gpu_id: int | None = None,
    ) -> dict[str, Any]:
        """Worker function for parallel execution.

        This is a static method so it can be pickled for ProcessPoolExecutor.

        Args:
            task: Training task dict
            forecast_horizon: Forecast horizon in days
            settings: Application settings
            gpu_id: GPU device ID to use (None = CPU fallback) - US-028 Phase 7 Batch 4

        Returns:
            Result dict with status, metrics, error, attempts
        """
        # US-028 Phase 7 Batch 4: Pin process to specific GPU
        import os

        # Prepare environment with GPU assignment
        # CRITICAL: Must set CUDA_VISIBLE_DEVICES in subprocess env, not parent process
        worker_env = os.environ.copy()
        if gpu_id is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(
                f"Worker assigned to GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})",
                extra={"symbol": task["symbol"], "window": task["window_label"]},
            )
        else:
            # CPU fallback (remove GPU from visibility)
            worker_env["CUDA_VISIBLE_DEVICES"] = ""
            logger.warning(
                "Worker using CPU (no GPU assigned)",
                extra={"symbol": task["symbol"], "window": task["window_label"]},
            )

        max_attempts = settings.parallel_retry_limit
        backoff_seconds = settings.parallel_retry_backoff_seconds

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(backoff_seconds)

            # Build command with explicit GPU device ID (US-028 Phase 7: Multi-GPU fix)
            # US-028 Phase 7 Afternoon: Add --output-dir flag (parallel mode fix)
            artifacts_path = Path(task["artifacts_path"])
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
                "--output-dir",
                str(artifacts_path),
            ]

            # Add explicit GPU device ID to ensure LightGBM uses correct GPU
            if gpu_id is not None:
                cmd.extend(["--gpu-device-id", str(gpu_id)])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    env=worker_env,  # Pass GPU assignment via environment
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

    parser.add_argument(
        "--max-failure-rate",
        type=float,
        help="Max acceptable failure rate (0.0-1.0, default: from settings)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load settings
    settings = Settings()  # type: ignore[call-arg]

    # Override max_failure_rate if provided via CLI (US-028 Phase 7)
    if args.max_failure_rate is not None:
        if not 0.0 <= args.max_failure_rate <= 1.0:
            logger.error(f"--max-failure-rate must be between 0.0 and 1.0 (got {args.max_failure_rate})")
            return 1
        settings.batch_training_max_failure_rate = args.max_failure_rate

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

    # Return exit code based on failure rate threshold (US-028 Phase 7 hardening)
    total_windows = summary["total_windows"]
    failed_windows = summary["failed"]
    failure_rate = failed_windows / total_windows if total_windows > 0 else 0.0
    max_failure_rate = settings.batch_training_max_failure_rate

    logger.info(
        f"Batch completion: {summary['completed']}/{total_windows} succeeded, "
        f"{failed_windows} failed, {summary['skipped']} skipped"
    )
    logger.info(f"Failure rate: {failure_rate:.2%} (threshold: {max_failure_rate:.2%})")

    if failure_rate > max_failure_rate:
        logger.error(
            f"Failure rate {failure_rate:.2%} exceeds threshold {max_failure_rate:.2%}. "
            f"{failed_windows}/{total_windows} windows failed after {settings.parallel_retry_limit} retries."
        )
        logger.error("Check batch status in data/state/teacher_batch.json for details")
        return 1

    if failed_windows > 0:
        logger.warning(
            f"Batch completed with {failed_windows} expected failures ({failure_rate:.2%} "
            f"≤ threshold {max_failure_rate:.2%})"
        )
    else:
        logger.info("All training windows completed successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
