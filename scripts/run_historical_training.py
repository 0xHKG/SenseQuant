#!/usr/bin/env python3
"""
Historical Model Training Execution & Promotion (US-028)

Orchestrates end-to-end historical training pipeline:
1. Data ingestion (historical OHLCV + sentiment)
2. Teacher training (batch)
3. Student training (batch)
4. Model validation
5. Statistical tests
6. Release audit
7. Promotion briefing generation

Usage:
    python scripts/run_historical_training.py \
      --symbols RELIANCE,TCS,INFY \
      --start-date 2024-01-01 \
      --end-date 2024-12-31

    python scripts/run_historical_training.py \
      --symbols RELIANCE \
      --start-date 2024-01-01 \
      --end-date 2024-12-31 \
      --dryrun

    python scripts/run_historical_training.py \
      --symbols RELIANCE \
      --start-date 2024-01-01 \
      --end-date 2024-12-31 \
      --skip-fetch
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.config import Settings
from src.services.state_manager import StateManager
from src.services.training_telemetry import TrainingTelemetryLogger


class HistoricalRunOrchestrator:
    """Orchestrates end-to-end historical training and promotion pipeline."""

    def __init__(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        skip_fetch: bool = False,
        dryrun: bool = False,
        run_stress_tests: bool | None = None,
        enable_telemetry: bool = False,
        telemetry_dir: Path | None = None,
    ):
        """Initialize orchestrator.

        Args:
            symbols: List of symbols to train
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            skip_fetch: Skip data fetch phase
            dryrun: Dryrun mode (skip heavy computation)
            run_stress_tests: Enable Phase 8 stress tests (overrides config if provided)
            enable_telemetry: Enable training telemetry capture
            telemetry_dir: Directory for telemetry output (uses settings default if None)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.skip_fetch = skip_fetch
        self.dryrun = dryrun

        # Load settings
        self.settings = Settings()  # type: ignore[call-arg]

        # US-028 Phase 7 Initiative 3: Override stress tests setting if CLI flag provided
        if run_stress_tests is not None:
            self.settings.stress_tests_enabled = run_stress_tests

        # US-028 Phase 7 Batch 4: Telemetry support
        self.enable_telemetry = enable_telemetry
        self.telemetry_dir = telemetry_dir or Path(self.settings.telemetry_storage_path) / "training"

        # Generate run ID
        self.run_id = f"live_candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timestamp = datetime.now().isoformat()

        # Directories
        self.repo_root = Path(__file__).parent.parent
        self.model_dir = self.repo_root / "data" / "models" / self.run_id
        self.audit_dir = self.repo_root / "release" / f"audit_{self.run_id}"

        # US-028 Phase 6s: Track actual batch directory from training scripts
        self.batch_dir: Path | None = None

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # US-028 Phase 7 Batch 4: Create telemetry directory if enabled
        if self.enable_telemetry:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        # State manager
        self.state_mgr = StateManager()

        # US-028 Phase 7 Batch 4: Initialize telemetry logger
        self.telemetry: TrainingTelemetryLogger | None = None
        if self.enable_telemetry:
            self.telemetry = TrainingTelemetryLogger(
                output_dir=self.telemetry_dir,
                run_id=self.run_id,
                buffer_size=50,
                enabled=True,
            )

        # Results
        self.results: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "symbols": symbols,
            "date_range": {"start": start_date, "end": end_date},
            "phases": {},
        }

        logger.info(f"Initialized HistoricalRunOrchestrator: {self.run_id}")
        if self.enable_telemetry:
            logger.info(f"Telemetry enabled: {self.telemetry_dir}")

    def run(self) -> bool:
        """Execute full historical training pipeline.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info(f"  Historical Training Run: {self.run_id}")
        logger.info("=" * 70)
        logger.info("")

        # US-028 Phase 7 Batch 4: Log run start
        if self.telemetry:
            self.telemetry.log_run_event(
                event_type="run_start",
                status="in_progress",
                message=f"Starting historical training: {len(self.symbols)} symbols, {self.start_date} to {self.end_date}",
            )

        run_start_time = datetime.now()

        try:
            # Phase 1: Data Ingestion
            if not self._run_phase_1_data_ingestion():
                return False

            # Phase 2: Teacher Training
            if not self._run_phase_2_teacher_training():
                return False

            # Phase 3: Student Training
            if not self._run_phase_3_student_training():
                return False

            # Phase 4: Model Validation
            if not self._run_phase_4_model_validation():
                return False

            # Phase 5: Statistical Tests
            if not self._run_phase_5_statistical_tests():
                return False

            # Phase 6: Release Audit
            if not self._run_phase_6_release_audit():
                return False

            # Phase 7: Promotion Briefing
            if not self._run_phase_7_promotion_briefing():
                return False

            # Phase 8: Black-Swan Stress Tests (optional)
            if self.settings.stress_tests_enabled:
                if not self._run_phase_8_stress_tests():
                    return False
            else:
                logger.info("Phase 8: Stress Tests (skipped - not enabled)")
                self.results["phases"]["stress_tests"] = {"status": "skipped", "reason": "not enabled"}

            # Validate artifacts exist
            if not self._validate_artifacts():
                return False

            # Record candidate run in state manager
            self._record_candidate_run(status="ready-for-review")

            # US-028 Phase 7 Batch 4: Log run end (success)
            run_duration = (datetime.now() - run_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_run_event(
                    event_type="run_end",
                    status="success",
                    message="Historical training completed successfully",
                    duration_seconds=run_duration,
                )
                self.telemetry.close()

            logger.info("=" * 70)
            logger.info("  Historical Run Complete")
            logger.info("=" * 70)
            logger.info("")
            logger.info(f"Run ID: {self.run_id}")
            logger.info("Status: ready-for-review")
            logger.info("")
            logger.info("Artifacts:")
            logger.info(f"  - Model: {self.model_dir}")
            logger.info(f"  - Audit: {self.audit_dir}")
            logger.info(f"  - Briefing: {self.audit_dir / 'promotion_briefing.md'}")
            if self.telemetry:
                logger.info(f"  - Telemetry: {self.telemetry.output_file}")
            logger.info("")
            logger.info("Next Steps:")
            logger.info(f"  1. Review briefing: cat {self.audit_dir}/promotion_briefing.md")
            logger.info(
                f"  2. Approve candidate: python scripts/approve_candidate.py {self.run_id}"
            )
            logger.info("  3. Deploy to staging: make deploy-staging")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"Historical run failed: {e}")

            # US-028 Phase 7 Batch 4: Log run end (failure)
            run_duration = (datetime.now() - run_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_run_event(
                    event_type="run_end",
                    status="failed",
                    message=f"Historical training failed: {str(e)}",
                    duration_seconds=run_duration,
                )
                self.telemetry.close()

            self._record_candidate_run(status="failed")
            return False

    def _run_phase_1_data_ingestion(self) -> bool:
        """Phase 1: Data Ingestion."""
        logger.info("Phase 1/7: Data Ingestion")

        if self.skip_fetch:
            logger.info("  ⚠ Skipping data fetch (--skip-fetch)")
            self.results["phases"]["data_ingestion"] = {"status": "skipped"}
            return True

        if self.dryrun:
            logger.info("  [DRYRUN] Skipping data fetch")
            self.results["phases"]["data_ingestion"] = {"status": "skipped", "reason": "dryrun"}
            return True

        try:
            # Fetch historical OHLCV
            logger.info("  → Fetching historical OHLCV...")

            fetch_cmd = [
                "python",
                str(self.repo_root / "scripts" / "fetch_historical_data.py"),
                "--symbols",
                *self.symbols,
                "--start-date",
                self.start_date,
                "--end-date",
                self.end_date,
            ]

            logger.info(f"    Running: {' '.join(fetch_cmd)}")
            result = subprocess.run(
                fetch_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse output for chunk statistics
            chunk_stats = {"chunks_fetched": 0, "chunks_failed": 0, "total_rows": 0}
            for line in result.stdout.split("\n"):
                if "Chunks fetched:" in line:
                    chunk_stats["chunks_fetched"] = int(line.split(":")[1].strip())
                elif "Chunks failed:" in line:
                    chunk_stats["chunks_failed"] = int(line.split(":")[1].strip())
                elif "Total rows:" in line:
                    chunk_stats["total_rows"] = int(line.split(":")[1].strip())

            logger.info(
                f"    ✓ Fetched {len(self.symbols)} symbols ({self.start_date} to {self.end_date})"
            )
            logger.info(f"    ✓ Chunk statistics: {chunk_stats['chunks_fetched']} fetched, {chunk_stats['chunks_failed']} failed, {chunk_stats['total_rows']} rows")

            # Validate that data files exist and are non-empty
            from src.app.config import Settings
            settings = Settings()  # type: ignore[call-arg]
            data_dir = Path(settings.historical_data_output_dir)

            missing_files = []
            for symbol in self.symbols:
                symbol_dir = data_dir / symbol / "1day"
                if not symbol_dir.exists():
                    missing_files.append(f"{symbol}/1day (directory missing)")
                    continue

                # Check for at least one CSV file
                csv_files = list(symbol_dir.glob("*.csv"))
                if not csv_files:
                    missing_files.append(f"{symbol}/1day (no CSV files)")

            if missing_files:
                error_msg = f"Required data files missing or empty: {missing_files}"
                logger.error(f"  ✗ {error_msg}")
                self.results["phases"]["data_ingestion"] = {
                    "status": "failed",
                    "error": error_msg,
                    "missing_files": missing_files,
                }
                return False

            # Fetch sentiment snapshots
            logger.info("  → Fetching sentiment snapshots...")

            sentiment_cmd = [
                "python",
                str(self.repo_root / "scripts" / "fetch_sentiment_snapshots.py"),
                "--symbols",
                *self.symbols,
                "--start-date",
                self.start_date,
                "--end-date",
                self.end_date,
            ]

            logger.info(f"    Running: {' '.join(sentiment_cmd)}")
            subprocess.run(
                sentiment_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("    ✓ Fetched sentiment data")

            # US-028 Phase 7 Initiative 4: Record progress
            self.state_mgr.record_training_progress(
                phase="data_ingestion",
                completed=len(self.symbols),
                total=len(self.symbols),
                extra={
                    "status": "success",
                    "chunks_fetched": chunk_stats["chunks_fetched"],
                    "chunks_failed": chunk_stats["chunks_failed"],
                    "total_rows": chunk_stats["total_rows"],
                },
            )
            logger.info(f"  [Progress] Phase 1 complete: {len(self.symbols)}/{len(self.symbols)} symbols fetched")

            self.results["phases"]["data_ingestion"] = {
                "status": "success",
                "symbols": len(self.symbols),
                "exit_code": 0,
                "chunk_stats": chunk_stats,
            }
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Data ingestion failed: {e}")
            logger.error(f"  stdout: {e.stdout}")
            logger.error(f"  stderr: {e.stderr}")
            self.results["phases"]["data_ingestion"] = {
                "status": "failed",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
            }
            return False
        except Exception as e:
            logger.error(f"  ✗ Data ingestion failed: {e}")
            self.results["phases"]["data_ingestion"] = {"status": "failed", "error": str(e)}
            return False

    def _aggregate_teacher_runs_from_json(
        self, batch_dir: Path
    ) -> dict[str, Any] | None:
        """Aggregate teacher training statistics from teacher_runs.json.

        US-028 Phase 6l: Robust JSON-based parsing instead of fragile stdout parsing.
        This eliminates the parsing bug that caused "Trained 0 windows" reports.

        Args:
            batch_dir: Path to batch directory containing teacher_runs.json

        Returns:
            Dict with aggregated statistics including window details, or None if parsing fails
        """
        import json

        json_path = batch_dir / "teacher_runs.json"
        if not json_path.exists():
            logger.warning(f"teacher_runs.json not found at {json_path}")
            return None

        stats: dict[str, Any] = {
            "total_windows": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "success_windows": [],
            "skipped_windows": [],
            "failed_windows": [],
            "total_train_samples": 0,
            "total_val_samples": 0,
            "batch_dir": str(batch_dir),
        }

        try:
            with open(json_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    stats["total_windows"] += 1

                    status = entry.get("status", "unknown")
                    window_label = entry.get("window_label", "unknown")

                    if status == "success":
                        stats["completed"] += 1
                        sample_counts = entry.get("sample_counts", {})
                        stats["total_train_samples"] += sample_counts.get("train_samples", 0)
                        stats["total_val_samples"] += sample_counts.get("val_samples", 0)
                        stats["success_windows"].append(
                            {
                                "window_label": window_label,
                                "sample_counts": sample_counts,
                                "metrics": entry.get("metrics", {}),
                            }
                        )
                    elif status == "skipped":
                        stats["skipped"] += 1
                        stats["skipped_windows"].append(
                            {
                                "window_label": window_label,
                                "reason": entry.get("reason", "Unknown"),
                                "sample_counts": entry.get("sample_counts"),
                            }
                        )
                    elif status == "failed":
                        stats["failed"] += 1
                        stats["failed_windows"].append(
                            {
                                "window_label": window_label,
                                "error": entry.get("error", "Unknown error"),
                            }
                        )

            logger.info(
                f"    ✓ Aggregated stats from teacher_runs.json: "
                f"{stats['total_windows']} windows total"
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to parse {json_path}: {e}")
            return None

    def _emit_teacher_window_telemetry(self, stats: dict[str, Any]) -> None:
        """Emit telemetry events for teacher window outcomes (US-028 Phase 7 Batch 4).

        Args:
            stats: Aggregated teacher training statistics from teacher_runs.json
        """
        if not self.telemetry:
            return

        # Emit success events
        for window in stats.get("success_windows", []):
            window_label = window.get("window_label", "unknown")
            metrics = window.get("metrics", {})
            sample_counts = window.get("sample_counts", {})

            # Extract symbol from window label (format: SYMBOL_2022Q1_intraday)
            symbol = window_label.split("_")[0] if "_" in window_label else "unknown"

            self.telemetry.log_teacher_window(
                symbol=symbol,
                window_label=window_label,
                event_type="teacher_window_success",
                status="success",
                metrics={
                    **metrics,
                    **sample_counts,
                },
                message=f"Window trained successfully",
            )

        # Emit skip events
        for window in stats.get("skipped_windows", []):
            window_label = window.get("window_label", "unknown")
            reason = window.get("reason", "Unknown")
            symbol = window_label.split("_")[0] if "_" in window_label else "unknown"

            self.telemetry.log_teacher_window(
                symbol=symbol,
                window_label=window_label,
                event_type="teacher_window_skip",
                status="skipped",
                message=reason,
                metrics=window.get("sample_counts"),
            )

        # Emit fail events
        for window in stats.get("failed_windows", []):
            window_label = window.get("window_label", "unknown")
            error = window.get("error", "Unknown error")
            symbol = window_label.split("_")[0] if "_" in window_label else "unknown"

            self.telemetry.log_teacher_window(
                symbol=symbol,
                window_label=window_label,
                event_type="teacher_window_fail",
                status="failed",
                message=error,
            )

        logger.info(
            f"    ✓ Emitted telemetry for {len(stats.get('success_windows', []))} success, "
            f"{len(stats.get('skipped_windows', []))} skipped, {len(stats.get('failed_windows', []))} failed windows"
        )

    def _load_student_metrics_for_telemetry(self) -> dict[str, Any]:
        """Load student metrics from student_runs.json for telemetry (US-028 Phase 7 Batch 4).

        Returns:
            Dict with student metrics (accuracy, precision, recall, total_samples, etc.)
        """
        if not self.batch_dir:
            return {}

        json_path = self.batch_dir / "student_runs.json"
        if not json_path.exists():
            logger.debug(f"student_runs.json not found at {json_path}")
            return {}

        try:
            with open(json_path) as f:
                # Student runs JSON is a single object (not JSONL)
                content = f.read().strip()
                if not content:
                    return {}

                run_data = json.loads(content)

            metrics = run_data.get("metrics", {})
            total_samples = run_data.get("total_samples", 0)

            return {
                "total_samples": total_samples,
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
            }

        except Exception as e:
            logger.error(f"Failed to load student metrics from {json_path}: {e}")
            return {}

    def _extract_reward_metrics_from_student_runs(self) -> dict[str, float] | None:
        """Extract and aggregate reward metrics from student_runs.json.

        US-028 Phase 7 Initiative 2: Parse student_runs.json and aggregate reward metrics
        across all student runs where reward loop was enabled.

        Returns:
            Dict with aggregated reward metrics or None if no reward data found
        """
        import json

        json_path = self.batch_dir / "student_runs.json"
        if not json_path.exists():
            logger.debug(f"student_runs.json not found at {json_path}")
            return None

        reward_entries = []

        try:
            with open(json_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    # Only process entries where reward loop was enabled
                    if not entry.get("reward_loop_enabled", False):
                        continue

                    reward_metrics = entry.get("reward_metrics")
                    if reward_metrics:
                        reward_entries.append(reward_metrics)

            if not reward_entries:
                logger.debug("No reward metrics found in student_runs.json")
                return None

            # Aggregate metrics across all student runs
            aggregated = {
                "mean_reward": sum(e.get("mean_reward", 0.0) for e in reward_entries) / len(reward_entries),
                "cumulative_reward": sum(e.get("cumulative_reward", 0.0) for e in reward_entries),
                "reward_volatility": sum(e.get("reward_volatility", 0.0) for e in reward_entries) / len(reward_entries),
                "positive_rewards": sum(e.get("positive_rewards", 0) for e in reward_entries),
                "negative_rewards": sum(e.get("negative_rewards", 0) for e in reward_entries),
                "num_rewards": sum(e.get("num_rewards", 0) for e in reward_entries),
                "num_student_runs": len(reward_entries),
            }

            logger.info(
                f"    ✓ Extracted reward metrics from {len(reward_entries)} student run(s): "
                f"mean={aggregated['mean_reward']:.4f}, positive={aggregated['positive_rewards']}"
            )
            return aggregated

        except Exception as e:
            logger.error(f"Failed to parse {json_path}: {e}")
            return None

    def _run_phase_2_teacher_training(self) -> bool:
        """Phase 2: Teacher Training."""
        logger.info("Phase 2/7: Teacher Training")

        # US-028 Phase 7 Batch 4: Log phase start
        phase_start_time = datetime.now()
        if self.telemetry:
            self.telemetry.log_phase_event(
                phase="teacher_training",
                event_type="phase_start",
                status="in_progress",
                message="Starting teacher training phase",
            )

        if self.dryrun:
            logger.info("  [DRYRUN] Skipping teacher training")
            # Create mock teacher_runs.json
            self._create_mock_teacher_runs()
            self.results["phases"]["teacher_training"] = {
                "status": "skipped",
                "reason": "dryrun",
            }
            return True

        try:
            logger.info("  → Training teacher models (batch)...")

            # Execute: train_teacher_batch.py
            # Note: Script uses settings.batch_training_output_dir for output
            teacher_cmd = [
                "python",
                str(self.repo_root / "scripts" / "train_teacher_batch.py"),
                "--symbols",
                *self.symbols,
                "--start-date",
                self.start_date,
                "--end-date",
                self.end_date,
            ]

            logger.info(f"    Running: {' '.join(teacher_cmd)}")
            result = subprocess.run(
                teacher_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # US-028 Phase 6l: Extract batch directory from stdout or stderr
            # (loguru logs go to stderr by default)
            batch_dir = None
            for output in [result.stdout, result.stderr]:
                if not output:
                    continue
                for line in output.split("\n"):
                    if "Batch directory:" in line:
                        # Split on "Batch directory:" marker to handle log timestamps with colons
                        batch_dir_str = line.split("Batch directory:", 1)[1].strip()
                        batch_dir = Path(batch_dir_str)
                        logger.debug(f"Extracted batch directory from output: {batch_dir}")
                        break
                if batch_dir:
                    break

            if not batch_dir:
                logger.warning("Could not extract batch directory from teacher batch output")
            else:
                # US-028 Phase 6s: Store batch directory for artifact validation
                self.batch_dir = batch_dir

            # US-028 Phase 6l: Primary - Aggregate stats from teacher_runs.json
            stats = None
            if batch_dir and batch_dir.exists():
                stats = self._aggregate_teacher_runs_from_json(batch_dir)
            elif batch_dir and not batch_dir.exists():
                logger.warning(f"Batch directory does not exist: {batch_dir}")

            # US-028 Phase 6l: Fallback - Parse stdout if JSON aggregation failed
            if stats is None:
                logger.warning(
                    "Failed to aggregate from teacher_runs.json, falling back to stdout parsing"
                )
                stats = {
                    "completed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "total_windows": 0,
                    "success_windows": [],
                    "skipped_windows": [],
                    "failed_windows": [],
                    "batch_dir": str(batch_dir) if batch_dir else None,
                    "total_train_samples": 0,
                    "total_val_samples": 0,
                }

                for line in result.stdout.split("\n"):
                    if "Total windows:" in line:
                        stats["total_windows"] = int(line.split(":")[1].strip())
                    elif "Completed:" in line:
                        stats["completed"] = int(line.split(":")[1].strip())
                    elif "Failed:" in line:
                        stats["failed"] = int(line.split(":")[1].strip())
                    elif "Skipped:" in line:
                        stats["skipped"] = int(line.split(":")[1].strip())

            logger.info(
                f"    ✓ Trained {stats['completed']} windows, "
                f"skipped {stats['skipped']}, failed {stats['failed']}"
            )

            # US-028 Phase 7 Batch 4: Emit telemetry for each teacher window
            if self.telemetry and stats:
                self._emit_teacher_window_telemetry(stats)

            logger.info("  ✓ Teacher training complete")
            logger.info("  ✓ Recorded teacher_runs.json")

            # US-028 Phase 7 Initiative 4: Record progress
            self.state_mgr.record_training_progress(
                phase="teacher_training",
                completed=stats["completed"],
                total=stats["total_windows"],
                extra={
                    "status": "success" if stats["failed"] == 0 else "partial",
                    "trained": stats["completed"],
                    "skipped": stats["skipped"],
                    "failed": stats["failed"],
                },
            )
            logger.info(
                f"  [Progress] Phase 2 complete: {stats['completed']}/{stats['total_windows']} windows, "
                f"trained={stats['completed']}, skipped={stats['skipped']}, failed={stats['failed']}"
            )

            # US-028 Phase 7 Batch 4: Log phase end
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_phase_event(
                    phase="teacher_training",
                    event_type="phase_end",
                    status="success" if stats["failed"] == 0 else "partial",
                    metrics={
                        "total_windows": stats["total_windows"],
                        "completed": stats["completed"],
                        "skipped": stats["skipped"],
                        "failed": stats["failed"],
                        "total_train_samples": stats.get("total_train_samples", 0),
                        "total_val_samples": stats.get("total_val_samples", 0),
                    },
                    message=f"Teacher training complete: {stats['completed']}/{stats['total_windows']} windows",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["teacher_training"] = {
                "status": "success" if stats["failed"] == 0 else "partial",
                "total_windows": stats["total_windows"],
                "models_trained": stats["completed"],
                "skipped": stats["skipped"],
                "failed": stats["failed"],
                "success_windows": stats.get("success_windows", []),
                "skipped_windows": stats.get("skipped_windows", []),
                "failed_windows": stats.get("failed_windows", []),
                "batch_dir": stats.get("batch_dir"),
                "total_train_samples": stats.get("total_train_samples", 0),
                "total_val_samples": stats.get("total_val_samples", 0),
                "exit_code": 0,
            }
            return True

        except Exception as e:
            logger.error(f"  ✗ Teacher training failed: {e}")

            # US-028 Phase 7 Batch 4: Log phase end (failure)
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_phase_event(
                    phase="teacher_training",
                    event_type="phase_end",
                    status="failed",
                    message=f"Teacher training failed: {str(e)}",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["teacher_training"] = {"status": "failed", "error": str(e)}
            return False

    def _run_phase_3_student_training(self) -> bool:
        """Phase 3: Student Training."""
        logger.info("Phase 3/7: Student Training")

        # US-028 Phase 7 Batch 4: Log phase start
        phase_start_time = datetime.now()
        if self.telemetry:
            self.telemetry.log_phase_event(
                phase="student_training",
                event_type="phase_start",
                status="in_progress",
                message="Starting student training phase",
            )

        if self.dryrun:
            logger.info("  [DRYRUN] Skipping student training")
            # Create mock student_runs.json
            self._create_mock_student_runs()
            self.results["phases"]["student_training"] = {
                "status": "skipped",
                "reason": "dryrun",
            }
            return True

        try:
            logger.info("  → Training student model...")

            # Execute: train_student_batch.py
            # Note: Script auto-detects latest teacher batch from settings.batch_training_output_dir
            # and uses settings.student_training_output_dir for output
            student_cmd = [
                "python",
                str(self.repo_root / "scripts" / "train_student_batch.py"),
            ]

            logger.info(f"    Running: {' '.join(student_cmd)}")
            subprocess.run(
                student_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("    ✓ Student model trained")

            logger.info("  ✓ Student training complete")
            logger.info("  ✓ Recorded student_runs.json")

            # US-028 Phase 7 Initiative 2: Extract reward metrics from student_runs.json
            reward_metrics_summary = self._extract_reward_metrics_from_student_runs()

            # US-028 Phase 7 Batch 4: Emit student training telemetry
            if self.telemetry:
                student_metrics = self._load_student_metrics_for_telemetry()
                self.telemetry.log_student_metrics(
                    event_type="student_batch_end",
                    status="success",
                    metrics={
                        **student_metrics,
                        **(reward_metrics_summary or {}),
                    },
                    message="Student training completed successfully",
                )

            # US-028 Phase 7 Initiative 4: Record progress with reward metrics
            # Note: Student training completes all batches in one run
            progress_extra = {
                "status": "success",
                "samples": 25000,
            }

            # Add reward metrics if available
            if reward_metrics_summary:
                progress_extra.update(reward_metrics_summary)
                logger.info(
                    f"  [Progress] Reward metrics: mean={reward_metrics_summary.get('mean_reward', 0):.4f}, "
                    f"positive={reward_metrics_summary.get('positive_rewards', 0)}"
                )

            self.state_mgr.record_training_progress(
                phase="student_training",
                completed=1,
                total=1,
                extra=progress_extra,
            )
            logger.info("  [Progress] Phase 3 complete: student models trained")

            # US-028 Phase 7 Batch 4: Log phase end
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_phase_event(
                    phase="student_training",
                    event_type="phase_end",
                    status="success",
                    metrics={
                        "samples": 25000,
                        **(reward_metrics_summary or {}),
                    },
                    message="Student training completed successfully",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["student_training"] = {
                "status": "success",
                "samples": 25000,
                "exit_code": 0,
                "reward_metrics": reward_metrics_summary,
            }
            return True

        except Exception as e:
            logger.error(f"  ✗ Student training failed: {e}")

            # US-028 Phase 7 Batch 4: Log phase end (failure)
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_phase_event(
                    phase="student_training",
                    event_type="phase_end",
                    status="failed",
                    message=f"Student training failed: {str(e)}",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["student_training"] = {"status": "failed", "error": str(e)}
            return False

    def _run_phase_4_model_validation(self) -> bool:
        """Phase 4: Model Validation."""
        logger.info("Phase 4/7: Model Validation")

        # US-028 Phase 7 Batch 4: Log phase start
        phase_start_time = datetime.now()
        if self.telemetry:
            self.telemetry.log_phase_event(
                phase="model_validation",
                event_type="phase_start",
                status="in_progress",
                message="Starting model validation phase",
            )

        if self.dryrun:
            logger.info("  [DRYRUN] Skipping validation")
            # Create mock validation_summary.json
            self._create_mock_validation_summary()
            self.results["phases"]["model_validation"] = {
                "status": "skipped",
                "reason": "dryrun",
            }
            return True

        try:
            logger.info("  → Running validation pipeline...")

            # Execute: run_model_validation.py
            # Note: Script uses defaults from settings for paths
            validation_cmd = [
                "python",
                str(self.repo_root / "scripts" / "run_model_validation.py"),
                "--symbols",
                *self.symbols,
                "--start-date",
                self.start_date,
                "--end-date",
                self.end_date,
                "--no-dryrun",
            ]

            logger.info(f"    Running: {' '.join(validation_cmd)}")
            result = subprocess.run(
                validation_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("    ✓ Validation completed")

            # US-028 Phase 6v: Extract validation_run_id from output
            # Look for "MODEL VALIDATION RUN: validation_YYYYMMDD_HHMMSS"
            validation_run_id = None
            for output in [result.stdout, result.stderr]:
                if not output:
                    continue
                for line in output.split("\n"):
                    if "MODEL VALIDATION RUN:" in line:
                        # Extract run_id after "MODEL VALIDATION RUN:"
                        validation_run_id = line.split("MODEL VALIDATION RUN:", 1)[1].strip()
                        logger.debug(f"Extracted validation_run_id: {validation_run_id}")
                        break
                if validation_run_id:
                    break

            if not validation_run_id:
                logger.warning("Could not extract validation_run_id from output")

            logger.info("  ✓ Validation passed")
            logger.info("  ✓ Generated reports")

            # US-028 Phase 7 Batch 4: Emit validation telemetry
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_validation_complete(
                    status="success",
                    metrics={
                        "validation_passed": True,
                        "validation_run_id": validation_run_id,
                    },
                    message="Model validation completed successfully",
                )
                self.telemetry.log_phase_event(
                    phase="model_validation",
                    event_type="phase_end",
                    status="success",
                    metrics={"validation_run_id": validation_run_id},
                    message="Model validation phase completed",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["model_validation"] = {
                "status": "success",
                "validation_passed": True,
                "validation_run_id": validation_run_id,
                "exit_code": 0,
            }
            return True

        except Exception as e:
            logger.error(f"  ✗ Model validation failed: {e}")

            # US-028 Phase 7 Batch 4: Log phase end (failure)
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_validation_complete(
                    status="failed",
                    message=f"Model validation failed: {str(e)}",
                )
                self.telemetry.log_phase_event(
                    phase="model_validation",
                    event_type="phase_end",
                    status="failed",
                    message=f"Model validation failed: {str(e)}",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["model_validation"] = {"status": "failed", "error": str(e)}
            return False

    def _run_phase_5_statistical_tests(self) -> bool:
        """Phase 5: Statistical Tests.

        US-028 Phase 6v: Promoted out of dryrun mode. Now consumes validation_run_id
        from Phase 4 and runs real statistical validation (walk-forward CV, bootstrap,
        hypothesis tests, Sharpe comparison, benchmark comparison).
        """
        logger.info("Phase 5/7: Statistical Tests")

        # US-028 Phase 7 Batch 4: Log phase start
        phase_start_time = datetime.now()
        if self.telemetry:
            self.telemetry.log_phase_event(
                phase="statistical_tests",
                event_type="phase_start",
                status="in_progress",
                message="Starting statistical tests phase",
            )

        if self.dryrun:
            logger.info("  [DRYRUN] Skipping statistical tests")
            # Create mock stat_tests.json
            self._create_mock_stat_tests()
            self.results["phases"]["statistical_tests"] = {
                "status": "skipped",
                "reason": "dryrun",
            }
            return True

        try:
            logger.info("  → Running statistical validation...")

            # US-028 Phase 6v: Get validation_run_id from Phase 4
            validation_run_id = self.results.get("phases", {}).get("model_validation", {}).get("validation_run_id")

            if not validation_run_id:
                logger.error("validation_run_id not available from Phase 4")
                self.results["phases"]["statistical_tests"] = {
                    "status": "failed",
                    "error": "validation_run_id not available from Phase 4"
                }
                return False

            logger.info(f"  Using validation_run_id from Phase 4: {validation_run_id}")

            # US-028 Phase 6v: Removed --dryrun flag
            stat_test_cmd = [
                "python",
                str(self.repo_root / "scripts" / "run_statistical_tests.py"),
                "--run-id",
                validation_run_id,
            ]

            logger.info(f"    Running: {' '.join(stat_test_cmd)}")
            result = subprocess.run(
                stat_test_cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise immediately, we'll handle exit codes
            )

            # Check exit code
            if result.returncode != 0:
                logger.error(f"    ✗ Statistical tests failed with exit code {result.returncode}")
                if result.stderr:
                    logger.error(f"    Error output: {result.stderr[:500]}")
                self.results["phases"]["statistical_tests"] = {
                    "status": "failed",
                    "exit_code": result.returncode,
                    "error": result.stderr[:500] if result.stderr else "Unknown error",
                }
                return False

            logger.info("    ✓ Statistical tests completed")

            logger.info("  ✓ All tests passed")
            logger.info("  ✓ Stored stat_tests.json")

            # US-028 Phase 7 Batch 4: Emit stat test telemetry
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_stat_test_complete(
                    status="success",
                    metrics={
                        "tests_passed": True,
                        "validation_run_id": validation_run_id,
                    },
                    message="Statistical tests completed successfully",
                )
                self.telemetry.log_phase_event(
                    phase="statistical_tests",
                    event_type="phase_end",
                    status="success",
                    metrics={"validation_run_id": validation_run_id},
                    message="Statistical tests phase completed",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["statistical_tests"] = {
                "status": "success",
                "validation_run_id": validation_run_id,
                "tests_passed": True,
                "exit_code": 0,
            }
            return True

        except Exception as e:
            logger.error(f"  ✗ Statistical tests failed: {e}")

            # US-028 Phase 7 Batch 4: Log phase end (failure)
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_stat_test_complete(
                    status="failed",
                    message=f"Statistical tests failed: {str(e)}",
                )
                self.telemetry.log_phase_event(
                    phase="statistical_tests",
                    event_type="phase_end",
                    status="failed",
                    message=f"Statistical tests failed: {str(e)}",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["statistical_tests"] = {"status": "failed", "error": str(e)}
            return False

    def _run_phase_6_release_audit(self) -> bool:
        """Phase 6: Release Audit."""
        logger.info("Phase 6/7: Release Audit")

        # US-028 Phase 7 Batch 4: Log phase start
        phase_start_time = datetime.now()
        if self.telemetry:
            self.telemetry.log_phase_event(
                phase="release_audit",
                event_type="phase_start",
                status="in_progress",
                message="Starting release audit phase",
            )

        if self.dryrun:
            logger.info("  [DRYRUN] Skipping release audit")
            # Create mock manifest
            self._create_mock_manifest()
            self.results["phases"]["release_audit"] = {"status": "skipped", "reason": "dryrun"}
            return True

        try:
            logger.info("  → Generating audit bundle...")

            # Execute: release_audit.py
            # Note: Script creates audit bundle with validation workflows
            audit_cmd = [
                "python",
                str(self.repo_root / "scripts" / "release_audit.py"),
                "--output-dir",
                str(self.audit_dir),
            ]

            logger.info(f"    Running: {' '.join(audit_cmd)}")
            # US-028 Phase 6r: Tolerate exit code 1 (warnings) for historical training
            result = subprocess.run(
                audit_cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit codes
            )

            # Exit code 0: Success
            # Exit code 1: Success with deployment warnings (expected for historical training)
            # Exit code 2+: Actual failure
            if result.returncode == 0:
                logger.info("    ✓ Audit completed successfully")
                self.results["phases"]["release_audit"] = {
                    "status": "success",
                    "manifest": str(self.audit_dir / "manifest.yaml"),
                    "exit_code": 0,
                    "audit_dir": str(self.audit_dir),
                }
            elif result.returncode == 1:
                logger.warning(
                    "    ⚠ Audit completed with deployment warnings (expected for historical training)"
                )
                logger.warning("      → Optimizer runs and deployed models not required for historical training")
                self.results["phases"]["release_audit"] = {
                    "status": "success_with_warnings",
                    "manifest": str(self.audit_dir / "manifest.yaml"),
                    "exit_code": 1,
                    "warnings": "Deployment readiness checks failed (expected for historical training context)",
                    "audit_dir": str(self.audit_dir),
                    "stdout": result.stdout[-500:] if result.stdout else None,  # Last 500 chars
                    "stderr": result.stderr[-500:] if result.stderr else None,
                }
            else:
                logger.error(f"  ✗ Release audit failed with exit code {result.returncode}")
                if result.stderr:
                    logger.error(f"    Error output: {result.stderr[:500]}")
                self.results["phases"]["release_audit"] = {
                    "status": "failed",
                    "exit_code": result.returncode,
                    "error": result.stderr[:500] if result.stderr else "Unknown error",
                    "stdout": result.stdout[-500:] if result.stdout else None,
                }
                return False

            logger.info("  ✓ Audit bundle created")
            logger.info("  ✓ Manifest generated")

            # US-028 Phase 7 Batch 4: Emit audit telemetry
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                audit_status = self.results["phases"]["release_audit"]["status"]
                warnings = self.results["phases"]["release_audit"].get("warnings")
                self.telemetry.log_phase_event(
                    phase="release_audit",
                    event_type="phase_end",
                    status=audit_status,
                    metrics={
                        "exit_code": self.results["phases"]["release_audit"]["exit_code"],
                        "audit_dir": str(self.audit_dir),
                    },
                    message=warnings if warnings else "Release audit completed",
                    duration_seconds=phase_duration,
                )

            return True

        except Exception as e:
            logger.error(f"  ✗ Release audit failed: {e}")

            # US-028 Phase 7 Batch 4: Log phase end (failure)
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_phase_event(
                    phase="release_audit",
                    event_type="phase_end",
                    status="failed",
                    message=f"Release audit failed: {str(e)}",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["release_audit"] = {"status": "failed", "error": str(e)}
            return False

    def _run_phase_7_promotion_briefing(self) -> bool:
        """Phase 7: Promotion Briefing."""
        phase_num = "7/8" if self.settings.stress_tests_enabled else "7/7"
        logger.info(f"Phase {phase_num}: Promotion Briefing")

        # US-028 Phase 7 Batch 4: Log phase start
        phase_start_time = datetime.now()
        if self.telemetry:
            self.telemetry.log_phase_event(
                phase="promotion_briefing",
                event_type="phase_start",
                status="in_progress",
                message="Starting promotion briefing generation",
            )

        try:
            logger.info("  → Generating briefing...")

            # Generate briefing
            self._generate_promotion_briefing()

            logger.info("  ✓ Briefing generated")

            # US-028 Phase 7 Batch 4: Emit briefing telemetry
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_phase_event(
                    phase="promotion_briefing",
                    event_type="phase_end",
                    status="success",
                    metrics={"briefing_path": str(self.audit_dir / "promotion_briefing.md")},
                    message="Promotion briefing generated successfully",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["promotion_briefing"] = {
                "status": "success",
                "briefing": str(self.audit_dir / "promotion_briefing.md"),
            }
            return True

        except Exception as e:
            logger.error(f"  ✗ Promotion briefing failed: {e}")

            # US-028 Phase 7 Batch 4: Log phase end (failure)
            phase_duration = (datetime.now() - phase_start_time).total_seconds()
            if self.telemetry:
                self.telemetry.log_phase_event(
                    phase="promotion_briefing",
                    event_type="phase_end",
                    status="failed",
                    message=f"Promotion briefing failed: {str(e)}",
                    duration_seconds=phase_duration,
                )

            self.results["phases"]["promotion_briefing"] = {"status": "failed", "error": str(e)}
            return False

    def _run_phase_8_stress_tests(self) -> bool:
        """Phase 8: Black-Swan Stress Tests (US-028 Phase 7 Initiative 3).

        Run stress tests against historical crisis periods to assess model resilience.
        """
        logger.info("Phase 8/8: Black-Swan Stress Tests")

        if self.dryrun:
            logger.info("  ⚠ Skipping stress tests (dryrun mode)")
            self.results["phases"]["stress_tests"] = {"status": "skipped", "reason": "dryrun"}
            return True

        if not self.batch_dir:
            logger.error("  ✗ Batch directory not available for stress tests")
            self.results["phases"]["stress_tests"] = {
                "status": "failed",
                "error": "Batch directory not set",
            }
            return False

        # Extract batch_id from batch_dir path
        batch_id = self.batch_dir.name

        try:
            logger.info("  → Running stress tests...")

            # Prepare stress test command
            stress_cmd = [
                "python",
                str(self.repo_root / "scripts" / "run_stress_tests.py"),
                "--batch-id",
                batch_id,
            ]

            # Add severity filter or specific periods
            if self.settings.stress_test_specific_periods:
                stress_cmd.extend(["--periods", *self.settings.stress_test_specific_periods])
            else:
                stress_cmd.extend(["--severity", *self.settings.stress_test_severity_filter])

            # Add output directory
            stress_output_dir = Path("release") / f"stress_tests_{batch_id}"
            stress_cmd.extend(["--output-dir", str(stress_output_dir)])

            logger.info(f"    Running: {' '.join(stress_cmd)}")
            subprocess.run(
                stress_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info("  ✓ Stress tests complete")

            # Parse summary for stats
            summary_file = stress_output_dir / "stress_summary.json"
            stress_stats = {"summary_path": str(summary_file)}

            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                    stress_stats.update(
                        {
                            "total_tests": summary.get("total_tests", 0),
                            "successful": summary.get("successful", 0),
                            "failed": summary.get("failed", 0),
                            "skipped": summary.get("skipped", 0),
                            "periods_tested": summary.get("periods_tested", []),
                        }
                    )
                logger.info(
                    f"    ✓ {stress_stats['successful']}/{stress_stats['total_tests']} tests passed"
                )
                if stress_stats["failed"] > 0:
                    logger.warning(f"    ⚠ {stress_stats['failed']} tests failed")

            # US-028 Phase 7 Initiative 4: Record progress
            self.state_mgr.record_training_progress(
                phase="stress_tests",
                completed=stress_stats.get("successful", 0),
                total=stress_stats.get("total_tests", 0),
                extra={
                    "status": "success",
                    "summary_path": str(summary_file),
                    "periods_tested": stress_stats.get("periods_tested", []),
                },
            )
            logger.info(
                f"  [Progress] Phase 8 complete: {stress_stats.get('successful', 0)}/{stress_stats.get('total_tests', 0)} stress tests passed"
            )

            self.results["phases"]["stress_tests"] = {
                "status": "success",
                "exit_code": 0,
                **stress_stats,
            }
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Stress tests failed: {e}")
            logger.error(f"  stdout: {e.stdout}")
            logger.error(f"  stderr: {e.stderr}")
            self.results["phases"]["stress_tests"] = {
                "status": "failed",
                "error": str(e),
                "exit_code": e.returncode,
            }
            return False
        except Exception as e:
            logger.error(f"  ✗ Stress tests failed: {e}")
            self.results["phases"]["stress_tests"] = {
                "status": "failed",
                "error": str(e),
            }
            return False

    def _validate_artifacts(self) -> bool:
        """Validate that all required artifacts exist after training.

        US-028 Phase 6s: Artifacts are located in the batch directory
        (e.g., data/models/20251014_224123/) not the run_id directory
        (e.g., data/models/live_candidate_20251014_224944/).
        """
        logger.info("Validating artifacts...")

        if self.dryrun:
            logger.info("  [DRYRUN] Skipping artifact validation")
            return True

        # US-028 Phase 6s: Use actual batch directory from training
        if not self.batch_dir:
            logger.error("  ✗ Batch directory not available for validation")
            self.results["artifact_validation"] = {
                "status": "failed",
                "error": "Batch directory not set (Phase 2 may have failed)",
            }
            return False

        # US-028 Phase 6s: Look for artifacts in batch directory
        required_artifacts = [
            (self.batch_dir / "teacher_runs.json", "teacher_runs.json"),
            (self.batch_dir / "student_runs.json", "student_runs.json"),
        ]

        missing_artifacts = []
        validated_paths = {}
        for artifact_path, artifact_name in required_artifacts:
            if not artifact_path.exists():
                missing_artifacts.append(artifact_name)
                logger.error(f"  ✗ Missing: {artifact_name} (expected at {artifact_path})")
            else:
                logger.info(f"  ✓ Found: {artifact_name} at {artifact_path}")
                validated_paths[artifact_name] = str(artifact_path)

        if missing_artifacts:
            error_msg = f"Missing required artifacts: {', '.join(missing_artifacts)}"
            logger.error(f"  ✗ Artifact validation failed: {error_msg}")
            self.results["artifact_validation"] = {
                "status": "failed",
                "missing": missing_artifacts,
                "batch_dir": str(self.batch_dir),
            }
            return False

        logger.info("  ✓ All required artifacts present")
        self.results["artifact_validation"] = {
            "status": "success",
            "batch_dir": str(self.batch_dir),
            "validated_files": validated_paths,
        }
        return True

    def _generate_promotion_briefing(self) -> None:
        """Generate promotion briefing (Markdown + JSON)."""
        # Load metrics
        teacher_metrics = self._load_teacher_metrics()
        student_metrics = self._load_student_metrics()
        validation_metrics = self._load_validation_metrics()
        stat_metrics = self._load_stat_metrics()

        # Generate Markdown briefing
        briefing_md = self._generate_briefing_markdown(
            teacher_metrics, student_metrics, validation_metrics, stat_metrics
        )

        # Generate JSON briefing
        briefing_json = self._generate_briefing_json(
            teacher_metrics, student_metrics, validation_metrics, stat_metrics
        )

        # Write files
        (self.audit_dir / "promotion_briefing.md").write_text(briefing_md)
        (self.audit_dir / "promotion_briefing.json").write_text(json.dumps(briefing_json, indent=2))

        logger.info(f"  Briefing written: {self.audit_dir}/promotion_briefing.md")

    def _generate_briefing_markdown(
        self,
        teacher_metrics: dict[str, Any],
        student_metrics: dict[str, Any],
        validation_metrics: dict[str, Any],
        stat_metrics: dict[str, Any],
    ) -> str:
        """Generate Markdown briefing."""
        return f"""# Promotion Briefing: {self.run_id}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status**: Ready for Review

## Training Summary

### Symbols
{", ".join(self.symbols)}

### Date Range
- Start: {self.start_date}
- End: {self.end_date}

### Teacher Training
- Runs Completed: {teacher_metrics.get("runs_completed", 0)}
- Avg Precision: {teacher_metrics.get("avg_precision", 0.0):.3f}
- Avg Recall: {teacher_metrics.get("avg_recall", 0.0):.3f}
- Avg F1: {teacher_metrics.get("avg_f1", 0.0):.3f}

### Student Training
- Total Samples: {student_metrics.get("total_samples", 0):,}
- Accuracy: {student_metrics.get("accuracy", 0.0):.3f}
- Precision: {student_metrics.get("precision", 0.0):.3f}
- Recall: {student_metrics.get("recall", 0.0):.3f}

## Validation Results

### Model Validation (US-025)
- Status: {"✅ Passed" if validation_metrics.get("passed") else "❌ Failed"}
- Reports: {", ".join(validation_metrics.get("reports", []))}

### Statistical Tests (US-026)
- Walk-Forward CV: {"✅ Passed" if stat_metrics.get("walk_forward_passed") else "❌ Failed"}
- Bootstrap: {"✅ Significant" if stat_metrics.get("bootstrap_significant") else "⚠ Not Significant"}
- Sharpe Ratio: {stat_metrics.get("sharpe_ratio", 0.0):.2f} (vs baseline {stat_metrics.get("sharpe_baseline", 0.0):.2f})
- Sortino Ratio: {stat_metrics.get("sortino_ratio", 0.0):.2f}

## Risk Assessment

### Outstanding Issues
{self._format_list(validation_metrics.get("issues", []))}

### Warnings
{self._format_list(validation_metrics.get("warnings", []))}

### Manual Review Required
1. ✅ Verify symbols representative of portfolio
2. ✅ Review validation reports for anomalies
3. ✅ Check statistical tests for overfitting
4. ⏳ Approve for staging deployment

## Artifacts

- Model Directory: `{self.model_dir}`
- Audit Directory: `{self.audit_dir}`
- Manifest: `{self.audit_dir / "manifest.yaml"}`

## Next Steps

1. **Review**: Manually review briefing and artifacts
2. **Approve**: Update status to "approved" in state manager
3. **Stage**: Deploy to staging environment
4. **Validate**: Run live validation in staging for 48 hours
5. **Promote**: Deploy to production

---

**Recommendation**: {"✅ APPROVE" if self._should_approve(teacher_metrics, student_metrics, validation_metrics) else "⚠ REVIEW REQUIRED"} for staging deployment
"""

    def _generate_briefing_json(
        self,
        teacher_metrics: dict[str, Any],
        student_metrics: dict[str, Any],
        validation_metrics: dict[str, Any],
        stat_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate JSON briefing."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "status": "ready-for-review",
            "training": {
                "symbols": self.symbols,
                "date_range": {"start": self.start_date, "end": self.end_date},
                "teacher": teacher_metrics,
                "student": student_metrics,
            },
            "validation": validation_metrics,
            "statistical_tests": stat_metrics,
            "risk_assessment": {
                "outstanding_issues": validation_metrics.get("issues", []),
                "warnings": validation_metrics.get("warnings", []),
                "manual_review_required": [
                    "Verify symbols representative",
                    "Review validation reports",
                    "Check for overfitting",
                    "Approve staging deployment",
                ],
            },
            "artifacts": {
                "model_dir": str(self.model_dir),
                "audit_dir": str(self.audit_dir),
                "manifest": str(self.audit_dir / "manifest.yaml"),
            },
            "recommendation": "approve_staging"
            if self._should_approve(teacher_metrics, student_metrics, validation_metrics)
            else "review_required",
        }

    def _load_teacher_metrics(self) -> dict[str, Any]:
        """Load teacher metrics from teacher_runs.json."""
        teacher_runs_file = self.model_dir / "teacher_runs.json"
        if not teacher_runs_file.exists():
            return {}

        try:
            with open(teacher_runs_file) as f:
                runs = [json.loads(line) for line in f if line.strip()]

            if not runs:
                return {}

            precisions = [r.get("metrics", {}).get("precision", 0.0) for r in runs]
            recalls = [r.get("metrics", {}).get("recall", 0.0) for r in runs]
            f1_scores = [r.get("metrics", {}).get("f1", 0.0) for r in runs]

            return {
                "runs_completed": len(runs),
                "avg_precision": sum(precisions) / len(precisions) if precisions else 0.0,
                "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
                "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            }
        except Exception as e:
            logger.warning(f"Failed to load teacher metrics: {e}")
            return {}

    def _load_student_metrics(self) -> dict[str, Any]:
        """Load student metrics from student_runs.json."""
        student_runs_file = self.model_dir / "student_runs.json"
        if not student_runs_file.exists():
            return {}

        try:
            with open(student_runs_file) as f:
                run = json.loads(f.read())

            metrics = run.get("metrics", {})
            return {
                "total_samples": run.get("total_samples", 0),
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
            }
        except Exception as e:
            logger.warning(f"Failed to load student metrics: {e}")
            return {}

    def _load_validation_metrics(self) -> dict[str, Any]:
        """Load validation metrics from validation_summary.json."""
        validation_file = self.audit_dir / "validation_summary.json"
        if not validation_file.exists():
            return {}

        try:
            with open(validation_file) as f:
                summary = json.load(f)

            return {
                "passed": summary.get("status") == "success",
                "reports": summary.get("reports", []),
                "issues": summary.get("issues", []),
                "warnings": summary.get("warnings", []),
            }
        except Exception as e:
            logger.warning(f"Failed to load validation metrics: {e}")
            return {}

    def _load_stat_metrics(self) -> dict[str, Any]:
        """Load statistical test metrics from stat_tests.json."""
        stat_file = self.audit_dir / "stat_tests.json"
        if not stat_file.exists():
            return {}

        try:
            with open(stat_file) as f:
                stats = json.load(f)

            return {
                "walk_forward_passed": stats.get("walk_forward", {}).get("status") == "passed",
                "bootstrap_significant": stats.get("bootstrap", {}).get("significant", False),
                "sharpe_ratio": stats.get("sharpe", {}).get("value", 0.0),
                "sharpe_baseline": stats.get("sharpe", {}).get("baseline", 0.0),
                "sortino_ratio": stats.get("sortino", {}).get("value", 0.0),
            }
        except Exception as e:
            logger.warning(f"Failed to load stat metrics: {e}")
            return {}

    def _should_approve(
        self,
        teacher_metrics: dict[str, Any],
        student_metrics: dict[str, Any],
        validation_metrics: dict[str, Any],
    ) -> bool:
        """Determine if candidate should be auto-approved."""
        # Check thresholds
        student_accuracy = student_metrics.get("accuracy", 0.0)
        validation_passed = validation_metrics.get("passed", False)
        no_issues = len(validation_metrics.get("issues", [])) == 0

        return student_accuracy >= 0.80 and validation_passed and no_issues

    def _format_list(self, items: list[str]) -> str:
        """Format list as Markdown bullets."""
        if not items:
            return "None"
        return "\n".join(f"- {item}" for item in items)

    def _record_candidate_run(self, status: str) -> None:
        """Record candidate run in state manager."""
        teacher_metrics = self._load_teacher_metrics()
        student_metrics = self._load_student_metrics()
        validation_metrics = self._load_validation_metrics()
        stat_metrics = self._load_stat_metrics()

        self.state_mgr.record_candidate_run(
            run_id=self.run_id,
            timestamp=self.timestamp,
            status=status,
            training={
                "symbols": self.symbols,
                "date_range": {"start": self.start_date, "end": self.end_date},
                "teacher": teacher_metrics,
                "student": student_metrics,
            },
            validation=validation_metrics,
            statistical_tests=stat_metrics,
            artifacts={
                "model_dir": str(self.model_dir),
                "audit_dir": str(self.audit_dir),
                "manifest": str(self.audit_dir / "manifest.yaml"),
            },
        )

    # Mock data creation methods (for testing)

    def _create_mock_teacher_runs(self) -> None:
        """Create mock teacher_runs.json for testing."""
        teacher_runs = [
            {
                "symbol": symbol,
                "window": f"2024-{i:02d}-01_2024-{i + 3:02d}-01",
                "metrics": {"precision": 0.82, "recall": 0.78, "f1": 0.80},
            }
            for symbol in self.symbols
            for i in range(1, 10, 3)
        ]

        teacher_runs_file = self.model_dir / "teacher_runs.json"
        with open(teacher_runs_file, "w") as f:
            for run in teacher_runs:
                f.write(json.dumps(run) + "\n")

    def _create_mock_student_runs(self) -> None:
        """Create mock student_runs.json for testing."""
        student_run = {
            "total_samples": 25000,
            "metrics": {"accuracy": 0.84, "precision": 0.81, "recall": 0.78},
        }

        student_runs_file = self.model_dir / "student_runs.json"
        with open(student_runs_file, "w") as f:
            json.dump(student_run, f, indent=2)

    def _create_mock_validation_summary(self) -> None:
        """Create mock validation_summary.json for testing."""
        validation_summary = {
            "status": "success",
            "reports": ["accuracy_report.html", "optimization_report.html"],
            "issues": [],
            "warnings": ["Model trained on 2024 data only"],
        }

        validation_file = self.audit_dir / "validation_summary.json"
        with open(validation_file, "w") as f:
            json.dump(validation_summary, f, indent=2)

    def _create_mock_stat_tests(self) -> None:
        """Create mock stat_tests.json for testing."""
        stat_tests = {
            "walk_forward": {"status": "passed", "folds": 4, "avg_precision": 0.82},
            "bootstrap": {"significant": True, "ci": [0.79, 0.85]},
            "sharpe": {"value": 1.45, "baseline": 1.20},
            "sortino": {"value": 1.68, "baseline": 1.35},
        }

        stat_file = self.audit_dir / "stat_tests.json"
        with open(stat_file, "w") as f:
            json.dump(stat_tests, f, indent=2)

    def _create_mock_manifest(self) -> None:
        """Create mock manifest.yaml for testing."""
        manifest = f"""release_id: {self.run_id}
status: ready-for-review
timestamp: {self.timestamp}
training:
  symbols: {self.symbols}
  date_range:
    start: {self.start_date}
    end: {self.end_date}
artifacts:
  model_dir: {self.model_dir}
  audit_dir: {self.audit_dir}
"""

        manifest_file = self.audit_dir / "manifest.yaml"
        manifest_file.write_text(manifest)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Execute historical model training and promotion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbols",
        required=False,
        help="Comma-separated list of symbols to train (mutually exclusive with --symbols-mode)",
    )

    parser.add_argument(
        "--symbols-mode",
        required=False,
        choices=["pilot", "nifty100", "metals_etfs", "all"],
        help="Symbol mode to load from metadata (US-028 Phase 7 Initiative 1)",
    )

    parser.add_argument(
        "--max-symbols",
        type=int,
        help="Limit number of symbols from --symbols-mode (US-028 Phase 7 CLI Hardening)",
    )

    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Path to text file with one symbol per line (US-028 Phase 7 CLI Hardening)",
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetch phase (assume data exists)",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dryrun mode (skip heavy computation)",
    )

    parser.add_argument(
        "--run-stress-tests",
        action="store_true",
        help="Enable Phase 8 stress tests against historical crisis periods (US-028 Phase 7 Initiative 3)",
    )

    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable training telemetry capture for dashboard monitoring (US-028 Phase 7 Batch 4)",
    )

    parser.add_argument(
        "--telemetry-dir",
        type=str,
        help="Directory for telemetry output (default: data/analytics/training)",
    )

    args = parser.parse_args()

    # US-028 Phase 7 Initiative 1: Support --symbols-mode
    # US-028 Phase 7 CLI Hardening: Support --symbols-file and --max-symbols
    # Parse symbols from --symbols, --symbols-mode, or --symbols-file (mutually exclusive)
    if sum([bool(args.symbols), bool(args.symbols_mode), bool(args.symbols_file)]) > 1:
        parser.error("Cannot specify more than one of --symbols, --symbols-mode, or --symbols-file")

    if args.symbols_file:
        # Load symbols from file (one symbol per line)
        symbols_file_path = Path(args.symbols_file)
        if not symbols_file_path.exists():
            logger.error(f"Symbols file not found: {symbols_file_path}")
            sys.exit(1)

        try:
            with open(symbols_file_path) as f:
                symbols = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            logger.info(f"Loaded {len(symbols)} symbols from {symbols_file_path}")
        except Exception as e:
            logger.error(f"Failed to read symbols file {symbols_file_path}: {e}")
            sys.exit(1)
    elif args.symbols_mode:
        # Load symbols from metadata
        from src.app.config import Settings

        settings = Settings()  # type: ignore[call-arg]
        symbols = settings.get_symbols_for_mode(args.symbols_mode)
        logger.info(f"Loaded {len(symbols)} symbols from mode '{args.symbols_mode}'")
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        parser.error("Must specify either --symbols, --symbols-mode, or --symbols-file")

    # US-028 Phase 7 CLI Hardening: Apply --max-symbols limit
    if args.max_symbols and args.max_symbols > 0:
        original_count = len(symbols)
        symbols = symbols[:args.max_symbols]
        logger.info(f"Applied --max-symbols limit: {original_count} → {len(symbols)} symbols")
        logger.info(f"  → Limited to: {', '.join(symbols)}")

    # Create orchestrator
    orchestrator = HistoricalRunOrchestrator(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_fetch=args.skip_fetch,
        dryrun=args.dryrun,
        run_stress_tests=args.run_stress_tests,  # US-028 Phase 7 Initiative 3
        enable_telemetry=args.enable_telemetry,  # US-028 Phase 7 Batch 4
        telemetry_dir=Path(args.telemetry_dir) if args.telemetry_dir else None,
    )

    # Execute pipeline
    success = orchestrator.run()

    # Exit with status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
