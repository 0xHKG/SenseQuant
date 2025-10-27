"""Batch Student training script from Teacher batch outputs (US-024 Phase 2).

This script orchestrates Student model training from Teacher batch metadata,
generating promotion checklists and recording results for each window.

Usage:
    # Train students from latest teacher batch
    python scripts/train_student_batch.py

    # Train with custom baseline
    python scripts/train_student_batch.py \\
        --baseline-precision 0.65 \\
        --baseline-recall 0.60

    # Resume partial batch
    python scripts/train_student_batch.py --resume

    # Specify teacher batch directory
    python scripts/train_student_batch.py \\
        --teacher-batch-dir data/models/20251012_190000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.config import Settings
from src.services.state_manager import StateManager
from src.services.teacher_student import StudentModel, TeacherLabeler


class StudentBatchTrainer:
    """Manages batch Student training from Teacher outputs."""

    def __init__(
        self,
        teacher_batch_dir: Path,
        baseline_precision: float,
        baseline_recall: float,
        resume: bool = False,
        incremental: bool = False,
        workers: int = 1,
    ):
        """Initialize student batch trainer (US-024 Phases 2-5).

        Args:
            teacher_batch_dir: Directory containing teacher batch artifacts
            baseline_precision: Baseline precision for promotion
            baseline_recall: Baseline recall for promotion
            resume: If True, skip already-trained students
            incremental: If True, incremental mode (US-024 Phase 4)
            workers: Number of parallel workers (US-024 Phase 5)
        """
        self.teacher_batch_dir = teacher_batch_dir
        self.baseline_precision = baseline_precision
        self.baseline_recall = baseline_recall
        self.resume = resume
        self.incremental = incremental
        self.workers = workers

        # Load teacher batch metadata
        self.teacher_metadata_file = teacher_batch_dir / "teacher_runs.json"
        self.teacher_runs = TeacherLabeler.load_batch_metadata(self.teacher_metadata_file)

        # Student metadata file
        self.student_metadata_file = teacher_batch_dir / "student_runs.json"

        # Extract batch ID from first teacher run
        if self.teacher_runs:
            self.batch_id = self.teacher_runs[0].get("batch_id", "unknown")
        else:
            self.batch_id = "unknown"

        # Statistics
        self.stats = {
            "total_teachers": len(self.teacher_runs),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "retries": 0,
        }

        # US-024 Phase 5: State management
        state_file = Path("data/state/student_batch.json")
        self.state_manager = StateManager(state_file)
        self.metadata_lock = Lock()

    def get_student_artifacts_path(self, teacher_run: dict[str, Any]) -> Path:
        """Get student artifacts path for a teacher run.

        Args:
            teacher_run: Teacher run metadata dict

        Returns:
            Path to student artifacts directory
        """
        window_label = teacher_run.get("window_label") or Path(teacher_run["artifacts_path"]).name
        return self.teacher_batch_dir / f"{window_label}_student"

    def is_already_trained(self, teacher_run: dict[str, Any]) -> bool:
        """Check if student model already trained for this teacher run.

        Args:
            teacher_run: Teacher run metadata dict

        Returns:
            True if student artifacts exist
        """
        student_path = self.get_student_artifacts_path(teacher_run)
        # Check for student model file
        return (student_path / "student_model.pkl").exists()

    def train_student(
        self,
        teacher_run: dict[str, Any],
    ) -> dict[str, Any]:
        """Train student model for a teacher run.

        Args:
            teacher_run: Teacher run metadata dict

        Returns:
            Result dict with status, metrics, error (if failed)
        """
        symbol = teacher_run["symbol"]
        teacher_artifacts_path = teacher_run["artifacts_path"]
        student_artifacts_path = self.get_student_artifacts_path(teacher_run)

        logger.info(f"Training student for {symbol} from {teacher_artifacts_path}")

        # Build command to invoke train_student.py
        cmd = [
            sys.executable,
            "scripts/train_student.py",
            "--teacher-dir",
            teacher_artifacts_path,
            "--output-dir",
            str(student_artifacts_path),
            "--batch-mode",
            "--baseline-precision",
            str(self.baseline_precision),
            "--baseline-recall",
            str(self.baseline_recall),
        ]

        # US-028 Phase 7 Initiative 2: Add reward loop flags from Settings
        from src.app.config import Settings
        settings = Settings()  # type: ignore[call-arg]

        if settings.reward_loop_enabled:
            cmd.append("--enable-reward-loop")
            cmd.extend(["--reward-horizon-days", str(settings.reward_horizon_days)])
            cmd.extend(["--reward-weighting-mode", settings.reward_weighting_mode])

            if settings.reward_ab_testing_enabled:
                cmd.append("--reward-ab-testing")

            logger.info(f"  Reward loop enabled: mode={settings.reward_weighting_mode}, horizon={settings.reward_horizon_days} days")

        try:
            # Execute training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per student
            )

            if result.returncode == 0:
                logger.info(f"✓ Student for {symbol} completed successfully")

                # Extract metrics from output (if available)
                metrics = self._extract_metrics_from_output(result.stdout)

                # US-028 Phase 7 Initiative 2: Extract reward metrics
                reward_metrics = self._extract_reward_metrics_from_output(result.stdout)

                # Find promotion checklist
                checklist_path = student_artifacts_path / "promotion_checklist.md"

                return {
                    "status": "success",
                    "metrics": metrics,
                    "promotion_checklist_path": str(checklist_path)
                    if checklist_path.exists()
                    else None,
                    "reward_metrics": reward_metrics,  # US-028 Phase 7 Initiative 2
                }
            else:
                logger.error(f"✗ Student for {symbol} failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr[:500],  # Truncate long errors
                    "metrics": None,
                    "promotion_checklist_path": None,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"✗ Student for {symbol} timed out")
            return {
                "status": "failed",
                "error": "Training timeout (10 minutes exceeded)",
                "metrics": None,
                "promotion_checklist_path": None,
            }
        except Exception as e:
            logger.error(f"✗ Student for {symbol} error: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "metrics": None,
                "promotion_checklist_path": None,
            }

    def _extract_metrics_from_output(self, stdout: str) -> dict[str, float] | None:
        """Extract metrics from training output.

        Args:
            stdout: Training script stdout

        Returns:
            Dict with precision, recall, f1 or None if not found
        """
        # Simple parsing - look for common metrics patterns
        metrics = {}

        for line in stdout.split("\n"):
            line_lower = line.lower()
            if "precision" in line_lower and ":" in line:
                try:
                    parts = line.split(":")
                    value = float(parts[-1].strip().rstrip("%"))
                    if value > 1:  # If percentage
                        value /= 100.0
                    metrics["precision"] = value
                except (ValueError, IndexError):
                    pass
            elif "recall" in line_lower and ":" in line:
                try:
                    parts = line.split(":")
                    value = float(parts[-1].strip().rstrip("%"))
                    if value > 1:
                        value /= 100.0
                    metrics["recall"] = value
                except (ValueError, IndexError):
                    pass

        return metrics if metrics else None

    def _extract_reward_metrics_from_output(self, stdout: str) -> dict[str, Any] | None:
        """Extract reward metrics from training output (US-028 Phase 7 Initiative 2).

        Args:
            stdout: Training script stdout

        Returns:
            Dict with reward metrics or None if not found
        """
        for line in stdout.split("\n"):
            if line.startswith("REWARD_METRICS_JSON:"):
                try:
                    json_str = line.split("REWARD_METRICS_JSON:", 1)[1].strip()
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"Failed to parse reward metrics JSON: {e}")
                    return None
        return None

    def log_metadata(
        self,
        teacher_run: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Log student batch metadata.

        Args:
            teacher_run: Teacher run metadata dict
            result: Student training result dict
        """
        teacher_run_id = teacher_run.get("window_label") or Path(teacher_run["artifacts_path"]).name

        # US-024 Phase 3: Get sentiment snapshot path from teacher run if available
        sentiment_snapshot_path = teacher_run.get("sentiment_snapshot_path")

        # US-028 Phase 7 Initiative 2: reward_loop_enabled is determined automatically from reward_metrics

        StudentModel.log_batch_metadata(
            metadata_file=self.student_metadata_file,
            batch_id=self.batch_id,
            symbol=teacher_run["symbol"],
            teacher_run_id=teacher_run_id,
            teacher_artifacts_path=teacher_run["artifacts_path"],
            student_artifacts_path=str(self.get_student_artifacts_path(teacher_run)),
            metrics=result.get("metrics"),
            promotion_checklist_path=result.get("promotion_checklist_path"),
            status=result["status"],
            error=result.get("error"),
            sentiment_snapshot_path=sentiment_snapshot_path,
            incremental=getattr(self, "incremental", False),  # US-024 Phase 4
            reward_metrics=result.get("reward_metrics"),  # US-028 Phase 7 Initiative 2
        )

    def run_batch(self) -> dict[str, Any]:
        """Run batch student training.

        Returns:
            Summary dict with statistics
        """
        # Filter to successful teacher runs only
        successful_teacher_runs = [
            run for run in self.teacher_runs if run.get("status") == "success"
        ]

        logger.info(
            f"Found {len(successful_teacher_runs)} successful teacher runs "
            f"(out of {len(self.teacher_runs)} total)"
        )

        # Initialize student metadata file if it doesn't exist
        if not self.student_metadata_file.exists():
            self.student_metadata_file.write_text("")

        # Train student for each successful teacher run
        # US-028 Phase 7 Initiative 4: Progress monitoring with tqdm
        with tqdm(total=len(successful_teacher_runs), desc="Training student models", unit="batch") as pbar:
            for i, teacher_run in enumerate(successful_teacher_runs, 1):
                symbol = teacher_run["symbol"]
                logger.info(f"[{i}/{len(successful_teacher_runs)}] Processing {symbol}")

                # Check resume
                if self.resume and self.is_already_trained(teacher_run):
                    logger.info(f"⊙ Student for {symbol} already trained (skipping)")
                    self.stats["skipped"] += 1
                    # US-028 Phase 7 Initiative 4: Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "symbol": symbol[:15],
                        "status": "skipped",
                        "trained": self.stats["completed"],
                        "skipped": self.stats["skipped"],
                        "failed": self.stats["failed"],
                    })
                    continue

                # Train student
                result = self.train_student(teacher_run)

                # Log metadata
                self.log_metadata(teacher_run, result)

                # Update stats
                if result["status"] == "success":
                    self.stats["completed"] += 1
                else:
                    self.stats["failed"] += 1

                # US-028 Phase 7 Initiative 4: Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "symbol": symbol[:15],
                    "status": result["status"],
                    "trained": self.stats["completed"],
                    "skipped": self.stats["skipped"],
                    "failed": self.stats["failed"],
                })

        # Generate summary
        trained_count = len(successful_teacher_runs)
        summary = {
            "batch_id": self.batch_id,
            "teacher_batch_dir": str(self.teacher_batch_dir),
            "total_teacher_runs": self.stats["total_teachers"],
            "successful_teacher_runs": len(successful_teacher_runs),
            "student_completed": self.stats["completed"],
            "student_failed": self.stats["failed"],
            "student_skipped": self.stats["skipped"],
            "success_rate": (self.stats["completed"] / trained_count if trained_count > 0 else 0.0),
        }

        return summary


def find_latest_teacher_batch(base_dir: Path) -> Path | None:
    """Find the latest teacher batch directory.

    Args:
        base_dir: Base models directory

    Returns:
        Path to latest batch directory or None if none found
    """
    # Find directories with timestamp format (YYYYMMDD_HHMMSS)
    batch_dirs = sorted(base_dir.glob("[0-9]*_[0-9]*"), reverse=True)

    for batch_dir in batch_dirs:
        # Check if it has teacher_runs.json
        if (batch_dir / "teacher_runs.json").exists():
            return batch_dir

    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Student training from Teacher batch outputs"
    )

    parser.add_argument(
        "--teacher-batch-dir",
        type=str,
        help="Teacher batch directory (default: latest in batch_training_output_dir)",
    )
    parser.add_argument(
        "--baseline-precision",
        type=float,
        help="Baseline precision for promotion (default: from settings)",
    )
    parser.add_argument(
        "--baseline-recall",
        type=float,
        help="Baseline recall for promotion (default: from settings)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partial batch (skip already-trained students)",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode (train only for new teacher runs)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load settings
    settings = Settings()  # type: ignore[call-arg]

    # Determine teacher batch directory
    if args.teacher_batch_dir:
        teacher_batch_dir = Path(args.teacher_batch_dir)
    else:
        # Find latest batch
        base_dir = Path(settings.batch_training_output_dir)
        teacher_batch_dir = find_latest_teacher_batch(base_dir)

        if teacher_batch_dir is None:
            logger.error(f"No teacher batch found in {base_dir}")
            logger.error("Run train_teacher_batch.py first or specify --teacher-batch-dir")
            return 1

    # Verify teacher batch directory exists
    if not teacher_batch_dir.exists():
        logger.error(f"Teacher batch directory not found: {teacher_batch_dir}")
        return 1

    # Verify teacher_runs.json exists
    teacher_metadata_file = teacher_batch_dir / "teacher_runs.json"
    if not teacher_metadata_file.exists():
        logger.error(f"Teacher metadata not found: {teacher_metadata_file}")
        return 1

    # Use baseline from args or settings
    baseline_precision = (
        args.baseline_precision
        if args.baseline_precision
        else settings.student_batch_baseline_precision
    )
    baseline_recall = (
        args.baseline_recall if args.baseline_recall else settings.student_batch_baseline_recall
    )

    logger.info("=" * 70)
    logger.info("BATCH STUDENT TRAINING")
    logger.info("=" * 70)
    logger.info(f"Teacher batch: {teacher_batch_dir}")
    logger.info(f"Baseline precision: {baseline_precision:.2%}")
    logger.info(f"Baseline recall: {baseline_recall:.2%}")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 70)

    # Create student batch trainer
    trainer = StudentBatchTrainer(
        teacher_batch_dir,
        baseline_precision,
        baseline_recall,
        resume=args.resume,
        incremental=getattr(args, "incremental", False),
        workers=getattr(args, "workers", 1),
    )

    # Run batch
    try:
        summary = trainer.run_batch()
    except Exception as e:
        logger.error(f"Fatal error during batch student training: {e}")
        return 1

    # Print summary
    logger.info("=" * 70)
    logger.info("STUDENT BATCH SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Batch ID: {summary['batch_id']}")
    logger.info(f"Teacher runs: {summary['total_teacher_runs']} total")
    logger.info(f"Successful teacher runs: {summary['successful_teacher_runs']}")
    logger.info(f"Student completed: {summary['student_completed']}")
    logger.info(f"Student failed: {summary['student_failed']}")
    logger.info(f"Student skipped: {summary['student_skipped']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info("=" * 70)
    logger.info(f"Student metadata: {trainer.student_metadata_file}")
    logger.info("=" * 70)

    # Return exit code based on failures
    if summary["student_failed"] > 0:
        logger.warning(f"{summary['student_failed']} student trainings failed")
        return 1

    logger.info("All student trainings completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
