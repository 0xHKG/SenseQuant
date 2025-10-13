"""Model Validation Orchestration Script (US-025).

Runs comprehensive model validation workflow:
1. Teacher & student batch training
2. Optimizer evaluation (read-only)
3. Report generation (notebooks -> HTML)
4. Validation summary creation
5. State recording for audit trail

Usage:
    # Dryrun mode (default)
    python scripts/run_model_validation.py

    # Real validation
    python scripts/run_model_validation.py --no-dryrun

    # Custom configuration
    python scripts/run_model_validation.py \\
        --symbols RELIANCE TCS \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.config import Settings
from src.services.state_manager import StateManager


class ModelValidationRunner:
    """Orchestrates model validation workflow."""

    def __init__(
        self,
        run_id: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
        dryrun: bool = True,
        skip_optimizer: bool = False,
        skip_reports: bool = False,
    ):
        """Initialize validation runner.

        Args:
            run_id: Unique validation run identifier
            symbols: List of stock symbols to validate
            start_date: Validation start date (YYYY-MM-DD)
            end_date: Validation end date (YYYY-MM-DD)
            dryrun: If True, run in dryrun mode
            skip_optimizer: If True, skip optimizer evaluation
            skip_reports: If True, skip report generation
        """
        self.run_id = run_id
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.dryrun = dryrun
        self.skip_optimizer = skip_optimizer
        self.skip_reports = skip_reports

        # Setup directories
        self.models_dir = Path("data/models") / run_id
        self.optimization_dir = Path("data/optimization") / run_id
        self.release_dir = Path("release") / f"audit_{run_id}"
        self.reports_dir = self.release_dir / "reports"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # State manager
        self.state_manager = StateManager(Path("data/state/validation_runs.json"))

        # Results
        self.results = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "date_range": {"start": start_date, "end": end_date},
            "dryrun": dryrun,
            "status": "running",
            "teacher_results": {},
            "student_results": {},
            "optimizer_results": {},
            "reports": [],
            "errors": [],
        }

    def run(self) -> dict[str, Any]:
        """Execute validation workflow.

        Returns:
            Dict with validation results
        """
        logger.info("=" * 70)
        logger.info(f"MODEL VALIDATION RUN: {self.run_id}")
        logger.info("=" * 70)
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Dryrun mode: {self.dryrun}")
        logger.info("=" * 70)

        try:
            # Step 1: Teacher batch training
            logger.info("Step 1/5: Teacher batch training")
            self._run_teacher_batch()

            # Step 2: Student batch training
            logger.info("Step 2/5: Student batch training")
            self._run_student_batch()

            # Step 3: Optimizer evaluation (optional)
            if not self.skip_optimizer:
                logger.info("Step 3/5: Optimizer evaluation")
                self._run_optimizer()
            else:
                logger.info("Step 3/5: Skipping optimizer")

            # Step 4: Report generation (optional)
            if not self.skip_reports:
                logger.info("Step 4/5: Report generation")
                self._generate_reports()
            else:
                logger.info("Step 4/5: Skipping reports")

            # Step 5: Summary generation
            logger.info("Step 5/5: Summary generation")
            self._generate_summary()

            # Mark as completed
            self.results["status"] = "completed"

            # Record in state
            self._record_state()

            logger.info("=" * 70)
            logger.info("VALIDATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results["status"] = "failed"
            self.results["errors"].append(str(e))
            self._record_state()
            raise

        return self.results

    def _run_teacher_batch(self) -> None:
        """Run teacher batch training."""
        cmd = [
            sys.executable,
            "scripts/train_teacher_batch.py",
            "--symbols",
            *self.symbols,
            "--start-date",
            self.start_date,
            "--end-date",
            self.end_date,
        ]

        if self.dryrun:
            logger.warning("Dryrun mode: Skipping actual teacher training")
            self.results["teacher_results"] = {"status": "skipped", "reason": "dryrun"}
            return

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("✓ Teacher batch training completed")
                self.results["teacher_results"] = {"status": "success"}
            else:
                raise RuntimeError(f"Teacher training failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Teacher training error: {e}")
            self.results["errors"].append(f"Teacher training: {str(e)}")
            raise

    def _run_student_batch(self) -> None:
        """Run student batch training."""
        # In dryrun, skip actual execution
        if self.dryrun:
            logger.warning("Dryrun mode: Skipping actual student training")
            self.results["student_results"] = {"status": "skipped", "reason": "dryrun"}
            return

        cmd = [sys.executable, "scripts/train_student_batch.py"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("✓ Student batch training completed")
                self.results["student_results"] = {"status": "success"}
            else:
                raise RuntimeError(f"Student training failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Student training error: {e}")
            self.results["errors"].append(f"Student training: {str(e)}")
            raise

    def _run_optimizer(self) -> None:
        """Run optimizer in read-only mode."""
        if self.dryrun:
            logger.warning("Dryrun mode: Skipping actual optimizer")
            self.results["optimizer_results"] = {
                "status": "skipped",
                "reason": "dryrun",
                "best_configs": {},
            }
            return

        # Create default search space for optimizer
        search_space_file = Path("config/search_space.yaml")
        if not search_space_file.exists():
            logger.warning("Search space file not found, using default parameters")
            # Create minimal search space
            search_space_file.parent.mkdir(parents=True, exist_ok=True)
            default_search_space = {
                "rsi_overbought": [65, 70, 75],
                "rsi_oversold": [25, 30, 35],
                "rsi_period": [14],
            }
            import yaml

            with open(search_space_file, "w") as f:
                yaml.dump(default_search_space, f)

        # Run optimizer script
        cmd = [
            sys.executable,
            "scripts/optimize.py",
            "--config",
            str(search_space_file),
            "--symbols",
            *self.symbols,
            "--start-date",
            self.start_date,
            "--end-date",
            self.end_date,
            "--output-dir",
            str(self.optimization_dir),
            "--top-n",
            "5",
        ]

        try:
            logger.info(f"Running optimizer: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

            if result.returncode == 0:
                logger.info("✓ Optimizer evaluation completed (read-only)")

                # Load optimizer results
                best_configs = {}
                summary_file = self.optimization_dir / "optimization_summary.json"
                configs_file = self.optimization_dir / "configs.json"

                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                        best_configs["summary"] = summary

                if configs_file.exists():
                    with open(configs_file) as f:
                        configs = json.load(f)
                        if configs:
                            # Get top config
                            best = configs[0]
                            best_configs["best_config"] = {
                                "config_id": best.get("config_id"),
                                "parameters": best.get("parameters", {}),
                                "score": best.get("score", 0.0),
                                "metrics": best.get("metrics", {}),
                            }

                self.results["optimizer_results"] = {
                    "status": "success",
                    "output_dir": str(self.optimization_dir),
                    "best_configs": best_configs,
                }
            else:
                logger.warning(f"Optimizer returned non-zero exit code: {result.returncode}")
                logger.warning(f"Stderr: {result.stderr}")
                self.results["optimizer_results"] = {
                    "status": "failed",
                    "error": result.stderr,
                    "best_configs": {},
                }

        except subprocess.TimeoutExpired:
            logger.error("Optimizer timed out after 2 hours")
            self.results["optimizer_results"] = {
                "status": "failed",
                "error": "Timeout after 2 hours",
                "best_configs": {},
            }
        except Exception as e:
            logger.error(f"Optimizer error: {e}")
            self.results["optimizer_results"] = {
                "status": "failed",
                "error": str(e),
                "best_configs": {},
            }

    def _generate_reports(self) -> None:
        """Generate HTML reports from notebooks."""
        if self.dryrun:
            logger.warning("Dryrun mode: Skipping report generation")
            self.results["reports"] = []
            return

        reports_generated = []

        # Define notebooks to export
        notebooks = [
            ("accuracy_report", "notebooks/accuracy_report.ipynb"),
            ("optimization_report", "notebooks/optimization_report.ipynb"),
        ]

        for notebook_name, notebook_path in notebooks:
            notebook_file = Path(notebook_path)

            if not notebook_file.exists():
                logger.warning(f"Notebook not found: {notebook_path}")
                continue

            try:
                # Use nbconvert to generate HTML
                output_file = self.reports_dir / f"{notebook_name}.html"

                cmd = [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "html",
                    "--execute",
                    "--no-input",  # Hide input cells
                    "--output-dir",
                    str(self.reports_dir),
                    "--output",
                    notebook_name,
                    str(notebook_file),
                ]

                logger.info(f"Generating {notebook_name}.html...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0:
                    logger.info(f"✓ Generated {output_file}")
                    reports_generated.append(str(output_file))
                else:
                    logger.warning(f"Failed to generate {notebook_name}: {result.stderr[:200]}")

            except subprocess.TimeoutExpired:
                logger.error(f"Notebook export timed out: {notebook_name}")
            except FileNotFoundError:
                logger.error("jupyter nbconvert not found. Install with: pip install nbconvert")
                break
            except Exception as e:
                logger.error(f"Error generating {notebook_name}: {e}")

        self.results["reports"] = reports_generated

    def _generate_summary(self) -> None:
        """Generate validation summary with optimizer results and accuracy metrics."""
        # Load teacher/student results if available
        teacher_metrics = self._load_teacher_metrics()
        student_metrics = self._load_student_metrics()

        # Get optimizer best configs
        optimizer_best = {}
        if self.results.get("optimizer_results", {}).get("best_configs"):
            optimizer_best = self.results["optimizer_results"]["best_configs"].get(
                "best_config", {}
            )

        # Calculate promotion recommendation
        promotion_approved = False
        promotion_reason = "Validation incomplete"

        if student_metrics and not self.dryrun:
            avg_accuracy = student_metrics.get("avg_accuracy", 0.0)
            avg_precision = student_metrics.get("avg_precision", 0.0)

            if avg_accuracy >= 0.80 and avg_precision >= 0.75:
                promotion_approved = True
                promotion_reason = (
                    f"Accuracy thresholds met (accuracy={avg_accuracy:.1%}, "
                    f"precision={avg_precision:.1%})"
                )
            else:
                promotion_reason = (
                    f"Accuracy thresholds not met (accuracy={avg_accuracy:.1%}, "
                    f"precision={avg_precision:.1%})"
                )
        elif self.dryrun:
            promotion_reason = "Dryrun mode - real validation required"

        # US-029 Phase 2: Load feature configuration
        settings = Settings()

        summary = {
            "run_id": self.run_id,
            "timestamp": self.results["timestamp"],
            "status": "completed" if not self.results["errors"] else "failed",
            "symbols": self.symbols,
            "date_range": {"start": self.start_date, "end": self.end_date},
            "dryrun": self.dryrun,
            "feature_set": {
                "order_book_enabled": settings.enable_order_book_features,
                "options_enabled": settings.enable_options_features,
                "macro_enabled": settings.enable_macro_features,
            },
            "teacher_results": {
                "status": self.results["teacher_results"].get("status", "unknown"),
                "metrics": teacher_metrics,
            },
            "student_results": {
                "status": self.results["student_results"].get("status", "unknown"),
                "metrics": student_metrics,
            },
            "optimizer_results": {
                "status": self.results["optimizer_results"].get("status", "unknown"),
                "best_config": optimizer_best,
                "output_dir": self.results["optimizer_results"].get("output_dir"),
            },
            "reports": self.results.get("reports", []),
            "promotion_recommendation": {
                "approved": promotion_approved,
                "reason": promotion_reason,
                "next_steps": [
                    "Review validation summary",
                    "Check optimizer best configurations",
                    "Verify accuracy metrics meet thresholds",
                    "Promote models if approved"
                    if promotion_approved
                    else "Re-run validation with adjustments",
                ],
            },
        }

        # Write summary JSON
        summary_json = self.release_dir / "validation_summary.json"
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        # Write summary Markdown
        summary_md = self.release_dir / "validation_summary.md"
        with open(summary_md, "w") as f:
            f.write(f"# Validation Summary: {self.run_id}\n\n")
            f.write(f"**Status**: {summary['status']}\n")
            f.write(f"**Timestamp**: {summary['timestamp']}\n")
            f.write(f"**Symbols**: {', '.join(self.symbols)}\n")
            f.write(f"**Date Range**: {self.start_date} to {self.end_date}\n")
            f.write(f"**Dryrun**: {self.dryrun}\n\n")

            # US-029 Phase 2: Include feature set configuration
            f.write("**Feature Set** (US-029 Phase 2):\n")
            f.write(
                f"- Order Book Features: {'Enabled' if summary['feature_set']['order_book_enabled'] else 'Disabled'}\n"
            )
            f.write(
                f"- Options Features: {'Enabled' if summary['feature_set']['options_enabled'] else 'Disabled'}\n"
            )
            f.write(
                f"- Macro Features: {'Enabled' if summary['feature_set']['macro_enabled'] else 'Disabled'}\n\n"
            )

            f.write("---\n\n")
            f.write("## Teacher Results\n\n")
            f.write(f"**Status**: {summary['teacher_results']['status']}\n\n")
            if teacher_metrics:
                f.write("**Metrics**:\n")
                for key, value in teacher_metrics.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

            f.write("## Student Results\n\n")
            f.write(f"**Status**: {summary['student_results']['status']}\n\n")
            if student_metrics:
                f.write("**Metrics**:\n")
                for key, value in student_metrics.items():
                    if isinstance(value, float):
                        f.write(f"- {key}: {value:.2%}\n")
                    else:
                        f.write(f"- {key}: {value}\n")
                f.write("\n")

            f.write("## Optimizer Results\n\n")
            f.write(f"**Status**: {summary['optimizer_results']['status']}\n\n")
            if optimizer_best:
                f.write("**Best Configuration**:\n")
                f.write(f"- Config ID: {optimizer_best.get('config_id', 'N/A')}\n")
                f.write(f"- Score: {optimizer_best.get('score', 0.0):.3f}\n")
                f.write("- Parameters:\n")
                for param, value in optimizer_best.get("parameters", {}).items():
                    f.write(f"  - {param}: {value}\n")
                f.write("\n")

            f.write("## Generated Reports\n\n")
            if summary["reports"]:
                for report in summary["reports"]:
                    f.write(f"- {report}\n")
            else:
                f.write("No reports generated\n")
            f.write("\n")

            f.write("---\n\n")
            f.write("## Promotion Recommendation\n\n")
            f.write(f"**Approved**: {promotion_approved}\n")
            f.write(f"**Reason**: {promotion_reason}\n\n")
            f.write("**Next Steps**:\n")
            for step in summary["promotion_recommendation"]["next_steps"]:
                f.write(f"- {step}\n")

        logger.info(f"✓ Summary generated: {summary_md}")
        self.results["summary"] = str(summary_json)

    def _load_teacher_metrics(self) -> dict[str, Any]:
        """Load teacher training metrics from teacher_runs.json."""
        teacher_runs_file = self.models_dir / "teacher_runs.json"
        if not teacher_runs_file.exists():
            return {}

        try:
            with open(teacher_runs_file) as f:
                runs = [json.loads(line) for line in f if line.strip()]
                if not runs:
                    return {}

                # Calculate aggregate metrics
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
        """Load student training metrics from student_runs.json."""
        student_runs_file = self.models_dir / "student_runs.json"
        if not student_runs_file.exists():
            return {}

        try:
            with open(student_runs_file) as f:
                runs = [json.loads(line) for line in f if line.strip()]
                if not runs:
                    return {}

                # Calculate aggregate metrics
                accuracies = [r.get("metrics", {}).get("accuracy", 0.0) for r in runs]
                precisions = [r.get("metrics", {}).get("precision", 0.0) for r in runs]
                recalls = [r.get("metrics", {}).get("recall", 0.0) for r in runs]

                return {
                    "runs_completed": len(runs),
                    "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
                    "avg_precision": sum(precisions) / len(precisions) if precisions else 0.0,
                    "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
                }
        except Exception as e:
            logger.warning(f"Failed to load student metrics: {e}")
            return {}

    def _record_state(self) -> None:
        """Record validation run in state manager."""
        self.state_manager.record_validation_run(
            run_id=self.run_id,
            timestamp=self.results["timestamp"],
            symbols=self.symbols,
            date_range={"start": self.start_date, "end": self.end_date},
            status=self.results["status"],
            dryrun=self.dryrun,
            results=self.results,
        )
        logger.info("✓ Validation run recorded in state")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run model validation workflow (US-025)")

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Stock symbols to validate",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Validation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Validation end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-dryrun",
        action="store_true",
        help="Disable dryrun mode (run with real data)",
    )
    parser.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="Skip optimizer evaluation",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip report generation",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Use command-line args or settings defaults
    symbols = args.symbols or ["RELIANCE", "TCS"]
    start_date = args.start_date or "2024-01-01"
    end_date = args.end_date or "2024-12-31"
    dryrun = not args.no_dryrun

    # Generate run ID
    run_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create runner
    runner = ModelValidationRunner(
        run_id=run_id,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        dryrun=dryrun,
        skip_optimizer=args.skip_optimizer,
        skip_reports=args.skip_reports,
    )

    # Run validation
    try:
        results = runner.run()
        logger.info(f"Validation results: {results}")
        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
