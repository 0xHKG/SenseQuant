#!/usr/bin/env python
"""Run black-swan stress tests against historical crisis periods (US-028 Phase 7 Initiative 3).

This script replays trained models (teacher/student) against historical stress periods
(2008 crash, 2020 COVID, etc.) to assess resilience and identify weaknesses.

Usage:
    # Test all high/extreme severity periods
    python scripts/run_stress_tests.py --batch-id batch_20251015_120000

    # Test specific periods
    python scripts/run_stress_tests.py \\
        --batch-id batch_20251015_120000 \\
        --periods covid_crash_2020 global_financial_crisis

    # Test all periods
    python scripts/run_stress_tests.py --batch-id batch_20251015_120000 --all

    # Filter by severity
    python scripts/run_stress_tests.py \\
        --batch-id batch_20251015_120000 \\
        --severity extreme high
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.config import Settings


class StressTestRunner:
    """Run stress tests against historical crisis periods."""

    def __init__(
        self,
        batch_id: str,
        output_dir: Path,
        settings: Settings | None = None,
    ):
        """Initialize stress test runner.

        Args:
            batch_id: Batch ID of models to test
            output_dir: Output directory for stress test results
            settings: Application settings (creates new if None)
        """
        self.batch_id = batch_id
        self.output_dir = output_dir
        self.settings = settings or Settings()  # type: ignore[call-arg]

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load teacher/student metadata
        self.batch_dir = Path(self.settings.student_batch_output_dir) / batch_id
        self.teacher_metadata_file = self.batch_dir / "teacher_runs.json"
        self.student_metadata_file = self.batch_dir / "student_runs.json"

        logger.info(f"Initialized stress test runner for batch: {batch_id}")
        logger.info(f"Output directory: {output_dir}")

    def load_teacher_student_pairs(self) -> list[dict[str, Any]]:
        """Load teacher-student model pairs from metadata.

        Returns:
            List of dicts with teacher/student paths and metadata
        """
        if not self.student_metadata_file.exists():
            logger.error(f"Student metadata not found: {self.student_metadata_file}")
            return []

        # Load student runs (JSONL format)
        pairs = []
        with open(self.student_metadata_file) as f:
            for line in f:
                if not line.strip():
                    continue
                student_run = json.loads(line)
                if student_run.get("status") == "success":
                    pairs.append(student_run)

        logger.info(f"Loaded {len(pairs)} successful student model pairs")
        return pairs

    def load_student_model(self, student_artifacts_path: Path) -> Any:
        """Load student model from artifacts.

        Args:
            student_artifacts_path: Path to student artifacts directory

        Returns:
            Loaded student model or None if failed
        """
        model_path = student_artifacts_path / "student_model.pkl"
        if not model_path.exists():
            logger.warning(f"Student model not found: {model_path}")
            return None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Failed to load student model from {model_path}: {e}")
            return None

    def load_historical_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """Load historical OHLCV data for stress period.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with historical data or None if not available
        """
        data_dir = Path(self.settings.historical_data_output_dir) / symbol / "1day"
        if not data_dir.exists():
            logger.warning(f"Historical data directory not found: {data_dir}")
            return None

        # Load all CSV files in date range
        csv_files = sorted(data_dir.glob("*.csv"))
        all_data = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, parse_dates=["timestamp"])
                # Filter by date range
                df = df[
                    (df["timestamp"] >= start_date)
                    & (df["timestamp"] <= end_date)
                ]
                if len(df) > 0:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")

        if not all_data:
            logger.warning(f"No data found for {symbol} in period {start_date} to {end_date}")
            return None

        # Combine and deduplicate
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return combined

    def calculate_resilience_metrics(
        self,
        predictions: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate resilience metrics for stress period.

        Args:
            predictions: DataFrame with predictions and timestamps
            price_data: DataFrame with actual price data

        Returns:
            Dict of resilience metrics
        """
        metrics = {}

        # Calculate returns based on predictions
        returns = []
        for _idx, row in predictions.iterrows():
            pred = row["prediction"]
            timestamp = pd.to_datetime(row["timestamp"])

            # Find next day's price
            future_prices = price_data[price_data["timestamp"] > timestamp]
            if len(future_prices) == 0:
                continue

            current_price = row.get("close", None)
            if current_price is None:
                # Look up in price data
                current_row = price_data[price_data["timestamp"] == timestamp]
                if len(current_row) > 0:
                    current_price = current_row.iloc[0]["close"]
                else:
                    continue

            future_price = future_prices.iloc[0]["close"]
            actual_return = (future_price - current_price) / current_price

            # Simulate strategy return based on prediction
            # 0=down (short), 1=neutral (hold), 2=up (long)
            if pred == 0:
                strategy_return = -actual_return  # Short position
            elif pred == 1:
                strategy_return = 0.0  # No position
            else:  # pred == 2
                strategy_return = actual_return  # Long position

            returns.append(strategy_return)

        if len(returns) == 0:
            return {
                "total_predictions": 0,
                "cumulative_return": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "sharpe_ratio": 0.0,
                "recovery_time_days": 0,
            }

        returns_array = np.array(returns)

        # Cumulative return
        cumulative_return = float(np.sum(returns_array))

        # Max drawdown
        cumulative = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Hit rate (% positive returns)
        hit_rate = float(np.sum(returns_array > 0) / len(returns_array)) if len(returns_array) > 0 else 0.0

        # Sharpe ratio (annualized, assuming daily returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe_ratio = float((mean_return / std_return) * np.sqrt(252)) if std_return > 0 else 0.0

        # Recovery time (days to recover from max drawdown)
        recovery_time = 0
        if max_drawdown > 0:
            max_dd_idx = int(np.argmax(drawdown))
            # Find when cumulative returns exceed running_max again
            for i in range(max_dd_idx + 1, len(cumulative)):
                if cumulative[i] >= running_max[max_dd_idx]:
                    recovery_time = i - max_dd_idx
                    break
            if recovery_time == 0:
                recovery_time = len(cumulative) - max_dd_idx  # Never recovered

        metrics = {
            "total_predictions": len(returns),
            "cumulative_return": cumulative_return,
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate,
            "sharpe_ratio": sharpe_ratio,
            "recovery_time_days": recovery_time,
            "mean_return": float(mean_return),
            "return_volatility": float(std_return),
            "positive_days": int(np.sum(returns_array > 0)),
            "negative_days": int(np.sum(returns_array < 0)),
        }

        return metrics

    def run_stress_test(
        self,
        student_run: dict[str, Any],
        stress_period: dict[str, Any],
    ) -> dict[str, Any]:
        """Run stress test for a single model against a stress period.

        Args:
            student_run: Student model metadata dict
            stress_period: Stress period metadata dict

        Returns:
            Dict with stress test results
        """
        symbol = student_run["symbol"]
        period_id = stress_period["id"]

        logger.info(f"Testing {symbol} against {period_id} ({stress_period['name']})")

        # Load student model
        student_path = Path(student_run["student_artifacts_path"])
        model = self.load_student_model(student_path)

        if model is None:
            return {
                "symbol": symbol,
                "period_id": period_id,
                "status": "failed",
                "error": "Failed to load model",
            }

        # Load historical data for stress period
        price_data = self.load_historical_data(
            symbol, stress_period["start_date"], stress_period["end_date"]
        )

        if price_data is None or len(price_data) == 0:
            return {
                "symbol": symbol,
                "period_id": period_id,
                "status": "skipped",
                "error": "No data available for stress period",
            }

        # Load teacher-generated labels/features for prediction
        teacher_path = Path(student_run["teacher_artifacts_path"])
        labels_file = teacher_path / "labels.csv.gz"

        if not labels_file.exists():
            return {
                "symbol": symbol,
                "period_id": period_id,
                "status": "failed",
                "error": "Teacher labels not found",
            }

        # Load labels (contains features)
        try:
            labels_df = pd.read_csv(labels_file, compression="gzip", parse_dates=["timestamp"])
        except Exception as e:
            return {
                "symbol": symbol,
                "period_id": period_id,
                "status": "failed",
                "error": f"Failed to load labels: {e}",
            }

        # Filter labels to stress period
        labels_df = labels_df[
            (labels_df["timestamp"] >= stress_period["start_date"])
            & (labels_df["timestamp"] <= stress_period["end_date"])
        ]

        if len(labels_df) == 0:
            return {
                "symbol": symbol,
                "period_id": period_id,
                "status": "skipped",
                "error": "No labels in stress period",
            }

        # Extract features (exclude non-feature columns)
        non_feature_cols = ["timestamp", "ts", "label", "symbol", "forward_return"]
        features = labels_df.drop(columns=[col for col in non_feature_cols if col in labels_df.columns])

        # Make predictions
        try:
            predictions = model.predict(features)
        except Exception as e:
            return {
                "symbol": symbol,
                "period_id": period_id,
                "status": "failed",
                "error": f"Prediction failed: {e}",
            }

        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            "timestamp": labels_df["timestamp"],
            "prediction": predictions,
            "close": labels_df.get("close", [None] * len(predictions)),
        })

        # Calculate resilience metrics
        metrics = self.calculate_resilience_metrics(predictions_df, price_data)

        result = {
            "symbol": symbol,
            "period_id": period_id,
            "period_name": stress_period["name"],
            "start_date": stress_period["start_date"],
            "end_date": stress_period["end_date"],
            "severity": stress_period.get("severity", "unknown"),
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def run_all_stress_tests(
        self,
        stress_periods: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run stress tests across all models and periods.

        Args:
            stress_periods: List of stress period dicts

        Returns:
            Summary dict with all results
        """
        # Load model pairs
        pairs = self.load_teacher_student_pairs()

        if not pairs:
            logger.error("No successful student models found")
            return {"status": "failed", "error": "No models to test"}

        # Run tests
        results = []
        for pair in pairs:
            for period in stress_periods:
                result = self.run_stress_test(pair, period)
                results.append(result)

        # Aggregate results
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        skipped = [r for r in results if r["status"] == "skipped"]

        summary = {
            "batch_id": self.batch_id,
            "total_tests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "timestamp": datetime.now().isoformat(),
            "periods_tested": [p["id"] for p in stress_periods],
            "symbols_tested": list({p["symbol"] for p in pairs}),
            "results": results,
        }

        # Calculate aggregate metrics
        if successful:
            summary["aggregate_metrics"] = self._calculate_aggregate_metrics(successful)

        return summary

    def _calculate_aggregate_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate aggregate metrics across all successful tests."""
        all_metrics = [r["metrics"] for r in results]

        return {
            "mean_cumulative_return": float(np.mean([m["cumulative_return"] for m in all_metrics])),
            "mean_max_drawdown": float(np.mean([m["max_drawdown"] for m in all_metrics])),
            "mean_hit_rate": float(np.mean([m["hit_rate"] for m in all_metrics])),
            "mean_sharpe_ratio": float(np.mean([m["sharpe_ratio"] for m in all_metrics])),
            "worst_drawdown": float(np.max([m["max_drawdown"] for m in all_metrics])),
            "best_sharpe": float(np.max([m["sharpe_ratio"] for m in all_metrics])),
            "worst_sharpe": float(np.min([m["sharpe_ratio"] for m in all_metrics])),
        }

    def save_results(self, summary: dict[str, Any]) -> None:
        """Save stress test results to JSON file.

        Args:
            summary: Summary dict with all results
        """
        # Save detailed JSON
        json_path = self.output_dir / "stress_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved detailed results to {json_path}")

        # Save human-readable summary
        txt_path = self.output_dir / "stress_summary.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BLACK-SWAN STRESS TEST RESULTS\n")
            f.write(f"Batch ID: {summary['batch_id']}\n")
            f.write(f"Generated: {summary['timestamp']}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Successful: {summary['successful']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Skipped: {summary['skipped']}\n\n")

            f.write(f"Periods Tested: {', '.join(summary['periods_tested'])}\n")
            f.write(f"Symbols Tested: {', '.join(summary['symbols_tested'])}\n\n")

            if "aggregate_metrics" in summary:
                f.write("AGGREGATE RESILIENCE METRICS\n")
                f.write("-" * 80 + "\n")
                agg = summary["aggregate_metrics"]
                f.write(f"Mean Cumulative Return: {agg['mean_cumulative_return']:.4f}\n")
                f.write(f"Mean Max Drawdown: {agg['mean_max_drawdown']:.4f}\n")
                f.write(f"Mean Hit Rate: {agg['mean_hit_rate']:.2%}\n")
                f.write(f"Mean Sharpe Ratio: {agg['mean_sharpe_ratio']:.4f}\n")
                f.write(f"Worst Drawdown: {agg['worst_drawdown']:.4f}\n")
                f.write(f"Best Sharpe: {agg['best_sharpe']:.4f}\n")
                f.write(f"Worst Sharpe: {agg['worst_sharpe']:.4f}\n\n")

            # Per-test details
            f.write("DETAILED RESULTS BY TEST\n")
            f.write("=" * 80 + "\n")
            for result in summary["results"]:
                if result["status"] != "success":
                    continue

                f.write(f"\n{result['symbol']} - {result['period_name']}\n")
                f.write(f"Period: {result['start_date']} to {result['end_date']} ({result['severity']})\n")
                f.write("-" * 40 + "\n")
                m = result["metrics"]
                f.write(f"  Predictions: {m['total_predictions']}\n")
                f.write(f"  Cumulative Return: {m['cumulative_return']:.4f}\n")
                f.write(f"  Max Drawdown: {m['max_drawdown']:.4f}\n")
                f.write(f"  Hit Rate: {m['hit_rate']:.2%}\n")
                f.write(f"  Sharpe Ratio: {m['sharpe_ratio']:.4f}\n")
                f.write(f"  Recovery Time: {m['recovery_time_days']} days\n")

        logger.info(f"Saved human-readable summary to {txt_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run black-swan stress tests (US-028 Phase 7 Initiative 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--batch-id",
        required=True,
        help="Batch ID of models to test",
    )

    parser.add_argument(
        "--periods",
        nargs="+",
        help="Specific stress period IDs to test (e.g., covid_crash_2020)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all stress periods",
    )

    parser.add_argument(
        "--severity",
        nargs="+",
        choices=["extreme", "high", "medium", "low"],
        default=["extreme", "high"],
        help="Filter by severity levels",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: release/stress_tests_<batch_id>)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load settings
    settings = Settings()  # type: ignore[call-arg]

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("release") / f"stress_tests_{args.batch_id}"

    # Load stress periods
    try:
        if args.all:
            stress_periods = settings.get_stress_periods()
        elif args.periods:
            stress_periods = settings.get_stress_periods(period_ids=args.periods)
        else:
            # Default: test high/extreme severity
            stress_periods = settings.get_stress_periods(severity_filter=args.severity)
    except Exception as e:
        logger.error(f"Failed to load stress periods: {e}")
        return 1

    if not stress_periods:
        logger.error("No stress periods to test")
        return 1

    logger.info(f"Testing {len(stress_periods)} stress periods")
    for period in stress_periods:
        logger.info(f"  - {period['id']}: {period['name']} ({period['severity']})")

    # Run stress tests
    runner = StressTestRunner(args.batch_id, output_dir, settings)
    summary = runner.run_all_stress_tests(stress_periods)

    if summary.get("status") == "failed":
        logger.error(f"Stress tests failed: {summary.get('error')}")
        return 1

    # Save results
    runner.save_results(summary)

    logger.info(f"✓ Stress tests complete: {summary['successful']}/{summary['total_tests']} successful")
    if summary.get("failed", 0) > 0:
        logger.warning(f"  {summary['failed']} tests failed")
    if summary.get("skipped", 0) > 0:
        logger.info(f"  {summary['skipped']} tests skipped (no data)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
