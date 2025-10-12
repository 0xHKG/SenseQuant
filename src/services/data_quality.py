"""Data Quality Service for historical and sentiment data validation (US-024 Phase 6).

Scans cached historical OHLCV and sentiment snapshot files to compute data quality metrics,
detect anomalies, and provide summary APIs for dashboard consumption.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class DataQualityService:
    """Service for data quality scanning and metrics computation.

    Scans historical OHLCV CSVs and sentiment JSONL files to detect:
    - Missing bars (gaps in timestamps)
    - Duplicate timestamps
    - Zero-volume bars
    - Sentiment gaps
    - Validation failures
    """

    def __init__(self, historical_dir: Path, sentiment_dir: Path):
        """Initialize data quality service.

        Args:
            historical_dir: Directory containing historical OHLCV data
            sentiment_dir: Directory containing sentiment snapshots
        """
        self.historical_dir = historical_dir
        self.sentiment_dir = sentiment_dir

    def scan_historical_quality(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Scan historical OHLCV data quality for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional

        Returns:
            Dict with quality metrics:
                - total_files: Number of CSV files
                - total_bars: Total bars across all files
                - missing_files: Files that should exist but don't
                - duplicate_timestamps: Number of duplicate timestamps
                - zero_volume_bars: Number of bars with zero volume
                - validation_errors: List of validation errors
        """
        symbol_dir = self.historical_dir / symbol
        if not symbol_dir.exists():
            return {
                "symbol": symbol,
                "total_files": 0,
                "total_bars": 0,
                "missing_files": 0,
                "duplicate_timestamps": 0,
                "zero_volume_bars": 0,
                "validation_errors": ["Symbol directory not found"],
            }

        metrics = {
            "symbol": symbol,
            "total_files": 0,
            "total_bars": 0,
            "missing_files": 0,
            "duplicate_timestamps": 0,
            "zero_volume_bars": 0,
            "validation_errors": [],
        }

        # Find all CSV files
        csv_files = list(symbol_dir.rglob("*.csv"))
        metrics["total_files"] = len(csv_files)

        # Scan each file
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Count bars
                metrics["total_bars"] += len(df)

                # Check for duplicates
                if "timestamp" in df.columns:
                    duplicates = df["timestamp"].duplicated().sum()
                    metrics["duplicate_timestamps"] += duplicates

                # Check for zero volume
                if "volume" in df.columns:
                    zero_volume = (df["volume"] == 0).sum()
                    metrics["zero_volume_bars"] += zero_volume

                # Validate required columns
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    metrics["validation_errors"].append(
                        f"{csv_file.name}: Missing columns {missing_cols}"
                    )

            except Exception as e:
                metrics["validation_errors"].append(f"{csv_file.name}: {str(e)}")

        # Check for missing files in date range if provided
        if start_date and end_date:
            expected_dates = self._get_expected_dates(start_date, end_date)
            actual_files = {f.stem for f in csv_files}
            missing_dates = [d for d in expected_dates if d not in actual_files]
            metrics["missing_files"] = len(missing_dates)

        return metrics

    def scan_sentiment_quality(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Scan sentiment snapshot quality for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional

        Returns:
            Dict with quality metrics:
                - total_files: Number of JSONL files
                - total_snapshots: Total snapshots across all files
                - missing_files: Files that should exist but don't
                - invalid_scores: Snapshots with invalid sentiment scores
                - low_confidence: Snapshots with low confidence (<0.5)
                - validation_errors: List of validation errors
        """
        symbol_dir = self.sentiment_dir / symbol
        if not symbol_dir.exists():
            return {
                "symbol": symbol,
                "total_files": 0,
                "total_snapshots": 0,
                "missing_files": 0,
                "invalid_scores": 0,
                "low_confidence": 0,
                "validation_errors": ["Symbol directory not found"],
            }

        metrics = {
            "symbol": symbol,
            "total_files": 0,
            "total_snapshots": 0,
            "missing_files": 0,
            "invalid_scores": 0,
            "low_confidence": 0,
            "validation_errors": [],
        }

        # Find all JSONL files
        jsonl_files = list(symbol_dir.glob("*.jsonl"))
        metrics["total_files"] = len(jsonl_files)

        # Scan each file
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file) as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            metrics["total_snapshots"] += 1

                            # Validate score range
                            score = data.get("score", 0)
                            if score < -1.0 or score > 1.0:
                                metrics["invalid_scores"] += 1

                            # Check confidence
                            confidence = data.get("confidence", 0)
                            if confidence < 0.5:
                                metrics["low_confidence"] += 1

                        except json.JSONDecodeError:
                            metrics["validation_errors"].append(
                                f"{jsonl_file.name}:L{line_num}: Invalid JSON"
                            )

            except Exception as e:
                metrics["validation_errors"].append(f"{jsonl_file.name}: {str(e)}")

        # Check for missing files in date range if provided
        if start_date and end_date:
            expected_dates = self._get_expected_dates(start_date, end_date)
            actual_files = {f.stem for f in jsonl_files}
            missing_dates = [d for d in expected_dates if d not in actual_files]
            metrics["missing_files"] = len(missing_dates)

        return metrics

    def get_summary_for_all_symbols(self) -> dict[str, Any]:
        """Get data quality summary for all symbols.

        Returns:
            Dict with overall quality metrics and per-symbol summaries
        """
        # Find all symbols in historical directory
        historical_symbols = (
            {d.name for d in self.historical_dir.iterdir() if d.is_dir()}
            if self.historical_dir.exists()
            else set()
        )

        # Find all symbols in sentiment directory
        sentiment_symbols = (
            {d.name for d in self.sentiment_dir.iterdir() if d.is_dir()}
            if self.sentiment_dir.exists()
            else set()
        )

        all_symbols = historical_symbols | sentiment_symbols

        summary = {
            "scan_timestamp": datetime.now().isoformat(),
            "total_symbols": len(all_symbols),
            "symbols": {},
        }

        for symbol in sorted(all_symbols):
            hist_metrics = self.scan_historical_quality(symbol)
            sent_metrics = self.scan_sentiment_quality(symbol)

            summary["symbols"][symbol] = {
                "historical": hist_metrics,
                "sentiment": sent_metrics,
            }

        return summary

    def check_quality_thresholds(
        self, metrics: dict[str, Any], thresholds: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Check if quality metrics exceed alert thresholds.

        Args:
            metrics: Quality metrics dict
            thresholds: Dict with threshold values

        Returns:
            List of alerts (empty if all thresholds pass)
        """
        alerts = []

        # Check missing files threshold
        if "missing_files" in metrics:
            threshold = thresholds.get("max_missing_files", 10)
            if metrics["missing_files"] > threshold:
                alerts.append(
                    {
                        "severity": "warning",
                        "metric": "missing_files",
                        "value": metrics["missing_files"],
                        "threshold": threshold,
                        "message": f"Missing {metrics['missing_files']} files (threshold: {threshold})",
                    }
                )

        # Check duplicate timestamps threshold
        if "duplicate_timestamps" in metrics:
            threshold = thresholds.get("max_duplicate_timestamps", 100)
            if metrics["duplicate_timestamps"] > threshold:
                alerts.append(
                    {
                        "severity": "warning",
                        "metric": "duplicate_timestamps",
                        "value": metrics["duplicate_timestamps"],
                        "threshold": threshold,
                        "message": f"Found {metrics['duplicate_timestamps']} duplicate timestamps",
                    }
                )

        # Check zero volume bars threshold
        if "zero_volume_bars" in metrics:
            threshold = thresholds.get("max_zero_volume_bars", 50)
            if metrics["zero_volume_bars"] > threshold:
                alerts.append(
                    {
                        "severity": "warning",
                        "metric": "zero_volume_bars",
                        "value": metrics["zero_volume_bars"],
                        "threshold": threshold,
                        "message": f"Found {metrics['zero_volume_bars']} zero-volume bars",
                    }
                )

        # Check validation errors
        if "validation_errors" in metrics and metrics["validation_errors"]:
            alerts.append(
                {
                    "severity": "error",
                    "metric": "validation_errors",
                    "value": len(metrics["validation_errors"]),
                    "threshold": 0,
                    "message": f"Found {len(metrics['validation_errors'])} validation errors",
                }
            )

        return alerts

    def _get_expected_dates(self, start_date: str, end_date: str) -> list[str]:
        """Get list of expected dates in range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of date strings (YYYY-MM-DD)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        return dates
